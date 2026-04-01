"""
V4: Improved SupCon fine-tune on V0 checkpoint.
Improvements over V3:
- CosineAnnealingWarmRestarts (3 cycles) for better exploration
- P=6, K=10 -> batch=60, 9 positives per sample (more than V3's 7)
- Stronger augmentation (preserve structure: no vertical flip for aircraft)
- Longer Stage 1 (60 ep) + Stage 2 with Mixup (40 ep)
- Label smoothing tuned to 0.15
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import DINOv2ContrastiveModel, SupConLoss
from pk_sampler import PKSampler
from visualize import save_badcases, plot_confusion_matrix
from gradcam_vis import plot_grid as plot_gradcam

CONFIG = {
    'data_root': '../data/fgvc_aircraft',
    'dino_weights': '/mnt/datasets/dinov2_vitb14.pth',
    'resume_from': 'output/best_model.pth',
    'output_dir': 'output-4',
    'lora_rank': 8,
    'proj_dim': 128,
    # Stage 1
    'P': 6, 'K': 10,           # batch=60, 9 pos + 50 neg per sample
    'stage1_epochs': 60,
    'stage1_lr': 8e-4,
    'stage1_lr_lora': 3e-4,
    'temperature': 0.07,        # sharper than V3's 0.1
    'T_0': 20,                  # warm restart period
    # Stage 2
    'stage2_epochs': 40,
    'stage2_lr': 0.01,
    'stage2_batch_size': 64,
    'label_smoothing': 0.15,
    'mixup_alpha': 0.2,
    # Common
    'weight_decay': 5e-4,
    'num_workers': 4,
    'seed': 42,
    'grad_clip': 1.0,
}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_cl_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomGrayscale(p=0.25),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def knn_eval(model, train_loader, test_loader, device, k=20):
    model.eval()
    def extract(loader):
        feats, labels = [], []
        with torch.no_grad(), autocast():
            for imgs, lbls in loader:
                f = model.encoder(imgs.to(device)).float()
                feats.append(F.normalize(f, dim=-1).cpu())
                labels.append(lbls)
        return torch.cat(feats), torch.cat(labels)
    tf, tl = extract(train_loader)
    ef, el = extract(test_loader)
    topk_labels = tl[( ef @ tf.T).topk(k, dim=1)[1]]
    return (torch.mode(topk_labels, dim=1)[0] == el).float().mean().item() * 100


def compute_per_class_accuracy(model, loader, num_classes, device):
    correct = torch.zeros(num_classes); total = torch.zeros(num_classes)
    model.eval()
    with torch.no_grad(), autocast():
        for images, labels in loader:
            preds = model(images.to(device))[2].argmax(1).cpu()
            for c in range(num_classes):
                m = labels == c; total[c] += m.sum(); correct[c] += (preds[m] == c).sum()
    return (correct / total.clamp(min=1) * 100).tolist()


def train():
    cfg = CONFIG
    seed_everything(cfg['seed'])
    device = torch.device("cuda")
    os.makedirs(cfg['output_dir'], exist_ok=True)
    with open(os.path.join(cfg['output_dir'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data
    cl_tf = get_cl_transforms()
    test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    train_ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), cl_tf)
    train_ds_plain = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), test_tf)
    test_ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'test'), test_tf)
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    bs = cfg['P'] * cfg['K']
    pk = PKSampler(train_ds.targets, p=cfg['P'], k=cfg['K'])
    train_loader_pk = DataLoader(train_ds, batch_sampler=pk, num_workers=cfg['num_workers'], pin_memory=True)
    train_loader_plain = DataLoader(train_ds_plain, batch_size=cfg['stage2_batch_size'], shuffle=True,
                                    num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    print(f"PK: P={cfg['P']}, K={cfg['K']} -> {bs}/batch")
    print(f"Each sample: {cfg['K']-1} positives + {(cfg['P']-1)*cfg['K']} negatives")

    # Model
    model = DINOv2ContrastiveModel(num_classes, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']).to(device)
    ckpt = torch.load(cfg['resume_from'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded V0: acc={ckpt['test_acc']:.2f}%")

    knn_before = knn_eval(model, train_loader_plain, test_loader, device)
    print(f"KNN before: {knn_before:.2f}%")

    scaler = GradScaler()
    supcon_fn = SupConLoss(temperature=cfg['temperature'])

    metrics = {k: [] for k in ['epoch','stage','loss','knn_acc','test_acc','lr','pos_sim','neg_sim','per_class_acc']}

    # ==================== Stage 1: SupCon with Warm Restarts ====================
    print(f"\n{'='*60}")
    print(f"Stage 1: SupCon ({cfg['stage1_epochs']}ep, CosineWarmRestarts T_0={cfg['T_0']})")
    print(f"{'='*60}")

    for p in model.classifier.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW([
        {'params': model.proj_head.parameters(), 'lr': cfg['stage1_lr'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params, 'lr': cfg['stage1_lr_lora'], 'weight_decay': 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=1, eta_min=1e-6)

    best_knn = knn_before
    for epoch in range(cfg['stage1_epochs']):
        model.train(); model.classifier.eval()
        ep_loss = 0; nb = 0; pos_s = []; neg_s = []
        optimizer.zero_grad()
        for images, labels in tqdm(train_loader_pk, desc=f"S1 {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                _, proj, _ = model(images)
                loss = supcon_fn(proj, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(epoch + nb / len(train_loader_pk))
            ep_loss += loss.item(); nb += 1
            with torch.no_grad():
                sim = proj.float() @ proj.float().T
                pm = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float(); pm.fill_diagonal_(0)
                nm = 1 - torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                pos_s.append((sim*pm).sum().item()/pm.sum().clamp(min=1).item())
                neg_s.append((sim*nm).sum().item()/nm.sum().clamp(min=1).item())

        knn = 0
        if (epoch+1) % 5 == 0 or epoch == 0:
            knn = knn_eval(model, train_loader_plain, test_loader, device)
            if knn > best_knn:
                best_knn = knn
                torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),'knn_acc':knn,'config':cfg},
                           os.path.join(cfg['output_dir'], 'best_stage1.pth'))

        metrics['epoch'].append(epoch+1); metrics['stage'].append(1)
        metrics['loss'].append(ep_loss/nb); metrics['knn_acc'].append(knn)
        metrics['test_acc'].append(0); metrics['lr'].append(optimizer.param_groups[0]['lr'])
        metrics['pos_sim'].append(np.mean(pos_s)); metrics['neg_sim'].append(np.mean(neg_s))

        knn_s = f"knn={knn:.1f}%" if knn>0 else ""
        print(f"  S1 {epoch+1:>3d}: cl={ep_loss/nb:.3f} pos={np.mean(pos_s):.3f} neg={np.mean(neg_s):.3f} lr={optimizer.param_groups[0]['lr']:.6f} {knn_s} best={best_knn:.1f}%")

    # Load best stage1
    s1_path = os.path.join(cfg['output_dir'], 'best_stage1.pth')
    if os.path.exists(s1_path):
        model.load_state_dict(torch.load(s1_path, map_location=device, weights_only=False)['model_state_dict'])
        print(f"Loaded best S1 (knn={best_knn:.1f}%)")

    print(f"\nStage 1 done. KNN: {knn_before:.1f}% -> {best_knn:.1f}% ({best_knn-knn_before:+.1f}%)")

    # ==================== Stage 2: CE + Mixup ====================
    print(f"\n{'='*60}")
    print(f"Stage 2: CE+Mixup ({cfg['stage2_epochs']}ep)")
    print(f"{'='*60}")

    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    opt2 = torch.optim.SGD(model.classifier.parameters(), lr=cfg['stage2_lr'], momentum=0.9, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg['stage2_epochs'], eta_min=1e-5)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    best_acc = 0
    for epoch in range(cfg['stage2_epochs']):
        model.train(); model.encoder.eval(); model.proj_head.eval()
        ep_ce = 0; correct = total = 0; nb = 0
        for images, labels in tqdm(train_loader_plain, desc=f"S2 {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            opt2.zero_grad()
            with autocast():
                with torch.no_grad():
                    feat = model.encoder(images).float()
                # Mixup on features
                if cfg['mixup_alpha'] > 0 and random.random() < 0.5:
                    feat_m, la, lb, lam = mixup_data(feat, labels, cfg['mixup_alpha'])
                    logits = model.classifier(feat_m)
                    loss = lam * ce_fn(logits, la) + (1-lam) * ce_fn(logits, lb)
                else:
                    logits = model.classifier(feat)
                    loss = ce_fn(logits, labels)
            scaler.scale(loss).backward(); scaler.step(opt2); scaler.update()
            ep_ce += loss.item(); correct += (model.classifier(feat).argmax(1)==labels).sum().item()
            total += len(labels); nb += 1
        sch2.step()
        train_acc = 100*correct/total

        model.eval()
        tc = tt = 0
        with torch.no_grad(), autocast():
            for imgs, lbls in test_loader:
                tc += (model(imgs.to(device))[2].argmax(1)==lbls.to(device)).sum().item()
                tt += len(lbls)
        test_acc = 100*tc/tt
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch':cfg['stage1_epochs']+epoch+1,'model_state_dict':model.state_dict(),
                        'test_acc':test_acc,'config':cfg},
                       os.path.join(cfg['output_dir'], 'best_model_4.pth'))

        metrics['epoch'].append(cfg['stage1_epochs']+epoch+1); metrics['stage'].append(2)
        metrics['loss'].append(ep_ce/nb); metrics['knn_acc'].append(0)
        metrics['test_acc'].append(test_acc); metrics['lr'].append(opt2.param_groups[0]['lr'])
        metrics['pos_sim'].append(0); metrics['neg_sim'].append(0)
        if (epoch+1)%10==0 or epoch==cfg['stage2_epochs']-1:
            metrics['per_class_acc'].append(compute_per_class_accuracy(model, test_loader, num_classes, device))

        print(f"  S2 {epoch+1:>3d}: ce={ep_ce/nb:.3f} train={train_acc:.1f}% test={test_acc:.2f}% best={best_acc:.2f}%")

    # ==================== Summary ====================
    print(f"\n{'='*60}")
    print(f"V4 complete! V0={ckpt['test_acc']:.2f}% -> V4={best_acc:.2f}% ({best_acc-ckpt['test_acc']:+.2f}%)")
    print(f"KNN: {knn_before:.1f}% -> {best_knn:.1f}%")
    print(f"{'='*60}")

    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Visualize
    bp = os.path.join(cfg['output_dir'], 'best_model_4.pth')
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    tl_vis = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    save_badcases(model, tl_vis, class_names, cfg['output_dir'], device)
    plot_confusion_matrix(model, tl_vis, class_names, cfg['output_dir'], device)
    plot_gradcam(model, test_ds, class_names, os.path.join(cfg['output_dir'], 'gradcam.png'), f'V4 GradCAM ({best_acc:.1f}%)')

    pca = compute_per_class_accuracy(model, tl_vis, num_classes, device)
    summary = {'v0':ckpt['test_acc'],'v4_best':best_acc,'knn_before':knn_before,'knn_after':best_knn,
               'classes_below_50':sum(1 for a in pca if a<50),
               'worst_10':[(class_names[i],f"{a:.1f}%") for i,a in sorted(enumerate(pca),key=lambda x:x[1])[:10]]}
    with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {cfg['output_dir']}/")

if __name__ == '__main__':
    train()
