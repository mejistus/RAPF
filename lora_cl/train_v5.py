"""
V5: Train from scratch — SupCon(PK P=8,K=8) → CE
No V0 checkpoint. Pure contrastive learning shapes LoRA features first.
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
    'output_dir': 'output-5',
    'lora_rank': 8,
    'proj_dim': 128,
    'P': 8, 'K': 8,  # batch=64
    # Stage 1: SupCon from scratch
    'stage1_epochs': 80,
    'stage1_lr': 1e-3,
    'stage1_lr_lora': 5e-4,
    'temperature': 0.1,
    'warmup_epochs': 5,
    # Stage 2: CE
    'stage2_epochs': 30,
    'stage2_lr': 0.01,
    'label_smoothing': 0.1,
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
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

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
    topk_labels = tl[(ef @ tf.T).topk(k, dim=1)[1]]
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
    train_loader_plain = DataLoader(train_ds_plain, batch_size=64, shuffle=True,
                                    num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    # Fresh model — NO checkpoint
    model = DINOv2ContrastiveModel(num_classes, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']).to(device)
    print(f"Fresh model (no checkpoint). PK: P={cfg['P']}, K={cfg['K']} -> {bs}/batch")

    knn_init = knn_eval(model, train_loader_plain, test_loader, device)
    print(f"KNN initial (frozen DINOv2): {knn_init:.2f}%")

    scaler = GradScaler()
    supcon_fn = SupConLoss(temperature=cfg['temperature'])

    metrics = {k: [] for k in ['epoch','stage','loss','knn_acc','test_acc','lr','pos_sim','neg_sim','per_class_acc']}

    # ==================== Stage 1: SupCon from scratch ====================
    print(f"\n{'='*60}")
    print(f"Stage 1: SupCon from scratch — {cfg['stage1_epochs']} epochs")
    print(f"{'='*60}")

    for p in model.classifier.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW([
        {'params': model.proj_head.parameters(), 'lr': cfg['stage1_lr'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params, 'lr': cfg['stage1_lr_lora'], 'weight_decay': 1e-4},
    ])
    warmup = cfg['warmup_epochs']
    s1ep = cfg['stage1_epochs']
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (s1ep - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_knn = knn_init
    for epoch in range(s1ep):
        model.train(); model.classifier.eval()
        ep_loss = 0; nb = 0; pos_s = []; neg_s = []
        optimizer.zero_grad()
        for images, labels in tqdm(train_loader_pk, desc=f"S1 {epoch+1}/{s1ep}", leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                _, proj, _ = model(images)
                loss = supcon_fn(proj, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            ep_loss += loss.item(); nb += 1
            with torch.no_grad():
                sim = proj.float() @ proj.float().T
                pm = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float(); pm.fill_diagonal_(0)
                nm = 1 - torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                pos_s.append((sim*pm).sum().item()/pm.sum().clamp(min=1).item())
                neg_s.append((sim*nm).sum().item()/nm.sum().clamp(min=1).item())
        scheduler.step()

        knn = 0
        if (epoch+1) % 5 == 0 or epoch == 0 or epoch == s1ep - 1:
            knn = knn_eval(model, train_loader_plain, test_loader, device)
            if knn > best_knn:
                best_knn = knn
                torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),'knn_acc':knn},
                           os.path.join(cfg['output_dir'], 'best_stage1.pth'))

        metrics['epoch'].append(epoch+1); metrics['stage'].append(1)
        metrics['loss'].append(ep_loss/nb); metrics['knn_acc'].append(knn)
        metrics['test_acc'].append(0); metrics['lr'].append(optimizer.param_groups[0]['lr'])
        metrics['pos_sim'].append(np.mean(pos_s)); metrics['neg_sim'].append(np.mean(neg_s))

        kstr = f"knn={knn:.1f}%" if knn > 0 else ""
        print(f"  S1 {epoch+1:>3d}: cl={ep_loss/nb:.3f} pos={np.mean(pos_s):.3f} neg={np.mean(neg_s):.3f} {kstr} best_knn={best_knn:.1f}%")

    # Load best stage1
    s1_path = os.path.join(cfg['output_dir'], 'best_stage1.pth')
    if os.path.exists(s1_path):
        model.load_state_dict(torch.load(s1_path, map_location=device, weights_only=False)['model_state_dict'])
    print(f"\nStage 1 done. KNN: {knn_init:.1f}% → {best_knn:.1f}% ({best_knn-knn_init:+.1f}%)")

    # ==================== Stage 2: CE ====================
    print(f"\n{'='*60}")
    print(f"Stage 2: CE classifier — {cfg['stage2_epochs']} epochs")
    print(f"{'='*60}")

    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    opt2 = torch.optim.SGD(model.classifier.parameters(), lr=cfg['stage2_lr'], momentum=0.9, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg['stage2_epochs'])
    ce_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    best_acc = 0
    for epoch in range(cfg['stage2_epochs']):
        model.train(); model.encoder.eval(); model.proj_head.eval()
        ep_ce = 0; correct = total = 0; nb = 0
        for images, labels in tqdm(train_loader_plain, desc=f"S2 {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            opt2.zero_grad()
            with autocast():
                with torch.no_grad(): feat = model.encoder(images)
                logits = model.classifier(feat.float())
                loss = ce_fn(logits, labels)
            scaler.scale(loss).backward(); scaler.step(opt2); scaler.update()
            ep_ce += loss.item()
            correct += (model.classifier(feat.float()).argmax(1)==labels).sum().item()
            total += len(labels); nb += 1
        sch2.step()
        train_acc = 100*correct/total

        model.eval(); tc = tt = 0
        with torch.no_grad(), autocast():
            for imgs, lbls in test_loader:
                tc += (model(imgs.to(device))[2].argmax(1)==lbls.to(device)).sum().item()
                tt += len(lbls)
        test_acc = 100*tc/tt
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch':s1ep+epoch+1,'model_state_dict':model.state_dict(),
                        'test_acc':test_acc,'config':cfg},
                       os.path.join(cfg['output_dir'], 'best_model_5.pth'))

        metrics['epoch'].append(s1ep+epoch+1); metrics['stage'].append(2)
        metrics['loss'].append(ep_ce/nb); metrics['knn_acc'].append(0)
        metrics['test_acc'].append(test_acc); metrics['lr'].append(opt2.param_groups[0]['lr'])
        metrics['pos_sim'].append(0); metrics['neg_sim'].append(0)
        if (epoch+1)%10==0 or epoch==cfg['stage2_epochs']-1:
            metrics['per_class_acc'].append(compute_per_class_accuracy(model, test_loader, num_classes, device))

        print(f"  S2 {epoch+1:>3d}: ce={ep_ce/nb:.3f} train={train_acc:.1f}% test={test_acc:.2f}% best={best_acc:.2f}%")

    # ==================== Summary ====================
    print(f"\n{'='*60}")
    print(f"V5 complete! KNN: {knn_init:.1f}% → {best_knn:.1f}%, Test: {best_acc:.2f}%")
    print(f"{'='*60}")

    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    bp = os.path.join(cfg['output_dir'], 'best_model_5.pth')
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    tl_vis = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    save_badcases(model, tl_vis, class_names, cfg['output_dir'], device)
    plot_confusion_matrix(model, tl_vis, class_names, cfg['output_dir'], device)
    plot_gradcam(model, test_ds, class_names, os.path.join(cfg['output_dir'], 'gradcam.png'), f'V5 GradCAM ({best_acc:.1f}%)')

    pca = compute_per_class_accuracy(model, tl_vis, num_classes, device)
    summary = {'knn_init': knn_init, 'knn_best': best_knn, 'best_test_acc': best_acc,
               'classes_below_50': sum(1 for a in pca if a < 50),
               'worst_10': [(class_names[i], f"{a:.1f}%") for i,a in sorted(enumerate(pca), key=lambda x:x[1])[:10]]}
    with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {cfg['output_dir']}/")

if __name__ == '__main__':
    train()
