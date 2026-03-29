"""
V3: Continue from V0 checkpoint, SupCon-only fine-tune with PK(P=8, K=16).
Single view (no TwoView needed — K=16 same-class images ARE natural positive pairs).
Each sample: 15 positives + 112 negatives in batch of 128.
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
from visualize import plot_training_curves, plot_confusion_matrix, save_badcases
from gradcam_vis import ViTGradCAM, plot_grid as plot_gradcam

CONFIG = {
    'data_root': '../data/fgvc_aircraft',
    'dino_weights': '/mnt/datasets/dinov2_vitb14.pth',
    'resume_from': 'output/best_model.pth',
    'output_dir': 'output-3',
    'lora_rank': 8,
    'proj_dim': 128,
    'P': 8,  # 8 classes per batch
    'K': 8,  # 8 samples per class -> batch=64
    'accum_steps': 1,   # no accumulation needed
    'epochs': 40,
    'lr_proj': 5e-4,
    'lr_lora': 2e-4,
    'temperature': 0.07,
    'warmup_epochs': 3,
    'weight_decay': 5e-4,
    'num_workers': 4,
    'seed': 42,
    'grad_clip': 1.0,
    # Stage 2: quick linear probe after contrastive
    'stage2_epochs': 30,
    'stage2_lr': 0.01,
}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_cl_transforms(img_size=224):
    """Augmentations specified by user."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.CenterCrop(img_size),   # after random resize, center crop back
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomRotation(15),
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
    train_f, train_l = extract(train_loader)
    test_f, test_l = extract(test_loader)
    sim = test_f @ train_f.T
    topk_labels = train_l[sim.topk(k, dim=1)[1]]
    preds = torch.mode(topk_labels, dim=1)[0]
    return (preds == test_l).float().mean().item() * 100


def compute_per_class_accuracy(model, loader, num_classes, device):
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    model.eval()
    with torch.no_grad(), autocast():
        for images, labels in loader:
            images = images.to(device)
            _, _, logits = model(images)
            preds = logits.argmax(1).cpu()
            for c in range(num_classes):
                mask = labels == c
                total[c] += mask.sum()
                correct[c] += (preds[mask] == c).sum()
    return (correct / total.clamp(min=1) * 100).tolist()


def train():
    cfg = CONFIG
    seed_everything(cfg['seed'])
    device = torch.device("cuda")
    os.makedirs(cfg['output_dir'], exist_ok=True)
    with open(os.path.join(cfg['output_dir'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data
    cl_transform = get_cl_transforms()
    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), cl_transform)
    train_ds_plain = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), test_transform)
    test_ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'test'), test_transform)
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    batch_size = cfg['P'] * cfg['K']  # 128
    sub_batch = batch_size // cfg['accum_steps']  # 64

    # PKSampler returns batches of 128, but we split into sub-batches of 64 for GPU
    pk_sampler = PKSampler(train_ds.targets, p=cfg['P'], k=cfg['K'])
    train_loader = DataLoader(train_ds, batch_sampler=pk_sampler,
                              num_workers=cfg['num_workers'], pin_memory=True)
    train_loader_plain = DataLoader(train_ds_plain, batch_size=64, shuffle=True,
                                    num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True)

    print(f"PK: P={cfg['P']}, K={cfg['K']} -> {batch_size}/batch")
    print(f"Each sample: {cfg['K']-1} positives + {(cfg['P']-1)*cfg['K']} negatives")
    print(f"Accum steps: {cfg['accum_steps']} (sub-batch={sub_batch})")

    # Model — load V0 checkpoint
    model = DINOv2ContrastiveModel(
        num_classes, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']
    ).to(device)
    ckpt = torch.load(cfg['resume_from'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded V0: epoch={ckpt['epoch']}, acc={ckpt['test_acc']:.2f}%")

    # Baseline KNN before training
    knn_before = knn_eval(model, train_loader_plain, test_loader, device)
    print(f"KNN before training: {knn_before:.2f}%")

    # ==================== Stage 1: SupCon ====================
    print(f"\n{'='*60}")
    print(f"Stage 1: SupCon fine-tune on V0 — {cfg['epochs']} epochs")
    print(f"{'='*60}")

    for p in model.classifier.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': model.proj_head.parameters(), 'lr': cfg['lr_proj'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params, 'lr': cfg['lr_lora'], 'weight_decay': 1e-4},
    ])
    warmup = cfg['warmup_epochs']
    total_ep = cfg['epochs']
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (total_ep - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    supcon_fn = SupConLoss(temperature=cfg['temperature'])

    metrics = {k: [] for k in [
        'epoch', 'stage', 'cl_loss', 'ce_loss', 'train_acc', 'test_acc',
        'knn_acc', 'lr', 'pos_sim_mean', 'neg_sim_mean',
        'num_pos_per_sample', 'per_class_acc',
    ]}
    accum = cfg['accum_steps']
    best_knn = knn_before

    for epoch in range(total_ep):
        model.train()
        model.classifier.eval()

        ep_cl = 0; n_batches = 0
        pos_sims = []; neg_sims = []; n_pos_list = []

        pbar = tqdm(train_loader, desc=f"S1 {epoch+1}/{total_ep}", leave=False)
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            B = len(labels)

            with autocast():
                _, proj, _ = model(images)
                loss = supcon_fn(proj, labels)

            scaler.scale(loss / accum).backward()

            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            ep_cl += loss.item(); n_batches += 1

            with torch.no_grad():
                sim = proj.float() @ proj.float().T
                pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                pos_mask.fill_diagonal_(0)
                neg_mask = 1 - torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                n_pos = pos_mask.sum(1).mean().item()
                n_pos_list.append(n_pos)
                pos_sims.append((sim * pos_mask).sum().item() / pos_mask.sum().clamp(min=1).item())
                neg_sims.append((sim * neg_mask).sum().item() / neg_mask.sum().clamp(min=1).item())

            pbar.set_postfix(cl=f"{loss.item():.3f}",
                           pos=f"{pos_sims[-1]:.3f}", neg=f"{neg_sims[-1]:.3f}",
                           npos=f"{n_pos:.0f}")

        scheduler.step()

        knn_acc = 0
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == total_ep - 1:
            knn_acc = knn_eval(model, train_loader_plain, test_loader, device)
            if knn_acc > best_knn:
                best_knn = knn_acc
                torch.save({
                    'epoch': epoch + 1, 'stage': 1,
                    'model_state_dict': model.state_dict(),
                    'knn_acc': knn_acc,
                    'config': cfg,
                }, os.path.join(cfg['output_dir'], 'best_stage1.pth'))

        metrics['epoch'].append(epoch + 1)
        metrics['stage'].append(1)
        metrics['cl_loss'].append(ep_cl / n_batches)
        metrics['ce_loss'].append(0)
        metrics['train_acc'].append(0)
        metrics['test_acc'].append(0)
        metrics['knn_acc'].append(knn_acc)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        metrics['pos_sim_mean'].append(np.mean(pos_sims))
        metrics['neg_sim_mean'].append(np.mean(neg_sims))
        metrics['num_pos_per_sample'].append(np.mean(n_pos_list))

        knn_str = f"knn={knn_acc:.1f}%" if knn_acc > 0 else ""
        print(f"  S1 {epoch+1:>3d}: cl={ep_cl/n_batches:.3f} | "
              f"pos={np.mean(pos_sims):.3f} neg={np.mean(neg_sims):.3f} "
              f"n_pos={np.mean(n_pos_list):.0f} | {knn_str} best_knn={best_knn:.1f}%")

    print(f"\nStage 1 done. KNN: {knn_before:.1f}% → {best_knn:.1f}% ({best_knn-knn_before:+.1f}%)")

    # Load best stage1
    best_s1 = os.path.join(cfg['output_dir'], 'best_stage1.pth')
    if os.path.exists(best_s1):
        ckpt_s1 = torch.load(best_s1, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_s1['model_state_dict'])
        print(f"Loaded best Stage1 (epoch {ckpt_s1['epoch']}, knn={ckpt_s1['knn_acc']:.1f}%)")

    # ==================== Stage 2: Linear probe ====================
    print(f"\n{'='*60}")
    print(f"Stage 2: Linear probe — {cfg['stage2_epochs']} epochs")
    print(f"{'='*60}")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt2 = torch.optim.SGD(model.classifier.parameters(), lr=cfg['stage2_lr'],
                           momentum=0.9, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg['stage2_epochs'])
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(cfg['stage2_epochs']):
        model.train(); model.encoder.eval(); model.proj_head.eval()
        ep_ce = 0; correct = total = 0; nb = 0
        for images, labels in tqdm(train_loader_plain, desc=f"S2 {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            opt2.zero_grad()
            with autocast():
                with torch.no_grad():
                    feat = model.encoder(images)
                logits = model.classifier(feat.float())
                loss = ce_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()
            ep_ce += loss.item(); correct += (logits.argmax(1)==labels).sum().item()
            total += len(labels); nb += 1
        sch2.step()
        train_acc = 100*correct/total

        model.eval()
        tc = tt = 0
        with torch.no_grad(), autocast():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                _, _, logits = model(imgs)
                tc += (logits.argmax(1)==lbls).sum().item()
                tt += len(lbls)
        test_acc = 100*tc/tt
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': total_ep + epoch + 1,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc, 'config': cfg,
            }, os.path.join(cfg['output_dir'], 'best_model_3.pth'))

        metrics['epoch'].append(total_ep + epoch + 1)
        metrics['stage'].append(2)
        metrics['cl_loss'].append(0); metrics['ce_loss'].append(ep_ce/nb)
        metrics['train_acc'].append(train_acc); metrics['test_acc'].append(test_acc)
        metrics['knn_acc'].append(0); metrics['lr'].append(opt2.param_groups[0]['lr'])
        metrics['pos_sim_mean'].append(0); metrics['neg_sim_mean'].append(0)
        metrics['num_pos_per_sample'].append(0)

        if (epoch+1) % 10 == 0 or epoch == cfg['stage2_epochs']-1:
            pca = compute_per_class_accuracy(model, test_loader, num_classes, device)
            metrics['per_class_acc'].append(pca)

        print(f"  S2 {epoch+1:>3d}: ce={ep_ce/nb:.3f} train={train_acc:.1f}% test={test_acc:.2f}% best={best_acc:.2f}%")

    # ==================== Summary ====================
    print(f"\n{'='*60}")
    print(f"V3 complete!")
    print(f"  V0 baseline:     78.67%")
    print(f"  KNN before→after: {knn_before:.1f}% → {best_knn:.1f}%")
    print(f"  V3 best test:    {best_acc:.2f}%")
    print(f"{'='*60}")

    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Visualize
    best_path = os.path.join(cfg['output_dir'], 'best_model_3.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()

    plot_training_curves(metrics, cfg['output_dir'])
    test_loader_vis = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    save_badcases(model, test_loader_vis, class_names, cfg['output_dir'], device)
    plot_confusion_matrix(model, test_loader_vis, class_names, cfg['output_dir'], device)

    # GradCAM
    indices = []; seen = set()
    for i, (_, l) in enumerate(test_ds):
        if l not in seen and len(indices) < 16:
            indices.append(i); seen.add(l)
    plot_gradcam(model, test_ds, class_names,
                 os.path.join(cfg['output_dir'], 'gradcam_diverse.png'),
                 title="V3 GradCAM")

    final_pca = compute_per_class_accuracy(model, test_loader_vis, num_classes, device)
    summary = {
        'v0_baseline': 78.67,
        'knn_before': knn_before, 'knn_after': best_knn,
        'best_test_acc': best_acc,
        'mean_per_class': np.mean(final_pca),
        'classes_below_50': sum(1 for a in final_pca if a < 50),
        'worst_10': [(class_names[i], f"{a:.1f}%") for i, a in sorted(enumerate(final_pca), key=lambda x: x[1])[:10]],
    }
    with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {cfg['output_dir']}/")


if __name__ == '__main__':
    train()
