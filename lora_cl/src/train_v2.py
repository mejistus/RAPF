"""
V2: Two-stage SupCon + CE with PK balanced sampling.
Stage 1 (50ep): SupCon only → LoRA + proj_head learn class-discriminative features
Stage 2 (30ep): Freeze encoder → classifier learns on fixed features
P=48 classes, K=1 per class → max class diversity per batch
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
from tqdm import tqdm
from torchvision import datasets as tv_ds

from model import DINOv2ContrastiveModel, SupConLoss
from dataset import TwoViewDataset, get_transforms
from pk_sampler import PKSampler
from visualize import (plot_attention_map, plot_training_curves,
                       save_badcases, plot_confusion_matrix)

CONFIG = {
    'data_root': '../data/fgvc_aircraft',
    'dino_weights': '/mnt/datasets/dinov2_vitb14.pth',
    'output_dir': 'output-2',
    'lora_rank': 8,
    'proj_dim': 128,
    # Stage 1: SupCon
    'P': 48,
    'K': 1,
    'stage1_epochs': 50,
    'stage1_lr': 1e-3,
    'stage1_lr_lora': 5e-4,
    'temperature': 0.1,
    'warmup_epochs': 5,
    # Stage 2: CE
    'stage2_epochs': 30,
    'stage2_lr': 5e-3,
    'stage2_batch_size': 64,
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
    torch.backends.cudnn.deterministic = True


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


def knn_eval(model, train_loader, test_loader, device, k=20):
    """Quick KNN accuracy on frozen features to monitor Stage 1 quality."""
    model.eval()
    train_feats, train_labels = [], []
    test_feats, test_labels = [], []
    with torch.no_grad(), autocast():
        for imgs, lbls in train_loader:
            feat = model.encoder(imgs.to(device)).float()
            train_feats.append(F.normalize(feat, dim=-1).cpu())
            train_labels.append(lbls)
        for imgs, lbls in test_loader:
            feat = model.encoder(imgs.to(device)).float()
            test_feats.append(F.normalize(feat, dim=-1).cpu())
            test_labels.append(lbls)
    train_feats = torch.cat(train_feats)
    train_labels = torch.cat(train_labels)
    test_feats = torch.cat(test_feats)
    test_labels = torch.cat(test_labels)

    sim = test_feats @ train_feats.T
    topk_idx = sim.topk(k, dim=1)[1]
    topk_labels = train_labels[topk_idx]
    preds = torch.mode(topk_labels, dim=1)[0]
    return (preds == test_labels).float().mean().item() * 100


def train():
    cfg = CONFIG
    seed_everything(cfg['seed'])
    device = torch.device("cuda")
    os.makedirs(cfg['output_dir'], exist_ok=True)

    with open(os.path.join(cfg['output_dir'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    train_ds = TwoViewDataset(os.path.join(cfg['data_root'], 'train'), train_transform)
    train_ds_plain = tv_ds.ImageFolder(os.path.join(cfg['data_root'], 'train'), test_transform)
    test_ds = tv_ds.ImageFolder(os.path.join(cfg['data_root'], 'test'), test_transform)
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    batch_size = cfg['P'] * cfg['K']  # 48
    pk_sampler = PKSampler(train_ds.targets, p=cfg['P'], k=cfg['K'])
    train_loader_pk = DataLoader(train_ds, batch_sampler=pk_sampler,
                                 num_workers=cfg['num_workers'], pin_memory=True)
    train_loader_plain = DataLoader(train_ds_plain, batch_size=cfg['stage2_batch_size'],
                                    shuffle=True, num_workers=cfg['num_workers'],
                                    pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True)

    print(f"PK: P={cfg['P']}, K={cfg['K']} -> {batch_size}/batch, {len(pk_sampler)} batches/epoch")
    print(f"2 views -> {batch_size * 2} projections in contrastive matrix")
    print(f"Each sample: 1 positive (twin view), {(cfg['P']-1)*cfg['K']*2-1} negatives")

    # Model
    model = DINOv2ContrastiveModel(
        num_classes, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']
    ).to(device)

    supcon_fn = SupConLoss(temperature=cfg['temperature'])
    scaler = GradScaler()

    metrics = {k: [] for k in [
        'epoch', 'stage', 'train_loss', 'cl_loss', 'ce_loss',
        'train_acc', 'test_acc', 'knn_acc', 'lr',
        'pos_sim_mean', 'neg_sim_mean',
        'per_class_acc', 'unique_classes_per_batch',
    ]}

    # ==================== Stage 1: SupCon Only ====================
    print(f"\n{'='*60}")
    print(f"Stage 1: SupCon only — {cfg['stage1_epochs']} epochs")
    print(f"Training: LoRA + proj_head | Frozen: classifier")
    print(f"{'='*60}")

    # Only train LoRA + proj_head, freeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = False

    s1_params = [
        {'params': model.proj_head.parameters(), 'lr': cfg['stage1_lr'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params, 'lr': cfg['stage1_lr_lora'], 'weight_decay': 1e-4},
    ]
    optimizer = torch.optim.AdamW(s1_params)

    warmup = cfg['warmup_epochs']
    s1_epochs = cfg['stage1_epochs']
    def lr_lambda_s1(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (s1_epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_s1)

    best_knn = 0
    for epoch in range(s1_epochs):
        model.train()
        # Keep classifier frozen
        model.classifier.eval()

        ep_cl = 0; n_batches = 0
        pos_sims = []; neg_sims = []; ucls_list = []

        pbar = tqdm(train_loader_pk, desc=f"S1 Ep {epoch+1}/{s1_epochs}", leave=False)
        for view1, view2, labels in pbar:
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            ucls_list.append(len(labels.unique()))

            optimizer.zero_grad()
            with autocast():
                _, proj1, _ = model(view1)
                _, proj2, _ = model(view2)
                all_proj = torch.cat([proj1, proj2], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                loss = supcon_fn(all_proj, all_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            ep_cl += loss.item(); n_batches += 1

            with torch.no_grad():
                sim = proj1.float() @ proj2.float().T
                pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                neg_mask = 1 - pos_mask
                pos_sims.append((sim * pos_mask).sum().item() / pos_mask.sum().clamp(min=1).item())
                neg_sims.append((sim * neg_mask).sum().item() / neg_mask.sum().clamp(min=1).item())

            pbar.set_postfix(cl=f"{loss.item():.3f}",
                           pos=f"{pos_sims[-1]:.3f}", neg=f"{neg_sims[-1]:.3f}",
                           ucls=f"{ucls_list[-1]}")

        scheduler.step()

        # KNN eval every 5 epochs (cheap proxy for feature quality)
        knn_acc = 0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            knn_acc = knn_eval(model, train_loader_plain, test_loader, device, k=20)
            if knn_acc > best_knn:
                best_knn = knn_acc

        metrics['epoch'].append(epoch + 1)
        metrics['stage'].append(1)
        metrics['train_loss'].append(ep_cl / n_batches)
        metrics['cl_loss'].append(ep_cl / n_batches)
        metrics['ce_loss'].append(0)
        metrics['train_acc'].append(0)
        metrics['test_acc'].append(0)
        metrics['knn_acc'].append(knn_acc)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        metrics['pos_sim_mean'].append(np.mean(pos_sims))
        metrics['neg_sim_mean'].append(np.mean(neg_sims))
        metrics['unique_classes_per_batch'].append(np.mean(ucls_list))

        knn_str = f"knn={knn_acc:.1f}%" if knn_acc > 0 else ""
        print(f"  S1 Ep {epoch+1:>3d}: cl={ep_cl/n_batches:.3f} | "
              f"pos={np.mean(pos_sims):.3f} neg={np.mean(neg_sims):.3f} | "
              f"classes/batch={np.mean(ucls_list):.0f} | {knn_str} best_knn={best_knn:.1f}%")

    # Save stage 1 checkpoint
    torch.save({
        'epoch': s1_epochs, 'stage': 1,
        'model_state_dict': model.state_dict(),
        'knn_acc': best_knn,
    }, os.path.join(cfg['output_dir'], 'stage1_model.pth'))
    print(f"\nStage 1 done. Best KNN acc: {best_knn:.2f}%")

    # ==================== Stage 2: CE Only ====================
    print(f"\n{'='*60}")
    print(f"Stage 2: CE classifier — {cfg['stage2_epochs']} epochs")
    print(f"Training: classifier only | Frozen: LoRA + encoder + proj_head")
    print(f"{'='*60}")

    # Freeze encoder + LoRA + proj_head, unfreeze classifier
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    s2_params = [{'params': model.classifier.parameters(), 'lr': cfg['stage2_lr']}]
    optimizer2 = torch.optim.SGD(s2_params, momentum=0.9, weight_decay=1e-4)

    s2_epochs = cfg['stage2_epochs']
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=s2_epochs, eta_min=1e-5)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    best_acc = 0
    for epoch in range(s2_epochs):
        model.train()
        # Only classifier is training
        model.encoder.eval()
        model.proj_head.eval()

        ep_ce = 0; correct = total = 0; n_batches = 0

        pbar = tqdm(train_loader_plain, desc=f"S2 Ep {epoch+1}/{s2_epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer2.zero_grad()
            with autocast():
                with torch.no_grad():
                    feat = model.encoder(images)
                logits = model.classifier(feat.float())
                loss = ce_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer2)
            scaler.update()

            ep_ce += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels); n_batches += 1
            pbar.set_postfix(ce=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")

        scheduler2.step()
        train_acc = 100 * correct / total

        # Eval
        model.eval()
        tc = tt = 0
        with torch.no_grad(), autocast():
            for images, labels_t in test_loader:
                images, labels_t = images.to(device), labels_t.to(device)
                _, _, logits = model(images)
                tc += (logits.argmax(1) == labels_t).sum().item()
                tt += len(labels_t)
        test_acc = 100 * tc / tt
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': s1_epochs + epoch + 1, 'stage': 2,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'config': cfg,
            }, os.path.join(cfg['output_dir'], 'best_model_2.pth'))

        global_ep = s1_epochs + epoch + 1
        metrics['epoch'].append(global_ep)
        metrics['stage'].append(2)
        metrics['train_loss'].append(ep_ce / n_batches)
        metrics['cl_loss'].append(0)
        metrics['ce_loss'].append(ep_ce / n_batches)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['knn_acc'].append(0)
        metrics['lr'].append(optimizer2.param_groups[0]['lr'])
        metrics['pos_sim_mean'].append(0)
        metrics['neg_sim_mean'].append(0)
        metrics['unique_classes_per_batch'].append(0)

        if (epoch + 1) % 5 == 0 or epoch == s2_epochs - 1:
            pca = compute_per_class_accuracy(model, test_loader, num_classes, device)
            metrics['per_class_acc'].append(pca)

        print(f"  S2 Ep {epoch+1:>3d}: ce={ep_ce/n_batches:.3f} | "
              f"train={train_acc:.1f}% test={test_acc:.2f}% best={best_acc:.2f}%")

    # ==================== Post ====================
    print(f"\n{'='*60}")
    print(f"V2 complete! Best test: {best_acc:.2f}%  (Stage1 KNN: {best_knn:.2f}%)")
    print(f"{'='*60}")

    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Load best & visualize
    best_path = os.path.join(cfg['output_dir'], 'best_model_2.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    plot_training_curves(metrics, cfg['output_dir'])

    indices = []; seen = set()
    for i, (_, l) in enumerate(test_ds):
        if l not in seen and len(indices) < 16:
            indices.append(i); seen.add(l)
    plot_attention_map(model,
        torch.stack([test_ds[i][0] for i in indices]),
        torch.tensor([test_ds[i][1] for i in indices]),
        class_names, cfg['output_dir'], device)

    test_loader_vis = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    save_badcases(model, test_loader_vis, class_names, cfg['output_dir'], device)
    plot_confusion_matrix(model, test_loader_vis, class_names, cfg['output_dir'], device)

    final_pca = compute_per_class_accuracy(model, test_loader_vis, num_classes, device)
    worst = sorted(enumerate(final_pca), key=lambda x: x[1])[:10]
    best_cls = sorted(enumerate(final_pca), key=lambda x: x[1], reverse=True)[:10]
    summary = {
        'best_test_acc': best_acc,
        'stage1_best_knn': best_knn,
        'mean_per_class': np.mean(final_pca),
        'classes_below_50': sum(1 for a in final_pca if a < 50),
        'worst_10': [(class_names[i], f"{a:.1f}%") for i, a in worst],
        'best_10': [(class_names[i], f"{a:.1f}%") for i, a in best_cls],
    }
    with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest: {best_acc:.2f}%, KNN(S1): {best_knn:.2f}%")
    print(f"Classes <50%: {summary['classes_below_50']}")
    print(f"Worst 5: {summary['worst_10'][:5]}")
    print(f"Saved to {cfg['output_dir']}/")


if __name__ == '__main__':
    train()
