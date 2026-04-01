"""
DINOv2 + LoRA Supervised Contrastive Learning for FGVC-Aircraft.
- LoRA rank=8 on DINOv2 ViT-B/14 attention (qkv, proj)
- SupCon + CE joint loss
- AMP fp16 training, aggressive augmentation
- Saves checkpoint, training metrics, attention maps, bad cases
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model import DINOv2ContrastiveModel, SupConLoss
from dataset import build_loaders, get_transforms
from visualize import (plot_attention_map, plot_training_curves,
                       save_badcases, plot_confusion_matrix)

# ==================== Config ====================
CONFIG = {
    'data_root': '../data/fgvc_aircraft',
    'dino_weights': '/mnt/datasets/dinov2_vitb14.pth',
    'output_dir': 'output',
    'lora_rank': 8,
    'proj_dim': 128,
    'batch_size': 48,  # per-GPU, effective=48*accum_steps  # large batch for contrastive learning
    'epochs': 80,
    'accum_steps': 2,  # gradient accumulation -> effective batch=96
    'lr': 1e-3,
    'lr_lora': 5e-4,
    'weight_decay': 5e-4,
    'temperature': 0.1,
    'lambda_cl': 0.5,     # contrastive loss weight
    'lambda_ce': 1.0,     # CE loss weight
    'label_smoothing': 0.1,
    'warmup_epochs': 5,
    'num_workers': 4,
    'seed': 42,
    'grad_clip': 1.0,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_lr_lambda(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


def compute_per_class_accuracy(model, loader, num_classes, device):
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    model.eval()
    with torch.no_grad(), autocast():
        for images, labels in loader:
            images, labels = images.to(device), labels
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

    # Save config
    with open(os.path.join(cfg['output_dir'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data
    train_loader, test_loader, num_classes, class_names = build_loaders(
        cfg['data_root'], cfg['batch_size'], cfg['num_workers']
    )
    print(f"Dataset: {num_classes} classes, batch_size={cfg['batch_size']}")

    # Model
    model = DINOv2ContrastiveModel(
        num_classes, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']
    ).to(device)

    # Optimizer: separate lr for LoRA and heads
    param_groups = [
        {'params': list(model.proj_head.parameters()) + list(model.classifier.parameters()),
         'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params,
         'lr': cfg['lr_lora'], 'weight_decay': 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(cfg['warmup_epochs'], cfg['epochs'])
    )
    scaler = GradScaler()

    # Losses
    supcon_loss_fn = SupConLoss(temperature=cfg['temperature'])
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    # Metrics tracking
    metrics = {k: [] for k in [
        'epoch', 'train_loss', 'ce_loss', 'cl_loss', 'train_acc', 'test_acc',
        'lr', 'lora_grad_norm', 'head_grad_norm', 'per_class_acc',
        'lora_weight_norm', 'lora_update_norm', 'feat_norm_mean', 'feat_norm_std',
        'logit_entropy', 'pos_sim_mean', 'neg_sim_mean',
    ]}

    best_acc = 0
    print(f"\nStarting training: {cfg['epochs']} epochs, lr={cfg['lr']}, lora_lr={cfg['lr_lora']}")
    print(f"Loss: {cfg['lambda_ce']}*CE + {cfg['lambda_cl']}*SupCon(T={cfg['temperature']})")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = epoch_ce = epoch_cl = 0
        correct = total = 0
        lora_grad_acc = head_grad_acc = 0
        feat_norms = []
        logit_entropies = []
        pos_sims = []
        neg_sims = []
        n_batches = 0

        accum_steps = cfg.get('accum_steps', 1)
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['epochs']}", leave=False)
        for batch_idx, (view1, view2, labels) in enumerate(pbar):
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            B = labels.shape[0]

            with autocast():
                # Forward both views
                feat1, proj1, logits1 = model(view1)
                feat2, proj2, logits2 = model(view2)

                # Contrastive loss on combined projections
                all_proj = torch.cat([proj1, proj2], dim=0)   # [2B, D]
                all_labels = torch.cat([labels, labels], dim=0)  # [2B]
                loss_cl = supcon_loss_fn(all_proj, all_labels)

                # CE loss on both views
                loss_ce = (ce_loss_fn(logits1, labels) + ce_loss_fn(logits2, labels)) / 2

                loss = cfg['lambda_ce'] * loss_ce + cfg['lambda_cl'] * loss_cl

            scaler.scale(loss / accum_steps).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                lora_gn = torch.stack([p.grad.norm() for p in model.lora_params if p.grad is not None]).mean().item() if any(p.grad is not None for p in model.lora_params) else 0
                head_gn = torch.stack([p.grad.norm() for p in model.classifier.parameters() if p.grad is not None]).mean().item()
                lora_grad_acc += lora_gn
                head_grad_acc += head_gn
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Stats
            epoch_loss += loss.item()
            epoch_ce += loss_ce.item()
            epoch_cl += loss_cl.item()
            preds1 = logits1.argmax(1)
            correct += (preds1 == labels).sum().item()
            total += B
            n_batches += 1

            # Extra metrics
            with torch.no_grad():
                feat_norms.append(feat1.float().norm(dim=1).mean().item())
                probs = F.softmax(logits1.float(), dim=1)
                ent = -(probs * (probs + 1e-8).log()).sum(dim=1).mean().item()
                logit_entropies.append(ent)
                # Positive/negative similarity
                sim = proj1.float() @ proj2.float().T
                pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                neg_mask = 1 - pos_mask
                pos_sims.append((sim * pos_mask).sum().item() / pos_mask.sum().clamp(min=1).item())
                neg_sims.append((sim * neg_mask).sum().item() / neg_mask.sum().clamp(min=1).item())

            pbar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{loss_ce.item():.3f}",
                           cl=f"{loss_cl.item():.3f}", acc=f"{100*correct/total:.1f}%")

        scheduler.step()
        train_acc = 100 * correct / total

        # Eval
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad(), autocast():
            for images, labels_t in test_loader:
                images, labels_t = images.to(device), labels_t.to(device)
                _, _, logits = model(images)
                test_correct += (logits.argmax(1) == labels_t).sum().item()
                test_total += len(labels_t)
        test_acc = 100 * test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': cfg,
            }, os.path.join(cfg['output_dir'], 'best_model.pth'))

        # LoRA weight stats
        lora_w_norm = torch.stack([p.data.norm() for p in model.lora_params]).mean().item()

        # Log metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(epoch_loss / n_batches)
        metrics['ce_loss'].append(epoch_ce / n_batches)
        metrics['cl_loss'].append(epoch_cl / n_batches)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        metrics['lora_grad_norm'].append(lora_grad_acc / n_batches)
        metrics['head_grad_norm'].append(head_grad_acc / n_batches)
        metrics['lora_weight_norm'].append(lora_w_norm)
        metrics['lora_update_norm'].append(lora_grad_acc / n_batches * cfg['lr_lora'])
        metrics['feat_norm_mean'].append(np.mean(feat_norms))
        metrics['feat_norm_std'].append(np.std(feat_norms))
        metrics['logit_entropy'].append(np.mean(logit_entropies))
        metrics['pos_sim_mean'].append(np.mean(pos_sims))
        metrics['neg_sim_mean'].append(np.mean(neg_sims))

        # Per-class accuracy every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == cfg['epochs'] - 1:
            pca = compute_per_class_accuracy(model, test_loader, num_classes, device)
            metrics['per_class_acc'].append(pca)

        print(f"  Ep {epoch+1:>3d}: loss={epoch_loss/n_batches:.3f} ce={epoch_ce/n_batches:.3f} "
              f"cl={epoch_cl/n_batches:.3f} | train={train_acc:.1f}% test={test_acc:.2f}% "
              f"best={best_acc:.2f}% | lr={optimizer.param_groups[0]['lr']:.6f} "
              f"lora_gn={lora_grad_acc/n_batches:.3f} pos_sim={np.mean(pos_sims):.3f} "
              f"neg_sim={np.mean(neg_sims):.3f}")

        # Save metrics checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w') as f:
                json.dump(metrics, f)

    # ==================== Post-training ====================
    print("\n" + "=" * 60)
    print(f"Training complete! Best test accuracy: {best_acc:.2f}%")
    print("=" * 60)

    # Save final metrics
    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Load best model for visualization
    ckpt = torch.load(os.path.join(cfg['output_dir'], 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"\nLoaded best model from epoch {ckpt['epoch']}, test_acc={ckpt['test_acc']:.2f}%")

    # 1. Training curves
    plot_training_curves(metrics, cfg['output_dir'])

    # 2. Attention maps on test set
    test_transform = get_transforms(train=False)
    from torchvision import datasets
    test_ds = datasets.ImageFolder(f"{cfg['data_root']}/test", test_transform)
    # Get a diverse sample (different classes)
    indices = []
    seen_classes = set()
    for i, (_, label) in enumerate(test_ds):
        if label not in seen_classes and len(indices) < 16:
            indices.append(i)
            seen_classes.add(label)
    sample_images = torch.stack([test_ds[i][0] for i in indices])
    sample_labels = torch.tensor([test_ds[i][1] for i in indices])
    plot_attention_map(model, sample_images, sample_labels, class_names,
                       cfg['output_dir'], device)

    # 3. Bad cases
    from torch.utils.data import DataLoader
    test_loader_vis = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    save_badcases(model, test_loader_vis, class_names, cfg['output_dir'], device)

    # 4. Confusion matrix
    plot_confusion_matrix(model, test_loader_vis, class_names, cfg['output_dir'], device)

    # 5. Final summary
    final_per_class = compute_per_class_accuracy(model, test_loader_vis, num_classes, device)
    worst_classes = sorted(enumerate(final_per_class), key=lambda x: x[1])[:10]
    best_classes = sorted(enumerate(final_per_class), key=lambda x: x[1], reverse=True)[:10]

    summary = {
        'best_test_acc': best_acc,
        'best_epoch': ckpt['epoch'],
        'final_test_acc': test_acc,
        'mean_per_class_acc': np.mean(final_per_class),
        'std_per_class_acc': np.std(final_per_class),
        'worst_10_classes': [(class_names[i], f"{acc:.1f}%") for i, acc in worst_classes],
        'best_10_classes': [(class_names[i], f"{acc:.1f}%") for i, acc in best_classes],
        'lora_rank': cfg['lora_rank'],
        'total_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Best test acc: {best_acc:.2f}% (epoch {ckpt['epoch']})")
    print(f"Mean per-class acc: {np.mean(final_per_class):.2f}% (std={np.std(final_per_class):.2f}%)")
    print(f"\nWorst 10 classes:")
    for cls_name, acc in summary['worst_10_classes']:
        print(f"  {cls_name:>25s}: {acc}")
    print(f"\nBest 10 classes:")
    for cls_name, acc in summary['best_10_classes']:
        print(f"  {cls_name:>25s}: {acc}")
    print(f"\nAll outputs saved to: {cfg['output_dir']}/")


if __name__ == '__main__':
    train()
