"""
V1: Continue training from best_model.pth with:
- CutMix augmentation
- Attention entropy regularization (penalize diffuse attention)
- Stronger RandomErasing
- Hard negative contrastive mining
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, time, math, random, copy
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
    'resume_from': 'output/best_model.pth',
    'output_dir': 'output-1',
    'lora_rank': 8,
    'proj_dim': 128,
    'batch_size': 48,
    'accum_steps': 2,
    'epochs': 20,
    'lr': 2e-4,           # lower lr for fine-tuning
    'lr_lora': 1e-4,
    'weight_decay': 5e-4,
    'temperature': 0.1,   # sharper temperature
    'lambda_cl': 0.5,      # increase contrastive weight
    'lambda_ce': 1.0,
    'lambda_attn_ent': 0.1,  # attention entropy regularization
    'label_smoothing': 0.15,
    'warmup_epochs': 2,
    'cutmix_prob': 0.5,
    'cutmix_alpha': 1.0,
    'num_workers': 4,
    'seed': 42,
    'grad_clip': 1.0,
}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ==================== CutMix ====================
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam_adj


# ==================== Stronger augmentation ====================
class RandomChannelShuffle:
    """Randomly permute RGB channels to prevent color-based classification."""
    def __init__(self, p=0.3):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            import PIL.Image
            channels = list(img.split())
            random.shuffle(channels)
            return PIL.Image.merge('RGB', channels)
        return img

class RandomColorInvert:
    """Randomly invert image colors."""
    def __init__(self, p=0.2):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            import PIL.ImageOps
            return PIL.ImageOps.invert(img.convert('RGB'))
        return img

class RandomChannelDrop:
    """Randomly zero-out one color channel."""
    def __init__(self, p=0.15):
        self.p = p
    def __call__(self, tensor):
        if random.random() < self.p:
            ch = random.randint(0, 2)
            tensor[ch] = 0
        return tensor

def get_strong_transforms(img_size=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(45),
        RandomChannelShuffle(p=0.3),     # shuffle RGB channels
        RandomColorInvert(p=0.2),         # invert colors
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15),
        transforms.ToTensor(),
        RandomChannelDrop(p=0.15),        # drop one channel
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.4, scale=(0.05, 0.25), ratio=(0.3, 3.3)),
    ])


# ==================== Attention Entropy Loss ====================
def attention_entropy_loss(model):
    """Penalize high-entropy (diffuse) attention. Lower entropy = more focused attention."""
    attn = model.encoder.get_last_attn()  # [B, heads, N, N]
    # CLS token attention to patches
    cls_attn = attn[:, :, 0, 1:]  # [B, heads, num_patches]
    # Compute entropy per head
    cls_attn = cls_attn + 1e-8
    cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True)
    entropy = -(cls_attn * cls_attn.log()).sum(dim=-1)  # [B, heads]
    # Average over batch and heads, normalize by max possible entropy
    max_entropy = math.log(cls_attn.shape[-1])
    return entropy.mean() / max_entropy  # 0-1 range


# ==================== Hard Negative SupCon ====================
class HardNegSupConLoss(nn.Module):
    """SupCon with emphasis on hard negatives (most similar different-class samples)."""
    def __init__(self, temperature=0.07, hard_neg_weight=1.5):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_weight = hard_neg_weight

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        neg_mask = 1 - pos_mask
        self_mask = 1 - torch.eye(B, device=device)
        pos_mask = pos_mask * self_mask

        logits = features @ features.T / self.temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * self_mask

        # Up-weight hard negatives (high similarity but different class)
        neg_sim = (features @ features.T).detach() * neg_mask
        # Top-k hardest negatives per sample
        hard_neg_mask = neg_mask.clone()
        k = min(5, int(neg_mask.sum(1).min().item()))
        if k > 0:
            _, topk_idx = neg_sim.topk(k, dim=1)
            hard_boost = torch.zeros_like(neg_mask)
            hard_boost.scatter_(1, topk_idx, self.hard_neg_weight - 1)
            weight_mask = self_mask + hard_boost * neg_mask
            exp_logits = torch.exp(logits) * weight_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        num_pos = pos_mask.sum(dim=1).clamp(min=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / num_pos
        return -mean_log_prob.mean()


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

    # Data with stronger augmentation
    from torchvision import datasets as tv_datasets
    from torch.utils.data import DataLoader

    strong_transform = get_strong_transforms()
    test_transform = get_transforms(train=False)

    # TwoView with strong augmentation
    from dataset import TwoViewDataset
    train_ds = TwoViewDataset(os.path.join(cfg['data_root'], 'train'), strong_transform)
    test_ds = tv_datasets.ImageFolder(os.path.join(cfg['data_root'], 'test'), test_transform)
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True)

    print(f"Dataset: {num_classes} classes, batch_size={cfg['batch_size']}")

    # Model - load from checkpoint
    model = DINOv2ContrastiveModel(
        num_classes, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']
    ).to(device)

    ckpt = torch.load(cfg['resume_from'], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    prev_acc = ckpt['test_acc']
    print(f"Resumed from {cfg['resume_from']}, epoch {ckpt['epoch']}, acc={prev_acc:.2f}%")

    # Optimizer
    param_groups = [
        {'params': list(model.proj_head.parameters()) + list(model.classifier.parameters()),
         'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params,
         'lr': cfg['lr_lora'], 'weight_decay': 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    total_epochs = cfg['epochs']
    warmup = cfg['warmup_epochs']
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (total_epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Losses
    supcon_loss_fn = HardNegSupConLoss(temperature=cfg['temperature'], hard_neg_weight=1.5)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    metrics = {k: [] for k in [
        'epoch', 'train_loss', 'ce_loss', 'cl_loss', 'attn_ent_loss',
        'train_acc', 'test_acc', 'lr', 'lora_grad_norm', 'head_grad_norm',
        'per_class_acc', 'pos_sim_mean', 'neg_sim_mean', 'attn_entropy',
        'cutmix_applied', 'hard_neg_sim',
    ]}

    best_acc = prev_acc
    print(f"\nV1 training: {total_epochs} epochs, lr={cfg['lr']}, lora_lr={cfg['lr_lora']}")
    print(f"CutMix prob={cfg['cutmix_prob']}, Attn entropy weight={cfg['lambda_attn_ent']}")
    print(f"Hard neg SupCon T={cfg['temperature']}, lambda_cl={cfg['lambda_cl']}")
    print("=" * 60)

    accum_steps = cfg.get('accum_steps', 1)

    for epoch in range(total_epochs):
        model.train()
        ep_loss = ep_ce = ep_cl = ep_attn = 0
        correct = total = 0
        lora_grad_acc = head_grad_acc = 0
        pos_sims = []; neg_sims = []; attn_ents = []; cutmix_count = 0
        hard_neg_sims = []
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{total_epochs}", leave=False)
        for batch_idx, (view1, view2, labels) in enumerate(pbar):
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            B = labels.shape[0]

            # Apply CutMix to view1
            apply_cutmix = random.random() < cfg['cutmix_prob']
            if apply_cutmix:
                view1, labels_a, labels_b, lam = cutmix_data(view1, labels, cfg['cutmix_alpha'])
                cutmix_count += 1

            with autocast():
                feat1, proj1, logits1 = model(view1, return_attn=True)
                feat2, proj2, logits2 = model(view2)

                # Contrastive loss (hard negative mining)
                all_proj = torch.cat([proj1, proj2], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                loss_cl = supcon_loss_fn(all_proj, all_labels)

                # CE loss (with CutMix mixing if applied)
                if apply_cutmix:
                    loss_ce = lam * ce_loss_fn(logits1, labels_a) + (1 - lam) * ce_loss_fn(logits1, labels_b)
                    loss_ce = (loss_ce + ce_loss_fn(logits2, labels)) / 2
                else:
                    loss_ce = (ce_loss_fn(logits1, labels) + ce_loss_fn(logits2, labels)) / 2

                # Attention entropy regularization
                loss_attn = attention_entropy_loss(model)

                loss = (cfg['lambda_ce'] * loss_ce +
                        cfg['lambda_cl'] * loss_cl +
                        cfg['lambda_attn_ent'] * loss_attn)

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

            ep_loss += loss.item(); ep_ce += loss_ce.item(); ep_cl += loss_cl.item()
            ep_attn += loss_attn.item()
            preds1 = logits1.argmax(1)
            correct += (preds1 == labels).sum().item(); total += B
            n_batches += 1

            with torch.no_grad():
                sim = proj1.float() @ proj2.float().T
                pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                neg_mask = 1 - pos_mask
                pos_sims.append((sim * pos_mask).sum().item() / pos_mask.sum().clamp(min=1).item())
                neg_sims.append((sim * neg_mask).sum().item() / neg_mask.sum().clamp(min=1).item())
                attn_ents.append(loss_attn.item())
                # Hard negative similarity (top-5 most similar wrong class)
                neg_sim_vals = sim * neg_mask
                if neg_sim_vals.shape[0] > 5:
                    topk_neg = neg_sim_vals.topk(5, dim=1)[0].mean().item()
                    hard_neg_sims.append(topk_neg)

            pbar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{loss_ce.item():.3f}",
                           cl=f"{loss_cl.item():.3f}", attn=f"{loss_attn.item():.3f}",
                           acc=f"{100*correct/total:.1f}%")

        scheduler.step()
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
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': cfg,
            }, os.path.join(cfg['output_dir'], 'best_model_1.pth'))

        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(ep_loss / n_batches)
        metrics['ce_loss'].append(ep_ce / n_batches)
        metrics['cl_loss'].append(ep_cl / n_batches)
        metrics['attn_ent_loss'].append(ep_attn / n_batches)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        metrics['lora_grad_norm'].append(lora_grad_acc / max(n_batches // accum_steps, 1))
        metrics['head_grad_norm'].append(head_grad_acc / max(n_batches // accum_steps, 1))
        metrics['pos_sim_mean'].append(np.mean(pos_sims))
        metrics['neg_sim_mean'].append(np.mean(neg_sims))
        metrics['attn_entropy'].append(np.mean(attn_ents))
        metrics['cutmix_applied'].append(cutmix_count)
        metrics['hard_neg_sim'].append(np.mean(hard_neg_sims) if hard_neg_sims else 0)

        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            pca = compute_per_class_accuracy(model, test_loader, num_classes, device)
            metrics['per_class_acc'].append(pca)

        print(f"  Ep {epoch+1:>3d}: loss={ep_loss/n_batches:.3f} ce={ep_ce/n_batches:.3f} "
              f"cl={ep_cl/n_batches:.3f} attn={ep_attn/n_batches:.3f} | "
              f"train={train_acc:.1f}% test={test_acc:.2f}% best={best_acc:.2f}% | "
              f"pos={np.mean(pos_sims):.3f} neg={np.mean(neg_sims):.3f} "
              f"hard_neg={np.mean(hard_neg_sims) if hard_neg_sims else 0:.3f} "
              f"attn_ent={np.mean(attn_ents):.3f} cutmix={cutmix_count}")

    # ==================== Post-training ====================
    print(f"\nV1 complete! Best: {best_acc:.2f}% (prev: {prev_acc:.2f}%)")

    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Load best model
    best_path = os.path.join(cfg['output_dir'], 'best_model_1.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Visualizations
    plot_training_curves(metrics, cfg['output_dir'])

    test_ds_vis = tv_datasets.ImageFolder(os.path.join(cfg['data_root'], 'test'), test_transform)
    indices = []
    seen = set()
    for i, (_, label) in enumerate(test_ds_vis):
        if label not in seen and len(indices) < 16:
            indices.append(i); seen.add(label)
    sample_images = torch.stack([test_ds_vis[i][0] for i in indices])
    sample_labels = torch.tensor([test_ds_vis[i][1] for i in indices])
    plot_attention_map(model, sample_images, sample_labels, class_names, cfg['output_dir'], device)

    test_loader_vis = DataLoader(test_ds_vis, batch_size=64, shuffle=False, num_workers=2)
    save_badcases(model, test_loader_vis, class_names, cfg['output_dir'], device)
    plot_confusion_matrix(model, test_loader_vis, class_names, cfg['output_dir'], device)

    # Final per-class
    final_pca = compute_per_class_accuracy(model, test_loader_vis, num_classes, device)
    worst = sorted(enumerate(final_pca), key=lambda x: x[1])[:10]
    best = sorted(enumerate(final_pca), key=lambda x: x[1], reverse=True)[:10]

    summary = {
        'best_test_acc': best_acc,
        'prev_best_acc': prev_acc,
        'improvement': best_acc - prev_acc,
        'worst_10': [(class_names[i], f"{a:.1f}%") for i, a in worst],
        'best_10': [(class_names[i], f"{a:.1f}%") for i, a in best],
        'mean_per_class': np.mean(final_pca),
        'std_per_class': np.std(final_pca),
        'classes_below_50': sum(1 for a in final_pca if a < 50),
    }
    with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest: {best_acc:.2f}% (improvement: {best_acc - prev_acc:+.2f}%)")
    print(f"Classes <50%: {summary['classes_below_50']}")
    print(f"Worst: {summary['worst_10'][:5]}")
    print(f"All saved to {cfg['output_dir']}/")


if __name__ == '__main__':
    train()
