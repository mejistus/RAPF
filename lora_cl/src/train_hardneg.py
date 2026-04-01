"""
DAY_1 Night: Hard Negative Mining experiment.
Based on DAY_1_009 (λ=4.0, 82.24%) config, add confusion-matrix-based hard neg reweighting.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model_gl import GlobalLocalContrastiveModel
from model import SupConLoss
from pk_sampler import PKSampler

# ==================== Hard Negative Pairs (from DAY_1 confusion matrix) ====================
# Bidirectional: if A→B confused, then B→A also hard
HARD_PAIRS = [
    ("747-100", "747-200"), ("737-300", "737-500"), ("BAE_146-200", "BAE_146-300"),
    ("DC-3", "C-47"), ("A340-200", "A340-300"), ("ERJ_145", "ERJ_135"),
    ("747-300", "747-400"), ("A320", "A319"), ("MD-11", "DC-10"),
    ("737-400", "737-300"), ("A330-200", "A330-300"), ("MD-90", "MD-80"),
    ("737-700", "737-600"), ("737-900", "737-800"), ("747-300", "747-200"),
    ("A319", "A318"), ("767-300", "767-200"), ("757-200", "757-300"),
    ("A321", "A320"), ("737-500", "737-400"),
]


class HardNegSupConLoss(nn.Module):
    """SupCon with confusion-matrix-based hard negative reweighting."""
    def __init__(self, temperature=0.1, hard_weight=3.0, class_names=None):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight

        # Build hard pair index: class_idx → set of hard negative class_idx
        self.hard_neg_map = {}
        if class_names is not None:
            name2idx = {n: i for i, n in enumerate(class_names)}
            for a, b in HARD_PAIRS:
                if a in name2idx and b in name2idx:
                    ia, ib = name2idx[a], name2idx[b]
                    self.hard_neg_map.setdefault(ia, set()).add(ib)
                    self.hard_neg_map.setdefault(ib, set()).add(ia)
            print(f"HardNeg: {len(self.hard_neg_map)} classes with hard negatives, weight={hard_weight}")

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        self_mask = 1 - torch.eye(B, device=device)
        pos_mask = pos_mask * self_mask
        neg_mask = (1 - torch.eq(labels, labels.T).float())

        # Build hard negative weight matrix
        weight = torch.ones(B, B, device=device)
        if self.hard_neg_map:
            labels_flat = labels.squeeze()
            for i in range(B):
                li = labels_flat[i].item()
                if li in self.hard_neg_map:
                    for j in range(B):
                        if labels_flat[j].item() in self.hard_neg_map[li]:
                            weight[i, j] = self.hard_weight

        logits = features @ features.T / self.temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Apply hard negative weighting to exp_logits
        exp_logits = torch.exp(logits) * self_mask * weight
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        num_pos = pos_mask.sum(dim=1).clamp(min=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / num_pos
        return -mean_log_prob.mean()


CONFIG = {
    'data_root': '../data/fgvc_aircraft',
    'dino_weights': '/mnt/datasets/dinov2_vitb14.pth',
    'lora_rank': 8,
    'proj_dim': 128,
    'P': 8, 'K': 8,
    'epochs': 80,
    'lr': 1e-3,
    'lr_lora': 5e-4,
    'warmup_epochs': 5,
    'temperature': 0.1,
    'lambda_cl': 4.0,
    'local_weight': 0.5,
    'label_smoothing': 0.1,
    'weight_decay': 5e-4,
    'num_workers': 4,
    'seed': 42,
    'grad_clip': 1.0,
}


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train(hard_weight, exp_name):
    cfg = CONFIG.copy()
    cfg['hard_weight'] = hard_weight
    out_dir = exp_name
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/config.json", 'w') as f: json.dump(cfg, f, indent=2)

    seed_everything(cfg['seed'])
    device = torch.device("cuda")

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
        transforms.RandomGrayscale(0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    tf_test = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    train_ds = datasets.ImageFolder(f"{cfg['data_root']}/train", tf_train)
    test_ds = datasets.ImageFolder(f"{cfg['data_root']}/test", tf_test)
    nc = len(train_ds.classes); class_names = train_ds.classes

    pk = PKSampler(train_ds.targets, p=cfg['P'], k=cfg['K'])
    train_loader = DataLoader(train_ds, batch_sampler=pk, num_workers=cfg['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    model = GlobalLocalContrastiveModel(nc, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']).to(device)

    param_groups = [
        {'params': list(model.classifier.parameters()) + list(model.global_proj.parameters()) +
                   list(model.local_proj.parameters()) + list(model.local_agg.parameters()),
         'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params, 'lr': cfg['lr_lora'], 'weight_decay': 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    warmup = cfg['warmup_epochs']; total_ep = cfg['epochs']
    def lr_lam(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/(total_ep-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lam)
    scaler = GradScaler()

    supcon_fn = HardNegSupConLoss(temperature=cfg['temperature'], hard_weight=hard_weight, class_names=class_names)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    lam = cfg['lambda_cl']; lw = cfg['local_weight']

    metrics = {k: [] for k in ['epoch','loss','ce','clg','cll','train_acc','test_acc','lr']}
    best_acc = 0; t0 = time.time()

    print(f"\n{'='*50}\n{exp_name}: hard_weight={hard_weight}\n{'='*50}")

    for epoch in range(total_ep):
        model.train()
        el = ec = eg = eL = 0; cor = tot = 0; nb = 0
        for images, labels in tqdm(train_loader, desc=f"Ep{epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits, zg, zl, _ = model(images)
                loss_ce = ce_fn(logits, labels)
                loss_g = supcon_fn(zg, labels)
                loss_l = supcon_fn(zl, labels)
                loss = loss_ce + lam * (loss_g + lw * loss_l)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer); scaler.update()
            el += loss.item(); ec += loss_ce.item(); eg += loss_g.item(); eL += loss_l.item()
            cor += (logits.argmax(1)==labels).sum().item(); tot += len(labels); nb += 1
        scheduler.step()
        ta = 100*cor/tot

        model.eval(); tc = tt = 0
        with torch.no_grad(), autocast():
            for imgs, lbls in test_loader:
                tc += (model(imgs.to(device))[0].argmax(1)==lbls.to(device)).sum().item()
                tt += len(lbls)
        te = 100*tc/tt
        if te > best_acc:
            best_acc = te
            torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),'test_acc':te,'config':cfg},
                       f"{out_dir}/best_model.pth")

        metrics['epoch'].append(epoch+1); metrics['loss'].append(el/nb)
        metrics['ce'].append(ec/nb); metrics['clg'].append(eg/nb); metrics['cll'].append(eL/nb)
        metrics['train_acc'].append(ta); metrics['test_acc'].append(te); metrics['lr'].append(optimizer.param_groups[0]['lr'])

        if (epoch+1) % 10 == 0:
            mins = (time.time()-t0)/60
            print(f"  Ep {epoch+1:>3d}: loss={el/nb:.3f} ce={ec/nb:.3f} | train={ta:.1f}% test={te:.2f}% best={best_acc:.2f}% | {mins:.0f}min")

    with open(f"{out_dir}/metrics.json", 'w') as f: json.dump(metrics, f)
    summary = {'hard_weight': hard_weight, 'best_test_acc': best_acc, 'total_min': (time.time()-t0)/60}
    with open(f"{out_dir}/summary.json", 'w') as f: json.dump(summary, f, indent=2)
    print(f"  {exp_name} DONE → {best_acc:.2f}% ({(time.time()-t0)/60:.0f}min)")

    del model; torch.cuda.empty_cache()
    return best_acc


if __name__ == '__main__':
    results = []
    # Baseline (no hard neg, same as DAY_1_009 but fresh run for fair comparison)
    acc0 = train(hard_weight=1.0, exp_name="DAY_1_012_hardneg_w1.0")
    results.append((1.0, acc0))

    # Hard negative experiments
    for hw in [2.0, 3.0, 5.0]:
        acc = train(hard_weight=hw, exp_name=f"DAY_1_{len(results)+12:03d}_hardneg_w{hw:.1f}")
        results.append((hw, acc))

    print(f"\n{'='*50}\nHard Negative Sweep Results\n{'='*50}")
    for hw, acc in results:
        marker = " ★" if acc == max(r[1] for r in results) else ""
        print(f"  w={hw:.1f} → {acc:.2f}%{marker}")
