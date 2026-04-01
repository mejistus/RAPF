"""
DAY_1: Lambda sweep for Global-Local SupCon + CE.
λ ∈ {0.01, 0.025, 0.063, 0.156, 0.391, 0.977}
Each: 80 epochs, from scratch, PK(P=8,K=8), cosine LR + warmup.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, math, random, time, shutil
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

# ==================== Config ====================
BASE_CONFIG = {
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
    'local_weight': 0.5,  # λ_local = λ * local_weight
    'label_smoothing': 0.1,
    'weight_decay': 5e-4,
    'num_workers': 4,
    'seed': 42,
    'grad_clip': 1.0,
}

LAMBDA_VALUES = [1.5, 2.5, 4.0, 6.5, 10.0]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_transforms_train():
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

def get_transforms_test():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def run_experiment(lam, exp_id, cfg):
    """Run one experiment with given lambda."""
    out_dir = f"DAY_1_{exp_id:03d}_lambda{lam:.3f}"
    os.makedirs(out_dir, exist_ok=True)

    exp_cfg = {**cfg, 'lambda_cl': lam, 'exp_id': exp_id, 'output_dir': out_dir}
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(exp_cfg, f, indent=2)

    seed_everything(cfg['seed'])
    device = torch.device("cuda")

    # Data
    train_ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), get_transforms_train())
    test_ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'test'), get_transforms_test())
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    pk = PKSampler(train_ds.targets, p=cfg['P'], k=cfg['K'])
    train_loader = DataLoader(train_ds, batch_sampler=pk, num_workers=cfg['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    # Model
    model = GlobalLocalContrastiveModel(num_classes, cfg['dino_weights'], cfg['lora_rank'], cfg['proj_dim']).to(device)

    # Optimizer: heads + LoRA
    param_groups = [
        {'params': list(model.classifier.parameters()) + list(model.global_proj.parameters()) +
                   list(model.local_proj.parameters()) + list(model.local_agg.parameters()),
         'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
        {'params': model.lora_params, 'lr': cfg['lr_lora'], 'weight_decay': 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    warmup = cfg['warmup_epochs']
    total_ep = cfg['epochs']
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (total_ep - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    supcon_fn = SupConLoss(temperature=cfg['temperature'])
    ce_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    local_w = cfg['local_weight']
    metrics = {k: [] for k in [
        'epoch', 'total_loss', 'ce_loss', 'cl_global', 'cl_local',
        'train_acc', 'test_acc', 'lr', 'pos_sim_g', 'neg_sim_g',
        'pos_sim_l', 'neg_sim_l', 'attn_entropy',
    ]}

    best_acc = 0
    t_start = time.time()

    for epoch in range(total_ep):
        model.train()
        ep_total = ep_ce = ep_clg = ep_cll = 0
        correct = total = 0; nb = 0
        pos_sg = []; neg_sg = []; pos_sl = []; neg_sl = []; attn_ents = []

        optimizer.zero_grad()
        for images, labels in tqdm(train_loader, desc=f"[{exp_id:03d} λ={lam}] Ep{epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)

            with autocast():
                logits, z_g, z_l, attn_w = model(images)

                loss_ce = ce_fn(logits, labels)
                loss_cl_g = supcon_fn(z_g, labels)
                loss_cl_l = supcon_fn(z_l, labels)

                loss = loss_ce + lam * (loss_cl_g + local_w * loss_cl_l)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            ep_total += loss.item(); ep_ce += loss_ce.item()
            ep_clg += loss_cl_g.item(); ep_cll += loss_cl_l.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels); nb += 1

            with torch.no_grad():
                # Global similarity stats
                sim_g = z_g.float() @ z_g.float().T
                pm = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float(); pm.fill_diagonal_(0)
                nm = 1 - torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
                pos_sg.append((sim_g*pm).sum().item()/pm.sum().clamp(min=1).item())
                neg_sg.append((sim_g*nm).sum().item()/nm.sum().clamp(min=1).item())
                # Local similarity stats
                sim_l = z_l.float() @ z_l.float().T
                pos_sl.append((sim_l*pm).sum().item()/pm.sum().clamp(min=1).item())
                neg_sl.append((sim_l*nm).sum().item()/nm.sum().clamp(min=1).item())
                # Attention entropy (how focused is local aggregation)
                ent = -(attn_w * (attn_w + 1e-8).log()).sum(dim=-1).mean().item()
                attn_ents.append(ent)

        scheduler.step()
        train_acc = 100 * correct / total

        # Eval
        model.eval(); tc = tt = 0
        with torch.no_grad(), autocast():
            for imgs, lbls in test_loader:
                logits_t = model(imgs.to(device))[0]
                tc += (logits_t.argmax(1) == lbls.to(device)).sum().item()
                tt += len(lbls)
        test_acc = 100 * tc / tt
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'test_acc': test_acc, 'lambda': lam, 'config': exp_cfg},
                       os.path.join(out_dir, 'best_model.pth'))

        # Log
        metrics['epoch'].append(epoch+1)
        metrics['total_loss'].append(ep_total/nb)
        metrics['ce_loss'].append(ep_ce/nb)
        metrics['cl_global'].append(ep_clg/nb)
        metrics['cl_local'].append(ep_cll/nb)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        metrics['pos_sim_g'].append(np.mean(pos_sg))
        metrics['neg_sim_g'].append(np.mean(neg_sg))
        metrics['pos_sim_l'].append(np.mean(pos_sl))
        metrics['neg_sim_l'].append(np.mean(neg_sl))
        metrics['attn_entropy'].append(np.mean(attn_ents))

        if (epoch+1) % 10 == 0 or epoch == 0:
            elapsed = (time.time() - t_start) / 60
            print(f"  [{exp_id:03d} λ={lam:.3f}] Ep {epoch+1:>3d}: "
                  f"loss={ep_total/nb:.3f} ce={ep_ce/nb:.3f} clg={ep_clg/nb:.3f} cll={ep_cll/nb:.3f} | "
                  f"train={train_acc:.1f}% test={test_acc:.2f}% best={best_acc:.2f}% | "
                  f"pos_g={np.mean(pos_sg):.3f} neg_g={np.mean(neg_sg):.3f} "
                  f"pos_l={np.mean(pos_sl):.3f} neg_l={np.mean(neg_sl):.3f} "
                  f"attn_ent={np.mean(attn_ents):.3f} | {elapsed:.1f}min")

    # Save metrics
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Summary
    summary = {'lambda': lam, 'best_test_acc': best_acc, 'exp_id': exp_id,
               'final_test_acc': test_acc, 'total_time_min': (time.time()-t_start)/60}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    print(f"\n  [{exp_id:03d}] λ={lam:.3f} DONE → best={best_acc:.2f}% ({(time.time()-t_start)/60:.1f}min)\n")
    return best_acc


def update_task_md(results):
    """Update TASK.md with experiment results."""
    lines = open('TASK.md').readlines()
    new_lines = []
    for line in lines:
        for exp_id, lam, acc in results:
            tag = f"| {exp_id:03d}    | {lam}"
            if tag in line:
                status = "DONE" if acc > 0 else "RUNNING"
                line = f"| {exp_id:03d}    | {lam:<5}| {status:<7}| {acc:.2f}%   |     |\n"
        new_lines.append(line)
    with open('TASK.md', 'w') as f:
        f.writelines(new_lines)


def main():
    print("="*60)
    print("DAY_1: Lambda Sweep — Global-Local SupCon + CE")
    print(f"λ values: {LAMBDA_VALUES}")
    print(f"Total: {len(LAMBDA_VALUES)} experiments × 80 epochs")
    print("="*60)

    results = []
    for i, lam in enumerate(LAMBDA_VALUES):
        exp_id = i + 7
        print(f"\n{'='*60}")
        print(f"Experiment {exp_id}/{len(LAMBDA_VALUES)}: λ = {lam:.3f}")
        print(f"{'='*60}")

        acc = run_experiment(lam, exp_id, BASE_CONFIG)
        results.append((exp_id, lam, acc))

        # Update TASK.md
        update_task_md(results)

    # Final summary
    print("\n" + "="*60)
    print("DAY_1 SWEEP COMPLETE")
    print("="*60)
    print(f"{'λ':>8s} {'Acc':>8s}")
    print("-"*18)
    best_lam, best_acc = 0, 0
    for exp_id, lam, acc in results:
        marker = " ★" if acc == max(r[2] for r in results) else ""
        print(f"{lam:>8.3f} {acc:>7.2f}%{marker}")
        if acc > best_acc:
            best_acc = acc; best_lam = lam
    print(f"\nBest: λ={best_lam:.3f} → {best_acc:.2f}%")

    # Save sweep summary
    sweep_summary = {
        'results': [{'exp_id': e, 'lambda': l, 'best_acc': a} for e, l, a in results],
        'best_lambda': best_lam, 'best_acc': best_acc,
    }
    with open('DAY_1_sweep_summary.json', 'w') as f:
        json.dump(sweep_summary, f, indent=2)


if __name__ == '__main__':
    main()
