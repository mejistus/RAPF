"""Train best config (GL-SupCon λ=4.0 + HardNeg w=3.0) on merged datasets."""
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
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from model_gl import GlobalLocalContrastiveModel
from train_hardneg import HardNegSupConLoss, HARD_PAIRS
from pk_sampler import PKSampler

MEAN = np.array([0.485,0.456,0.406]); STD = np.array([0.229,0.224,0.225])
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def denorm(t): return np.clip(t.cpu().numpy().transpose(1,2,0)*STD+MEAN, 0, 1)

BASE = {
    'dino_weights': '/mnt/datasets/dinov2_vitb14.pth',
    'lora_rank': 8, 'proj_dim': 128,
    'P': 8, 'K': 8,
    'epochs': 80,
    'lr': 1e-3, 'lr_lora': 5e-4,
    'warmup_epochs': 5,
    'temperature': 0.1,
    'lambda_cl': 4.0, 'local_weight': 0.5,
    'hard_weight': 3.0,
    'label_smoothing': 0.1,
    'weight_decay': 5e-4,
    'num_workers': 4, 'seed': 42, 'grad_clip': 1.0,
}

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def run(data_root, out_dir, tag):
    cfg = {**BASE, 'data_root': data_root}
    os.makedirs(out_dir, exist_ok=True)

    seed_all(cfg['seed']); device = torch.device("cuda")

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
        transforms.RandomGrayscale(0.2), transforms.ColorJitter(0.4,0.4,0.4,0.15),
        transforms.GaussianBlur(5, sigma=(0.1,2.0)),
        transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist()),
    ])
    tf_test = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist()),
    ])

    train_ds = datasets.ImageFolder(f"{data_root}/train", tf_train)
    test_ds = datasets.ImageFolder(f"{data_root}/test", tf_test)
    nc = len(train_ds.classes); cn = train_ds.classes
    print(f"\n{'='*60}\n{tag}: {nc} classes, {len(train_ds)} train, {len(test_ds)} test\n{'='*60}")

    # Adjust P if fewer classes
    P = min(cfg['P'], nc)
    K = cfg['K']
    pk = PKSampler(train_ds.targets, p=P, k=K)
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
    w = cfg['warmup_epochs']; T = cfg['epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda ep: (ep+1)/w if ep < w else 0.5*(1+math.cos(math.pi*(ep-w)/(T-w))))
    scaler = GradScaler()

    supcon_fn = HardNegSupConLoss(cfg['temperature'], cfg['hard_weight'], cn)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    lam = cfg['lambda_cl']; lw = cfg['local_weight']

    metrics = {k: [] for k in ['epoch','loss','ce','clg','cll','train_acc','test_acc']}
    best_acc = 0; t0 = time.time()

    for epoch in range(T):
        model.train()
        el=ec=eg=eL=0; cor=tot=0; nb=0
        for images, labels in tqdm(train_loader, desc=f"Ep{epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits, zg, zl, _ = model(images)
                lce = ce_fn(logits, labels)
                lg = supcon_fn(zg, labels)
                ll = supcon_fn(zl, labels)
                loss = lce + lam*(lg + lw*ll)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer); scaler.update()
            el+=loss.item(); ec+=lce.item(); eg+=lg.item(); eL+=ll.item()
            cor+=(logits.argmax(1)==labels).sum().item(); tot+=len(labels); nb+=1
        scheduler.step()

        model.eval(); tc=tt=0
        with torch.no_grad(), autocast():
            for imgs, lbls in test_loader:
                tc+=(model(imgs.to(device))[0].argmax(1)==lbls.to(device)).sum().item(); tt+=len(lbls)
        te = 100*tc/tt
        if te > best_acc:
            best_acc = te
            torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),'test_acc':te},
                       f"{out_dir}/best_model.pth")

        for k,v in zip(['epoch','loss','ce','clg','cll','train_acc','test_acc'],
                        [epoch+1,el/nb,ec/nb,eg/nb,eL/nb,100*cor/tot,te]):
            metrics[k].append(v)

        if (epoch+1)%10==0:
            print(f"  Ep {epoch+1}: ce={ec/nb:.3f} | train={100*cor/tot:.1f}% test={te:.2f}% best={best_acc:.2f}% | {(time.time()-t0)/60:.0f}min")

    # Save metrics
    with open(f"{out_dir}/metrics.json",'w') as f: json.dump(metrics, f)

    # === Analysis ===
    model.load_state_dict(torch.load(f"{out_dir}/best_model.pth", map_location=device, weights_only=False)['model_state_dict'])
    model.eval()

    # Confusion matrix
    ap,al=[],[]
    with torch.no_grad(), autocast():
        for imgs,lbls in test_loader:
            ap.extend(model(imgs.to(device))[0].argmax(1).cpu().numpy()); al.extend(lbls.numpy())
    cm = confusion_matrix(al, ap)
    pca = np.array([(cm[i,i]/cm[i].sum()*100 if cm[i].sum()>0 else 0) for i in range(nc)])

    # Per-class accuracy plot
    si = np.argsort(pca)
    fig,ax = plt.subplots(figsize=(14,max(8, nc*0.12)))
    colors = ['#d32f2f' if a<50 else '#ff9800' if a<70 else '#4caf50' if a<90 else '#1b5e20' for a in pca[si]]
    ax.barh(range(nc), pca[si], color=colors, height=0.8)
    ax.set_yticks(range(nc)); ax.set_yticklabels([cn[i] for i in si], fontsize=5)
    ax.set_title(f'{tag} Per-class Acc (mean={pca.mean():.1f}%, {nc} classes)', fontsize=12)
    ax.axvline(x=pca.mean(), color='k', linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(f"{out_dir}/per_class_accuracy.png", dpi=150); plt.close()

    # Confusion matrix plot
    fig,ax = plt.subplots(figsize=(max(12,nc*0.18), max(10,nc*0.16)))
    ax.imshow(cm, cmap='Blues'); ax.set_title(f'{tag} ({nc} cls, {best_acc:.1f}%)')
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels(cn, rotation=90, fontsize=max(3,6-nc//20)); ax.set_yticklabels(cn, fontsize=max(3,6-nc//20))
    plt.tight_layout(); plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=200); plt.close()

    # Clustering
    cf = torch.zeros(nc, 768); cc = torch.zeros(nc)
    with torch.no_grad(), autocast():
        for imgs,lbls in test_loader:
            cls_t = model.forward_features(imgs.to(device))[0].float().cpu()
            for i in range(len(lbls)): cf[lbls[i]]+=cls_t[i]; cc[lbls[i]]+=1
    cf = F.normalize(cf/cc.unsqueeze(1).clamp(min=1), dim=-1)
    sim = (cf@cf.T).numpy(); dist = 1-sim; np.fill_diagonal(dist,0); dist=(dist+dist.T)/2; dist=np.maximum(dist,0)
    Z = linkage(squareform(dist), method='ward')
    fig,axes = plt.subplots(1,2, figsize=(24,max(10,nc*0.12)))
    dd = dendrogram(Z, labels=cn, ax=axes[0], leaf_rotation=90, leaf_font_size=max(3,6-nc//20))
    axes[0].set_title(f'{tag} Clustering')
    order=dd['leaves']; so=sim[np.ix_(order,order)]
    axes[1].imshow(so, cmap='RdBu_r', vmin=-0.2, vmax=1)
    axes[1].set_xticks(range(nc)); axes[1].set_yticks(range(nc))
    axes[1].set_xticklabels([cn[i] for i in order], rotation=90, fontsize=max(3,5-nc//25))
    axes[1].set_yticklabels([cn[i] for i in order], fontsize=max(3,5-nc//25))
    plt.tight_layout(); plt.savefig(f"{out_dir}/clustering.png", dpi=200); plt.close()

    summary = {'tag': tag, 'num_classes': nc, 'best_acc': best_acc,
               'mean_pca': float(pca.mean()), 'std_pca': float(pca.std()),
               'below_50': int((pca<50).sum()), 'above_90': int((pca>=90).sum()),
               'time_min': (time.time()-t0)/60}
    with open(f"{out_dir}/summary.json",'w') as f: json.dump(summary, f, indent=2)
    print(f"\n  {tag} DONE: {best_acc:.2f}%, mean_pca={pca.mean():.1f}%, <50%:{(pca<50).sum()}, >=90%:{(pca>=90).sum()}")

    with open(f"{out_dir}/config.json",'w') as f: json.dump({**cfg, 'tag': tag, 'num_classes': nc}, f, indent=2)
    del model; torch.cuda.empty_cache()
    return best_acc


if __name__ == '__main__':
    results = []
    for data, out, tag in [
        ("../data/fgvc_mild", "DAY_2_mild_97cls", "Mild (97 cls)"),
        ("../data/fgvc_moderate", "DAY_2_moderate_76cls", "Moderate (76 cls)"),
        ("../data/fgvc_aggressive", "DAY_2_aggressive_60cls", "Aggressive (60 cls)"),
    ]:
        acc = run(data, out, tag)
        results.append((tag, acc))

    print(f"\n{'='*60}\nMerge Comparison\n{'='*60}")
    print(f"  {'Original (100 cls)':>25s}: 82.99% (DAY_1_014)")
    for tag, acc in results:
        print(f"  {tag:>25s}: {acc:.2f}%")
