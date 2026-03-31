"""Generate analysis for all 5 new lambda experiments (007-011)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json, torch, numpy as np
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from model_gl import GlobalLocalContrastiveModel

DEVICE = "cuda"
DATA_ROOT = "../data/fgvc_aircraft"
DINO_WEIGHTS = "/mnt/datasets/dinov2_vitb14.pth"
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
def denorm(t): return np.clip(t.cpu().numpy().transpose(1,2,0)*STD+MEAN, 0, 1)

test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist())])
test_ds = datasets.ImageFolder(f"{DATA_ROOT}/test", test_tf)
train_ds = datasets.ImageFolder(f"{DATA_ROOT}/train", test_tf)
class_names = test_ds.classes; nc = len(class_names)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

EXPS = [
    ("DAY_1_007_lambda1.500", 1.5),
    ("DAY_1_008_lambda2.500", 2.5),
    ("DAY_1_009_lambda4.000", 4.0),
    ("DAY_1_010_lambda6.500", 6.5),
    ("DAY_1_011_lambda10.000", 10.0),
]

# Diverse sample indices
div_idx = []; seen = set()
for i, (_, l) in enumerate(test_ds):
    if l not in seen and len(div_idx) < 16: div_idx.append(i); seen.add(l)
div_imgs = torch.stack([test_ds[i][0] for i in div_idx])
div_lbls = torch.tensor([test_ds[i][1] for i in div_idx])

def find_gt(cls, ds):
    for i, (_, l) in enumerate(ds):
        if l == cls: return ds[i][0]
    return None

for exp_dir, lam in EXPS:
    ckpt_path = f"{exp_dir}/best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"SKIP {exp_dir}"); continue
    out = f"{exp_dir}/analysis"; os.makedirs(out, exist_ok=True)

    model = GlobalLocalContrastiveModel(nc, DINO_WEIGHTS, 8, 128).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    acc = ckpt['test_acc']
    print(f"\n{'='*50}\n{exp_dir}: λ={lam}, acc={acc:.2f}%\n{'='*50}")

    # --- Confusion matrix + per-class acc ---
    ap, al = [], []
    with torch.no_grad(), autocast():
        for imgs, lbls in test_loader:
            ap.extend(model(imgs.to(DEVICE))[0].argmax(1).cpu().numpy())
            al.extend(lbls.numpy())
    cm = confusion_matrix(al, ap)
    pca = np.array([(cm[i,i]/cm[i].sum()*100 if cm[i].sum()>0 else 0) for i in range(nc)])

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.imshow(cm, cmap='Blues'); ax.set_title(f'λ={lam}, acc={acc:.1f}%', fontsize=13)
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels(class_names, rotation=90, fontsize=3.5); ax.set_yticklabels(class_names, fontsize=3.5)
    plt.tight_layout(); plt.savefig(f"{out}/confusion_matrix.png", dpi=200); plt.close()

    # Per-class accuracy
    si = np.argsort(pca)
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#d32f2f' if a<50 else '#ff9800' if a<70 else '#4caf50' if a<90 else '#1b5e20' for a in pca[si]]
    ax.barh(range(nc), pca[si], color=colors, height=0.8)
    ax.set_yticks(range(nc)); ax.set_yticklabels([class_names[i] for i in si], fontsize=4.5)
    ax.set_title(f'Per-class Acc λ={lam} (mean={pca.mean():.1f}%)', fontsize=12)
    ax.axvline(x=pca.mean(), color='k', linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(f"{out}/per_class_accuracy.png", dpi=150); plt.close()

    # --- GradCAM + Local attention ---
    imgs_d = div_imgs.to(DEVICE).requires_grad_(True)
    logits_d = model(imgs_d)[0]
    model.zero_grad()
    oh = torch.zeros_like(logits_d)
    for i in range(16): oh[i, div_lbls[i]] = 1
    (logits_d.float() * oh).sum().backward()
    grad = imgs_d.grad; ps = 14
    gp = grad.unfold(2,ps,ps).unfold(3,ps,ps).contiguous().view(grad.shape[0],3,16,16,-1)
    cam = gp.norm(dim=(1,4))
    for i in range(16): cam[i] = (cam[i]-cam[i].min())/(cam[i].max()-cam[i].min()+1e-8)
    cam = cam.detach().cpu()
    with torch.no_grad(), autocast():
        _, _, _, attn_w = model(div_imgs.to(DEVICE))
    attn_map = attn_w.cpu().float().reshape(-1, 16, 16)
    preds_d = logits_d.argmax(1).detach().cpu()

    fig, axes = plt.subplots(4, 12, figsize=(36, 16))
    for i in range(16):
        r, c = i//4, (i%4)*3
        img = denorm(div_imgs[i])
        axes[r,c].imshow(img); axes[r,c].set_title(f"GT:{class_names[div_lbls[i]]}", fontsize=6); axes[r,c].axis('off')
        cu = F.interpolate(cam[i:i+1].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze().numpy()
        axes[r,c+1].imshow(img); axes[r,c+1].imshow(cu, cmap='jet', alpha=0.5)
        clr = 'green' if preds_d[i]==div_lbls[i] else 'red'
        axes[r,c+1].set_title(f"GradCAM P:{class_names[preds_d[i]]}", fontsize=6, color=clr); axes[r,c+1].axis('off')
        au = F.interpolate(attn_map[i:i+1].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze().numpy()
        au = (au-au.min())/(au.max()-au.min()+1e-8)
        axes[r,c+2].imshow(img); axes[r,c+2].imshow(au, cmap='hot', alpha=0.5)
        axes[r,c+2].set_title("Local Attn", fontsize=6); axes[r,c+2].axis('off')
    plt.suptitle(f'GradCAM + Local Attention (λ={lam}, {acc:.1f}%)', fontsize=13)
    plt.tight_layout(); plt.savefig(f"{out}/attention_maps.png", dpi=200); plt.close()

    # --- Bad cases vs GT ---
    bcs = []
    with torch.no_grad(), autocast():
        for imgs_b, lbls_b in test_loader:
            imgs_b = imgs_b.to(DEVICE); logits_b = model(imgs_b)[0]
            probs_b = F.softmax(logits_b.float(), dim=1); preds_b = probs_b.argmax(1).cpu()
            for idx in (preds_b != lbls_b).nonzero(as_tuple=True)[0]:
                i = idx.item()
                bcs.append({'img':imgs_b[i].cpu(), 'true':lbls_b[i].item(), 'pred':preds_b[i].item(), 'conf':probs_b[i].max().item()})
            if len(bcs) >= 10: break
    n = min(len(bcs), 8)
    fig, axes = plt.subplots(n, 4, figsize=(18, n*3.2))
    if n == 1: axes = axes[np.newaxis, :]
    for i in range(n):
        bc = bcs[i]; img = denorm(bc['img'])
        axes[i,0].imshow(img); axes[i,0].set_title(f"T:{class_names[bc['true']]}\nP:{class_names[bc['pred']]} ({bc['conf']:.2f})", fontsize=6, color='red'); axes[i,0].axis('off')
        bc_img = bc['img'].to(DEVICE).unsqueeze(0).requires_grad_(True)
        lo = model(bc_img)[0]; model.zero_grad()
        oh2 = torch.zeros_like(lo); oh2[0, bc['true']] = 1
        (lo.float()*oh2).sum().backward()
        gp2 = bc_img.grad.unfold(2,14,14).unfold(3,14,14).contiguous().view(1,3,16,16,-1)
        cm2 = gp2.norm(dim=(1,4)); cm2 = (cm2-cm2.min())/(cm2.max()-cm2.min()+1e-8)
        cu2 = F.interpolate(cm2.cpu().unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze().numpy()
        axes[i,1].imshow(img); axes[i,1].imshow(cu2, cmap='jet', alpha=0.5); axes[i,1].set_title('GradCAM(true)', fontsize=6); axes[i,1].axis('off')
        gt_t = find_gt(bc['true'], train_ds)
        if gt_t is not None: axes[i,2].imshow(denorm(gt_t))
        axes[i,2].set_title(f"GT: {class_names[bc['true']]}", fontsize=6, color='green'); axes[i,2].axis('off')
        gt_p = find_gt(bc['pred'], train_ds)
        if gt_p is not None: axes[i,3].imshow(denorm(gt_p))
        axes[i,3].set_title(f"GT: {class_names[bc['pred']]}", fontsize=6, color='orange'); axes[i,3].axis('off')
    plt.suptitle(f'Bad Cases vs GT (λ={lam})', fontsize=12)
    plt.tight_layout(); plt.savefig(f"{out}/badcases_vs_gt.png", dpi=200); plt.close()

    # --- Hierarchical clustering ---
    cf = torch.zeros(nc, 768); cc = torch.zeros(nc)
    with torch.no_grad(), autocast():
        for imgs_c, lbls_c in test_loader:
            cls_t = model.forward_features(imgs_c.to(DEVICE))[0].float().cpu()
            for i in range(len(lbls_c)): cf[lbls_c[i]] += cls_t[i]; cc[lbls_c[i]] += 1
    cf = F.normalize(cf / cc.unsqueeze(1).clamp(min=1), dim=-1)
    sim = (cf @ cf.T).numpy(); dist = 1-sim; np.fill_diagonal(dist, 0)
    dist = (dist + dist.T)/2; dist = np.maximum(dist, 0)
    Z = linkage(squareform(dist), method='ward')
    fig, axes = plt.subplots(1, 2, figsize=(26, 13))
    dd = dendrogram(Z, labels=class_names, ax=axes[0], leaf_rotation=90, leaf_font_size=4.5)
    axes[0].set_title(f'Hierarchical Clustering (λ={lam})', fontsize=11)
    order = dd['leaves']; so = sim[np.ix_(order, order)]
    axes[1].imshow(so, cmap='RdBu_r', vmin=-0.2, vmax=1)
    axes[1].set_xticks(range(nc)); axes[1].set_yticks(range(nc))
    axes[1].set_xticklabels([class_names[i] for i in order], rotation=90, fontsize=3.5)
    axes[1].set_yticklabels([class_names[i] for i in order], fontsize=3.5)
    axes[1].set_title('Similarity Matrix', fontsize=11)
    plt.tight_layout(); plt.savefig(f"{out}/class_clustering.png", dpi=200); plt.close()

    print(f"  All saved to {out}/")
    del model; torch.cuda.empty_cache()

# --- Grand summary plot ---
all_results = []
for d in sorted(os.listdir('.')):
    if d.startswith('DAY_1_') and os.path.isfile(f"{d}/summary.json"):
        s = json.load(open(f"{d}/summary.json"))
        all_results.append((s['lambda'], s['best_test_acc']))
all_results.sort()
fig, ax = plt.subplots(figsize=(10, 6))
lams, accs = zip(*all_results)
ax.plot(lams, accs, 'bo-', markersize=8, linewidth=2)
best_i = np.argmax(accs)
ax.plot(lams[best_i], accs[best_i], 'r*', markersize=20)
for l, a in zip(lams, accs):
    ax.annotate(f'{a:.1f}%', (l, a), textcoords="offset points", xytext=(0,12), fontsize=8, ha='center')
ax.set_xlabel('λ (Contrastive Loss Weight)', fontsize=12)
ax.set_ylabel('Best Test Accuracy (%)', fontsize=12)
ax.set_title('Global-Local SupCon λ Sweep (DINOv2+LoRA r=8)', fontsize=13)
ax.grid(True, alpha=0.3); ax.set_xscale('log')
plt.tight_layout(); plt.savefig('DAY_1_lambda_sweep_full.png', dpi=150); plt.close()
print(f"\nSaved DAY_1_lambda_sweep_full.png")
print("Done!")
