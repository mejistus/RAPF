"""Generate comprehensive analysis for the best model (DAY_1_006 λ=0.977, 81.73%)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from model_gl import GlobalLocalContrastiveModel

DEVICE = "cuda"
BEST_DIR = "DAY_1_006_lambda0.977"
DATA_ROOT = "../data/fgvc_aircraft"
DINO_WEIGHTS = "/mnt/datasets/dinov2_vitb14.pth"
OUT = f"{BEST_DIR}/analysis"
os.makedirs(OUT, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denorm(t):
    return np.clip(t.cpu().numpy().transpose(1,2,0)*STD+MEAN, 0, 1)

# Load
test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist())])
test_ds = datasets.ImageFolder(f"{DATA_ROOT}/test", test_tf)
train_ds = datasets.ImageFolder(f"{DATA_ROOT}/train", test_tf)
class_names = test_ds.classes
num_classes = len(class_names)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)

model = GlobalLocalContrastiveModel(num_classes, DINO_WEIGHTS, 8, 128).to(DEVICE)
ckpt = torch.load(f"{BEST_DIR}/best_model.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded: λ={ckpt['lambda']}, acc={ckpt['test_acc']:.2f}%, epoch={ckpt['epoch']}")

# ==================== 1. Training curves ====================
print("1. Training curves...")
metrics = json.load(open(f"{BEST_DIR}/metrics.json"))
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0,0].plot(metrics['epoch'], metrics['ce_loss'], 'r-', label='CE'); axes[0,0].plot(metrics['epoch'], metrics['cl_global'], 'b-', label='CL_global'); axes[0,0].plot(metrics['epoch'], metrics['cl_local'], 'g-', label='CL_local')
axes[0,0].set_title('Losses'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(metrics['epoch'], metrics['train_acc'], 'b-', label='Train'); axes[0,1].plot(metrics['epoch'], metrics['test_acc'], 'r-', label='Test')
axes[0,1].set_title('Accuracy'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[0,2].plot(metrics['epoch'], metrics['pos_sim_g'], 'b-', label='Pos(G)'); axes[0,2].plot(metrics['epoch'], metrics['neg_sim_g'], 'b--', label='Neg(G)')
axes[0,2].plot(metrics['epoch'], metrics['pos_sim_l'], 'r-', label='Pos(L)'); axes[0,2].plot(metrics['epoch'], metrics['neg_sim_l'], 'r--', label='Neg(L)')
axes[0,2].set_title('Contrastive Similarities'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

axes[1,0].plot(metrics['epoch'], metrics['lr']); axes[1,0].set_title('Learning Rate'); axes[1,0].grid(True, alpha=0.3)
axes[1,1].plot(metrics['epoch'], metrics['attn_entropy']); axes[1,1].set_title('Local Attention Entropy'); axes[1,1].grid(True, alpha=0.3)

# Lambda sweep comparison
sweep = json.load(open('DAY_1_sweep_summary.json'))
lams = [r['lambda'] for r in sweep['results']]; accs = [r['best_acc'] for r in sweep['results']]
axes[1,2].plot(lams, accs, 'bo-', markersize=8); axes[1,2].set_xlabel('λ'); axes[1,2].set_ylabel('Best Acc (%)')
axes[1,2].set_title('Lambda Sweep'); axes[1,2].grid(True, alpha=0.3)
for l, a in zip(lams, accs): axes[1,2].annotate(f'{a:.1f}%', (l, a), textcoords="offset points", xytext=(0,10), fontsize=8, ha='center')

plt.tight_layout(); plt.savefig(f"{OUT}/training_curves.png", dpi=150); plt.close()
print(f"  Saved {OUT}/training_curves.png")

# ==================== 2. Confusion matrix ====================
print("2. Confusion matrix...")
all_preds, all_labels = [], []
with torch.no_grad(), autocast():
    for imgs, lbls in test_loader:
        logits = model(imgs.to(DEVICE))[0]
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(lbls.numpy())
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Confusion Matrix (λ=0.977, acc={ckpt["test_acc"]:.1f}%)', fontsize=14)
ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=90, fontsize=4); ax.set_yticklabels(class_names, fontsize=4)
plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=200); plt.close()
print(f"  Saved {OUT}/confusion_matrix.png")

# Top confused pairs
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
pairs = []
for _ in range(20):
    idx = np.unravel_index(cm_off.argmax(), cm_off.shape)
    pairs.append((class_names[idx[0]], class_names[idx[1]], int(cm_off[idx])))
    cm_off[idx] = 0
with open(f"{OUT}/confused_pairs.txt", 'w') as f:
    f.write("Top 20 Most Confused Pairs:\n\n")
    for t, p, c in pairs: f.write(f"  {t:>25s} → {p:<25s} ({c} times)\n")

# ==================== 3. Per-class accuracy ====================
print("3. Per-class accuracy...")
pca = np.array([(cm[i,i]/cm[i].sum()*100 if cm[i].sum()>0 else 0) for i in range(num_classes)])
sorted_idx = np.argsort(pca)

fig, ax = plt.subplots(figsize=(16, 10))
colors = ['#d32f2f' if a < 50 else '#ff9800' if a < 70 else '#4caf50' if a < 90 else '#1b5e20' for a in pca[sorted_idx]]
bars = ax.barh(range(num_classes), pca[sorted_idx], color=colors, height=0.8)
ax.set_yticks(range(num_classes)); ax.set_yticklabels([class_names[i] for i in sorted_idx], fontsize=5)
ax.set_xlabel('Accuracy (%)', fontsize=11); ax.set_title(f'Per-class Accuracy (mean={pca.mean():.1f}%, std={pca.std():.1f}%)', fontsize=13)
ax.axvline(x=pca.mean(), color='k', linestyle='--', alpha=0.5, label=f'Mean={pca.mean():.1f}%')
ax.legend(); ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/per_class_accuracy.png", dpi=150); plt.close()
print(f"  Saved {OUT}/per_class_accuracy.png")

# ==================== 4. GradCAM attention maps ====================
print("4. GradCAM attention maps...")
from gradcam_vis import ViTGradCAM

class GLGradCAM:
    """GradCAM adapted for GlobalLocalContrastiveModel."""
    def __init__(self, model):
        self.model = model
    @torch.enable_grad()
    def compute(self, images, target_class=None):
        self.model.eval()
        images = images.to(DEVICE).requires_grad_(True)
        logits = self.model(images)[0]
        preds = logits.argmax(dim=1)
        confs = F.softmax(logits.float(), dim=1).max(dim=1)[0]
        if target_class is None: target_class = preds
        self.model.zero_grad()
        oh = torch.zeros_like(logits)
        for i in range(len(images)): oh[i, target_class[i]] = 1
        (logits.float() * oh).sum().backward()
        grad = images.grad
        ps = 14; B, C, H, W = grad.shape
        nH, nW = H//ps, W//ps
        gp = grad.unfold(2,ps,ps).unfold(3,ps,ps).contiguous().view(B,C,nH,nW,-1)
        cam = gp.norm(dim=(1,4))
        for i in range(B): cam[i] = (cam[i]-cam[i].min())/(cam[i].max()-cam[i].min()+1e-8)
        return cam.detach().cpu(), preds.detach().cpu(), confs.detach().cpu()

gcam = GLGradCAM(model)
# Diverse samples
indices = []; seen = set()
for i,(_, l) in enumerate(test_ds):
    if l not in seen and len(indices) < 16: indices.append(i); seen.add(l)
imgs = torch.stack([test_ds[i][0] for i in indices])
lbls = torch.tensor([test_ds[i][1] for i in indices])
cam, preds, confs = gcam.compute(imgs, lbls)

# Also get local attention maps
with torch.no_grad(), autocast():
    _, _, _, attn_w = model(imgs.to(DEVICE))  # attn_w: [B, N]
attn_map = attn_w.cpu().float().reshape(-1, 16, 16)

fig, axes = plt.subplots(4, 12, figsize=(36, 16))
for i in range(16):
    r, c = i//4, (i%4)*3
    img = denorm(imgs[i])
    # Original
    axes[r,c].imshow(img); axes[r,c].set_title(f"GT:{class_names[lbls[i]]}", fontsize=6); axes[r,c].axis('off')
    # GradCAM
    cam_up = F.interpolate(cam[i:i+1].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze().numpy()
    axes[r,c+1].imshow(img); axes[r,c+1].imshow(cam_up, cmap='jet', alpha=0.5)
    color = 'green' if preds[i]==lbls[i] else 'red'
    axes[r,c+1].set_title(f"GradCAM P:{class_names[preds[i]]}", fontsize=6, color=color); axes[r,c+1].axis('off')
    # Local attention
    attn_up = F.interpolate(attn_map[i:i+1].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze().numpy()
    attn_up = (attn_up - attn_up.min())/(attn_up.max()-attn_up.min()+1e-8)
    axes[r,c+2].imshow(img); axes[r,c+2].imshow(attn_up, cmap='hot', alpha=0.5)
    axes[r,c+2].set_title("Local Attn", fontsize=6); axes[r,c+2].axis('off')

plt.suptitle(f'GradCAM + Local Attention (λ=0.977, {ckpt["test_acc"]:.1f}%)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{OUT}/attention_maps.png", dpi=200); plt.close()
print(f"  Saved {OUT}/attention_maps.png")

# ==================== 5. Bad cases with GT comparison ====================
print("5. Bad cases analysis...")
badcases = []
with torch.no_grad(), autocast():
    for imgs_b, lbls_b in test_loader:
        imgs_b = imgs_b.to(DEVICE)
        logits_b = model(imgs_b)[0]
        probs_b = F.softmax(logits_b.float(), dim=1)
        preds_b = probs_b.argmax(1).cpu()
        wrong = preds_b != lbls_b
        for idx in wrong.nonzero(as_tuple=True)[0]:
            i = idx.item()
            badcases.append({'img': imgs_b[i].cpu(), 'true': lbls_b[i].item(),
                            'pred': preds_b[i].item(), 'conf': probs_b[i].max().item(),
                            'true_conf': probs_b[i, lbls_b[i]].item()})
        if len(badcases) >= 30: break

# Find GT examples for confused classes
def find_gt_example(class_idx, dataset):
    for i, (_, l) in enumerate(dataset):
        if l == class_idx: return dataset[i][0]
    return None

n = min(len(badcases), 10)
fig, axes = plt.subplots(n, 4, figsize=(20, n*3.5))
col_titles = ['Bad Case', 'GradCAM', 'GT: True Class', 'GT: Pred Class']
for i in range(n):
    bc = badcases[i]
    img = denorm(bc['img'])
    # Bad case
    axes[i,0].imshow(img)
    axes[i,0].set_title(f"True: {class_names[bc['true']]}\nPred: {class_names[bc['pred']]} ({bc['conf']:.2f})", fontsize=7, color='red')
    axes[i,0].axis('off')
    # GradCAM of bad case
    cam_bc, _, _ = gcam.compute(bc['img'].unsqueeze(0), torch.tensor([bc['true']]))
    cam_up = F.interpolate(cam_bc[:1].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze().numpy()
    axes[i,1].imshow(img); axes[i,1].imshow(cam_up, cmap='jet', alpha=0.5)
    axes[i,1].set_title('GradCAM (true class)', fontsize=7); axes[i,1].axis('off')
    # GT true class example
    gt_true = find_gt_example(bc['true'], train_ds)
    if gt_true is not None:
        axes[i,2].imshow(denorm(gt_true))
        axes[i,2].set_title(f"GT: {class_names[bc['true']]}", fontsize=7, color='green')
    axes[i,2].axis('off')
    # GT pred class example
    gt_pred = find_gt_example(bc['pred'], train_ds)
    if gt_pred is not None:
        axes[i,3].imshow(denorm(gt_pred))
        axes[i,3].set_title(f"GT: {class_names[bc['pred']]}", fontsize=7, color='orange')
    axes[i,3].axis('off')

plt.suptitle('Bad Cases with Ground Truth Comparison', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{OUT}/badcases_vs_gt.png", dpi=200); plt.close()
print(f"  Saved {OUT}/badcases_vs_gt.png")

# ==================== 6. Hierarchical class similarity clustering ====================
print("6. Class feature clustering...")
class_feats = torch.zeros(num_classes, 768)
class_counts = torch.zeros(num_classes)
with torch.no_grad(), autocast():
    for imgs_c, lbls_c in test_loader:
        cls_tok = model.forward_features(imgs_c.to(DEVICE))[0].float().cpu()
        for i in range(len(lbls_c)):
            class_feats[lbls_c[i]] += cls_tok[i]
            class_counts[lbls_c[i]] += 1
class_feats = class_feats / class_counts.unsqueeze(1).clamp(min=1)
class_feats = F.normalize(class_feats, dim=-1)

# Cosine similarity matrix
sim = (class_feats @ class_feats.T).numpy()

# Hierarchical clustering
dist = 1 - sim
np.fill_diagonal(dist, 0)
dist = np.maximum(dist, 0)
dist = (dist + dist.T) / 2; condensed = squareform(dist)
Z = linkage(condensed, method='ward')

fig, axes = plt.subplots(1, 2, figsize=(28, 14))
# Dendrogram
dendro = dendrogram(Z, labels=class_names, ax=axes[0], leaf_rotation=90, leaf_font_size=5,
                     color_threshold=0.7*max(Z[:,2]))
axes[0].set_title('Hierarchical Clustering (Ward linkage on CLS features)', fontsize=12)
axes[0].set_ylabel('Distance')
# Reorder similarity matrix by clustering
order = dendro['leaves']
sim_ordered = sim[np.ix_(order, order)]
im = axes[1].imshow(sim_ordered, cmap='RdBu_r', vmin=-0.2, vmax=1)
axes[1].set_xticks(range(num_classes)); axes[1].set_yticks(range(num_classes))
axes[1].set_xticklabels([class_names[i] for i in order], rotation=90, fontsize=4)
axes[1].set_yticklabels([class_names[i] for i in order], fontsize=4)
axes[1].set_title('Class Similarity Matrix (cosine, clustered)', fontsize=12)
plt.colorbar(im, ax=axes[1])
plt.tight_layout(); plt.savefig(f"{OUT}/class_clustering.png", dpi=200); plt.close()
print(f"  Saved {OUT}/class_clustering.png")

# ==================== 7. t-SNE of test features ====================
print("7. t-SNE visualization...")
from sklearn.manifold import TSNE

all_feats, all_lbls = [], []
with torch.no_grad(), autocast():
    for imgs_t, lbls_t in test_loader:
        cls_t = model.forward_features(imgs_t.to(DEVICE))[0].float().cpu()
        all_feats.append(cls_t); all_lbls.append(lbls_t)
all_feats = torch.cat(all_feats).numpy()
all_lbls = torch.cat(all_lbls).numpy()

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
emb = tsne.fit_transform(all_feats)

fig, ax = plt.subplots(figsize=(14, 14))
scatter = ax.scatter(emb[:, 0], emb[:, 1], c=all_lbls, cmap='tab20', s=5, alpha=0.6)
ax.set_title(f't-SNE of Test Set CLS Features (100 classes, {ckpt["test_acc"]:.1f}%)', fontsize=13)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/tsne.png", dpi=150); plt.close()
print(f"  Saved {OUT}/tsne.png")

# ==================== Summary ====================
summary = {
    'model': f'DAY_1_006 λ=0.977',
    'test_acc': ckpt['test_acc'],
    'mean_per_class_acc': float(pca.mean()),
    'std_per_class_acc': float(pca.std()),
    'classes_below_50': int((pca < 50).sum()),
    'classes_above_90': int((pca >= 90).sum()),
    'top_confused': pairs[:10],
    'files': os.listdir(OUT),
}
with open(f"{OUT}/analysis_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"All analysis saved to {OUT}/")
print(f"Test acc: {ckpt['test_acc']:.2f}%")
print(f"Per-class: mean={pca.mean():.1f}% std={pca.std():.1f}%")
print(f"Classes <50%: {(pca<50).sum()}, ≥90%: {(pca>=90).sum()}")
print(f"{'='*60}")
