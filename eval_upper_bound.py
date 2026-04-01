"""Evaluate upper-bound accuracy of CLIP / DINOv2 / CLIP+DINOv2 on FGVC-Aircraft (no incremental learning)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import clip
import sys
sys.path.insert(0, os.path.dirname(__file__))
from continual_clip.models import DINOv2Encoder

DEVICE = "cuda"
DATA_ROOT = "data/fgvc_aircraft"
DINO_WEIGHTS = "/mnt/datasets/dinov2_vitb14.pth"
BATCH_SIZE = 128
NUM_WORKERS = 4

# ---- Transforms ----
clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
dino_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
base_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=base_transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=base_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

num_classes = len(train_ds.classes)
print(f"Dataset: {num_classes} classes, {len(train_ds)} train, {len(test_ds)} test")

# ---- Load models ----
clip_model, _ = clip.load("ViT-B/16", device=DEVICE, jit=False)
clip_model.eval()
clip_visual = clip_model.visual
clip_dtype = clip_model.dtype

dino_model = DINOv2Encoder(DINO_WEIGHTS, device=DEVICE).to(DEVICE)
dino_model.eval()

# ---- Feature extraction ----
@torch.no_grad()
def extract_features(loader, desc=""):
    clip_feats, dino_feats, labels = [], [], []
    for imgs, tgts in tqdm(loader, desc=desc):
        imgs = imgs.to(DEVICE)
        # CLIP
        clip_input = clip_normalize(imgs).to(clip_dtype)
        cf = clip_visual(clip_input).float()
        # DINOv2
        dino_input = dino_normalize(imgs)
        df = dino_model(dino_input)
        
        clip_feats.append(cf.cpu())
        dino_feats.append(df.cpu())
        labels.append(tgts)
    return torch.cat(clip_feats), torch.cat(dino_feats), torch.cat(labels)

print("Extracting train features...")
train_clip, train_dino, train_labels = extract_features(train_loader, "Train")
print("Extracting test features...")
test_clip, test_dino, test_labels = extract_features(test_loader, "Test")

# L2-normalize features
def l2norm(x):
    return x / x.norm(dim=-1, keepdim=True)

train_clip_n = l2norm(train_clip)
test_clip_n  = l2norm(test_clip)
train_dino_n = l2norm(train_dino)
test_dino_n  = l2norm(test_dino)

# Concatenated
train_both = torch.cat([train_clip_n, train_dino_n], dim=-1)
test_both  = torch.cat([test_clip_n, test_dino_n], dim=-1)
train_both_n = l2norm(train_both)
test_both_n  = l2norm(test_both)

print(f"\nFeature dims: CLIP={train_clip.shape[1]}, DINOv2={train_dino.shape[1]}, Concat={train_both.shape[1]}")

# ---- 1. Zero-shot CLIP ----
from continual_clip.utils import AIRCRAFT_DESCRIPTIVE_NAMES
class_names = train_ds.classes  # folder names

# Build text features
descriptive = [AIRCRAFT_DESCRIPTIVE_NAMES.get(c, c) for c in class_names]
text_tokens = clip.tokenize([f"a photograph of a {d}." for d in descriptive]).to(DEVICE)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens).float()
    text_features = l2norm(text_features)

zs_logits = test_clip_n @ text_features.cpu().t()
zs_preds = zs_logits.argmax(dim=1)
zs_acc = (zs_preds == test_labels).float().mean().item() * 100
print(f"\n[Zero-shot CLIP + descriptive names] Acc: {zs_acc:.2f}%")

# Also try generic prompt
text_tokens_g = clip.tokenize([f"a photo of a {c} aircraft." for c in class_names]).to(DEVICE)
with torch.no_grad():
    text_features_g = clip_model.encode_text(text_tokens_g).float()
    text_features_g = l2norm(text_features_g)
zs_logits_g = test_clip_n @ text_features_g.cpu().t()
zs_acc_g = (zs_logits_g.argmax(dim=1) == test_labels).float().mean().item() * 100
print(f"[Zero-shot CLIP + generic names]      Acc: {zs_acc_g:.2f}%")

# ---- 2. KNN (k=20) ----
def knn_eval(train_f, test_f, train_y, test_y, k=20, name=""):
    # cosine similarity
    sim = test_f @ train_f.t()
    topk_sim, topk_idx = sim.topk(k, dim=1)
    topk_labels = train_y[topk_idx]
    # weighted vote
    preds = []
    for i in range(len(test_f)):
        counts = torch.zeros(num_classes)
        for j in range(k):
            counts[topk_labels[i, j]] += topk_sim[i, j]
        preds.append(counts.argmax())
    preds = torch.stack(preds)
    acc = (preds == test_y).float().mean().item() * 100
    print(f"[KNN k={k} {name}] Acc: {acc:.2f}%")
    return acc

print()
knn_eval(train_clip_n, test_clip_n, train_labels, test_labels, k=20, name="CLIP")
knn_eval(train_dino_n, test_dino_n, train_labels, test_labels, k=20, name="DINOv2")
knn_eval(train_both_n, test_both_n, train_labels, test_labels, k=20, name="CLIP+DINOv2")

# ---- 3. Linear probe (SGD, 100 epochs) ----
def linear_probe(train_f, test_f, train_y, test_y, name="", epochs=100, lr=0.1):
    dim = train_f.shape[1]
    clf = nn.Linear(dim, num_classes).cuda()
    optimizer = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_f_gpu = train_f.cuda()
    train_y_gpu = train_y.cuda()
    
    bs = 256
    best_acc = 0
    for ep in range(epochs):
        clf.train()
        perm = torch.randperm(len(train_f_gpu))
        total_loss = 0
        for i in range(0, len(train_f_gpu), bs):
            idx = perm[i:i+bs]
            logits = clf(train_f_gpu[idx])
            loss = nn.functional.cross_entropy(logits, train_y_gpu[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            clf.eval()
            with torch.no_grad():
                test_logits = clf(test_f.cuda())
                acc = (test_logits.argmax(1) == test_y.cuda()).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc
            print(f"  [{name}] Epoch {ep+1}: acc={acc:.2f}%  (best={best_acc:.2f}%)")
    
    print(f"[Linear Probe {name}] Best Acc: {best_acc:.2f}%")
    return best_acc

print("\n--- Linear Probe ---")
linear_probe(train_clip_n, test_clip_n, train_labels, test_labels, name="CLIP", epochs=100)
linear_probe(train_dino_n, test_dino_n, train_labels, test_labels, name="DINOv2", epochs=100)
linear_probe(train_both_n, test_both_n, train_labels, test_labels, name="CLIP+DINOv2", epochs=100)

# ---- 4. Linear probe on raw (unnormalized) concat with different weighting ----
print("\n--- Weighted Concat Linear Probe ---")
for alpha in [0.3, 0.5, 0.7, 1.0]:
    wf_train = torch.cat([train_clip_n * alpha, train_dino_n * (1-alpha)], dim=-1)
    wf_test  = torch.cat([test_clip_n * alpha,  test_dino_n * (1-alpha)], dim=-1)
    linear_probe(wf_train, wf_test, train_labels, test_labels, name=f"CLIP*{alpha}+DINO*{1-alpha:.1f}", epochs=100)

print("\nDone!")
