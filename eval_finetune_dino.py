"""LoRA fine-tune DINOv2 on FGVC-Aircraft. Two-phase: warm up head first, then joint."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import sys, math

sys.path.insert(0, os.path.dirname(__file__))
from continual_clip.models import DINOv2Encoder

DEVICE = "cuda"
DATA_ROOT = "data/fgvc_aircraft"
DINO_WEIGHTS = "/mnt/datasets/dinov2_vitb14.pth"
NUM_WORKERS = 4


class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.orig = orig
        d_in, d_out = orig.in_features, orig.out_features
        self.lora_A = nn.Parameter(torch.empty(d_in, rank))
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank

    def forward(self, x):
        return self.orig(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def inject_lora(model, rank=8, alpha=16, targets=("qkv", "proj")):
    count = 0
    for block in model.blocks:
        for name in targets:
            if name == "qkv":
                block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
                count += 1
            elif name == "proj":
                block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)
                count += 1
            elif name == "fc1":
                block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, alpha)
                count += 1
            elif name == "fc2":
                block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, alpha)
                count += 1
    for p in model.parameters():
        p.requires_grad = False
    lora_params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            lora_params.extend([m.lora_A, m.lora_B])
    total = sum(p.numel() for p in lora_params)
    print(f"LoRA: {count} modules, {total:,} trainable, rank={rank}")
    return lora_params


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes, rank=8, targets=("qkv", "proj")):
        super().__init__()
        self.encoder = DINOv2Encoder(DINO_WEIGHTS, device="cpu")
        self.lora_params = inject_lora(self.encoder, rank=rank, alpha=rank*2, targets=targets)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


# Data
dino_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(), dino_norm,
])
test_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), dino_norm,
])
train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=test_transform)
num_classes = len(train_ds.classes)
print(f"Dataset: {num_classes} classes, {len(train_ds)} train, {len(test_ds)} test")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += len(labels)
    return 100 * correct / total

def run(rank, targets, head_epochs=10, joint_epochs=40, head_lr=0.05, lora_lr=5e-4):
    print(f"\n{'='*60}")
    print(f"rank={rank}, targets={targets}, head_ep={head_epochs}, joint_ep={joint_epochs}")
    print(f"{'='*60}")
    model = DINOv2Classifier(num_classes, rank=rank, targets=targets).to(DEVICE)

    # Phase 1: train head only (LoRA frozen at zero)
    print("Phase 1: Train head only")
    for p in model.lora_params:
        p.requires_grad = False
    opt1 = torch.optim.SGD(model.head.parameters(), lr=head_lr, momentum=0.9, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=head_epochs)
    best = 0
    for ep in range(head_epochs):
        model.train()
        correct = total = 0
        for imgs, labels in tqdm(train_loader, desc=f"H-{ep+1}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = F.cross_entropy(model(imgs), labels)
            opt1.zero_grad(); loss.backward(); opt1.step()
            correct += (model(imgs).detach().argmax(1) == labels).sum().item()
            total += len(labels)
        sch1.step()
        acc = evaluate(model)
        best = max(best, acc)
        print(f"  H-Ep {ep+1:>2d}: train={100*correct/total:.1f}%, test={acc:.2f}%, best={best:.2f}%")

    # Phase 2: joint train LoRA + head
    print("Phase 2: Joint LoRA + head")
    for p in model.lora_params:
        p.requires_grad = True
    opt2 = torch.optim.AdamW([
        {"params": model.lora_params, "lr": lora_lr},
        {"params": model.head.parameters(), "lr": lora_lr * 2},
    ], weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=joint_epochs, eta_min=1e-6)
    for ep in range(joint_epochs):
        model.train()
        correct = total = 0
        for imgs, labels in tqdm(train_loader, desc=f"J-{ep+1}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            correct += (logits.detach().argmax(1) == labels).sum().item()
            total += len(labels)
        sch2.step()
        acc = evaluate(model)
        best = max(best, acc)
        print(f"  J-Ep {ep+1:>2d}: train={100*correct/total:.1f}%, test={acc:.2f}%, best={best:.2f}%")

    print(f"\n>>> rank={rank} targets={targets} | Best: {best:.2f}% <<<\n")
    del model, opt1, opt2; torch.cuda.empty_cache()
    return best

results = {}
# LoRA on attention (qkv+proj)
for r in [4, 8, 16]:
    results[f"attn_r{r}"] = run(r, ("qkv", "proj"))

# LoRA on all (qkv+proj+fc1+fc2), rank=8
results["all_r8"] = run(8, ("qkv", "proj", "fc1", "fc2"))

print("\n" + "="*60)
print("SUMMARY")
for k, v in results.items():
    print(f"  {k:>15s}: {v:.2f}%")
print("="*60)
