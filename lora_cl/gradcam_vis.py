"""GradCAM visualization — improved for ViT with CLS-token classifier."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from model import DINOv2ContrastiveModel

DEVICE = "cuda"
DATA_ROOT = "../data/fgvc_aircraft"
DINO_WEIGHTS = "/mnt/datasets/dinov2_vitb14.pth"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


class ViTGradCAM:
    """
    Rollout-weighted GradCAM for ViT.
    Instead of hooking activations, we use attention rollout * gradient
    to get per-patch importance that reflects classification decision.
    """
    def __init__(self, model):
        self.model = model
        self.attentions = []
        # Hook ALL blocks to capture attention
        for blk in model.encoder.blocks:
            blk.attn.register_forward_hook(self._save_attn)

    def _save_attn(self, module, input, output):
        # Store attention AFTER softmax
        if hasattr(module, '_attn_map'):
            self.attentions.append(module._attn_map)

    @torch.enable_grad()
    def compute(self, images, target_class=None):
        self.model.eval()
        images = images.to(DEVICE).requires_grad_(True)

        # Forward with attention recording
        self.attentions = []
        # Need to enable return_attn for ALL blocks
        for blk in self.model.encoder.blocks:
            blk._return_attn = True

        # Override forward to capture all attentions
        feat = self._forward_with_attn(images)
        logits = self.model.classifier(feat)
        probs = F.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)
        confs = probs.max(dim=1)[0]

        if target_class is None:
            target_class = preds

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        for i in range(len(images)):
            one_hot[i, target_class[i]] = 1
        (logits.float() * one_hot).sum().backward()

        # Method: use input gradient w.r.t. patch embeddings
        # Get gradient of logits w.r.t. the image
        img_grad = images.grad  # [B, 3, 224, 224]

        # Compute per-patch importance from image gradient
        patch_size = 14
        B, C, H, W = img_grad.shape
        nH, nW = H // patch_size, W // patch_size

        # Reshape to patches and compute gradient magnitude
        grad_patches = img_grad.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # [B, 3, nH, nW, patch_size, patch_size]
        grad_patches = grad_patches.contiguous().view(B, C, nH, nW, -1)
        # Per-patch L2 norm of gradient
        cam = grad_patches.norm(dim=(1, 4))  # [B, nH, nW]

        # Normalize per sample
        for i in range(B):
            c = cam[i]
            cam[i] = (c - c.min()) / (c.max() - c.min() + 1e-8)

        return cam.detach().cpu(), preds.detach().cpu(), confs.detach().cpu()

    def _forward_with_attn(self, x):
        """Forward through encoder, recording attention at every layer."""
        enc = self.model.encoder
        B = x.shape[0]
        x = enc.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([enc.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + enc.pos_embed
        for blk in enc.blocks:
            x = blk(x, return_attn=True)
        x = enc.norm(x)
        return x[:, 0]


def plot_grid(model, test_ds, class_names, save_path, title, num=16):
    cam_tool = ViTGradCAM(model)

    indices = []
    seen = set()
    for i in range(len(test_ds)):
        _, l = test_ds[i]
        if l not in seen and len(indices) < num:
            indices.append(i); seen.add(l)

    images = torch.stack([test_ds[i][0] for i in indices])
    labels = torch.tensor([test_ds[i][1] for i in indices])

    cam, preds, confs = cam_tool.compute(images, target_class=labels)

    fig, axes = plt.subplots(4, 8, figsize=(28, 16))
    for i in range(min(num, 16)):
        row, col = i // 4, (i % 4) * 2
        img = denormalize(images[i])

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"GT: {class_names[labels[i]]}", fontsize=7)
        axes[row, col].axis('off')

        cam_up = F.interpolate(cam[i:i+1].unsqueeze(0).float(),
                               size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
        axes[row, col+1].imshow(img)
        axes[row, col+1].imshow(cam_up, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        color = 'green' if preds[i] == labels[i] else 'red'
        axes[row, col+1].set_title(f"P: {class_names[preds[i]]} ({confs[i]:.2f})", fontsize=7, color=color)
        axes[row, col+1].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_badcases(model, test_ds, class_names, save_path, title, max_cases=8):
    cam_tool = ViTGradCAM(model)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    bad_imgs, bad_lbls, bad_preds_list = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            _, _, logits = model(imgs.to(DEVICE))
            ps = logits.argmax(1).cpu()
            for idx in (ps != lbls).nonzero(as_tuple=True)[0]:
                bad_imgs.append(imgs[idx])
                bad_lbls.append(lbls[idx].item())
                bad_preds_list.append(ps[idx].item())
            if len(bad_imgs) >= max_cases:
                break

    n = min(len(bad_imgs), max_cases)
    images = torch.stack(bad_imgs[:n])
    labels = torch.tensor(bad_lbls[:n])
    pred_labels = torch.tensor(bad_preds_list[:n])

    cam_true, _, _ = cam_tool.compute(images, target_class=labels)
    cam_pred, preds, confs = cam_tool.compute(images, target_class=pred_labels)

    fig, axes = plt.subplots(n, 5, figsize=(22, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]
    col_titles = ["Original", "GradCAM (true class)", "GradCAM (pred class)", "Difference", "Heatmap (true)"]

    for i in range(n):
        img = denormalize(images[i])
        ct = F.interpolate(cam_true[i:i+1].unsqueeze(0).float(), size=(224,224),
                           mode='bilinear', align_corners=False).squeeze().numpy()
        cp = F.interpolate(cam_pred[i:i+1].unsqueeze(0).float(), size=(224,224),
                           mode='bilinear', align_corners=False).squeeze().numpy()

        axes[i,0].imshow(img)
        axes[i,0].set_title(f"GT:{class_names[labels[i]]}\nPred:{class_names[pred_labels[i]]} ({confs[i]:.2f})", fontsize=7, color='red')
        axes[i,0].axis('off')

        axes[i,1].imshow(img); axes[i,1].imshow(ct, cmap='jet', alpha=0.5)
        axes[i,1].set_title(col_titles[1], fontsize=7, color='green'); axes[i,1].axis('off')

        axes[i,2].imshow(img); axes[i,2].imshow(cp, cmap='jet', alpha=0.5)
        axes[i,2].set_title(col_titles[2], fontsize=7, color='red'); axes[i,2].axis('off')

        diff = np.abs(ct - cp)
        axes[i,3].imshow(img); axes[i,3].imshow(diff, cmap='hot', alpha=0.6)
        axes[i,3].set_title(col_titles[3], fontsize=7); axes[i,3].axis('off')

        axes[i,4].imshow(ct, cmap='jet'); axes[i,4].set_title(col_titles[4], fontsize=7); axes[i,4].axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, 'test'), test_transform)
    class_names = test_ds.classes
    num_classes = len(class_names)

    ckpt_path = "output/best_model.pth"
    out_dir = "output"

    print(f"Loading {ckpt_path}")
    model = DINOv2ContrastiveModel(num_classes, DINO_WEIGHTS, lora_rank=8, proj_dim=128).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded: epoch={ckpt['epoch']}, acc={ckpt['test_acc']:.2f}%")

    plot_grid(model, test_ds, class_names,
              os.path.join(out_dir, "gradcam_diverse.png"),
              "GradCAM — Diverse Classes (input gradient magnitude)")

    plot_badcases(model, test_ds, class_names,
                  os.path.join(out_dir, "gradcam_badcases.png"),
                  "Bad Cases — True vs Predicted GradCAM")

    print("Done!")

if __name__ == '__main__':
    main()
