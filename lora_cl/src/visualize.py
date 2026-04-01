"""Visualization: attention maps, training curves, bad cases."""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert normalized tensor back to displayable image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def plot_attention_map(model, images, labels, class_names, save_dir, device, num_samples=16):
    """Generate and save attention maps from the last transformer block."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    with torch.no_grad(), torch.cuda.amp.autocast():
        feat, proj, logits = model(images, return_attn=True)
        preds = logits.argmax(dim=1).cpu()

    attn = model.encoder.get_last_attn()  # [B, heads, N, N]
    # Average over heads, take CLS token attention to patches
    attn_cls = attn[:, :, 0, 1:].mean(dim=1)  # [B, num_patches]
    num_patches_side = int(attn_cls.shape[1] ** 0.5)  # 16
    attn_map = attn_cls.reshape(-1, num_patches_side, num_patches_side)  # [B, 16, 16]

    fig, axes = plt.subplots(4, 8, figsize=(24, 14))
    for i in range(min(num_samples, 16)):
        row, col = i // 4, (i % 4) * 2

        # Original image
        ax_img = axes[row, col]
        img = denormalize(images[i])
        ax_img.imshow(img)
        true_name = class_names[labels[i]]
        pred_name = class_names[preds[i]]
        color = 'green' if preds[i] == labels[i] else 'red'
        ax_img.set_title(f"GT: {true_name}", fontsize=7, color='black')
        ax_img.axis('off')

        # Attention overlay
        ax_attn = axes[row, col + 1]
        attn_resized = F.interpolate(
            attn_map[i:i+1].unsqueeze(0).float(),
            size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

        ax_attn.imshow(img)
        ax_attn.imshow(attn_resized, cmap='jet', alpha=0.5)
        ax_attn.set_title(f"Pred: {pred_name}", fontsize=7, color=color)
        ax_attn.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_maps.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Attention maps saved to {save_dir}/attention_maps.png")


def plot_training_curves(metrics, save_dir):
    """Plot loss, accuracy, lr curves from training metrics."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Train Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Total Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. CE Loss vs Contrastive Loss
    ax = axes[0, 1]
    ax.plot(metrics['epoch'], metrics['ce_loss'], 'r-', label='CE Loss')
    ax.plot(metrics['epoch'], metrics['cl_loss'], 'g-', label='SupCon Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('CE vs Contrastive Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Accuracy
    ax = axes[0, 2]
    ax.plot(metrics['epoch'], metrics['train_acc'], 'b-', label='Train Acc')
    ax.plot(metrics['epoch'], metrics['test_acc'], 'r-', label='Test Acc')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Learning Rate
    ax = axes[1, 0]
    ax.plot(metrics['epoch'], metrics['lr'], 'k-')
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR'); ax.set_title('Learning Rate')
    ax.grid(True, alpha=0.3)

    # 5. Gradient Norms
    ax = axes[1, 1]
    ax.plot(metrics['epoch'], metrics['lora_grad_norm'], 'b-', label='LoRA Grad')
    ax.plot(metrics['epoch'], metrics['head_grad_norm'], 'r-', label='Head Grad')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Grad Norm'); ax.set_title('Gradient Norms')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. Per-class accuracy distribution
    ax = axes[1, 2]
    if 'per_class_acc' in metrics and len(metrics['per_class_acc']) > 0:
        last_per_class = metrics['per_class_acc'][-1]
        ax.bar(range(len(last_per_class)), sorted(last_per_class, reverse=True), color='steelblue', alpha=0.7)
        ax.set_xlabel('Class (sorted)'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Per-class Acc (mean={np.mean(last_per_class):.1f}%)')
        ax.axhline(y=np.mean(last_per_class), color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def save_badcases(model, test_loader, class_names, save_dir, device, max_cases=50):
    """Save misclassified samples with predictions and confidences."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    badcases = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits = model(images)
            probs = F.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1)[0]

            wrong_mask = preds != labels
            for i in wrong_mask.nonzero(as_tuple=True)[0]:
                idx = i.item()
                badcases.append({
                    'image': images[idx].cpu(),
                    'true': labels[idx].item(),
                    'pred': preds[idx].item(),
                    'conf': confs[idx].item(),
                    'true_conf': probs[idx, labels[idx]].item(),
                    'top5_preds': probs[idx].topk(5)[1].cpu().tolist(),
                    'top5_confs': probs[idx].topk(5)[0].cpu().tolist(),
                })
            if len(badcases) >= max_cases:
                break

    badcases = badcases[:max_cases]

    # Save summary text
    with open(os.path.join(save_dir, "badcases.txt"), 'w') as f:
        f.write(f"Total bad cases shown: {len(badcases)}\n\n")
        for i, bc in enumerate(badcases):
            f.write(f"Case {i+1}:\n")
            f.write(f"  True: {class_names[bc['true']]} (conf={bc['true_conf']:.3f})\n")
            f.write(f"  Pred: {class_names[bc['pred']]} (conf={bc['conf']:.3f})\n")
            f.write(f"  Top5: {[(class_names[p], f'{c:.3f}') for p, c in zip(bc['top5_preds'], bc['top5_confs'])]}\n\n")

    # Plot badcases grid
    n = min(len(badcases), 25)
    rows = 5
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for i in range(n):
        ax = axes[i // cols, i % cols]
        img = denormalize(badcases[i]['image'])
        ax.imshow(img)
        true_name = class_names[badcases[i]['true']]
        pred_name = class_names[badcases[i]['pred']]
        conf = badcases[i]['conf']
        ax.set_title(f"T:{true_name}\nP:{pred_name} ({conf:.2f})", fontsize=7, color='red')
        ax.axis('off')
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "badcases.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bad cases saved to {save_dir}/badcases.txt and badcases.png")
    return badcases


def plot_confusion_matrix(model, test_loader, class_names, save_dir, device):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_preds, all_labels = [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels in test_loader:
            images = images.to(device)
            _, _, logits = model(images)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (100 classes)', fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")

    # Save most confused pairs
    np.fill_diagonal(cm, 0)
    confused_pairs = []
    for _ in range(20):
        idx = np.unravel_index(cm.argmax(), cm.shape)
        confused_pairs.append((class_names[idx[0]], class_names[idx[1]], cm[idx[0], idx[1]]))
        cm[idx[0], idx[1]] = 0

    with open(os.path.join(save_dir, "confused_pairs.txt"), 'w') as f:
        f.write("Top 20 Most Confused Class Pairs:\n\n")
        for true_cls, pred_cls, count in confused_pairs:
            f.write(f"  {true_cls:>25s} -> {pred_cls:<25s} ({count} times)\n")
    print(f"Confused pairs saved to {save_dir}/confused_pairs.txt")
