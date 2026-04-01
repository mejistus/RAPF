"""DINOv2 + LoRA with supervised contrastive learning for FGVC-Aircraft."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== LoRA ====================
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


def inject_lora(encoder, rank=8, alpha=16, targets=("qkv", "proj")):
    count = 0
    for block in encoder.blocks:
        for name in targets:
            if name == "qkv":
                block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha); count += 1
            elif name == "proj":
                block.attn.proj = LoRALinear(block.attn.proj, rank, alpha); count += 1
            elif name == "fc1":
                block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, alpha); count += 1
            elif name == "fc2":
                block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, alpha); count += 1
    for p in encoder.parameters():
        p.requires_grad = False
    lora_params = []
    for m in encoder.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            lora_params.extend([m.lora_A, m.lora_B])
    total = sum(p.numel() for p in lora_params)
    print(f"LoRA: {count} modules, {total:,} trainable, rank={rank}")
    return lora_params


# ==================== DINOv2 Encoder ====================
class DINOv2Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if return_attn:
            self._attn_map = attn.detach()
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DINOv2MLP(nn.Module):
    def __init__(self, dim=768, hidden_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DINOv2Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv2Attention(dim, num_heads)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOv2MLP(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x, return_attn=False):
        x = x + self.ls1 * self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DINOv2Encoder(nn.Module):
    def __init__(self, weights_path=None, input_size=224):
        super().__init__()
        self.embed_dim = 768
        self.patch_size = 14
        self.input_size = input_size
        num_patches = (input_size // self.patch_size) ** 2

        self.patch_embed = nn.Sequential()
        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, 768))
        self.blocks = nn.ModuleList([DINOv2Block(768, 12, 4.0) for _ in range(12)])
        self.norm = nn.LayerNorm(768)

        if weights_path:
            import os
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict.pop("mask_token", None)
                pos_embed = state_dict["pos_embed"]
                cls_pe = pos_embed[:, :1]
                patch_pe = pos_embed[:, 1:]
                old_size = int(patch_pe.shape[1] ** 0.5)
                new_size = input_size // self.patch_size
                patch_pe = patch_pe.reshape(1, old_size, old_size, 768).permute(0, 3, 1, 2)
                patch_pe = nn.functional.interpolate(patch_pe.float(), size=(new_size, new_size), mode="bicubic", align_corners=False)
                patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_size * new_size, 768)
                state_dict["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)
                new_sd = {}
                for k, v in state_dict.items():
                    if ".ls1.gamma" in k: new_sd[k.replace(".ls1.gamma", ".ls1")] = v
                    elif ".ls2.gamma" in k: new_sd[k.replace(".ls2.gamma", ".ls2")] = v
                    else: new_sd[k] = v
                self.load_state_dict(new_sd, strict=False)
                print(f"DINOv2 loaded: {len(new_sd)} keys")

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for i, blk in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1)
            x = blk(x, return_attn=(return_attn and is_last))
        x = self.norm(x)
        return x[:, 0]  # CLS token

    def get_last_attn(self):
        """Return last block attention map [B, heads, N, N]."""
        return self.blocks[-1].attn._attn_map


# ==================== Full Model ====================
class DINOv2ContrastiveModel(nn.Module):
    def __init__(self, num_classes, dino_weights, lora_rank=8, proj_dim=128):
        super().__init__()
        self.encoder = DINOv2Encoder(dino_weights)
        self.lora_params = inject_lora(self.encoder, rank=lora_rank, alpha=lora_rank * 2, targets=("qkv", "proj"))
        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, proj_dim),
        )
        # Classification head
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x, return_attn=False):
        feat = self.encoder(x, return_attn=return_attn)  # [B, 768]
        proj = self.proj_head(feat)  # [B, proj_dim]
        proj = F.normalize(proj, dim=-1)
        logits = self.classifier(feat)  # [B, num_classes]
        return feat, proj, logits

    def get_trainable_params(self):
        params = list(self.proj_head.parameters()) + list(self.classifier.parameters()) + self.lora_params
        return params


# ==================== Supervised Contrastive Loss ====================
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss (Khosla et al. 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [B, D] L2-normalized
        labels: [B]
        """
        device = features.device
        B = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B] same-class mask

        # Compute logits
        logits = features @ features.T / self.temperature  # [B, B]
        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        num_positives = mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / num_positives
        loss = -mean_log_prob.mean()
        return loss
