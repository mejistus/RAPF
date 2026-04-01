"""Global-Local Supervised Contrastive Model for fine-grained aircraft classification."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DINOv2Encoder, LoRALinear, inject_lora, SupConLoss


class LocalAggregator(nn.Module):
    """CLS-guided soft attention over patch tokens to extract object-aware local features."""
    def __init__(self, dim=768, tau=0.1):
        super().__init__()
        self.tau = tau
        # Learnable query/key projections for attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, cls_token, patch_tokens):
        """
        cls_token: [B, D]
        patch_tokens: [B, N, D]
        Returns: local_feat [B, D]
        """
        q = self.q_proj(cls_token).unsqueeze(1)    # [B, 1, D]
        k = self.k_proj(patch_tokens)               # [B, N, D]
        # Cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q * k).sum(dim=-1) / self.tau       # [B, N]
        attn = attn.softmax(dim=-1)                  # [B, N]
        # Weighted sum of patch tokens
        local_feat = (attn.unsqueeze(-1) * patch_tokens).sum(dim=1)  # [B, D]
        return local_feat, attn


class GlobalLocalContrastiveModel(nn.Module):
    """
    DINOv2 + LoRA with Global-Local Supervised Contrastive Learning.

    Architecture:
      Image → DINOv2+LoRA → [CLS], [patch_tokens]
        ├── classifier(CLS) → logits → CE
        ├── global_proj(CLS) → z_g → SupCon
        └── local_aggregator(CLS, patches) → local_feat
              └── local_proj → z_l → SupCon
    """
    def __init__(self, num_classes, dino_weights, lora_rank=8, proj_dim=128):
        super().__init__()
        self.encoder = DINOv2Encoder(dino_weights)
        self.lora_params = inject_lora(self.encoder, rank=lora_rank, alpha=lora_rank*2, targets=("qkv", "proj"))

        # Classification head (on CLS token)
        self.classifier = nn.Linear(768, num_classes)

        # Global contrastive projector (on CLS token)
        self.global_proj = nn.Sequential(
            nn.Linear(768, 768), nn.BatchNorm1d(768), nn.ReLU(inplace=True),
            nn.Linear(768, proj_dim),
        )

        # Local aggregation + projector
        self.local_agg = LocalAggregator(dim=768, tau=0.1)
        self.local_proj = nn.Sequential(
            nn.Linear(768, 768), nn.BatchNorm1d(768), nn.ReLU(inplace=True),
            nn.Linear(768, proj_dim),
        )

    def forward_features(self, x):
        """Forward through encoder, return CLS and patch tokens."""
        enc = self.encoder
        B = x.shape[0]
        x = enc.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([enc.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + enc.pos_embed
        for blk in enc.blocks:
            x = blk(x)
        x = enc.norm(x)
        cls_token = x[:, 0]       # [B, 768]
        patch_tokens = x[:, 1:]   # [B, N, 768]
        return cls_token, patch_tokens

    def forward(self, x):
        cls_feat, patch_tokens = self.forward_features(x)

        # Classification
        logits = self.classifier(cls_feat)

        # Global contrastive projection
        z_global = F.normalize(self.global_proj(cls_feat), dim=-1)

        # Local contrastive projection
        local_feat, attn_weights = self.local_agg(cls_feat, patch_tokens)
        z_local = F.normalize(self.local_proj(local_feat), dim=-1)

        return logits, z_global, z_local, attn_weights

    def get_trainable_params(self):
        return (list(self.classifier.parameters()) +
                list(self.global_proj.parameters()) +
                list(self.local_proj.parameters()) +
                list(self.local_agg.parameters()) +
                self.lora_params)
