"""LoRA (Low-Rank Adaptation) module for DINOv2."""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper around a frozen nn.Linear."""
    def __init__(self, orig: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.orig = orig  # frozen
        d_in, d_out = orig.in_features, orig.out_features
        self.lora_A = nn.Parameter(torch.empty(d_in, rank))
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank

    def forward(self, x):
        return self.orig(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def inject_lora(dino_encoder, rank=16, alpha=32, targets=("qkv", "proj")):
    """Inject LoRA into DINOv2 encoder attention layers. Returns list of LoRA params."""
    count = 0
    for block in dino_encoder.blocks:
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

    # Freeze all, then unfreeze LoRA
    for p in dino_encoder.parameters():
        p.requires_grad = False
    lora_params = []
    for m in dino_encoder.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            lora_params.extend([m.lora_A, m.lora_B])

    total = sum(p.numel() for p in lora_params)
    print(f"LoRA injected: {count} modules, {total:,} trainable params, rank={rank}")
    return lora_params


def get_lora_state_dict(dino_encoder):
    """Extract LoRA parameters as a state dict for saving."""
    state = {}
    for name, module in dino_encoder.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.data.clone()
            state[f"{name}.lora_B"] = module.lora_B.data.clone()
    return state


def load_lora_state_dict(dino_encoder, state):
    """Load LoRA parameters from a saved state dict."""
    for name, module in dino_encoder.named_modules():
        if isinstance(module, LoRALinear):
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state:
                module.lora_A.data.copy_(state[key_a])
                module.lora_B.data.copy_(state[key_b])
