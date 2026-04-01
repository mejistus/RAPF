# FGVC-Aircraft Fine-grained Classification with DINOv2+LoRA+Contrastive Learning

## 目标
在 DINOv2 ViT-B/14 + LoRA(r=8) 上实现 **Global-Local Supervised Contrastive Learning**，
超越当前最优 80.05% (V3)。

## 核心方法
```
Image → DINOv2+LoRA → [CLS], [patch_tokens]
  ├── classifier(CLS) → CE loss
  ├── global_proj(CLS) → z_g → SupCon_global
  └── local_aggregator(softmax(cos(CLS, patches)/τ) · patches)
        └── local_proj → z_l → SupCon_local

Loss = CE + λ * (SupCon_global + 0.5 * SupCon_local)
```

## 实验计划

### DAY_1: Lambda Sweep (6 experiments × 80 epochs)
| Exp ID | λ    | Status | Test Acc | KNN |
|--------|------|--------|----------|-----|
| 001    | 0.01 | DONE   | 77.26%   |     |
| 002    | 0.025| DONE   | 76.15%   |     |
| 003    | 0.063| DONE   | 77.68%   |     |
| 004    | 0.156| DONE   | 78.52%   |     |
| 005    | 0.391| DONE   | 80.74%   |     |
| 006    | 0.977| DONE   | 81.73%   |     |

### 共同配置
- Backbone: DINOv2 ViT-B/14 + LoRA rank=8 (qkv, proj)
- PK采样: P=8, K=8 (batch=64)
- 80 epochs, cosine LR + 5 epoch warmup
- lr=1e-3 (heads), lr_lora=5e-4
- SupCon temperature=0.1
- Local aggregation: CLS-guided soft attention over patch tokens

### 后续计划
- [ ] 找到最优λ后，加入hard negative mining
- [ ] 基于confusion matrix对易混淆类对加权
- [ ] 尝试更大rank(16)
- [ ] 数据增强消融实验

## 历史最优
| 版本 | 方法 | Acc |
|------|------|-----|
| V0   | CE+SupCon端到端(随机采样) | 78.67% |
| V3   | V0→SupCon(PK)→CE两阶段 | 80.05% |
| V5   | 从头SupCon(PK)→CE两阶段 | 78.52% |
