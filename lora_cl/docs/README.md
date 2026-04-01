# DINOv2 + LoRA + Contrastive Learning for FGVC-Aircraft

Fine-grained aircraft variant recognition using DINOv2 ViT-B/14 with LoRA adaptation and supervised contrastive learning.

## Architecture

- **Backbone**: DINOv2 ViT-B/14 (frozen)
- **Adaptation**: LoRA rank=8, alpha=16, targets: qkv + proj (all 12 blocks)
- **Projection Head**: 768 -> 768 (BN+ReLU) -> 128 (for contrastive learning)
- **Classifier**: 768 -> 100 classes
- **Trainable params**: ~1.2M

## Dataset

FGVC-Aircraft (100 variants), train/test split, input 224x224.

---

## Version History

### V0 — Baseline: LoRA + SupCon (output/)

**Config**: 80 epochs, lr=1e-3, lr_lora=5e-4, batch=48x2(accum), T=0.1, lambda_cl=0.5, lambda_ce=1.0, label_smoothing=0.1

**Results**:
| Metric | Value |
|---|---|
| Best test acc | **78.67%** (epoch 73) |
| Mean per-class acc | 78.67% |
| Std per-class acc | 18.65% |

**Worst 10 classes**: 737-300 (30.3%), A320 (33.3%), 777-200 (39.4%), 737-500 (41.2%), 767-300 (41.2%), 747-100 (42.4%), BAE_146-200 (42.4%), C-47 (45.5%), 747-200 (47.1%), 757-200 (50.0%)

**Top confused pairs**: BAE_146-200↔BAE_146-300 (19), C-47↔DC-3 (18+13), 737-300↔737-400 (13), 747-100↔747-200 (13), MD-11↔DC-10 (12)

**Observations**:
- Contrastive learning provides strong representation, pos_sim ~0.62, neg_sim ~0.14
- Worst classes are same-series variants (737-300/400/500, 747-100/200/300)
- C-47/DC-3 confusion is fundamentally difficult (same airframe, different designation)
- Attention maps show some patches attend to background/environment instead of aircraft structure

---

### V1 — CutMix + Attention Entropy + Hard Negatives (output-1/)

**Base**: resume from V0 best_model.pth

**Config**: 20 epochs, lr=2e-4, lr_lora=1e-4, cutmix_prob=0.5, lambda_attn_ent=0.1, hard_neg SupCon (top-5, weight=1.5), label_smoothing=0.15, stronger augmentation (channel shuffle/invert/drop, perspective, affine)

**Results**:
| Metric | Value |
|---|---|
| Best test acc | **78.67%** (no improvement) |
| Mean per-class acc | 76.90% (dropped 1.77%) |
| Std per-class acc | 19.45% (increased) |
| Classes <50% | 11 (was 10) |

**Worst 10 classes**: 737-300 (24.2%), A320 (30.3%), 777-200 (36.4%), 767-300 (38.2%), 757-200 (41.2%), MD-80 (41.2%), 747-200 (44.1%), 747-100 (45.5%), BAE_146-200 (45.5%), 747-300 (48.5%)

**Analysis**:
- 20 epochs too short for the model to adapt to aggressive augmentation
- CutMix + channel shuffle/drop may be too destructive for fine-grained features
- Attention entropy regularization (lambda=0.1) didn't significantly change attention patterns (entropy stayed ~0.68 throughout)
- Hard negative mining didn't help — hard negatives in FGVC are genuinely confusable
- 737-300 accuracy dropped from 30.3% to 24.2% — more augmentation hurt discrimination

---

## Key Problems

### 1. Same-series variant confusion (fundamental)
737-300/400/500, 747-100/200/300/400, BAE_146-200/300, A340-200/300 etc. are nearly identical visually — differences are in fuselage length, engine type, or winglet shape, often invisible at typical photo angles.

### 2. C-47 / DC-3 (label noise)
C-47 is the military variant of DC-3 — structurally identical. Human accuracy is near-random. This is essentially label noise.

### 3. Background attention
Attention maps in V0 show CLS token attending to runway, sky, buildings instead of aircraft structure. CutMix (V1) was intended to fix this but 20 epochs was insufficient and the augmentation was too aggressive.

---

## Planned Experiments (V2+)

Each version incrementally fine-tunes from the previous best checkpoint for **40 epochs** (increased from 20).

### V2 — Focus: Longer training + Gentler attention guidance
- Resume from V0 (not V1, since V1 didn't improve and may have degraded representations)
- **40 epochs**, lr=1e-4, lr_lora=5e-5 (lower, incremental refinement)
- Remove destructive augmentations (channel shuffle/invert/drop) — keep standard augmentation
- Attention entropy regularization with lower weight (lambda=0.03) — gentle guidance
- Keep SupCon (original, not hard neg version — hard neg didn't help)
- Consider class-balanced sampling to help tail classes

### V3 — Focus: Hierarchical/Group-aware learning
- Merge confusable series into super-classes for auxiliary loss (e.g., "737-family")
- Hierarchical CE: coarse-level (family) + fine-level (variant) joint training
- This acknowledges that 737-300→737-400 is a much smaller error than 737-300→A320

### V4 — Focus: Part-based attention
- Multi-crop strategy: force model to attend to discriminative parts (nose, engines, tail, winglets)
- Token selection: use only top-k attended patch tokens (not just CLS) for classification
- GradCAM-guided crop: augment with crops around discriminative regions

### V5 — Focus: Test-time strategies
- Test-time augmentation (TTA): multi-crop + flip ensemble
- Feature ensemble: average features from multiple crops
- Confusion-aware calibration: post-hoc calibration for confusable pairs

### Other ideas to explore
- **Label smoothing per-pair**: higher smoothing for confusable pairs (C-47/DC-3)
- **Merge C-47/DC-3**: treat as same class since they're physically identical
- **Focal loss**: reduce easy-class dominance, focus on hard classes
- **Mixup on features** (not images): mix embeddings to create harder training signal
- **Progressive unfreezing**: gradually unfreeze deeper DINOv2 blocks

---

## File Structure

```
lora_cl/
├── train.py          # V0: baseline training (80 epochs)
├── train_v1.py       # V1: CutMix + attn entropy + hard neg (20 epochs)
├── model.py          # DINOv2 + LoRA + contrastive model
├── dataset.py        # TwoView dataset + augmentation
├── visualize.py      # Attention maps, curves, badcases, confusion matrix
├── output/           # V0 results
│   ├── best_model.pth
│   ├── metrics.json, summary.json, config.json
│   ├── attention_maps.png, confusion_matrix.png, training_curves.png
│   ├── badcases.txt, badcases.png, confused_pairs.txt
├── output-1/         # V1 results (no improvement)
│   └── (same structure, no best_model saved since no improvement)
└── README.md
```
