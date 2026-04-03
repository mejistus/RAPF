# FGVC-70 Incremental Learning Benchmark

## Dataset: fgvc_moderate_v2 (70 classes)
FGVC-Aircraft with same-type sub-variant merges only.
Merged: 737 Classic/NG, 747, 757, 767, 777, A320 Family, A330, A340, MD-80 Series, ERJ, DHC8, E-Jet, CRJ, DC-3/C-47, BAE 146.
All different aircraft types (707, DC-8, DC-10, MD-11, A300, A310, Boeing 717, DC-9, etc.) kept as separate classes.

---

## Results Summary

### Setting 1: init=10, inc=10 (7 tasks)

| Method | Backbone | Last Acc | Avg Acc |
|--------|----------|:--------:|:-------:|
| MOS (AAAI'25) | ViT-B/16 (IN1K) | 49.41% | 55.96% |
| RAPF | CLIP ViT-B/16 + DINOv2 ViT-B/14 + LoRA r16 | 55.15% | 75.31% |
| **MOS (AAAI'25)** | **DINOv2 ViT-B/14** | **86.08%** | **88.81%** |

### Setting 2: init=40, inc=2 (16 tasks)

| Method | Backbone | Last Acc | Avg Acc |
|--------|----------|:--------:|:-------:|
| MOS (AAAI'25) | ViT-B/16 (IN1K) | 53.62% | 55.27% |
| RAPF | CLIP ViT-B/16 + DINOv2 ViT-B/14 + LoRA r16 | 54.97% | 66.10% |
| **MOS (AAAI'25)** | **DINOv2 ViT-B/14** | **87.55%** | **88.86%** |

---

## Detailed Curves

### RAPF (CLIP+DINOv2+LoRA) — 10-10

| Task | Classes | Acc | Avg Acc | Forgetting | BWT |
|:----:|:-------:|:---:|:-------:|:----------:|:---:|
| 0 | 10 | 94.75% | 94.75% | 0.00 | 0.00 |
| 1 | 20 | 81.03% | 87.89% | 0.00 | 0.00 |
| 2 | 30 | 80.81% | 85.53% | 1.31 | -0.88 |
| 3 | 40 | 78.03% | 83.65% | 0.20 | -0.41 |
| 4 | 50 | 69.50% | 80.82% | 9.63 | -2.16 |
| 5 | 60 | 67.91% | 78.67% | 10.07 | -2.33 |
| 6 | 70 | 55.15% | 75.31% | 20.47 | -5.52 |

### MOS (ViT-B/16) — 10-10

top1 curve: [56.61, 65.97, 60.26, 54.74, 54.89, 49.84, 49.41]
Average Accuracy: 55.96%

### MOS (DINOv2) — 10-10

top1 curve: [87.28, 89.96, 90.80, 88.88, 88.54, 86.15, 86.08]
Average Accuracy: 88.81%

### RAPF (CLIP+DINOv2+LoRA) — 40-2

| Task | Classes | Acc | Avg Acc | Forgetting | BWT |
|:----:|:-------:|:---:|:-------:|:----------:|:---:|
| 0 | 40 | 88.32% | 88.32% | 0.00 | 0.00 |
| 5 | 50 | 63.46% | 74.64% | 33.12 | -24.91 |
| 10 | 60 | 61.01% | 69.90% | 15.41 | -20.22 |
| 15 | 70 | 54.97% | 66.10% | 38.17 | -24.41 |

### MOS (ViT-B/16) — 40-2

top1 curve: [35.11, 57.15, 58.53, 60.0, 60.79, 60.85, 60.64, 58.55, 57.81, 56.3, 55.16, 55.11, 54.42, 53.82, 53.75, 53.62]
Average Accuracy: 55.27%

### MOS (DINOv2) — 40-2

top1 curve: [78.35, 90.85, 90.96, 91.33, 91.32, 90.74, 90.88, 89.95, 89.29, 88.38, 88.54, 88.91, 88.35, 87.84, 87.94, 87.55]
Average Accuracy: 88.86%

---

## Key Findings

1. **Backbone is king**: Switching MOS from ViT-B/16(IN1K) to DINOv2 improved Last Acc by +36.7% (10-10) and +33.9% (40-2). This dwarfs any algorithmic improvement.

2. **DINOv2's self-supervised features are superior for fine-grained CIL**: DINOv2 learns dense, part-aware features that generalize across aircraft types, enabling near-zero forgetting even with 16 incremental tasks.

3. **RAPF's dual-backbone (CLIP+DINOv2) doesn't beat single DINOv2 in MOS**: Despite using two encoders, RAPF underperforms MOS+DINOv2 because MOS's adapter merging + self-refined retrieval better leverages DINOv2's strong features.

4. **MOS+DINOv2 achieves remarkable stability**: In the 40-2 setting, accuracy only drops from 91.3%→87.6% across 16 tasks — just 3.7% degradation, indicating near-complete knowledge retention.
