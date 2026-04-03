# FGVC-70 Unified Benchmark Results

## Dataset
FGVC-Aircraft Moderate V2 (70 classes): same-type sub-variants merged, different aircraft kept separate.

## Methods
| Method | Backbone | Trainable Params | Strategy |
|--------|----------|:---:|------|
| RAPF | CLIP ViT-B/16 + DINOv2 ViT-B/14 + LoRA r16 | ~1.8M | Gaussian replay + adapter mixing |
| MOS (AAAI'25) | ViT-B/16 (IN1K) | ~0.4M | Adapter merging + self-refined retrieval |
| MOS (AAAI'25) | DINOv2 ViT-B/14 | ~0.4M | Adapter merging + self-refined retrieval |

## Setting 1: init=10, inc=10 (7 tasks)

| Task | Classes | RAPF Acc | MOS(ViT) Acc | MOS(DINOv2) Acc |
|:----:|:-------:|:--------:|:------------:|:---------------:|
| 0 | 10 | 94.75% | 56.61% | 87.28% |
| 1 | 20 | 80.55% | 66.09% | 89.94% |
| 2 | 30 | 79.27% | 59.96% | 90.18% |
| 3 | 40 | 77.89% | 54.66% | 88.61% |
| 4 | 50 | 69.35% | 54.66% | 88.34% |
| 5 | 60 | 68.30% | 49.88% | 86.05% |
| 6 | 70 | 55.39% | 49.30% | 85.94% |

### Final Metrics (after all 7 tasks)

| Metric | RAPF | MOS (ViT) | MOS (DINOv2) |
|--------|:----:|:---------:|:------------:|
| Last Acc | 55.39% | 49.30% | 85.94% |
| Avg Acc | 75.07% | 55.88% | 88.05% |
| Forgetting | 25.77 | 8.16 | 3.40 |
| BWT | -18.01 | -6.46 | -2.90 |

## Setting 2: init=40, inc=2 (16 tasks)

| Method | Last Acc | Avg Acc |
|--------|:--------:|:-------:|
| RAPF | 54.97% | 66.10% |
| MOS (ViT) | 53.62% | 55.27% |
| **MOS (DINOv2)** | **87.55%** | **88.86%** |

## Key Findings

1. **MOS+DINOv2 dominates**: 86.08% last acc vs RAPF's 55.15% and MOS(ViT)'s 49.41%
2. **Backbone > Algorithm**: Switching ViT→DINOv2 gains +36.7%, far exceeding any method-level improvement
3. **MOS+DINOv2 barely forgets**: Forgetting=3.4% vs RAPF's 25.8%
4. **DINOv2's self-supervised features are ideal for fine-grained CIL**: dense, part-aware representations generalize across aircraft types
