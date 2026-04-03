# MOS (AAAI 2025) on FGVC-70 (Moderate V2)

Backbone: ViT-B/16 (ImageNet-1K pretrained) + Adapter
Method: MOS (Model Surgery) — task-specific adapters + adapter merging + self-refined retrieval

## Exp 1: init=10, inc=10 (7 tasks)

| Task | Classes | Last Acc | Avg Acc |
|:----:|:-------:|:--------:|:-------:|
| 0 | 10 | 56.61% | 56.61% |
| 1 | 20 | 65.97% | 61.29% |
| 2 | 30 | 60.26% | 60.95% |
| 3 | 40 | 54.74% | 59.40% |
| 4 | 50 | 54.89% | 58.49% |
| 5 | 60 | 49.84% | 57.05% |
| **6** | **70** | **49.41%** | **55.96%** |

## Exp 2: init=40, inc=2 (16 tasks)

| Task | Classes | Last Acc | Avg Acc |
|:----:|:-------:|:--------:|:-------:|
| 0 | 40 | 35.11% | 35.11% |
| 5 | 50 | 60.79% | 54.32% |
| 10 | 60 | 55.16% | 56.45% |
| **15** | **70** | **53.62%** | **55.27%** |

top1 curve (40-2): [35.11, 57.15, 58.53, 60.0, 60.79, 60.85, 60.64, 58.55, 57.81, 56.3, 55.16, 55.11, 54.42, 53.82, 53.75, 53.62]

## Comparison: RAPF vs MOS on FGVC-70

| Method | Setting | Last Acc | Avg Acc |
|--------|---------|:--------:|:-------:|
| **RAPF (ours)** | **10-10** | **55.15%** | **75.31%** |
| MOS | 10-10 | 49.41% | 55.96% |
| **RAPF (ours)** | **40-2** | **54.97%** | **66.10%** |
| MOS | 40-2 | 53.62% | 55.27% |

RAPF outperforms MOS significantly on both settings, especially on Avg Acc.
