# DAY3: RAPF on FGVC-70 (Moderate V2 — corrected merge)

## Dataset: fgvc_moderate_v2 (70 classes)
Merged same-type sub-variants only (737 Classic/NG, 747, 767, 777, A320 Family, A330, A340, MD-80 Series, etc.)
All different aircraft types kept separate (707≠DC-8, DC-10≠MD-11, A300≠A310, etc.)

## Exp 1: initial=10, increment=10 (7 tasks)

| Task | Classes | Acc | Avg Acc | Forgetting | BWT |
|:----:|:-------:|:---:|:-------:|:----------:|:---:|
| 0 | 10 | 94.75% | 94.75% | 0.00 | 0.00 |
| 1 | 20 | 81.03% | 87.89% | 0.00 | 0.00 |
| 2 | 30 | 80.81% | 85.53% | 1.31 | -0.88 |
| 3 | 40 | 78.03% | 83.65% | 0.20 | -0.41 |
| 4 | 50 | 69.50% | 80.82% | 9.63 | -2.16 |
| 5 | 60 | 67.91% | 78.67% | 10.07 | -2.33 |
| 6 | 70 | **55.15%** | **75.31%** | 20.47 | -5.52 |

## Exp 2: initial=40, increment=2 (16 tasks)

| Task | Classes | Acc | Avg Acc | Forgetting | BWT |
|:----:|:-------:|:---:|:-------:|:----------:|:---:|
| 0 | 40 | 88.32% | 88.32% | 0.00 | 0.00 |
| 5 | 50 | 70.53% | 76.88% | 31.94 | -20.81 |
| 10 | 60 | 60.72% | 68.32% | 38.11 | -28.39 |
| 15 | 70 | **54.97%** | **66.10%** | 38.17 | -24.41 |

## Config
- Backbone: DINOv2 ViT-B/14 + LoRA rank=16
- CLIP+DINOv2 fusion (1280d → 512d adapter)
- PK sampling not used (RAPF framework)
- 20 epochs/task, cosine LR, lr=5e-4
