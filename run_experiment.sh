#!/bin/bash
# RAPF experiments with DINOv2 fusion for FGVC-Aircraft
# Usage: bash run_experiment.sh

export ALL_PROXY=127.0.0.1:7890
PYTHON=/mnt/conda/envs/continual_clip/bin/python
cd "$(dirname "$0")"

echo "=== FGVC-Aircraft CLIP Baseline (10-10) ==="
$PYTHON main.py \
    --config-path configs/class \
    --config-name fgvc_10-10.yaml \
    dataset_root=data/fgvc_aircraft \
    class_order=class_orders/fgvc_aircraft_order.yaml

echo ""
echo "=== FGVC-Aircraft DINOv2 Fusion (10-10) ==="
$PYTHON main.py \
    --config-path configs/class \
    --config-name fgvc_dino_10-10.yaml \
    dataset_root=data/fgvc_aircraft \
    class_order=class_orders/fgvc_aircraft_order.yaml

echo ""
echo "=== FGVC-Aircraft DINOv2 Fusion Shuffled (10-10) ==="
$PYTHON main.py \
    --config-path configs/class \
    --config-name fgvc_dino_10-10.yaml \
    dataset_root=data/fgvc_aircraft \
    class_order=class_orders/fgvc_aircraft_order_shuffle.yaml
