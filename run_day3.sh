#!/bin/bash
# DAY3: RAPF on FGVC-60 (aggressive merged dataset)
export ALL_PROXY=127.0.0.1:7890
PYTHON=/mnt/conda/envs/continual_clip/bin/python
cd ~/Documents/Github/RAPF

echo "=========================================="
echo "DAY3 Exp 1: FGVC-60 initial=10 increment=10"
echo "=========================================="
$PYTHON main.py \
    --config-path configs/class \
    --config-name fgvc60_10-10.yaml \
    dataset_root=data/fgvc_aggressive \
    class_order=class_orders/fgvc_aggressive_order_shuffle.yaml

echo ""
echo "=========================================="
echo "DAY3 Exp 2: FGVC-60 initial=40 increment=2"
echo "=========================================="
$PYTHON main.py \
    --config-path configs/class \
    --config-name fgvc60_40-2.yaml \
    dataset_root=data/fgvc_aggressive \
    class_order=class_orders/fgvc_aggressive_order_shuffle.yaml

echo "DAY3 complete!"
