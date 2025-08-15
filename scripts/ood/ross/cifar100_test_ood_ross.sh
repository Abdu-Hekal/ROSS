#!/bin/bash
# sh scripts/ood/ross/cifar100_test_ood_ross.sh

PYTHONPATH='.':$PYTHONPATH \
python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ross \
   --save-score --save-csv
