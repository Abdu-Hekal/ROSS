#!/bin/bash
# sh scripts/ood/ross/cifar10_test_ood_ross.sh

PYTHONPATH='.':$PYTHONPATH \

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor ross \
   --save-score --save-csv --use_cache
