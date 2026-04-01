# ROSS - A Robust OOD Detector via Syngergistic Smoothing
Accepted in CVPR Findings 2026

## Overview

ROSS enhances out-of-distribution (OOD) detection through a synergistic, post-hoc process. It first smooths base OOD scores by taking their median across a set of noisy input samples. It then reuses these samples to calculate the Median Absolute Deviation (MAD), using this measure of score instability to better differentiate between in-distribution and OOD inputs. This dual use of noisy samples allows ROSS to achieve strong performance on both clean and robust accuracy benchmarks.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_org/ROB-OPENOOD.git
   cd ROB-OPENOOD
   ```

2. (Optional) Create and activate a Conda environment:
   ```bash
   conda create -n ross python=3.9
   conda activate ross
   ```

3. Install the package:
   ```bash
   cd foolbox
   pip install .
   cd ..
   pip install .
   ```

4. Prepare model checkpoints (choose one option):
   Option A (automatic): Download test datasets and official checkpoints:
   ```bash
   python ./scripts/download/download.py \
     --contents 'datasets' 'checkpoints' \
     --datasets 'all' \
     --checkpoints 'all' \
     --save_dir './data' './results' \
     --dataset_mode 'benchmark'
   ```
   Option B (manual): Place pre-trained model checkpoints under the `results/` directory:
   ```bash
   results/
     cifar10_resnet18_32x32_base_e100_lr0.1_default/
       s0/best.ckpt
       s1/best.ckpt
       s2/best.ckpt
     cifar100_resnet18_32x32_base_e100_lr0.1_default/...
     imagenet200_resnet18_32x32_base_e100_lr0.1_default/...
   ```
   You may train your own models or download official checkpoints from the OpenOOD releases page:
   https://github.com/thuml/OpenOOD/releases

## Usage

### 1) OOD Evaluation with `eval_ood`

Run OOD detection on CIFAR-10 using ROSS:
```bash
python3 scripts/eval_ood.py \
  --root results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
  --postprocessor ross \
  --id-data cifar10 \
  --batch-size 100 \
  --save-csv
```
The results will be saved to:
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/ood/ross.csv
```

### 2) Adversarial Attack Evaluation with `attack_ood`

Run PGD-based OOD attack evaluation:
```bash
python3 scripts/attack_ood.py \
  --root results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
  --postprocessor ross \
  --id-data cifar10 \
  --batch-size 100 \
  --attack-method LinfPGD \
  --eps 0.007843137 \
  --ood-objective max \
  --attack-base-pp \
  --save-csv \
  --steps 40
```
The results will be saved to:
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/attack_ood/ross_LinfPGD.csv
```

### 3) Automated Experiments with `run_experiments.py`
**Note:** The results produced by `run_experiments.py` may vary slightly due to random noise in the post-processing steps. Variation tends to decrease when using a larger number of noisy samples.

Run all tables and figures from the paper in one script:
```bash
python3 scripts/ross/run_experiments.py -e all
```

Or run a specific experiment (e.g., `table1`):
```bash
python3 scripts/ross/run_experiments.py -e table1
```

Available experiments:
```
table1, table2, table3, table4, table5, table6, table7, table8,
table9, table10, table11, table12, table13, figure2, figure3
```

---

For questions, issues, or contributions, please open an issue or pull request on the project repository.