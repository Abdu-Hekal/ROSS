#!/usr/bin/env python3
"""
table9

Generate Table 9: OOD detection robustness vs. number of samples (N) for noise σ=0.05.
Runs eval_ood for no attack and attack_ood under PGD-min/max, averages over all benchmarks, and writes table9.csv.
"""
import os
import shutil
import subprocess
from io import StringIO
import pandas as pd
import yaml
import copy

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
EXP_ROOT = "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
eval_ood_cli = os.path.join(PROJECT_ROOT, 'scripts', 'eval_ood.py')
attack_ood_cli = os.path.join(PROJECT_ROOT, 'scripts', 'attack_ood.py')
ross_cfg_path = os.path.join(PROJECT_ROOT, 'configs', 'postprocessors', 'ross.yml')
tables_dir = os.path.join(SCRIPT_DIR, 'tables')
os.makedirs(tables_dir, exist_ok=True)

# Settings
ID_DATA = 'cifar10'
BATCH_SIZE = 100
NUM_SAMPLES = [5, 10, 25, 50, 100]
EPSILONS = [('2/255', '0.007843137'),
            ('4/255', '0.0156862745'),
            ('8/255', '0.031372549'),
            ('16/255', '0.062745098')]
OBJECTIVES = [('min', 'Min'), ('max', 'Max')]
DATASETS = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]

# Backup original ross config
bak_path = ross_cfg_path + '.bak'
shutil.copy(ross_cfg_path, bak_path)
with open(bak_path) as f:
    base_cfg = yaml.safe_load(f)

# helper to parse ross block
def parse_ross_csv(path):
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == 'ross':
            header = lines[i+1]
            j = i+2
            data_lines = []
            while j < len(lines) and lines[j].strip():
                data_lines.append(lines[j])
                j += 1
            df = pd.read_csv(StringIO("\n".join([header] + data_lines)))
            return df.set_index('dataset')
    raise ValueError('ross block not found in CSV')

# functions to run eval and attack
def run_eval():
    subprocess.run([
        'python3', eval_ood_cli,
        '--root', EXP_ROOT,
        '--postprocessor', 'ross',
        '--id-data', ID_DATA,
        '--batch-size', str(BATCH_SIZE),
        '--save-csv'
    ], check=True)
    csv_path = os.path.join(EXP_ROOT, 'ood', 'ross.csv')
    df = parse_ross_csv(csv_path)
    fpr = df['FPR@95'].str.split('±').str[0].astype(float)
    au = df['AUROC'].str.split('±').str[0].astype(float)
    return fpr, au

def run_attack(eps_val, objective):
    attack_csv = os.path.join(EXP_ROOT, 'attack_ood', 'ross_LinfPGD.csv')
    if os.path.isfile(attack_csv):
        os.remove(attack_csv)
    subprocess.run([
        'python3', attack_ood_cli,
        '--root', EXP_ROOT,
        '--postprocessor', 'ross',
        '--id-data', ID_DATA,
        '--batch-size', str(BATCH_SIZE),
        '--attack-method', 'LinfPGD',
        '--eps', eps_val,
        '--ood-objective', objective,
        '--attack-base-pp',
        '--save-csv',
        '--steps', '40'
    ], check=True)
    df = parse_ross_csv(attack_csv)
    fpr = df['FPR@95'].str.split('±').str[0].astype(float)
    au = df['AUROC'].str.split('±').str[0].astype(float)
    return fpr, au

# Prepare table
columns = ['No Attack'] + [f"{eps}_{lbl}" for eps, _ in EPSILONS for _, lbl in OBJECTIVES]
table = pd.DataFrame(index=[str(n) for n in NUM_SAMPLES], columns=columns)

# run for each sample count
for n in NUM_SAMPLES:
    # update config num_samples
    cfg = copy.deepcopy(base_cfg)
    cfg['postprocessor']['postprocessor_args']['num_samples'] = n
    with open(ross_cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    # no attack
    fpr_na, au_na = run_eval()
    na_mean_fpr = fpr_na[DATASETS].mean()
    na_mean_au = au_na[DATASETS].mean()
    table.at[str(n), 'No Attack'] = f"{na_mean_fpr:.2f}/{na_mean_au:.2f}"
    # attacks
    for eps, eps_val in EPSILONS:
        for obj, lbl in OBJECTIVES:
            fpr_a, au_a = run_attack(eps_val, obj)
            mean_fpr = fpr_a[DATASETS].mean()
            mean_au = au_a[DATASETS].mean()
            col = f"{eps}_{lbl}"
            table.at[str(n), col] = f"{mean_fpr:.2f}/{mean_au:.2f}"

# restore original config
shutil.move(bak_path, ross_cfg_path)

# save table9
out_path = os.path.join(tables_dir, 'table9.csv')
table.to_csv(out_path, index_label='N')
print(table)
print(f"Saved Table 9 CSV to {out_path}") 