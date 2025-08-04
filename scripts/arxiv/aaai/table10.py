#!/usr/bin/env python3
"""
table10

Generate Table 10: OOD detection robustness vs. noise magnitude (σ) for sample size N=25.
Runs eval_ood for no attack and attack_ood under PGD-min/max, varying noise magnitude, averages over benchmarks, and writes table10.csv.
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
NOISES = [0.025, 0.05, 0.1, 0.25]
EPSILONS = [('2/255', '0.007843137'),
            ('4/255', '0.0156862745'),
            ('8/255', '0.031372549'),
            ('16/255', '0.062745098')]
OBJECTIVES = [('min', 'Min'), ('max', 'Max')]
DATASETS = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]

# Backup original config
bak = ross_cfg_path + '.bak'
shutil.copy(ross_cfg_path, bak)
with open(bak) as f:
    base_cfg = yaml.safe_load(f)

# helper to parse 'ross' block from CSV
def parse_ross_csv(path):
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == 'ross':
            header = lines[i+1]
            j = i+2
            data_lines = []
            while j < len(lines) and lines[j].strip():
                data_lines.append(lines[j]); j += 1
            df = pd.read_csv(StringIO("\n".join([header] + data_lines)))
            return df.set_index('dataset')
    raise RuntimeError("ross block not found in CSV")

# run eval_ood
def run_eval():
    subprocess.run([
        'python3', eval_ood_cli,
        '--root', EXP_ROOT,
        '--postprocessor', 'ross',
        '--id-data', ID_DATA,
        '--batch-size', str(BATCH_SIZE),
        '--save-csv'
    ], check=True)
    df = parse_ross_csv(os.path.join(EXP_ROOT,'ood','ross.csv'))
    fpr = df['FPR@95'].str.split('±').str[0].astype(float)
    au = df['AUROC'].str.split('±').str[0].astype(float)
    return fpr, au

# run attack_ood
def run_attack(eps_val, objective):
    csv_path = os.path.join(EXP_ROOT, 'attack_ood', 'ross_LinfPGD.csv')
    if os.path.exists(csv_path): os.remove(csv_path)
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
    df = parse_ross_csv(csv_path)
    fpr = df['FPR@95'].str.split('±').str[0].astype(float)
    au = df['AUROC'].str.split('±').str[0].astype(float)
    return fpr, au

# Prepare table
columns = ['No Attack'] + [f"{eps}_{lbl}" for eps,_ in EPSILONS for _,lbl in OBJECTIVES]
index = [str(n) for n in NOISES]
table = pd.DataFrame(index=index, columns=columns)

# Iterate noise magnitudes
for sigma in NOISES:
    # update config
    cfg = copy.deepcopy(base_cfg)
    cfg['postprocessor']['postprocessor_args']['noise_magnitude'] = sigma
    with open(ross_cfg_path,'w') as f: yaml.safe_dump(cfg,f)
    # no attack
    fpr_na, au_na = run_eval()
    table.at[str(sigma),'No Attack'] = f"{fpr_na[DATASETS].mean():.2f}/{au_na[DATASETS].mean():.2f}"
    # attacks
    for eps, eps_val in EPSILONS:
        for obj,lbl in OBJECTIVES:
            fpr_a, au_a = run_attack(eps_val, obj)
            col = f"{eps}_{lbl}"
            table.at[str(sigma),col] = f"{fpr_a[DATASETS].mean():.2f}/{au_a[DATASETS].mean():.2f}"

# Restore config
shutil.move(bak, ross_cfg_path)

# Save
out = os.path.join(tables_dir,'table10.csv')
table.to_csv(out, index_label='noise')
print(table)
print(f"Saved Table 10 CSV to {out}") 