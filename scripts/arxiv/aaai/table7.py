#!/usr/bin/env python3
"""
table7

Generate Table 7: analyze ROSS-fDBD post-processor on CIFAR-100 with different lambda values.
Runs eval_ood for each lambda, extracts only the 'ross' confidence metric, and writes table7.csv.
"""
import os
import shutil
import subprocess
from io import StringIO
import pandas as pd
import yaml

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
EXP_ROOT = "./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
eval_ood_cli = os.path.join(PROJECT_ROOT, 'scripts', 'eval_ood.py')
ross_cfg_path = os.path.join(PROJECT_ROOT, 'configs', 'postprocessors', 'ross.yml')
tables_dir = os.path.join(SCRIPT_DIR, 'tables')
os.makedirs(tables_dir, exist_ok=True)

# Settings
ID_DATA = 'cifar100'
BATCH_SIZE = 100
LAMBDAS = [0.005, 0.01, 0.02, 0.05]
DATASETS = ["cifar10", "tin", "mnist", "svhn", "texture", "places365"]

# Backup original config
dot_bak = ross_cfg_path + '.bak'
shutil.copy(ross_cfg_path, dot_bak)
with open(dot_bak) as f:
    base_cfg = yaml.safe_load(f)

# Prepare result table
index = [str(l) for l in LAMBDAS]
columns = DATASETS + ["Avg"]
table = pd.DataFrame(index=index, columns=columns)

try:
    for lam in LAMBDAS:
        # update lambda in config
        cfg = base_cfg.copy()
        cfg['postprocessor']['postprocessor_args']['lambda_'] = lam
        with open(ross_cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        # run eval_ood
        subprocess.run([
            'python3', eval_ood_cli,
            '--root', EXP_ROOT,
            '--postprocessor', 'ross',
            '--id-data', ID_DATA,
            '--batch-size', str(BATCH_SIZE),
            '--save-csv'
        ], check=True)
        # parse ross.csv
        csv_path = os.path.join(EXP_ROOT, 'ood', 'ross.csv')
        lines = open(csv_path).read().splitlines()
        # find 'ross' block
        for i, line in enumerate(lines):
            if line.strip() == 'ross':
                header = lines[i+1]
                j = i+2
                data_lines = []
                while j < len(lines) and lines[j].strip():
                    data_lines.append(lines[j])
                    j += 1
                df_conf = pd.read_csv(StringIO("\n".join([header] + data_lines)))
                break
        df_conf = df_conf.set_index('dataset')
        # extract means
        fpr = df_conf['FPR@95'].str.split('±').str[0].astype(float)
        au = df_conf['AUROC'].str.split('±').str[0].astype(float)
        # fill row
        for ds in DATASETS:
            table.at[str(lam), ds] = f"{fpr[ds]:.2f}/{au[ds]:.2f}"
        avg_fpr = fpr[DATASETS].mean()
        avg_au = au[DATASETS].mean()
        table.at[str(lam), 'Avg'] = f"{avg_fpr:.2f}/{avg_au:.2f}"
finally:
    # restore original config
    shutil.move(dot_bak, ross_cfg_path)

# Save to CSV
out_path = os.path.join(tables_dir, 'table7.csv')
table.to_csv(out_path, index_label='lambda')
print(table)
print(f"Saved Table 7 CSV to {out_path}") 