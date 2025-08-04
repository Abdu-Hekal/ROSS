#!/usr/bin/env python3
"""
table13

Generate Table 13: same as table1 but using ROSS with base postprocessor GEN instead of fDBD.
"""
import os
import shutil
import subprocess
import pandas as pd
from io import StringIO
import yaml

# Paths
tab_dir = os.path.dirname(os.path.abspath(__file__))
eval_script = os.path.abspath(os.path.join(tab_dir, "../../scripts/eval_ood.py"))
cfg_path = os.path.abspath(os.path.join(tab_dir, "../../configs/postprocessors/ross.yml"))
bak_path = cfg_path + ".bak"

# Backup and patch config
auto_backup = shutil.copy(cfg_path, bak_path)
cfg = yaml.safe_load(open(bak_path))
cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = 'gen'
with open(cfg_path, 'w') as f:
    yaml.safe_dump(cfg, f)

# Settings
exp_root = "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
id_data = "cifar10"
pp_name = "ross"
batch_size = 100

try:
    # Run eval_ood and save CSV
    subprocess.run([
        "python3", eval_script,
        "--root", exp_root,
        "--postprocessor", pp_name,
        "--id-data", id_data,
        "--batch-size", str(batch_size),
        "--save-csv"
    ], check=True)

    # Read CSV
    csv_path = os.path.join(exp_root, "ood", f"{pp_name}.csv")
    lines = open(csv_path).read().splitlines()

    # Parse metrics blocks
    metric_labels = ["median", "mad", "cov", "ross"]
    blocks = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line in metric_labels:
            header = lines[i+1]
            j = i+2
            data_lines = []
            while j < len(lines) and lines[j].strip():
                data_lines.append(lines[j]); j += 1
            blocks[line] = pd.read_csv(StringIO("\n".join([header] + data_lines)))
            i = j
        else:
            i += 1

    # Build table
    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    table = pd.DataFrame(index=metric_labels, columns=datasets + ["Avg"])
    for label in metric_labels:
        df_conf = blocks[label].set_index('dataset')
        fpr = df_conf['FPR@95'].str.split('±').str[0].astype(float)
        au = df_conf['AUROC'].str.split('±').str[0].astype(float)
        for ds in datasets:
            table.at[label, ds] = f"{fpr[ds]:.2f}/{au[ds]:.2f}"
        avg_fpr = fpr[datasets].mean()
        avg_au = au[datasets].mean()
        table.at[label, 'Avg'] = f"{avg_fpr:.2f}/{avg_au:.2f}"

    # Save to CSV
    tables_dir = os.path.join(tab_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, 'table13.csv')
    table.to_csv(out_path, index=True)
    print(table)
    print(f"Saved Table 13 CSV to {out_path}")
finally:
    # Restore original config
    shutil.move(bak_path, cfg_path) 