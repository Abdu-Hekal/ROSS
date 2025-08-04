#!/usr/bin/env python3
"""
table4

Generate Table 4 by averaging ROSS postprocessor results across multiple seeds using the existing eval_ood script.
"""
import os
import subprocess
import pandas as pd
from io import StringIO

# Paths
tab_dir = os.path.dirname(os.path.abspath(__file__))
eval_script = os.path.abspath(os.path.join(tab_dir, "../../scripts/eval_ood.py"))
# Path to the experiment root directory containing subfolders s0, s1, s2 (update this path accordingly)
exp_root = "./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
# Output directory for tables
tables_dir = os.path.join(tab_dir, "tables")
os.makedirs(tables_dir, exist_ok=True)

# Settings
id_data = "cifar100"
pp_name = "ross"
batch_size = 100    

# Run eval_ood to compute and average metrics across seeds, saving CSV
subprocess.run([
    "python3", eval_script,
    "--root", exp_root,
    "--postprocessor", pp_name,
    "--id-data", id_data,
    "--batch-size", str(batch_size),
    "--save-csv"
], check=True)

# Read the generated CSV file
csv_path = os.path.join(exp_root, "ood", f"{pp_name}.csv")
with open(csv_path) as f:
    lines = f.read().splitlines()

# Parse blocks for each metric
metric_labels = ["median", "mad", "cov", "ross"]
blocks = {}
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line in metric_labels:
        label = line
        header = lines[i+1]
        j = i+2
        data_lines = []
        while j < len(lines) and lines[j].strip():
            data_lines.append(lines[j])
            j += 1
        df_conf = pd.read_csv(StringIO("\n".join([header] + data_lines)))
        blocks[label] = df_conf
        i = j
    else:
        i += 1

# Build the table DataFrame
datasets = ["cifar10", "tin", "mnist", "svhn", "texture", "places365"]
table = pd.DataFrame(index=metric_labels, columns=datasets + ["Avg"])
for label in metric_labels:
    df_conf = blocks[label].set_index("dataset")
    # Extract numeric mean before '±' and convert to float
    fpr = df_conf["FPR@95"].str.split("±").str[0].astype(float)
    auroc = df_conf["AUROC"].str.split("±").str[0].astype(float)
    for ds in datasets:
        table.at[label, ds] = f"{fpr[ds]:.2f}/{auroc[ds]:.2f}"
    avg_fpr = fpr[datasets].mean()
    avg_auroc = auroc[datasets].mean()
    table.at[label, "Avg"] = f"{avg_fpr:.2f}/{avg_auroc:.2f}"

# Save to CSV
out_path = os.path.join(tables_dir, "table4.csv")
table.to_csv(out_path, index=True)
print(table)
print(f"Saved averaged table to {out_path}") 
