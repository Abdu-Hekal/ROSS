#!/usr/bin/env python3
"""
figure3

Generate Figure 3: AUROC vs. attack magnitude (ε) for PGD-Max only, across noise magnitudes σ.
Runs eval_ood for no attack and attack_ood for PGD-Max at each ε, varying noise magnitude in ross config, and plots curves.
"""
import os
import shutil
import subprocess
from io import StringIO
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
EXP_ROOT = os.path.join(PROJECT_ROOT, 'results', 'cifar10_resnet18_32x32_base_e100_lr0.1_default')
EVAL_CLI = os.path.join(PROJECT_ROOT, 'scripts', 'eval_ood.py')
ATTACK_CLI = os.path.join(PROJECT_ROOT, 'scripts', 'attack_ood.py')
ROSS_CFG = os.path.join(PROJECT_ROOT, 'configs', 'postprocessors', 'ross.yml')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Settings
ID_DATA = 'cifar10'
BATCH_SIZE = 100
NOISES = [0.025, 0.05, 0.1, 0.25]
EPSILONS = [('2/255', '0.007843137'),
            ('4/255', '0.0156862745'),
            ('8/255', '0.031372549'),
            ('16/255', '0.062745098')]

# Backup original ross config
bak = ROSS_CFG + '.bak'
shutil.copy(ROSS_CFG, bak)
with open(bak) as f:
    base_cfg = yaml.safe_load(f)

# Helper to parse ross block
from io import StringIO
import pandas as pd

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
    raise RuntimeError('ross block not found')

# Storage for AUROC
results = {}
for sigma in NOISES:
    # update noise in config
    cfg = base_cfg.copy()
    cfg['postprocessor']['postprocessor_args']['noise_magnitude'] = sigma
    with open(ROSS_CFG, 'w') as f:
        yaml.safe_dump(cfg, f)
    # No attack: run eval_ood
    subprocess.run([
        'python3', EVAL_CLI,
        '--root', EXP_ROOT,
        '--postprocessor', 'ross',
        '--id-data', ID_DATA,
        '--batch-size', str(BATCH_SIZE),
        '--save-csv'
    ], check=True)
    df_na = parse_ross_csv(os.path.join(EXP_ROOT, 'ood', 'ross.csv'))
    au_na = df_na['AUROC'].str.split('±').str[0].astype(float).mean()
    # Attacks: PGD-Max only
    au_list = [au_na]
    for _, eps_val in EPSILONS:
        subprocess.run([
            'python3', ATTACK_CLI,
            '--root', EXP_ROOT,
            '--postprocessor', 'ross',
            '--id-data', ID_DATA,
            '--batch-size', str(BATCH_SIZE),
            '--attack-method', 'LinfPGD',
            '--eps', eps_val,
            '--ood-objective', 'max',
            '--attack-base-pp',
            '--save-csv',
            '--steps', '40'
        ], check=True)
        df_atk = parse_ross_csv(os.path.join(EXP_ROOT, 'attack_ood', 'ross_LinfPGD.csv'))
        au_atk = df_atk['AUROC'].str.split('±').str[0].astype(float).mean()
        au_list.append(au_atk)
    results[sigma] = au_list

# Restore config
shutil.move(bak, ROSS_CFG)

# Plot
eps_labels = ['0'] + [eps for eps,_ in EPSILONS]
fig, ax = plt.subplots(figsize=(8,6))
for sigma, au_list in results.items():
    ax.plot(eps_labels, au_list, marker='o', label=str(sigma))

ax.set_xlabel('Attack Magnitude (ε)')
ax.set_ylabel('AUROC (%)')
ax.set_title('OOD Detection AUROC vs Attack Magnitude (Max only)')
ax.legend(title='Noise Magnitude')
ax.grid(True)
fig.tight_layout()
out_file = os.path.join(FIG_DIR, 'figure3.png')
fig.savefig(out_file)
print(f"Saved Figure 3 to {out_file}") 