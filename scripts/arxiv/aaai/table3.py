#!/usr/bin/env python3
"""
table3

Generate Table 3: robustness of CIFAR-100 OOD detection under PGD-min and PGD-max attacks.
Runs eval_ood for no attack and attack_ood for attacks, averages over seeds.
Outputs CSV to scripts/ross/tables/table3.csv
"""
import os
import shutil
import subprocess
import pandas as pd
import yaml

# === User-updated paths ===
# Root directory for ROSS experiment runs containing s0, s1, s2
EXP_ROOT = "./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"

# Path to project root (adjust if necessary)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Paths to CLI scripts and config
eval_ood_cli = os.path.join(PROJECT_ROOT, 'scripts', 'eval_ood.py')
attack_ood_cli = os.path.join(PROJECT_ROOT, 'scripts', 'attack_ood.py')
ross_cfg = os.path.join(PROJECT_ROOT, 'configs', 'postprocessors', 'ross.yml')
# Output folder for tables
tables_dir = os.path.join(SCRIPT_DIR, 'tables')
os.makedirs(tables_dir, exist_ok=True)

# Settings
ID_DATA = 'cifar100'
BATCH_SIZE = 100
ATTACK_METHOD = 'LinfPGD'
EPSILONS = [('2/255', '0.007843137'),
            ('4/255', '0.0156862745'),
            ('8/255', '0.031372549'),
            ('16/255', '0.062745098')]
OBJECTIVES = [('min', 'PGD-Min'), ('max', 'PGD-Max')]
# Baseline postprocessors: (display, tag)
BASE_PPS = [
    ('MSP', 'msp'),
    ('EBO', 'ebo'),
    ('GEN', 'gen'),
    ('ODIN', 'odin'),
    ('fDBD', 'fdbd'),
    ('PRO-fDBD', 'pro')
]
# ROSS variants: (display, base_tag)
ROSS_PPS = [
    ('ROSS-MSP', 'msp'),
    ('ROSS-EBO', 'ebo'),
    ('ROSS-GEN', 'gen'),
    ('ROSS-fDBD', 'fdbd'),
]

# Helpers to run commands
def run_eval(pp_tag):
    # no attack: run eval_ood
    subprocess.run([
        'python3', eval_ood_cli,
        '--root', EXP_ROOT,
        '--postprocessor', pp_tag,
        '--id-data', ID_DATA,
        '--batch-size', str(BATCH_SIZE),
        '--save-csv'
    ], check=True)
    # read CSV at EXP_ROOT/ood/{pp_tag}.csv
    csv_path = os.path.join(EXP_ROOT, 'ood', f'{pp_tag}.csv')
    # parse first block
    df = pd.read_csv(csv_path, skiprows=1)
    # drop aggregate rows (nearood, farood)
    df = df[~df['dataset'].isin(['nearood','farood'])]
    return df['FPR@95'].astype(float).mean(), df['AUROC'].astype(float).mean()


def run_attack(pp_tag, eps_val, objective, attack_base=False):
    # cleanup old CSV
    attack_dir = os.path.join(EXP_ROOT, 'attack_ood')
    out_csv = os.path.join(attack_dir, f'{pp_tag}_{ATTACK_METHOD}.csv')
    if os.path.isfile(out_csv):
        os.remove(out_csv)
    # build attack command
    cmd = [
        'python3', attack_ood_cli,
        '--root', EXP_ROOT,
        '--postprocessor', pp_tag,
        '--id-data', ID_DATA,
        '--batch-size', str(BATCH_SIZE),
        '--attack-method', ATTACK_METHOD,
        '--eps', eps_val,
        '--ood-objective', objective,
        '--steps', '40'
    ]
    if attack_base:
        cmd.append('--attack-base-pp')
    cmd.append('--save-csv')
    subprocess.run(cmd, check=True)
    # read CSV
    df = pd.read_csv(out_csv, skiprows=1)
    df = df[~df['dataset'].isin(['nearood','farood'])]
    return df['FPR@95'].astype(float).mean(), df['AUROC'].astype(float).mean()

# Table assembly
columns = ['No Attack'] + [f"{eps}_{lbl}" for eps, _ in EPSILONS for _, lbl in OBJECTIVES]
rows = [disp for disp, _ in BASE_PPS] + [disp for disp, _ in ROSS_PPS]
results = pd.DataFrame(index=rows, columns=columns)

# 1) baseline no attack
for disp, tag in BASE_PPS:
    fpr, au = run_eval(tag)
    results.at[disp, 'No Attack'] = f'{fpr:.2f}/{au:.2f}'
# 2) ROSS variants no attack (update config for each base)
ross_cfg_bak = ross_cfg + '.bak'
shutil.copy(ross_cfg, ross_cfg_bak)
with open(ross_cfg, 'r') as f:
    cfg = yaml.safe_load(f)
for disp, base in ROSS_PPS:
    # set base_pp
    cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = base
    with open(ross_cfg, 'w') as f:
        yaml.safe_dump(cfg, f)
    fpr, au = run_eval('ross')
    results.at[disp, 'No Attack'] = f'{fpr:.2f}/{au:.2f}'
# restore ROSS config
shutil.move(ross_cfg_bak, ross_cfg)

# 3) attacks: baseline
for disp, tag in BASE_PPS + [('ROSS-'+b.upper(), 'ross') for _, b in ROSS_PPS]:
    # ROSS variants: need to update config first if disp startswith 'ROSS'
    is_ross = disp.startswith('ROSS')
    if is_ross:
        # backup & update config
        shutil.copy(ross_cfg, ross_cfg_bak)
        cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = disp.split('-')[1].lower()
        with open(ross_cfg, 'w') as f:
            yaml.safe_dump(cfg, f)
        pp_call = 'ross'
    else:
        pp_call = tag
    for eps, eps_val in EPSILONS:
        for obj, lbl in OBJECTIVES:
            # use --attack-base-pp for composite or odin
            attack_base = pp_call in ['ross', 'pro', 'odin']
            fpr, au = run_attack(pp_call, eps_val, obj, attack_base=attack_base)
            col = f"{eps}_{lbl}"
            results.at[disp, col] = f'{fpr:.2f}/{au:.2f}'
    if is_ross:
        # restore config
        shutil.move(ross_cfg_bak, ross_cfg)

# Save table2
out_path = os.path.join(tables_dir, 'table2.csv')
results.to_csv(out_path, index_label='Post-processor')
print(f"Saved Table 3 CSV to {out_path}") 