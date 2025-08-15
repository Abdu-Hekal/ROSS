#!/usr/bin/env python3
"""
run_experiments: unified entrypoint for ROSS experiment scripts
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import torch
import numpy as np
import os
from openood.networks import ResNet18_32x32
from openood.evaluation_api.postprocessor import get_postprocessor

from ross_utils import (
    SCRIPT_DIR,
    PROJECT_ROOT,
    RESULTS_ROOT,
    ROSS_CFG_PATH,
    ensure_dir,
    parse_csv_blocks,
    parse_csv_skiprows,
    run_eval_ood,
    run_attack_ood,
    backup_config,
    restore_config,
    load_yaml,
    save_yaml,
    setup_model_and_loaders,
    collect_scores,
    build_avg_table,
    run_and_parse_blocks,
    reset_config_to_default
)

# Settings
ID_DATA = "cifar10"
BATCH_SIZE = 100

# Attack settings
EPSILONS = [
    ('2/255', '0.007843137'),
    ('4/255', '0.0156862745'),
    ('8/255', '0.031372549'),
    ('16/255', '0.062745098')
]
OBJECTIVES = [('min', 'PGD-Min'), ('max', 'PGD-Max')]


def _format_avg_fpr_auroc(df: pd.DataFrame) -> str:
    """Return formatted mean of FPR@95 and AUROC columns which are strings like 'xx.xx ± yy.yy'."""
    fpr_mean = df["FPR@95"].astype(str).str.split("±").str[0].astype(float).mean()
    auroc_mean = df["AUROC"].astype(str).str.split("±").str[0].astype(float).mean()
    return f"{fpr_mean:.2f}/{auroc_mean:.2f}"


def generate_table1():
    """Generate Table 1: average ROSS postprocessor metrics across seeds."""
    labels = ["median", "mad", "cov", "ross"]
    # Run eval and parse blocks
    blocks = run_and_parse_blocks("ross", labels, id_data=ID_DATA, batch_size=BATCH_SIZE, root=RESULTS_ROOT)

    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    table = build_avg_table(blocks, datasets)

    out_path = SCRIPT_DIR / "tables" / "table1.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index=True)
    print(table)
    print(f"Saved Table 1 to {out_path}")


def generate_table2():
    """Generate Table 2: robustness under PGD attacks for various postprocessors."""
    # Backup and prepare
    bak_cfg = backup_config()
    BASE_PPS = [('MSP', 'msp'), ('EBO', 'ebo'), ('GEN', 'gen'), ('ODIN', 'odin'), ('fDBD', 'fdbd'), ('PRO-fDBD', 'pro')]
    ROSS_PPS = [('ROSS-MSP', 'msp'), ('ROSS-EBO', 'ebo'), ('ROSS-GEN', 'gen'), ('ROSS-fDBD', 'fdbd')]

    columns = ['No Attack'] + [f"{eps}_{lbl}" for eps, _ in EPSILONS for _, lbl in OBJECTIVES]
    rows = [disp for disp, _ in BASE_PPS] + [disp for disp, _ in ROSS_PPS]
    results = pd.DataFrame(index=rows, columns=columns)

    # No attack: baselines
    for disp, tag in BASE_PPS:
        run_eval_ood(tag, ID_DATA, BATCH_SIZE)
        df = parse_csv_skiprows(str(RESULTS_ROOT / 'ood' / f"{tag}.csv"), skiprows=1, drop_datasets=['nearood','farood'])
        results.at[disp, 'No Attack'] = _format_avg_fpr_auroc(df)

    # No attack: ROSS variants
    cfg = load_yaml(ROSS_CFG_PATH)
    for disp, base in ROSS_PPS:
        cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = base
        save_yaml(cfg, ROSS_CFG_PATH)
        run_eval_ood('ross', ID_DATA, BATCH_SIZE)
        df = parse_csv_skiprows(str(RESULTS_ROOT / 'ood' / 'ross.csv'), skiprows=1, drop_datasets=['nearood','farood'])
        results.at[disp, 'No Attack'] = _format_avg_fpr_auroc(df)
    restore_config(bak_cfg)

    # Attacks
    for disp, tag in BASE_PPS + [(f"ROSS-{b.upper()}", 'ross') for _, b in ROSS_PPS]:
        is_ross = disp.startswith('ROSS')
        if is_ross:
            bak2 = backup_config()
            cfg2 = load_yaml(ROSS_CFG_PATH)
            cfg2['postprocessor']['postprocessor_args']['score_postprocessor'] = disp.split('-')[1].lower()
            save_yaml(cfg2, ROSS_CFG_PATH)
            pp_call = 'ross'
        else:
            pp_call = tag

        for eps, eps_val in EPSILONS:
            for obj, lbl in OBJECTIVES:
                attack_base = pp_call in ['ross','pro','odin']
                run_attack_ood(pp_call, eps_val, obj, attack_base=attack_base)
                df = parse_csv_skiprows(
                    str(RESULTS_ROOT / 'attack_ood' / f"{pp_call}_LinfPGD.csv"),
                    skiprows=1,
                    drop_datasets=['nearood','farood']
                )
                results.at[disp, f"{eps}_{lbl}"] = _format_avg_fpr_auroc(df)

        if is_ross:
            restore_config(bak2)

    out_path = SCRIPT_DIR / 'tables' / 'table2.csv'
    ensure_dir(out_path.parent)
    results.to_csv(out_path, index_label='Post-processor')
    print(f"Saved Table 2 to {out_path}")


def generate_table3():
    """Generate Table 3: robustness under PGD-min and PGD-max attacks for CIFAR-100."""
    bak_cfg = backup_config()
    BASE_PPS = [('MSP', 'msp'), ('EBO', 'ebo'), ('GEN', 'gen'), ('ODIN', 'odin'), ('fDBD', 'fdbd'), ('PRO-fDBD', 'pro')]
    ROSS_PPS = [('ROSS-MSP', 'msp'), ('ROSS-EBO', 'ebo'), ('ROSS-GEN', 'gen'), ('ROSS-fDBD', 'fdbd')]
    root = PROJECT_ROOT / "results" / "cifar100_resnet18_32x32_base_e100_lr0.1_default"

    columns = ['No Attack'] + [f"{eps}_{lbl}" for eps, _ in EPSILONS for _, lbl in OBJECTIVES]
    rows = [disp for disp, _ in BASE_PPS] + [disp for disp, _ in ROSS_PPS]
    results = pd.DataFrame(index=rows, columns=columns)

    # no attack: baselines
    for disp, tag in BASE_PPS:
        run_eval_ood(tag, id_data="cifar100", batch_size=BATCH_SIZE, root=root)
        df = parse_csv_skiprows(str(root / 'ood' / f"{tag}.csv"), skiprows=1, drop_datasets=['nearood','farood'])
        results.at[disp, 'No Attack'] = _format_avg_fpr_auroc(df)

    # no attack: ROSS variants
    cfg = load_yaml(ROSS_CFG_PATH)
    for disp, base in ROSS_PPS:
        cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = base
        save_yaml(cfg, ROSS_CFG_PATH)
        run_eval_ood('ross', id_data="cifar100", batch_size=BATCH_SIZE, root=root)
        df = parse_csv_skiprows(str(root / 'ood' / 'ross.csv'), skiprows=1, drop_datasets=['nearood','farood'])
        results.at[disp, 'No Attack'] = _format_avg_fpr_auroc(df)
    restore_config(bak_cfg)

    # attacks
    for disp, tag in BASE_PPS + [(f"ROSS-{b.upper()}", 'ross') for _, b in ROSS_PPS]:
        is_ross = disp.startswith('ROSS')
        if is_ross:
            bak2 = backup_config()
            cfg2 = load_yaml(ROSS_CFG_PATH)
            cfg2['postprocessor']['postprocessor_args']['score_postprocessor'] = disp.split('-')[1].lower()
            save_yaml(cfg2, ROSS_CFG_PATH)
            pp_call = 'ross'
        else:
            pp_call = tag

        for eps, eps_val in EPSILONS:
            for obj, lbl in OBJECTIVES:
                attack_base = pp_call in ['ross','pro','odin']
                run_attack_ood(pp_call, eps_val, obj, attack_base=attack_base, id_data="cifar100", batch_size=BATCH_SIZE, root=root)
                df = parse_csv_skiprows(str(root / 'attack_ood' / f"{pp_call}_LinfPGD.csv"), skiprows=1, drop_datasets=['nearood','farood'])
                results.at[disp, f"{eps}_{lbl}"] = _format_avg_fpr_auroc(df)
        if is_ross:
            restore_config(bak2)

    out_path = SCRIPT_DIR / 'tables' / 'table3.csv'
    ensure_dir(out_path.parent)
    results.to_csv(out_path, index_label='Post-processor')
    print(f"Saved Table 3 to {out_path}")

# Add remaining table generators

def generate_table4():
    """Generate Table 4: average ROSS postprocessor metrics for CIFAR-100 across seeds."""
    labels = ["median", "mad", "cov", "ross"]
    root = PROJECT_ROOT / "results" / "cifar100_resnet18_32x32_base_e100_lr0.1_default"
    # Run eval and parse blocks
    blocks = run_and_parse_blocks("ross", labels, id_data="cifar100", batch_size=BATCH_SIZE, root=root)
    datasets = ["cifar10", "tin", "mnist", "svhn", "texture", "places365"]
    table = build_avg_table(blocks, datasets)
    out_path = SCRIPT_DIR / "tables" / "table4.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index=True)
    print(table)
    print(f"Saved Table 4 to {out_path}")


def generate_table5():
    """Generate Table 5: average ROSS postprocessor metrics for ImageNet200 across seeds."""
    labels = ["median", "mad", "cov", "ross"]
    root = PROJECT_ROOT / "results" / "imagenet200_resnet18_224x224_base_e90_lr0.1_default"
    # Run eval and parse blocks
    blocks = run_and_parse_blocks("ross", labels, id_data="imagenet200", batch_size=BATCH_SIZE, root=root)
    datasets = ["ssb-hard", "ninco", "inaturalist", "textures", "openImage_o"]
    table = build_avg_table(blocks, datasets)
    out_path = SCRIPT_DIR / "tables" / "table5.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index=True)
    print(table)
    print(f"Saved Table 5 to {out_path}")


def generate_table6():
    """Generate Table 6: ROSS-fDBD with varied lambda on CIFAR-10."""
    LAMBDAS = [0.005, 0.01, 0.02, 0.05]
    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    root = RESULTS_ROOT
    bak = backup_config()
    base_cfg = load_yaml(bak)
    table = pd.DataFrame(index=[str(l) for l in LAMBDAS], columns=datasets + ["Avg"])
    for lam in LAMBDAS:
        cfg = base_cfg.copy()
        cfg['postprocessor']['postprocessor_args']['lambda_'] = lam
        save_yaml(cfg, ROSS_CFG_PATH)
        run_eval_ood("ross", ID_DATA, BATCH_SIZE, root=root)
        blocks = parse_csv_blocks(str(root / "ood" / "ross.csv"), ["ross"])
        df = blocks["ross"].set_index("dataset")
        fpr = df["FPR@95"].str.split("±").str[0].astype(float)
        auroc = df["AUROC"].str.split("±").str[0].astype(float)
        for ds in datasets:
            table.at[str(lam), ds] = f"{fpr[ds]:.2f}/{auroc[ds]:.2f}"
        table.at[str(lam), "Avg"] = f"{fpr[datasets].mean():.2f}/{auroc[datasets].mean():.2f}"
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table6.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index_label="lambda")
    print(table)
    print(f"Saved Table 6 to {out_path}")


def generate_table7():
    """Generate Table 7: ROSS-fDBD with varied lambda on CIFAR-100."""
    LAMBDAS = [0.005, 0.01, 0.02, 0.05]
    datasets = ["cifar10", "tin", "mnist", "svhn", "texture", "places365"]
    root = PROJECT_ROOT / "results" / "cifar100_resnet18_32x32_base_e100_lr0.1_default"
    bak = backup_config()
    base_cfg = load_yaml(bak)
    table = pd.DataFrame(index=[str(l) for l in LAMBDAS], columns=datasets + ["Avg"])
    for lam in LAMBDAS:
        cfg = base_cfg.copy()
        cfg['postprocessor']['postprocessor_args']['lambda_'] = lam
        save_yaml(cfg, ROSS_CFG_PATH)
        run_eval_ood("ross", id_data="cifar100", batch_size=BATCH_SIZE, root=root)
        blocks = parse_csv_blocks(str(root / "ood" / "ross.csv"), ["ross"])
        df = blocks["ross"].set_index("dataset")
        fpr = df["FPR@95"].str.split("±").str[0].astype(float)
        auroc = df["AUROC"].str.split("±").str[0].astype(float)
        for ds in datasets:
            table.at[str(lam), ds] = f"{fpr[ds]:.2f}/{auroc[ds]:.2f}"
        table.at[str(lam), "Avg"] = f"{fpr[datasets].mean():.2f}/{auroc[datasets].mean():.2f}"
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table7.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index_label="lambda")
    print(table)
    print(f"Saved Table 7 to {out_path}")


def generate_table8():
    """Generate Table 8: ROSS-fDBD with varied lambda on ImageNet200."""
    LAMBDAS = [0.005, 0.01, 0.02, 0.05]
    datasets = ["ssb-hard", "ninco", "inaturalist", "textures", "openImage_o"]
    root = PROJECT_ROOT / "results" / "imagenet200_resnet18_224x224_base_e90_lr0.1_default"
    bak = backup_config()
    base_cfg = load_yaml(bak)
    table = pd.DataFrame(index=[str(l) for l in LAMBDAS], columns=datasets + ["Avg"])
    for lam in LAMBDAS:
        cfg = base_cfg.copy()
        cfg['postprocessor']['postprocessor_args']['lambda_'] = lam
        save_yaml(cfg, ROSS_CFG_PATH)
        run_eval_ood("ross", id_data="imagenet200", batch_size=BATCH_SIZE, root=root)
        blocks = parse_csv_blocks(str(root / "ood" / "ross.csv"), ["ross"])
        df = blocks["ross"].set_index("dataset")
        fpr = df["FPR@95"].str.split("±").str[0].astype(float)
        auroc = df["AUROC"].str.split("±").str[0].astype(float)
        for ds in datasets:
            table.at[str(lam), ds] = f"{fpr[ds]:.2f}/{auroc[ds]:.2f}"
        table.at[str(lam), "Avg"] = f"{fpr[datasets].mean():.2f}/{auroc[datasets].mean():.2f}"
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table8.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index_label="lambda")
    print(table)
    print(f"Saved Table 8 to {out_path}")


def generate_table9():
    """Generate Table 9: robustness vs number of samples for σ=0.05 on CIFAR-10."""
    NUM_SAMPLES = [5, 10, 25, 50, 100]
    EPS = EPSILONS
    OBJECTS = [('min','Min'),('max','Max')]
    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    root = RESULTS_ROOT
    bak = backup_config()
    base_cfg = load_yaml(bak)
    columns = ['No Attack'] + [f"{eps}_{lbl}" for eps,_ in EPS for _,lbl in OBJECTS]
    table = pd.DataFrame(index=[str(n) for n in NUM_SAMPLES], columns=columns)
    for n in NUM_SAMPLES:
        cfg = base_cfg.copy()
        cfg['postprocessor']['postprocessor_args']['num_samples'] = n
        save_yaml(cfg, ROSS_CFG_PATH)
        run_eval_ood('ross', ID_DATA, BATCH_SIZE, root=root)
        blocks_na = parse_csv_blocks(str(root/'ood'/'ross.csv'), ['ross'])['ross'].set_index('dataset')
        fpr_na = blocks_na['FPR@95'].str.split('±').str[0].astype(float)
        au_na = blocks_na['AUROC'].str.split('±').str[0].astype(float)
        table.at[str(n), 'No Attack'] = f"{fpr_na[datasets].mean():.2f}/{au_na[datasets].mean():.2f}"
        for eps, eps_val in EPS:
            for obj, lbl in OBJECTS:
                run_attack_ood('ross', eps_val, obj, attack_base=True, root=root)
                blocks_atk = parse_csv_blocks(str(root/'attack_ood'/'ross_LinfPGD.csv'), ['ross'])['ross'].set_index('dataset')
                fpr_a = blocks_atk['FPR@95'].str.split('±').str[0].astype(float)
                au_a = blocks_atk['AUROC'].str.split('±').str[0].astype(float)
                table.at[str(n), f"{eps}_{lbl}"] = f"{fpr_a[datasets].mean():.2f}/{au_a[datasets].mean():.2f}"
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table9.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index_label='N')
    print(table)
    print(f"Saved Table 9 to {out_path}")


def generate_table10():
    """Generate Table 10: robustness vs noise magnitude for N=25 on CIFAR-10."""
    NOISES = [0.025, 0.05, 0.1, 0.25]
    OBJECTS = [('min','Min'),('max','Max')]
    EPS = EPSILONS
    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    root = RESULTS_ROOT
    bak = backup_config()
    base_cfg = load_yaml(bak)
    columns = ['No Attack'] + [f"{eps}_{lbl}" for eps,_ in EPS for _,lbl in OBJECTS]
    table = pd.DataFrame(index=[str(s) for s in NOISES], columns=columns)
    for sigma in NOISES:
        cfg = base_cfg.copy()
        cfg['postprocessor']['postprocessor_args']['noise_magnitude'] = sigma
        save_yaml(cfg, ROSS_CFG_PATH)
        run_eval_ood('ross', ID_DATA, BATCH_SIZE, root=root)
        blocks_na = parse_csv_blocks(str(root/'ood'/'ross.csv'), ['ross'])['ross'].set_index('dataset')
        fpr_na = blocks_na['FPR@95'].str.split('±').str[0].astype(float)
        au_na = blocks_na['AUROC'].str.split('±').str[0].astype(float)
        table.at[str(sigma), 'No Attack'] = f"{fpr_na[datasets].mean():.2f}/{au_na[datasets].mean():.2f}"
        for eps, eps_val in EPS:
            for obj, lbl in OBJECTS:
                run_attack_ood('ross', eps_val, obj, attack_base=True, root=root)
                blocks_atk = parse_csv_blocks(str(root/'attack_ood'/'ross_LinfPGD.csv'), ['ross'])['ross'].set_index('dataset')
                fpr_a = blocks_atk['FPR@95'].str.split('±').str[0].astype(float)
                au_a = blocks_atk['AUROC'].str.split('±').str[0].astype(float)
                table.at[str(sigma), f"{eps}_{lbl}"] = f"{fpr_a[datasets].mean():.2f}/{au_a[datasets].mean():.2f}"
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table10.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index_label='noise')
    print(table)
    print(f"Saved Table 10 to {out_path}")


def generate_table11():
    """Generate Table 11: ROSS with base MSP on CIFAR-10."""
    labels = ["median", "mad", "cov", "ross"]
    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    root = RESULTS_ROOT
    bak = backup_config()
    cfg = load_yaml(bak)
    cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = 'msp'
    save_yaml(cfg, ROSS_CFG_PATH)
    # Run eval with MSP variant and parse
    blocks = run_and_parse_blocks('ross', labels, id_data=ID_DATA, batch_size=BATCH_SIZE, root=root)
    table = build_avg_table(blocks, datasets)
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table11.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index=True)
    print(table)
    print(f"Saved Table 11 to {out_path}")


def generate_table12():
    """Generate Table 12: ROSS with base EBO on CIFAR-10."""
    labels = ["median", "mad", "cov", "ross"]
    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    root = RESULTS_ROOT
    bak = backup_config()
    cfg = load_yaml(bak)
    cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = 'ebo'
    save_yaml(cfg, ROSS_CFG_PATH)
    # Run eval with EBO variant and parse
    blocks = run_and_parse_blocks('ross', labels, id_data=ID_DATA, batch_size=BATCH_SIZE, root=root)
    table = build_avg_table(blocks, datasets)
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table12.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index=True)
    print(table)
    print(f"Saved Table 12 to {out_path}")


def generate_table13():
    """Generate Table 13: ROSS with base GEN on CIFAR-10."""
    labels = ["median", "mad", "cov", "ross"]
    datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    root = RESULTS_ROOT
    bak = backup_config()
    cfg = load_yaml(bak)
    cfg['postprocessor']['postprocessor_args']['score_postprocessor'] = 'gen'
    save_yaml(cfg, ROSS_CFG_PATH)
    # Run eval with GEN variant and parse
    blocks = run_and_parse_blocks('ross', labels, id_data=ID_DATA, batch_size=BATCH_SIZE, root=root)
    table = build_avg_table(blocks, datasets)
    restore_config(bak)
    out_path = SCRIPT_DIR / "tables" / "table13.csv"
    ensure_dir(out_path.parent)
    table.to_csv(out_path, index=True)
    print(table)
    print(f"Saved Table 13 to {out_path}")


def generate_figure2():
    """Generate Figure 2: histograms of median and MAD scores."""
    fig_dir = SCRIPT_DIR / 'figures'
    ensure_dir(fig_dir)

    # Load model and data
    net, loaders = setup_model_and_loaders(
        dataset_name='cifar10',
        model_cls=ResNet18_32x32,
        num_classes=10
    )
    postpp = get_postprocessor(str(PROJECT_ROOT / 'configs'), 'ross', 'cifar10')
    postpp.setup(net, loaders['id'], loaders['ood'])

    datasets = ['cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365']
    N_SAMPLES = 1000
    scores = {'id': {}, **{ds: {} for ds in datasets}}
    scores['id']['median'], scores['id']['mad'] = collect_scores(postpp, net, loaders['id']['test'], N_SAMPLES)
    for ds in datasets:
        dl = loaders['ood']['near'].get(ds) or loaders['ood']['far'][ds]
        scores[ds]['median'], scores[ds]['mad'] = collect_scores(postpp, net, dl, N_SAMPLES)

    # Truncate
    for key in ['id'] + datasets:
        for metric in ['median', 'mad']:
            scores[key][metric] = scores[key][metric][:N_SAMPLES]

    # Plot
    fig, axes = plt.subplots(2, len(datasets), figsize=(4*len(datasets), 8))
    for j, ds in enumerate(datasets):
        axes[0, j].hist(scores['id']['median'], bins=50, alpha=0.5, label='cifar10')
        axes[0, j].hist(scores[ds]['median'], bins=50, alpha=0.5, label=ds)
        if j == 0:
            axes[0, j].legend()
        axes[0, j].set_title(ds)
        axes[0, j].set_yticks([])
        axes[0, j].set_xticks([])

        axes[1, j].hist(scores['id']['mad'], bins=50, alpha=0.5, label='cifar10')
        axes[1, j].hist(scores[ds]['mad'], bins=50, alpha=0.5, label=ds)
        if j == 0:
            axes[1, j].legend()
        axes[1, j].set_yticks([])
        axes[1, j].set_xticks([])

    fig.tight_layout(rect=[0.05, 0.03, 1, 0.97])
    fig.text(0.04, 0.75, 'Median', va='center', rotation='vertical', fontsize=14)
    fig.text(0.04, 0.25, '-MAD', va='center', rotation='vertical', fontsize=14)
    out_file = fig_dir / 'figure2.png'
    fig.savefig(out_file)
    print(f"Saved Figure 2 to {out_file}")


def generate_figure3():
    """Generate Figure 3: AUROC vs attack magnitude for PGD-Max varying noise."""
    fig_dir = SCRIPT_DIR / 'figures'
    ensure_dir(fig_dir)

    # Backup config
    bak = backup_config()
    base_cfg = load_yaml(bak)
    NOISES = [0.025, 0.05, 0.1, 0.25]
    results = {}

    for sigma in NOISES:
        cfg = base_cfg.copy()
        cfg['postprocessor']['postprocessor_args']['noise_magnitude'] = sigma
        save_yaml(cfg, ROSS_CFG_PATH)

        # No attack
        run_eval_ood('ross', ID_DATA, BATCH_SIZE)
        df_na = parse_csv_blocks(str(RESULTS_ROOT / 'ood' / 'ross.csv'), ['ross'])['ross'].set_index('dataset')
        au_na = df_na['AUROC'].str.split('±').str[0].astype(float).mean()
        au_list = [au_na]

        # Attacks: PGD-Max only
        for _, eps_val in EPSILONS:
            run_attack_ood('ross', eps_val, 'max', attack_base=True)
            df_atk = parse_csv_blocks(str(RESULTS_ROOT / 'attack_ood' / 'ross_LinfPGD.csv'), ['ross'])['ross'].set_index('dataset')
            au_atk = df_atk['AUROC'].str.split('±').str[0].astype(float).mean()
            au_list.append(au_atk)

        results[sigma] = au_list

    restore_config(bak)

    # Plot
    eps_labels = ['0'] + [eps for eps, _ in EPSILONS]
    fig, ax = plt.subplots(figsize=(8, 6))
    for sigma, au_list in results.items():
        ax.plot(eps_labels, au_list, marker='o', label=str(sigma))

    ax.set_xlabel('Attack Magnitude (ε)')
    ax.set_ylabel('AUROC (%)')
    ax.set_title('OOD Detection AUROC vs Attack Magnitude (Max only)')
    ax.legend(title='Noise Magnitude')
    ax.grid(True)
    fig.tight_layout()
    out_file = fig_dir / 'figure3.png'
    fig.savefig(out_file)
    print(f"Saved Figure 3 to {out_file}")

# Mapping experiments to functions
EXPERIMENTS = {
    'figure2': generate_figure2,
    'table1': generate_table1,
    'figure3': generate_figure3,
    'table2': generate_table2,
    'table3': generate_table3,
    'table4': generate_table4,
    'table5': generate_table5,
    'table6': generate_table6,
    'table7': generate_table7,
    'table8': generate_table8,
    'table9': generate_table9,
    'table10': generate_table10,
    'table11': generate_table11,
    'table12': generate_table12,
    'table13': generate_table13,
}


def main():
    parser = argparse.ArgumentParser(description='Run ROSS experiments')
    parser.add_argument('--experiment', '-e', choices=list(EXPERIMENTS.keys()) + ['all'], default='all', help='Which experiment to run')
    args = parser.parse_args()

    if args.experiment == 'all':
        for name, func in EXPERIMENTS.items():
            print(f"Running {name}...")
            reset_config_to_default()
            func()
    else:
        print(f"Running {args.experiment}...")
        reset_config_to_default()
        EXPERIMENTS[args.experiment]()

    print('Done!')


if __name__ == '__main__':
    main() 