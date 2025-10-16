#!/usr/bin/env python3
"""
ross_utils: common helpers for ROSS experiment scripts
"""
import os
import subprocess
import shutil
import yaml
import pandas as pd
from io import StringIO
from pathlib import Path


# Directories and default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "cifar10_resnet18_32x32_base_e100_lr0.1_default"
EVAL_OOD_CLI = PROJECT_ROOT / "scripts" / "eval_ood.py"
ATTACK_OOD_CLI = PROJECT_ROOT / "scripts" / "attack_ood.py"
ROSS_CFG_PATH = PROJECT_ROOT / "configs" / "postprocessors" / "ross.yml"
DATA_ROOT = PROJECT_ROOT / "data"

# Path to default ROSS config shipped with experiments
DEFAULT_ROSS_CFG = SCRIPT_DIR / "ross_default_config.yml"


def ensure_dir(path):
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def reset_config_to_default(cfg_path=ROSS_CFG_PATH, default_path=DEFAULT_ROSS_CFG):
    """Replace the ROSS config file with the default copy before experiments."""
    # Copy default config into place
    import shutil as _shutil
    _shutil.copy(default_path, cfg_path)


def parse_csv_blocks(csv_path, labels):
    """
    Parse a CSV containing multiple labeled blocks.
    Returns dict mapping each label to its pandas DataFrame.
    """
    lines = open(csv_path).read().splitlines()
    blocks = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line in labels:
            label = line
            header = lines[i+1]
            j = i+2
            data_lines = []
            while j < len(lines) and lines[j].strip():
                data_lines.append(lines[j])
                j += 1
            df = pd.read_csv(StringIO("\n".join([header] + data_lines)))
            blocks[label] = df
            i = j
        else:
            i += 1
    return blocks


def parse_csv_skiprows(csv_path, skiprows=1, drop_datasets=None):
    """
    Read a CSV skipping a number of initial rows and optionally dropping specific datasets.
    """
    df = pd.read_csv(csv_path, skiprows=skiprows)
    # Drop any fully-empty rows
    df = df.dropna(how='all')
    # Some generated CSVs may repeat headers within the file; remove such stray header rows
    if 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() != 'dataset']
    if drop_datasets:
        df = df[~df['dataset'].isin(drop_datasets)]
    return df


def run_eval_ood(postprocessor, id_data="cifar10", batch_size=100, root=RESULTS_ROOT, save_csv=True):
    """Run eval_ood CLI for a given postprocessor."""
    cmd = [
        "python3", str(EVAL_OOD_CLI),
        "--root", str(root),
        "--postprocessor", postprocessor,
        "--id-data", id_data,
        "--batch-size", str(batch_size)
    ]
    if save_csv:
        cmd.append("--save-csv")
    subprocess.run(cmd, check=True)


def run_attack_ood(postprocessor, eps_val, objective, attack_method="LinfPGD", steps=40, attack_base=False, id_data="cifar10", batch_size=100, root=RESULTS_ROOT, save_csv=True):
    """Run attack_ood CLI for a given postprocessor and attack settings."""
    attack_dir = root / "attack_ood"
    attack_dir.mkdir(parents=True, exist_ok=True)
    out_csv = attack_dir / f"{postprocessor}_{attack_method}.csv"
    if out_csv.exists():
        out_csv.unlink()
    cmd = [
        "python3", str(ATTACK_OOD_CLI),
        "--root", str(root),
        "--postprocessor", postprocessor,
        "--id-data", id_data,
        "--batch-size", str(batch_size),
        "--attack-method", attack_method,
        "--eps", eps_val,
        "--ood-objective", objective,
        "--steps", str(steps)
    ]
    if attack_base:
        cmd.append("--attack-base-pp")
    if save_csv:
        cmd.append("--save-csv")
    subprocess.run(cmd, check=True)


def backup_config(cfg_path=ROSS_CFG_PATH):
    """Make a backup copy of the ROSS YAML config."""
    bak = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    shutil.copy(cfg_path, bak)
    return bak


def restore_config(bak_path, cfg_path=ROSS_CFG_PATH):
    """Restore the ROSS YAML config from backup."""
    shutil.move(str(bak_path), str(cfg_path))

# Helpers for figure scripts
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openood.evaluation_api.datasets import get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
from openood.evaluation_api.postprocessor import get_postprocessor


def load_yaml(path):
    """Load a YAML file and return its contents."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(cfg, path):
    """Save a dict to a YAML file at the given path."""
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def setup_model_and_loaders(dataset_name, ckpt_subpath="s0/best.ckpt", batch_size=100, num_workers=4, model_cls=None, num_classes=None):
    """
    Load a model checkpoint and prepare ID/OOD dataloaders.
    """
    assert model_cls is not None and num_classes is not None, "Provide model_cls and num_classes"
    net = model_cls(num_classes=num_classes)
    ckpt_path = RESULTS_ROOT / ckpt_subpath
    state = torch.load(str(ckpt_path), map_location="cpu")
    net.load_state_dict(state)
    net.cuda().eval()

    preproc = get_default_preprocessor(dataset_name)
    loaders = get_id_ood_dataloader(
        dataset_name, str(DATA_ROOT), preproc,
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return net, loaders


def collect_scores(postpp, net, loader, n_samples):
    """
    Collect median and MAD scores up to n_samples from a data loader.
    """
    med_list, mad_list = [], []
    collected = 0
    from tqdm import tqdm
    pbar = tqdm(total=n_samples, desc="Collecting scores", unit="sample")
    for batch in loader:
        data = batch['data'].cuda()
        with torch.no_grad():
            _, metrics = postpp.postprocess(net, data)
        med, mad = metrics[0], metrics[1]
        bs = data.size(0)
        for i in range(bs):
            if collected >= n_samples:
                pbar.close()
                return med_list, mad_list
            med_list.append(med[i].cpu().item())
            mad_list.append(mad[i].cpu().item())
            collected += 1
            pbar.update(1)
    pbar.close()
    return med_list, mad_list 


def build_avg_table(blocks, datasets):
    """Build a DataFrame of averaged FPR@95/AUROC metrics from parsed blocks."""
    import pandas as _pd
    labels = list(blocks.keys())
    table = _pd.DataFrame(index=labels, columns=datasets + ["Avg"])
    for label, df in blocks.items():
        df2 = df.set_index("dataset")
        fpr = df2["FPR@95"].str.split("±").str[0].astype(float)
        auroc = df2["AUROC"].str.split("±").str[0].astype(float)
        for ds in datasets:
            table.at[label, ds] = f"{fpr[ds]:.2f}/{auroc[ds]:.2f}"
        avg_fpr = fpr[datasets].mean()
        avg_auroc = auroc[datasets].mean()
        table.at[label, "Avg"] = f"{avg_fpr:.2f}/{avg_auroc:.2f}"
    return table 


def run_and_parse_blocks(postprocessor, labels, id_data="cifar10", batch_size=100, root=RESULTS_ROOT):
    """Run evaluation for a postprocessor and parse its CSV into labeled blocks."""
    # Run eval_ood CLI
    run_eval_ood(postprocessor, id_data=id_data, batch_size=batch_size, root=root)
    # Parse CSV blocks
    csv_path = Path(root) / "ood" / f"{postprocessor}.csv"
    return parse_csv_blocks(str(csv_path), labels) 