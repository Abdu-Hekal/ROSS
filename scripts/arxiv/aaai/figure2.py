#!/usr/bin/env python3
"""
figure2

Generate Figure 2: histograms of ID vs OOD median and MAD scores for ROSS postprocessor on CIFAR-10.
Creates a 2×6 grid (median on top row, negated MAD on bottom row) and saves to scripts/ross/figures/figure2.png.
"""
import os
import sys
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openood.evaluation_api.datasets import get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
from openood.evaluation_api.postprocessor import get_postprocessor
from openood.networks import ResNet18_32x32
from tqdm import tqdm

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


ckpt = "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"

# Load model
net = ResNet18_32x32(num_classes=10)
state = torch.load(ckpt, map_location='cpu')
net.load_state_dict(state)
net.cuda().eval()

# Data loaders
data_root = os.path.join(PROJECT_ROOT, 'data')
preproc = get_default_preprocessor('cifar10')
loader_kwargs = {'batch_size': 100, 'shuffle': False, 'num_workers': 4}
loaders = get_id_ood_dataloader('cifar10', data_root, preproc, **loader_kwargs)

# Postprocessor
config_root = os.path.join(PROJECT_ROOT, 'configs')
postpp = get_postprocessor(config_root, 'ross', 'cifar10')
postpp.setup(net, loaders['id'], loaders['ood'])

# OOD datasets list
datasets = ['cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365']

# Number of examples per dataset for histograms
N_SAMPLES = 1000

def collect_scores(loader, name=''):
    med_list, mad_list = [], []
    collected = 0
    pbar = tqdm(total=N_SAMPLES, desc=f"Collect {name}", unit="sample")
    for batch in loader:
        data = batch['data'].cuda()
        with torch.no_grad():
            _, metrics = postpp.postprocess(net, data)
        med, mad = metrics[0], metrics[1]
        bs = data.size(0)
        for i in range(bs):
            if collected >= N_SAMPLES:
                pbar.close()
                return med_list, mad_list
            med_list.append(med[i].cpu().item())
            mad_list.append(mad[i].cpu().item())
            collected += 1
            pbar.update(1)
    pbar.close()
    return med_list, mad_list

# Collect scores using sampling function
scores = {}
# ID
scores['id'] = {}
scores['id']['median'], scores['id']['mad'] = collect_scores(loaders['id']['test'], name='ID')
# OOD
for ds in datasets:
    dl = loaders['ood']['near'].get(ds, None) or loaders['ood']['far'][ds]
    scores[ds] = {}
    scores[ds]['median'], scores[ds]['mad'] = collect_scores(dl, name=ds)

# Limit number of examples per dataset to 1000 for histograms
n_samples = 1000
for key in ['id'] + datasets:
    for metric in ['median', 'mad']:
        scores[key][metric] = scores[key][metric][:n_samples]

# Plot
fig, axes = plt.subplots(2, len(datasets), figsize=(4*len(datasets), 8))
for j, ds in enumerate(datasets):
    # top: median
    axes[0, j].hist(scores['id']['median'], bins=50, alpha=0.5, label='cifar10')
    axes[0, j].hist(scores[ds]['median'], bins=50, alpha=0.5, label=ds)
    if j == 0:
        axes[0, j].legend()
    axes[0, j].set_title(ds)
    axes[0, j].set_yticks([])
    axes[0, j].set_xticks([])
    # bottom: MAD (negated)
    axes[1, j].hist(scores['id']['mad'], bins=50, alpha=0.5, label='cifar10')
    axes[1, j].hist(scores[ds]['mad'], bins=50, alpha=0.5, label=ds)
    if j == 0:
        axes[1, j].legend()
    axes[1, j].set_yticks([])
    axes[1, j].set_xticks([])

fig.tight_layout(rect=[0.05, 0.03, 1, 0.97])
# Add row labels on the left
fig.text(0.04, 0.75, 'Median', va='center', rotation='vertical', fontsize=14)
fig.text(0.04, 0.25, '-MAD', va='center', rotation='vertical', fontsize=14)
out_file = os.path.join(FIG_DIR, 'figure2.png')
fig.savefig(out_file)
print(f"Saved Figure 2 to {out_file}") 