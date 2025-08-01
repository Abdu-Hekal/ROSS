#!/usr/bin/env python3
"""
visualize_scores.py: Visualize base vs noisy sample scores and mean vs coefficient of variation
for ID and OOD datasets using the VariancePostprocessor.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(ROOT_DIR)

from openood.evaluation_api.datasets import get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
from openood.evaluation_api.postprocessor import get_postprocessor
from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50

# mapping for model architectures
NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet200': 200,
}
MODEL_ARCH = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize scores for ID/OOD samples')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (best.ckpt)')
    parser.add_argument('--id-data', type=str, required=True, choices=list(NUM_CLASSES.keys()), help='ID dataset name')
    parser.add_argument('--data-root', type=str, default=os.path.join(ROOT_DIR, 'data'), help='Root directory for datasets')
    parser.add_argument('--config-root', type=str, default=os.path.join(ROOT_DIR, 'configs'), help='Root directory for config files')
    parser.add_argument('--postprocessor', type=str, default='variance', help='Name of postprocessor (must be variance)')
    parser.add_argument('--ood-split', type=str, choices=['near', 'far'], default='near', help='OOD split to sample from')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples per dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle datasets when sampling')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for sampling')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--output-dir', type=str, default=os.path.join(ROOT_DIR, 'plots'), help='Directory to save plots')
    return parser.parse_args()


def load_model(id_data, checkpoint_path):
    if id_data not in NUM_CLASSES:
        raise ValueError(f'Unsupported id-data: {id_data}')
    num_classes = NUM_CLASSES[id_data]
    arch = MODEL_ARCH[id_data]
    net = arch(num_classes=num_classes)
    # load state dict
    state = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(state)
    net.cuda()
    net.eval()
    return net


def sample_and_plot(net, postpp, loader, group_name, args):
    """Compute requested metrics using VariancePostprocessor over multi-sample batches."""
    # prepare output folder
    out_group = os.path.join(args.output_dir, group_name)
    os.makedirs(out_group, exist_ok=True)
    # initialize stats dynamically (list of lists, one per metric)
    stats = []
    loader_iter = iter(loader)
    collected = 0
    # collect until desired number of samples
    while collected < args.n_samples:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        data = batch['data'].cuda()
        # vectorized metric computation
        _, metrics = postpp.postprocess(net, data)
        # metrics: list of tensors shape (batch_size,)
        batch_size = data.size(0)
        # initialize stats shape on first batch
        if not stats:
            num_metrics = len(metrics)
            stats = [[] for _ in range(num_metrics)]
        for j in range(batch_size):
            if collected >= args.n_samples:
                break
            # collect each metric value for sample j
            for idx, metric in enumerate(metrics):
                stats[idx].append(metric[j].cpu().item())
            collected += 1
    return stats


def main():
    args = parse_args()
    # load model
    net = load_model(args.id_data, args.checkpoint)
    # prepare dataloaders (batch_size 1 for sampling)
    preproc = get_default_preprocessor(args.id_data)
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'num_workers': args.num_workers,
    }
    loaders = get_id_ood_dataloader(args.id_data, args.data_root, preproc, **loader_kwargs)
    # prepare postprocessor
    postpp = get_postprocessor(args.config_root, args.postprocessor, args.id_data)
    # run setup to compute any needed hyperparams
    try:
        postpp.setup(net, loaders['id'], loaders['ood'])
    except AttributeError:
        pass
    # visualize ID
    # collect mean and cov for each dataset
    all_stats = {}
    all_stats['ID'] = sample_and_plot(net, postpp, loaders['id']['test'], 'ID', args)

    # visualize each OOD dataset
    split = args.ood_split
    for ds, dl in loaders['ood'][split].items():
        all_stats[ds] = sample_and_plot(net, postpp, dl, ds, args)

    # combined scatter plots for metrics: plot each metric vs mean
    metric_labels = postpp.metric_labels
    num_metrics = len(metric_labels)
    for metric_idx in range(1, num_metrics):
        fig, ax = plt.subplots()
        for ds, stats in all_stats.items():
            x_vals = stats[0]
            y_vals = stats[metric_idx]
            if x_vals:
                ax.scatter(x_vals, y_vals, label=ds)
        x_label = metric_labels[1]
        y_label = metric_labels[metric_idx]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Combined {x_label} vs {y_label}')
        ax.legend()
        fig.savefig(os.path.join(args.output_dir, f'combined_{x_label}_vs_{y_label}.png'))
        plt.close(fig)

if __name__ == '__main__':
    main() 