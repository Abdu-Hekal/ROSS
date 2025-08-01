#!/usr/bin/env python3
"""
plot_variance_histograms.py: Run VariancePostprocessor with fdbd base and plot histograms for score_median and score_base_mad for ID vs OOD datasets.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(ROOT_DIR)

from openood.evaluation_api.datasets import get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.utils.config import Config
from openood.postprocessors.variance_postprocessor import VariancePostprocessor

NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet200': 200
}

MODEL_ARCH = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot score_median and score_base_mad histograms using VariancePostprocessor with fdbd base.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--id-data', type=str, required=True, choices=list(NUM_CLASSES.keys()), help='ID dataset name')
    parser.add_argument('--data-root', type=str, default=os.path.join(ROOT_DIR, 'data'), help='Root dir for datasets')
    parser.add_argument('--config-root', type=str, default=os.path.join(ROOT_DIR, 'configs'), help='Root dir for configs')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for data loaders')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loaders')
    parser.add_argument('--noise-samples', type=int, default=25, help='Number of noisy samples for variance')
    parser.add_argument('--noise-magnitude', type=float, default=0.05, help='Noise magnitude for variance')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of examples per dataset for histogram')
    parser.add_argument('--output-dir', type=str, default=os.path.join(ROOT_DIR, 'figures'), help='Directory to save figures')
    return parser.parse_args()

def load_model(id_data, checkpoint_path):
    num_classes = NUM_CLASSES[id_data]
    arch = MODEL_ARCH[id_data]
    net = arch(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(state)
    net.cuda()
    net.eval()
    return net

def collect_scores(net, postpp, loader, n_samples):
    median_list = []
    mad_list = []
    collected = 0
    it = iter(loader)
    while collected < n_samples:
        try:
            batch = next(it)
        except StopIteration:
            break
        data = batch['data'].cuda()
        with torch.no_grad():
            _, metrics = postpp.postprocess(net, data)
        idx_med = postpp.metric_labels.index('score_median')
        idx_mad = postpp.metric_labels.index('score_base_mad')
        m_med = metrics[idx_med]
        m_mad = metrics[idx_mad]
        bs = data.size(0)
        for i in range(bs):
            if collected >= n_samples:
                break
            median_list.append(m_med[i].cpu().item())
            mad_list.append(m_mad[i].cpu().item())
            collected += 1
    return np.array(median_list), np.array(mad_list)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    net = load_model(args.id_data, args.checkpoint)
    preproc = get_default_preprocessor(args.id_data)
    loaders = get_id_ood_dataloader(
        args.id_data, args.data_root, preproc,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # configure VariancePostprocessor with fdbd base
    var_cfg = Config(os.path.join(args.config_root, 'postprocessors', 'variance.yml'))
    var_cfg.postprocessor.postprocessor_args.score_postprocessor = 'fdbd'
    var_cfg.postprocessor.postprocessor_args.num_samples = args.noise_samples
    var_cfg.postprocessor.postprocessor_args.noise_magnitude = args.noise_magnitude
    var_cfg.dataset = {'name': args.id_data}

    postpp = VariancePostprocessor(var_cfg)
    try:
        postpp.setup(net, loaders['id'], loaders['ood'])
    except Exception:
        pass

    # prepare loaders for ID and OOD (both near and far splits)
    id_loader = loaders['id']['test']
    groups = {args.id_data: id_loader}
    for split in ['near', 'far']:
        for ds, dl in loaders['ood'][split].items():
            groups[ds] = dl

    # collect and plot
    all_med = {}
    all_mad = {}
    for name, loader in groups.items():
        med, mad = collect_scores(net, postpp, loader, args.n_samples)
        all_med[name] = med
        all_mad[name] = mad

    # plot individual histograms for each OOD dataset
    for ds in groups:
        if ds == args.id_data:
            continue
        for metric, scores_dict in [('score_median', all_med), ('score_base_mad', all_mad)]:
            plt.figure(figsize=(8, 6))
            id_scores = scores_dict[args.id_data]
            ood_scores = scores_dict[ds]
            plt.hist(id_scores, bins=50, alpha=0.5, label=args.id_data)
            plt.hist(ood_scores, bins=50, alpha=0.5, label=ds)
            # remove title and axis labels/ticks
            plt.xticks([])
            plt.yticks([])
            plt.legend(fontsize=32)
            filename = f"hist_{metric.replace(' ', '_').lower()}_{args.id_data}_vs_{ds}.png"
            out_path = os.path.join(args.output_dir, filename)
            plt.savefig(out_path)
            plt.close()

if __name__ == '__main__':
    main() 