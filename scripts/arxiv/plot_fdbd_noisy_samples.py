#!/usr/bin/env python3
"""
plot_fdbd_noisy_samples.py: run the fDBD postprocessor on noisy perturbations of ID (cifar-10) and OOD (e.g. cifar-100) images
and plot the fDBD score versus noisy sample index, including median and MAD shading.
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
# set serif font and grid style manually for academic look
matplotlib.rcParams.update({
    # high-resolution, serif font
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],  # use built-in serif font
    'font.size': 14,
    # no grid, minimal spines
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': False,
    'axes.linewidth': 1.0,
    'axes.labelsize': 14,
    # legend without frame
    'legend.frameon': False,
    'legend.fontsize': 12,
    # thicker lines and moderate markers
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    # remove x ticks, keep inward y ticks
    'xtick.bottom': False,
    'xtick.top': False,
    'xtick.labelbottom': False,
    'ytick.right': False,
    'ytick.direction': 'in'
})

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
    parser = argparse.ArgumentParser(description='Plot fDBD scores for noisy samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--id-data', type=str, required=True,
                        choices=list(NUM_CLASSES.keys()),
                        help='ID dataset name (e.g. cifar10)')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(ROOT_DIR, 'data'),
                        help='Root directory for datasets')
    parser.add_argument('--config-root', type=str,
                        default=os.path.join(ROOT_DIR, 'configs'),
                        help='Root directory for config files')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ROOT_DIR, 'plots_fdbd'),
                        help='Directory to save plots')
    parser.add_argument('--ood-split', type=str, choices=['near', 'far'],
                        default='near', help='OOD split to sample from')
    parser.add_argument('--n-noisy-samples', type=int, default=25,
                        help='Number of noisy perturbations per image')
    parser.add_argument('--noise-std', type=float, default=0.05,
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--n-id-images', type=int, default=50,
                        help='Number of ID images to process')
    parser.add_argument('--n-ood-images', type=int, default=50,
                        help='Number of OOD images to process per dataset')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of DataLoader workers')
    return parser.parse_args()


def load_model(id_data, checkpoint_path):
    if id_data not in NUM_CLASSES:
        raise ValueError(f'Unsupported id-data: {id_data}')
    num_classes = NUM_CLASSES[id_data]
    arch = MODEL_ARCH[id_data]
    net = arch(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(state)
    net.cuda()
    net.eval()
    return net


def sample_and_plot_batch(data, ds_name, category, net, postpp, args, start_idx):
    """
    Generate noisy samples for a single image tensor and plot fDBD scores.
    data: tensor of shape [1, C, H, W]
    ds_name: dataset name (e.g. cifar10 or cifar100)
    category: 'ID' or 'OOD'
    image_idx: index of the image sample (for filename)
    """
    device = next(net.parameters()).device
    base_imgs = data.cuda()  # [B, C, H, W]
    B = base_imgs.size(0)
    n = args.n_noisy_samples
    # generate Gaussian noise for each image sample
    noise = torch.randn((B, n, *base_imgs.shape[1:]), device=device) * args.noise_std
    imgs = base_imgs.unsqueeze(1).repeat(1, n, 1, 1, 1)
    # shape [B*n, C, H, W]
    data_noisy = (imgs + noise).view(B * n, *base_imgs.shape[1:])
    # compute scores for all noisy samples at once
    _, scores_flat = postpp.postprocess(net, data_noisy)
    scores = scores_flat.view(B, n).cpu().numpy()
    # compute medians and MADs per image
    medians = np.median(scores, axis=1)
    mads = np.mean(np.abs(scores - medians[:, None]), axis=1)
    # plot each image's noisy sample scores
    for i in range(B):
        median = medians[i]
        mad = mads[i]
        fig, ax = plt.subplots(figsize=(6, 4))
        # already hide x ticks, no grid
        ax.plot(np.arange(n), scores[i], marker='x', linestyle='-')
        # draw median horizontal line with value
        ax.axhline(median, color='black', linestyle='--', label=f'Median = {median:.3f}')
        # shade region for MAD around median
        x_vals = np.arange(n)
        ax.fill_between(x_vals, median - mad, median + mad, color='gray', alpha=0.3, label=f'MAD = {mad:.3f}')
        ax.set_ylabel('fDBD score')
        # removed x-axis title and plot title as per request
        # show legend with values in upper right
        ax.legend(loc='lower right')
        fig.tight_layout()
        out_dir = os.path.join(args.output_dir, category, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f'image{start_idx + i}.png'))
        plt.close(fig)
        # return computed medians and mads for this batch
        return medians, mads


# Function to overlay ID and OOD distributions for special cases
def overlay_plot(id_data, ood_data, id_ds_name, ood_ds_name, net, postpp, args, id_idx, ood_idx, out_dir):
    device = next(net.parameters()).device
    # prepare single images
    id_img = id_data.unsqueeze(0).cuda()
    ood_img = ood_data.unsqueeze(0).cuda()
    n = args.n_noisy_samples
    # generate noisy samples
    noise_id = torch.randn((n, *id_img.shape[1:]), device=device) * args.noise_std
    data_id = id_img.repeat(n,1,1,1) + noise_id
    noise_ood = torch.randn((n, *ood_img.shape[1:]), device=device) * args.noise_std
    data_ood = ood_img.repeat(n,1,1,1) + noise_ood
    # compute scores
    _, scores_id = postpp.postprocess(net, data_id)
    _, scores_ood = postpp.postprocess(net, data_ood)
    scores_id = scores_id.cpu().numpy()
    scores_ood = scores_ood.cpu().numpy()
    # compute medians and MADs
    med_id = np.median(scores_id)
    mad_id = np.median(np.abs(scores_id - med_id))
    med_ood = np.median(scores_ood)
    mad_ood = np.median(np.abs(scores_ood - med_ood))
    # plotting overlay
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(n)
    # colors matching eval_ood histograms (soft blue and orange)
    id_color = '#1f77b4'
    ood_color = '#ff7f0e'
    # plot ID samples
    ax.plot(x, scores_id, marker='o', linestyle='-', color=id_color, markerfacecolor='white', markeredgecolor=id_color, label=f'ID')
    ax.fill_between(x, med_id - mad_id, med_id + mad_id, color=id_color, alpha=0.3, label=f'ID MAD = {mad_id:.3f}')
    ax.axhline(med_id, color=id_color, linestyle='--', label=f'ID Median = {med_id:.3f}')
    # plot OOD samples
    ax.plot(x, scores_ood, marker='x', linestyle='--', color=ood_color, label=f'OOD')
    ax.fill_between(x, med_ood - mad_ood, med_ood + mad_ood, color=ood_color, alpha=0.3, hatch='//', label=f'OOD MAD = {mad_ood:.3f}')
    ax.axhline(med_ood, color=ood_color, linestyle=':', label=f'OOD Median = {med_ood:.3f}')
    # labels and legend
    ax.set_ylabel('fDBD score')
    # place legend above plot, horizontal layout
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False)
    fig.tight_layout()
    # save
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'special_ID{id_idx}_OOD{ood_ds_name}{ood_idx}.png'))
    plt.close(fig)

def main():
    args = parse_args()
    # load model
    net = load_model(args.id_data, args.checkpoint)
    # prepare data loaders (batch_size=1 for individual samples)
    preproc = get_default_preprocessor(args.id_data)
    loader_kwargs = {
        'batch_size': 100,
        'shuffle': True,
        'num_workers': args.num_workers,
    }
    loaders = get_id_ood_dataloader(args.id_data, args.data_root, preproc, **loader_kwargs)
    # prepare postprocessor
    postpp = get_postprocessor(args.config_root, 'fdbd', args.id_data)
    # run setup if available
    try:
        postpp.setup(net, loaders['id'], loaders['ood'])
    except AttributeError:
        pass
    # collect metrics and raw images for comparison
    id_metrics, id_images = [], []
    ood_metrics, ood_images = {}, {}
    # process ID images
    processed = 0
    for batch in loaders['id']['test']:
        data = batch['data']  # [B, C, H, W]
        remain = args.n_id_images - processed
        if remain <= 0:
            break
        if data.size(0) > remain:
            data = data[:remain]
        # store raw ID images
        id_images.extend([img.clone() for img in data])
        # compute and plot ID batch
        med_batch, mad_batch = sample_and_plot_batch(data, args.id_data, 'ID', net, postpp, args, processed)
        # record metrics
        for idx, (med, mad) in enumerate(zip(med_batch, mad_batch)):
            id_metrics.append((processed + idx, med, mad))
        processed += len(med_batch)
        if processed >= args.n_id_images:
            break
    # process OOD images
    split = args.ood_split
    for ds_name, ood_loader in loaders['ood'][split].items():
        ood_metrics[ds_name] = []
        ood_images[ds_name] = []
        processed = 0
        for batch in ood_loader:
            data = batch['data']
            remain = args.n_ood_images - processed
            if remain <= 0:
                break
            if data.size(0) > remain:
                data = data[:remain]
            # store raw OOD images
            ood_images[ds_name].extend([img.clone() for img in data])
            # compute and plot OOD batch
            med_batch, mad_batch = sample_and_plot_batch(data, ds_name, 'OOD', net, postpp, args, processed)
            # record metrics
            for idx, (med, mad) in enumerate(zip(med_batch, mad_batch)):
                ood_metrics[ds_name].append((processed + idx, med, mad))
            processed += len(med_batch)
            if processed >= args.n_ood_images:
                break
    # identify and print special instances
    special_dir = os.path.join(args.output_dir, 'special')
    os.makedirs(special_dir, exist_ok=True)
    for id_idx, id_med, id_mad in id_metrics:
        for ds_name, metrics in ood_metrics.items():
            for ood_idx, ood_med, ood_mad in metrics:
                if id_med < ood_med and id_mad < ood_mad:
                    print(f"Special case ID {id_idx} vs OOD {ds_name} {ood_idx}")
                    overlay_plot(id_images[id_idx], ood_images[ds_name][ood_idx],
                                 args.id_data, ds_name, net, postpp, args,
                                 id_idx, ood_idx, special_dir)

if __name__ == '__main__':
    main() 