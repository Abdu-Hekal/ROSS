#!/usr/bin/env python3
"""
Visualize score landscapes for different OOD detection methods under input perturbation.
Methods: MSP, EBO, fDBD, GEN.
Default: CIFAR10 in-distribution, SVHN out-of-distribution, eps=0.05, grid_size=25.
"""
import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils.data import DataLoader, Dataset
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import matplotlib
# apply professional styling
matplotlib.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 14,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': False,
    'axes.linewidth': 1.0,
    'axes.labelsize': 14,
    'legend.frameon': False,
    'legend.fontsize': 12,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'xtick.bottom': False,
    'xtick.top': False,
    'xtick.labelbottom': False,
    'ytick.right': False,
    'ytick.direction': 'in'
})
from torchvision import datasets, transforms

# Import network and postprocessors from openood
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.postprocessors.maxlogit_postprocessor import MaxLogitPostprocessor
from openood.postprocessors.ebo_postprocessor import EBOPostprocessor
from openood.postprocessors.fdbd_postprocessor import fDBDPostprocessor
from openood.postprocessors.gen_postprocessor import GENPostprocessor


# Dummy config to initialize postprocessors
class DummyConfig:
    pass

def get_dummy_config():
    config = DummyConfig()
    class PP:
        pass
    config.postprocessor = PP()
    args = PP()
    args.temperature = 1.0
    args.gamma = 0.1
    args.M = 100
    args.distance_as_normalizer = False
    config.postprocessor.postprocessor_args = args
    config.postprocessor.postprocessor_sweep = None
    return config

# Dataset wrapper to provide dict batches for fDBD
class DictDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]
        return {'data': data, 'label': label}

# Utility to load datasets by name
def load_dataset(name, data_root, transform):
    name = name.lower()
    if name == 'cifar10':
        return datasets.CIFAR10(root=data_root, train=False, transform=transform, download=True)
    elif name == 'svhn':
        return datasets.SVHN(root=data_root, split='test', transform=transform, download=True)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=data_root, train=False, transform=transform, download=True)
    else:
        raise ValueError(f'Unknown dataset: {name}')


def load_model(model_path, device):
    model = ResNet18_32x32()
    checkpoint = torch.load(model_path, map_location=device)
    # handle PyTorch Lightning or raw state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # strip leading 'model.' if present
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state[k[len('model.'):]] = v
            else:
                new_state[k] = v
        state_dict = new_state
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_surface(x, model, postprocessor, device, eps, grid_size):
    # x: single sample tensor [C,H,W]
    x = x.to(device)
    # sample two random orthonormal directions
    flat = x.view(-1)
    u = torch.randn_like(flat)
    u = u / u.norm()
    v = torch.randn_like(flat)
    v = v - (u * v).sum() * u
    v = v / v.norm()
    u = u.view_as(x)
    v = v.view_as(x)

    a_vals = torch.linspace(-eps, eps, grid_size)
    b_vals = torch.linspace(-eps, eps, grid_size)
    Z = np.zeros((grid_size, grid_size))
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            x_pert = x + a * u + b * v
            x_pert = x_pert.clamp(0, 1).unsqueeze(0)
            with torch.no_grad():
                pred, score = postprocessor.postprocess(model, x_pert)
            Z[i, j] = score.cpu().item()
    return a_vals.cpu().numpy(), b_vals.cpu().numpy(), Z


def main():
    parser = argparse.ArgumentParser(
        description="Visualize score landscapes under input perturbations"
    )
    parser.add_argument(
        '--model_path', type=str,
        default='results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
        help='Path to s0 model checkpoint'
    )
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--ood_dataset', type=str, default='svhn',
        choices=['svhn', 'cifar100'],
        help='Out-of-distribution dataset'
    )
    parser.add_argument(
        '--ood_datasets', type=str, default='svhn',
        help='Comma-separated list of OOD dataset names (overrides --ood_dataset)'
    )
    parser.add_argument(
        '--eps', type=float, default=0.05,
        help='Max perturbation epsilon'
    )
    parser.add_argument(
        '--grid_size', type=int, default=25,
        help='Grid size for landscape'
    )
    parser.add_argument(
        '--sample_index', type=int, default=0,
        help='Index of sample to visualize (same for ID and OOD)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility (auto-generated if not set)'
    )
    parser.add_argument(
        '--save_path', type=str, default='score_landscape.png',
        help='Path to save the output figure'
    )
    args = parser.parse_args()
    # Set random seed for reproducibility and randomness
    if args.seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
    else:
        seed = args.seed
    print(f"Using random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms (normalize for CIFAR10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ])

    # Build list of dataset names (ID first, then OOD)
    if args.ood_datasets:
        ood_list = [s.strip() for s in args.ood_datasets.split(',')]
    else:
        ood_list = [args.ood_dataset]
    dataset_names = ['cifar10'] + ood_list
    datasets_list = [load_dataset(name, args.data_root, transform) for name in dataset_names]

    # Train loader for fDBD
    base_train_dataset = datasets.CIFAR10(
        root=args.data_root, train=True,
        transform=transform, download=True
    )
    # Wrap dataset to yield dict batches for fDBD
    train_dataset = DictDataset(base_train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=128,
        shuffle=False, num_workers=4
    )

    # Load model
    model = load_model(args.model_path, device)

    # Create dummy config and initialize postprocessors
    config = get_dummy_config()
    msp_proc = MaxLogitPostprocessor(config)
    ebo_proc = EBOPostprocessor(config)
    fdbd_proc = fDBDPostprocessor(config)
    gen_proc = GENPostprocessor(config)
    # Setup fDBD (compute training mean)
    fdbd_proc.setup(model, {'train': train_loader}, {})

    methods = [
        ('MSP', msp_proc),
        ('EBO', ebo_proc),
        ('FDBD', fdbd_proc),
        ('GEN', gen_proc),
    ]

    # Select sample from each dataset
    data_samples = {}
    for name, ds in zip(dataset_names, datasets_list):
        x, _ = ds[args.sample_index]
        data_samples[name] = x

    # Compute landscapes for each dataset and method
    results = {}
    for method_name, proc in methods:
        for name in dataset_names:
            a, b, z = compute_surface(
                data_samples[name], model, proc, device, args.eps, args.grid_size
            )
            results[(name, method_name)] = (a, b, z)

    # Plot all landscapes in grid
    # Define ID and OOD colours for deviation colormap (from plot_fdbd style)
    id_color = '#1f77b4'
    ood_color = '#ff7f0e'
    n_rows = len(dataset_names)
    n_cols = len(methods)
    # dynamic sizing: approx 3" width per column, 2.5" height per row
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 2.5), constrained_layout=True)
    # prepare deviation colormap: ID colour at zero deviation, OOD colour at max
    cmap_abs = LinearSegmentedColormap.from_list('absdev', [id_color, ood_color])
    for i, name in enumerate(dataset_names):
        for j, (method_name, proc) in enumerate(methods):
            a, b, z = results[(name, method_name)]
            A, B = np.meshgrid(a, b)
            ax = fig.add_subplot(n_rows, n_cols, i*n_cols + j + 1, projection='3d')
            # reduce number of ticks
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.zaxis.set_major_locator(MaxNLocator(3))
            # deviation-based facecolors
            dev_abs = np.abs(z - np.median(z))
            norm_abs = Normalize(vmin=0, vmax=dev_abs.max())
            facecolors = cmap_abs(norm_abs(dev_abs))
            ax.plot_surface(A, B, z, facecolors=facecolors, shade=True)
            # titles and row labels
            if i == 0:
                ax.set_title(method_name.upper())
            if j == 0:
                ax.set_ylabel(name.upper(), labelpad=10)
            # clear axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            # remove numeric tick labels (keep tick marks and grid)
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
    # Save and show
    plt.savefig(args.save_path)
    print(f"Saved figure to {args.save_path}")
    plt.show()

if __name__ == "__main__":
    main()
