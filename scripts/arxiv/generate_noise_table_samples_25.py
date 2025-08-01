#!/usr/bin/env python3
import os
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def load_metric_section(file_path, section_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # locate section header
    section_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == section_name:
            section_idx = idx
            break
    if section_idx is None:
        raise ValueError(f"Section '{section_name}' not found in {file_path}")
    # header is next line
    header = lines[section_idx + 1].strip()
    # collect data lines until blank
    data_lines = []
    for line in lines[section_idx + 2:]:
        if not line.strip():
            break
        data_lines.append(line.strip())
    # parse into DataFrame
    df = pd.read_csv(StringIO("\n".join([header] + data_lines)))
    # extract FPR95 and AUROC means
    df['FPR95_mean'] = df['FPR@95'].apply(lambda x: float(x.split('±')[0].strip()))
    df['AUROC_mean'] = df['AUROC'].apply(lambda x: float(x.split('±')[0].strip()))
    result = df.set_index('dataset')[['FPR95_mean', 'AUROC_mean']]
    return result.drop(['nearood', 'farood'], errors='ignore')


def main():
    # base directories
    no_attack_dir = os.path.join('scripts', 'experiments', 'outputs', 'variance_sweep')
    attack_base_dir = os.path.join('scripts', 'experiments', 'outputs', 'attack_ood', 'cifar10', 'LinfPGD')

    # fixed sample size
    sample = 25
    # attack epsilons
    eps_map = {'2/255': '0.007843137', '4/255': '0.0156862745', '8/255': '0.031372549', '16/255': '0.062745098'}

    # detect noise magnitudes for sample size
    noise_files = [f for f in os.listdir(no_attack_dir) if f.startswith('variance_n') and f.endswith(f'_s{sample}.csv')]
    noise_vals = sorted({f.split('_')[1][1:] for f in noise_files}, key=lambda x: float(x))
    # exclude noise magnitude 0.01 from results
    noise_vals = [n for n in noise_vals if n != '0.01']

    # section of interest
    section = 'score_min_max_95_alt_6_median'

    # collect metrics
    table = {}
    for noise in noise_vals:
        # no-attack metrics
        na_file = os.path.join(no_attack_dir, f'variance_n{noise}_s{sample}.csv')
        if os.path.exists(na_file):
            df_na = load_metric_section(na_file, section)
            na_mean = df_na.mean()
        else:
            na_mean = pd.Series({'FPR95_mean': float('nan'), 'AUROC_mean': float('nan')})

        # attacked metrics
        atk_dict = {}
        subdir = os.path.join(attack_base_dir, f'variance_{sample}_samples', 'variance_fdbd')
        for label, pref in eps_map.items():
            for atk in ['min', 'max']:
                atk_file = os.path.join(subdir, f'variance_fdbd_noise{noise}_LinfPGD_eps{pref}_{atk}.csv')
                if os.path.exists(atk_file):
                    df_atk = load_metric_section(atk_file, section)
                    atk_mean = df_atk.mean()
                else:
                    atk_mean = pd.Series({'FPR95_mean': float('nan'), 'AUROC_mean': float('nan')})
                atk_dict[(label, atk)] = atk_mean

        table[noise] = {'No Attack': na_mean, **atk_dict}

    # generate LaTeX table
    tex_path = os.path.join('scripts', 'experiments', 'combined_attacks.tex')
    with open(tex_path, 'w') as f:
        f.write(r"""\begin{table*}[ht!]
\centering
\caption{%
    Robustness of OOD detection scores for CIFAR-10 against PGD-\textbf{min} and PGD-\textbf{max} attacks with sample size 25. 
    Results are average over all benchmarks and performance is reported as FPR95 (\%) $\downarrow$ / AUROC (\%) $\uparrow$.}
\label{tab:combined_attacks}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|c|cc|cc|cc|cc}
\toprule
\textbf{Noise Magnitude} & \textbf{No Attack} & \multicolumn{2}{c|}{\textbf{$\epsilon=2/255$}} & \multicolumn{2}{c|}{\textbf{$\epsilon=4/255$}} & \multicolumn{2}{c|}{\textbf{$\epsilon=8/255$}} & \multicolumn{2}{c}{\textbf{$\epsilon=16/255$}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
& ($\epsilon=0$) & Min & Max & Min & Max & Min & Max & Min & Max \\
\midrule""")
        f.write('\n')
        for noise in noise_vals:
            row_vals = [noise, f"{table[noise]['No Attack']['FPR95_mean']:.2f}/{table[noise]['No Attack']['AUROC_mean']:.2f}"]
            for label in eps_map:
                for atk in ['min', 'max']:
                    m = table[noise][(label, atk)]
                    row_vals.append(f"{m['FPR95_mean']:.2f}/{m['AUROC_mean']:.2f}")
            f.write(' & '.join(row_vals) + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    print(f"Saved LaTeX table to {tex_path}")
    # generate AUROC vs attack magnitude plots (including no-attack at 0)
    eps_labels = ['0'] + list(eps_map.keys())
    x = list(range(len(eps_labels)))
    # define distinct markers for noise levels
    markers = ['o', 's', '^', 'D', 'v', 'x', 'P', '*', 'h', '+']
    
    for atk in ['min', 'max']:
        plt.figure()
        # plot fdbd curves for each noise
        for idx, noise in enumerate(noise_vals):
            y_vals = [table[noise]['No Attack']['AUROC_mean']] + [table[noise][(label, atk)]['AUROC_mean'] for label in eps_map.keys()]
            marker = markers[idx % len(markers)]
            plt.plot(x, y_vals, marker=marker, linestyle='-', label=noise)
        
        plt.xlabel('Attack Magnitude (ϵ)')
        plt.ylabel('AUROC (%)')
        plt.xticks(x, eps_labels)
        plt.gca().tick_params(axis='x', bottom=True, labelbottom=True)
        plt.legend(title='Noise Magnitude', frameon=True)
        plt.grid(True)
        plot_path = os.path.join('scripts', 'experiments', f'auroc_vs_attack_{atk}.png')
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    main() 