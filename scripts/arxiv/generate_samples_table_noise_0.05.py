#!/usr/bin/env python3
import os
import pandas as pd
from io import StringIO

def load_metric_section(file_path, section_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    section_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == section_name:
            section_idx = idx
            break
    if section_idx is None:
        raise ValueError(f"Section '{section_name}' not found in {file_path}")
    header = lines[section_idx + 1].strip()
    data_lines = []
    for line in lines[section_idx + 2:]:
        if not line.strip():
            break
        data_lines.append(line.strip())
    df = pd.read_csv(StringIO("\n".join([header] + data_lines)))
    df['FPR95_mean'] = df['FPR@95'].apply(lambda x: float(x.split('±')[0].strip()))
    df['AUROC_mean'] = df['AUROC'].apply(lambda x: float(x.split('±')[0].strip()))
    result = df.set_index('dataset')[['FPR95_mean', 'AUROC_mean']]
    return result.drop(['nearood', 'farood'], errors='ignore')

def main():
    noise = '0.05'
    samples = [5, 10, 25, 50, 100]
    no_attack_dir = os.path.join('scripts', 'experiments', 'outputs', 'variance_sweep')
    attack_base_dir = os.path.join('scripts', 'experiments', 'outputs', 'attack_ood', 'cifar10', 'LinfPGD')
    section = 'score_min_max_95_alt_6_median'
    eps_map = {'2/255': '0.007843137', '4/255': '0.0156862745', '8/255': '0.031372549', '16/255': '0.062745098'}

    table = {}
    for s in samples:
        # no attack
        na_file = os.path.join(no_attack_dir, f'variance_n{noise}_s{s}.csv')
        df_na = load_metric_section(na_file, section)
        na_mean = df_na.mean()
        # attacks
        atk_dict = {}
        for label, pref in eps_map.items():
            for atk in ['min', 'max']:
                if s == 25:
                    subdir = os.path.join(attack_base_dir, f'variance_{s}_samples', 'variance_fdbd')
                else:
                    subdir = os.path.join(attack_base_dir, f'variance_{s}_samples')
                atk_file = os.path.join(subdir, f'variance_fdbd_noise{noise}_LinfPGD_eps{pref}_{atk}.csv')
                if os.path.exists(atk_file):
                    df_atk = load_metric_section(atk_file, section)
                    atk_dict[(label, atk)] = df_atk.mean()
                else:
                    atk_dict[(label, atk)] = pd.Series({'FPR95_mean': float('nan'), 'AUROC_mean': float('nan')})
        table[s] = {'No Attack': na_mean, **atk_dict}

    out_path = os.path.join('scripts', 'experiments', 'table_samples_noise_0.05.tex')
    with open(out_path, 'w') as f:
        f.write(r"""\begin{table*}[ht!]
\centering
\caption{%
    OOD detection robustness vs sample size for noise=0.05. 
    Results averaged over all benchmarks and reported as FPR95 (\%) $\downarrow$ / AUROC (\%) $\uparrow$.}
\label{tab:table_samples_noise_0.05}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|c|cc|cc|cc|cc}
\toprule
\textbf{Number of samples} & \textbf{No Attack} & \multicolumn{2}{c|}{\textbf{$\epsilon=2/255$}} & \multicolumn{2}{c|}{\textbf{$\epsilon=4/255$}} & \multicolumn{2}{c|}{\textbf{$\epsilon=8/255$}} & \multicolumn{2}{c}{\textbf{$\epsilon=16/255$}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
& ($\epsilon=0$) & Min & Max & Min & Max & Min & Max & Min & Max \\
\midrule""")
        f.write('\n')
        for s in samples:
            row = [str(s), f"{table[s]['No Attack']['FPR95_mean']:.2f}/{table[s]['No Attack']['AUROC_mean']:.2f}"]
            for label in ['2/255', '4/255', '8/255', '16/255']:
                for atk in ['min', 'max']:
                    m = table[s][(label, atk)]
                    row.append(f"{m['FPR95_mean']:.2f}/{m['AUROC_mean']:.2f}")
            f.write(' & '.join(row) + r" \\ " + "\n")
        f.write(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    print(f"Saved LaTeX table to {out_path}")

if __name__ == '__main__':
    main() 