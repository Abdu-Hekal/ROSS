#!/usr/bin/env python3
import pandas as pd, os

BENCHMARK = "cifar10"
# Base dir for attack-OOD results
BASE = os.path.join(
    os.path.dirname(__file__),
    "experiments", "outputs", "attack_ood", BENCHMARK , "LinfPGD"
)

if BENCHMARK == "cifar100":
    # Your known No-Attack FPR95/AUROC values
    no_attack = {
        "MSP":        (57.40, 78.60),
        "EBO":        (56.26, 80.15),
        "GEN":        (55.95, 80.23),
        "ODIN":       (58.54, 79.49),
        "FDBD":       (54.63, 80.48),
        "PRO-FDBD":   (54.26, 80.51),
        "ROSS-MSP":   (60.04, 79.60),
        "ROSS-EBO":   (57.59, 77.94),
        "ROSS-GEN":   (56.94, 77.92),
        "ROSS-FDBD":  (54.76, 79.23),
    }
elif BENCHMARK == "cifar10":
    # Your known No-Attack FPR95/AUROC values
    no_attack = {
        "MSP":        (37.21, 89.83),
        "EBO":        (55.20, 90.00),
        "GEN":        (41.04, 90.3),
        "ODIN":       (63.81, 86.26),
        "FDBD":       (27.71, 92.3),
        "PRO-FDBD":   (27.83, 92.43),
        "ROSS-MSP":   (34.64, 88.64),
        "ROSS-EBO":   (49.54, 87.91),
        "ROSS-GEN":   (36.04, 89.04),
        "ROSS-FDBD":  (30.95, 90.62),
    }

# (Latex label, CSV tag or None for ROSS)
postprocs = [
    ("MSP",       "msp"),
    ("EBO",       "ebo"),
    ("GEN",       "gen"),
    ("ODIN",      "odin"),
    ("FDBD",      "fdbd"),
    ("PRO-FDBD",  "pro_fdbd"),
    ("ROSS-MSP", None),
    ("ROSS-EBO", None),
    ("ROSS-GEN", None),
    ("ROSS-FDBD", None),
]

# Attack radii
epsilons = [
    ("2/255",  "0.007843137"),
    ("4/255",  "0.0156862745"),
    ("8/255",  "0.031372549"),
    ("16/255", "0.0627410098"),
]

def mean_from_csv(path):
    df = pd.read_csv(path, skiprows=1)
    df = df[~df["dataset"].isin(["nearood", "farood"])]
    fpr = df["FPR@95"].str.split("±").str[0].astype(float).mean()
    au  = df["AUROC"].str.split("±").str[0].astype(float).mean()
    return fpr, au

def mean_mm95(path):
    lines = open(path).read().splitlines()
    out, cap = [], False
    for l in lines:
        if l.strip() == "score_min_max_95_alt_6_median":
            cap = True
            continue
        if cap and l.startswith("score_"):
            break
        if cap and l and not l.startswith("dataset"):
            ds, fpr, au, *_ = [x.strip() for x in l.split(",")]
            if ds not in ("nearood","farood"):
                out.append((float(fpr.split("±")[0]), float(au.split("±")[0])))
    arr = pd.DataFrame(out, columns=("FPR","AUROC"))
    return arr["FPR"].mean(), arr["AUROC"].mean()

def mean_pro_v2(path):
    """
    Parse the 'score_pro_v2' block from PRO-FDBD CSV and return mean FPR@95 and AUROC.
    """
    lines = open(path).read().splitlines()
    out, cap = [], False
    for l in lines:
        if l.strip() == "score_pro_v2":
            cap = True
            continue
        if cap and l.startswith("score_"):
            break
        if cap and l and "," in l and not l.startswith("dataset"):
            ds, fpr, au, *_ = [x.strip() for x in l.split(",")]
            if ds not in ("nearood", "farood"):
                out.append((float(fpr.split("±")[0]), float(au.split("±")[0])))
    dfp = pd.DataFrame(out, columns=("FPR","AUROC"))
    return dfp["FPR"].mean(), dfp["AUROC"].mean()

# ——————— Print LaTeX ———————
print("\\begin{table*}[ht!]")
print("\\centering")
print("\\caption{Robustness of OOD detection scores for BENCHMARK …}")
print("\\resizebox{\\textwidth}{!}{%")
print("\\begin{tabular}{l|c|cc|cc|cc|cc}")
print("\\toprule")
# Header
print("Post-processor & No Attack", end="")
for eps,_ in epsilons:
    print(f" & \\multicolumn{{2}}{{c|}}{{$\\epsilon={eps}$}}", end="")
print(" \\\\")
# cmidrules
print("\\cmidrule(lr){2-2}" + "".join(
    f" \\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(epsilons))
))
print("& ($\\epsilon=0$) " + " ".join(["& Min & Max"]*len(epsilons)) + " \\\\")
print("\\midrule")

for name, tag in postprocs:
    gray = "\\rowcolor{gray!15} " if name.startswith("ROSS") else ""
    f0, a0 = no_attack[name]
    print(f"{gray}{name} & {f0:.2f}/{a0:.2f}", end=" ")
    for _, val in epsilons:
        if tag == "pro_fdbd":
            fn_min = os.path.join(BASE, f"{tag}_LinfPGD_eps{val}_min.csv")
            fn_max = os.path.join(BASE, f"{tag}_LinfPGD_eps{val}_max.csv")
            fmin, amin = mean_pro_v2(fn_min)
            fmax, amax = mean_pro_v2(fn_max)
        elif tag:
            fn_min = os.path.join(BASE, f"{tag}_LinfPGD_eps{val}_min.csv")
            fn_max = os.path.join(BASE, f"{tag}_LinfPGD_eps{val}_max.csv")
            fmin, amin = mean_from_csv(fn_min)
            fmax, amax = mean_from_csv(fn_max)
        else:
            # Dynamically construct variance filename based on post-processor name
            suffix = name.split("ROSS-")[1].lower()
            base = os.path.join(
                BASE,
                "variance_25_samples",
                f"variance_{suffix}",
                f"variance_{suffix}_noise0.05_LinfPGD_eps{val}"
            )
            fmin, amin = mean_mm95(base + "_min.csv")
            fmax, amax = mean_mm95(base + "_max.csv")
        print(f"& {fmin:5.2f}/{amin:5.2f} & {fmax:5.2f}/{amax:5.2f}", end=" ")
    print("\\\\")
print("\\bottomrule")
print("\\end{tabular}%")
print("}")
print("\\end{table*}")