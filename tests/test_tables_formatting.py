import os
import sys
import tempfile
import unittest
import pandas as pd

# Ensure script directory is importable as top-level for absolute imports used in modules
_THIS_DIR = os.path.dirname(__file__)
_ROSS_SCRIPTS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "scripts", "ross"))
if _ROSS_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _ROSS_SCRIPTS_DIR)

# Import functions under test from modules that use absolute imports
from run_experiments import _format_avg_fpr_auroc
from ross_utils import parse_csv_blocks, parse_csv_skiprows, build_avg_table


DATASETS = ["cifar100", "tin", "mnist", "svhn", "texture", "places365", "nearood", "farood"]


def _block_csv(label: str, rows: list) -> str:
    lines = [label, "dataset,FPR@95,AUROC,AUPR_IN,AUPR_OUT,ACC"]
    lines.extend(rows)
    lines.append("")
    return "\n".join(lines)


def _make_rows(fpr_values, auroc_values):
    rows = []
    for ds, f, a in zip(DATASETS, fpr_values, auroc_values):
        rows.append(f"{ds},{f:.2f} ± 0.10,{a:.2f} ± 0.10,0,0,0")
    return rows


class TestTablesFormatting(unittest.TestCase):
    def test_format_avg_fpr_auroc_tolerates_stray_header(self):
        df = pd.DataFrame({
            "dataset": ["cifar100", "header", "tin"],
            "FPR@95": ["10.00 ± 0.1", "FPR@95", "20.00 ± 0.2"],
            "AUROC": ["90.00 ± 0.5", "AUROC", "80.00 ± 0.4"],
        })
        out = _format_avg_fpr_auroc(df)
        self.assertEqual(out, "15.00/85.00")

    def test_parse_csv_skiprows_drops_repeated_headers(self):
        content = "\n".join([
            "ignored header line",
            "dataset,FPR@95,AUROC,AUPR_IN,AUPR_OUT,ACC",
            "dataset,FPR@95,AUROC,AUPR_IN,AUPR_OUT,ACC",  # repeated header in body
            "cifar100,10.00 ± 0.1,90.00 ± 0.1,0,0,0",
            "tin,20.00 ± 0.1,80.00 ± 0.1,0,0,0",
        ])
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "msp.csv")
            with open(fp, "w") as f:
                f.write(content)
            df = parse_csv_skiprows(fp, skiprows=1)
            # The stray header row should be removed
            self.assertEqual(len(df), 2)
            self.assertEqual(_format_avg_fpr_auroc(df), "15.00/85.00")

    def test_pro_uses_score_pro_v2_block(self):
        # Construct pro.csv with three blocks; we only care about score_pro_v2
        fprs = [10, 20, 30, 40, 50, 60, 999, 999]
        aurocs = [90, 80, 70, 60, 50, 40, 999, 999]
        rows_v2 = _make_rows(fprs, aurocs)
        rows_other = _make_rows([1]*8, [2]*8)
        content = "\n".join([
            _block_csv("score_base", rows_other),
            _block_csv("score_pro", rows_other),
            _block_csv("score_pro_v2", rows_v2),
        ])
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "pro.csv")
            with open(fp, "w") as f:
                f.write(content)
            blocks = parse_csv_blocks(fp, ["score_pro_v2"])
            df = blocks["score_pro_v2"]
            df = df[~df["dataset"].isin(["nearood", "farood"])]
            self.assertEqual(len(df), 6)
            # Expected mean of first 6 fprs and aurocs
            self.assertEqual(_format_avg_fpr_auroc(df), f"{sum(fprs[:6])/6:.2f}/{sum(aurocs[:6])/6:.2f}")

    def test_ross_uses_ross_block(self):
        fprs = [12, 22, 32, 42, 52, 62, 999, 999]
        aurocs = [88, 78, 68, 58, 48, 38, 999, 999]
        rows_ross = _make_rows(fprs, aurocs)
        rows_other = _make_rows([3]*8, [4]*8)
        content = "\n".join([
            _block_csv("median", rows_other),
            _block_csv("mad", rows_other),
            _block_csv("cov", rows_other),
            _block_csv("ross", rows_ross),
        ])
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "ross.csv")
            with open(fp, "w") as f:
                f.write(content)
            blocks = parse_csv_blocks(fp, ["median","mad","cov","ross"])
            df = blocks["ross"]
            df = df[~df["dataset"].isin(["nearood", "farood"])]
            self.assertEqual(len(df), 6)
            self.assertEqual(_format_avg_fpr_auroc(df), f"{sum(fprs[:6])/6:.2f}/{sum(aurocs[:6])/6:.2f}")

    def test_build_avg_table_uses_all_blocks(self):
        datasets = ["cifar100", "tin", "mnist", "svhn", "texture", "places365"]
        def df_for(vals_fpr, vals_au):
            return pd.DataFrame({
                "dataset": datasets,
                "FPR@95": [f"{v:.2f} ± 0.1" for v in vals_fpr],
                "AUROC": [f"{v:.2f} ± 0.1" for v in vals_au],
            })
        blocks = {
            "median": df_for([10, 20, 30, 40, 50, 60], [90, 80, 70, 60, 50, 40]),
            "mad": df_for([11, 21, 31, 41, 51, 61], [89, 79, 69, 59, 49, 39]),
            "cov": df_for([12, 22, 32, 42, 52, 62], [88, 78, 68, 58, 48, 38]),
            "ross": df_for([13, 23, 33, 43, 53, 63], [87, 77, 67, 57, 47, 37]),
        }
        table = build_avg_table(blocks, datasets)
        # Check Avg for 'ross'
        expected_fpr_avg = sum([13,23,33,43,53,63]) / 6
        expected_au_avg = sum([87,77,67,57,47,37]) / 6
        self.assertEqual(table.loc["ross", "Avg"], f"{expected_fpr_avg:.2f}/{expected_au_avg:.2f}")
        # Check one dataset cell formatted
        self.assertEqual(table.loc["ross", "cifar100"], "13.00/87.00")

    def test_attack_block_selection_examples(self):
        # pro attack file with score_pro_v2
        fprs = [14, 24, 34, 44, 54, 64, 999, 999]
        aurocs = [86, 76, 66, 56, 46, 36, 999, 999]
        rows_v2 = _make_rows(fprs, aurocs)
        content_pro = _block_csv("score_pro_v2", rows_v2)
        # ross attack file with multiple blocks
        rows_ross = _make_rows([15,25,35,45,55,65,999,999], [85,75,65,55,45,35,999,999])
        rows_other = _make_rows([0]*8, [0]*8)
        content_ross = "\n".join([
            _block_csv("median", rows_other),
            _block_csv("mad", rows_other),
            _block_csv("cov", rows_other),
            _block_csv("ross", rows_ross),
        ])
        with tempfile.TemporaryDirectory() as td:
            pro_fp = os.path.join(td, "pro_LinfPGD.csv")
            ross_fp = os.path.join(td, "ross_LinfPGD.csv")
            with open(pro_fp, "w") as f:
                f.write(content_pro)
            with open(ross_fp, "w") as f:
                f.write(content_ross)
            # PRO
            blocks = parse_csv_blocks(pro_fp, ["score_pro_v2"])
            df = blocks["score_pro_v2"]
            df = df[~df["dataset"].isin(["nearood","farood"])]
            self.assertEqual(_format_avg_fpr_auroc(df), f"{sum(fprs[:6])/6:.2f}/{sum(aurocs[:6])/6:.2f}")
            # ROSS
            blocks = parse_csv_blocks(ross_fp, ["median","mad","cov","ross"])
            df = blocks["ross"]
            df = df[~df["dataset"].isin(["nearood","farood"])]
            self.assertEqual(_format_avg_fpr_auroc(df), "40.00/60.00")


if __name__ == "__main__":
    unittest.main()


