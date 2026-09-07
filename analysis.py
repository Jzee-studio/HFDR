#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis & Visualization Script for HFDR / LFCM Experiment Logs
================================================================
Reads training record logs, test evaluation logs and OOD logs from the
`result/` directory (CIFAR10 at root, CIFAR100/, TinyImageNet/ and
ResNet18/ subdirs), generates a comprehensive markdown report and
publication-quality figures.

Usage:
    python analysis.py [--result-dir ./result] [--output-dir ./result/figures]

Dependencies: pandas, matplotlib, seaborn (all in environment.yaml)
"""

import os
import re
import sys
import copy
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (12, 8),
})

# ── Colour palettes ───────────────────────────────────────────────────────
PALETTE_CANON = {"nocanon": "#e74c3c", "weak": "#f39c12", "baseline": "#2ecc71", "strong": "#3498db"}
PALETTE_CODEBOOK = {"K32": "#9b59b6", "K64": "#2ecc71", "K128": "#e67e22",
                    "K256": "#16a085", "K512": "#c0392b"}
PALETTE_METHOD = {"AT": "#95a5a6", "HFDR": "#3498db", "LFCM": "#2ecc71", "Natural": "#e74c3c"}
PALETTE_METRIC = {"Clean": "#2ecc71", "PGD-1": "#3498db", "PGD-20": "#f39c12", "PGD-100": "#e67e22", "CW-20": "#9b59b6", "AA": "#e74c3c"}
PALETTE_DATASET = {"CIFAR10": "#3498db", "CIFAR100": "#e67e22", "TinyImageNet": "#9b59b6", "ResNet18": "#16a085"}

STAGE_COLORS = {"Stage1": "#ebf5fb", "Stage2": "#fef9e7", "Stage3": "#eafaf1"}

# ==========================================================================
# 1. EXPERIMENT METADATA
# ==========================================================================

# Dataset -> directory name under result/ ("" = result root)
DATASET_DIRS = {
    "CIFAR10": "",
    "CIFAR100": "CIFAR100",
    "TinyImageNet": "TinyImageNet",
    "ResNet18": "ResNet18",
}

# Baseline LFCM stem per dataset (used to inject K=64 into codebook ablations)
LFCM_BASELINE_STEMS = {
    "CIFAR10": "WRN34_10_LFCM",
    "CIFAR100": "WRN34_100_LFCM",
    "TinyImageNet": "WRN34_200_LFCM",
    "ResNet18": "ResNet18_10_LFCM_K64",
}

# Dataset -> {filename_stem: (display_name, group, variant_label)}
EXPERIMENT_META = {
    "CIFAR10": {
        "WRN34_10_F":                ("AT (WRN34_10_F)",         "method",   "AT"),
        "WRN34_10_F_HFDR":           ("HFDR",                     "method",   "HFDR"),
        "WRN34_10_F_Natural":        ("Natural",                  "method",   "Natural"),
        "WRN34_10_LFCM":             ("LFCM (baseline, K=64)",    "canon",    "baseline"),
        "WRN34_10_LFCM_K32":         ("LFCM K=32",                "codebook", "K32"),
        "WRN34_10_LFCM_K128":        ("LFCM K=128",               "codebook", "K128"),
        "WRN34_10_LFCM_K64_strong":  ("LFCM strong (w_canon=1.0)", "canon",   "strong"),
        "WRN34_10_LFCM_K64_weak":    ("LFCM weak (w_canon=0.25)", "canon",    "weak"),
        "WRN34_10_LFCM_K64_nocanon": ("LFCM nocanon (w_canon=0)", "canon",    "nocanon"),
        "WRN34_10_LFCM_K64_early_canon": ("LFCM early_canon",     "schedule", "early_canon"),
    },
    "CIFAR100": {
        "WRN34_100_AT":        ("AT",        "method", "AT"),
        "WRN34_100_HFDR":      ("HFDR",      "method", "HFDR"),
        "WRN34_100_LFCM":      ("LFCM (K=64)", "method", "LFCM"),
        "WRN34_100_Natural":   ("Natural",   "method", "Natural"),
        "WRN34_100_LFCM_K128": ("LFCM K=128", "codebook", "K128"),
        "WRN34_100_LFCM_K256": ("LFCM K=256", "codebook", "K256"),
        "WRN34_100_LFCM_K512": ("LFCM K=512", "codebook", "K512"),
    },
    "TinyImageNet": {
        "WRN34_200_AT":    ("AT (94/110 ep, incomplete)", "method", "AT"),
        "WRN34_200_LFCM":  ("LFCM",   "method", "LFCM"),
        "WRN34_200_Natural": ("Natural", "method", "Natural"),
    },
    "ResNet18": {
        "ResNet18_10_AT":       ("AT",           "method", "AT"),
        "ResNet18_10_HFDR":     ("HFDR",         "method", "HFDR"),
        "ResNet18_10_LFCM_K64": ("LFCM (K=64)",  "method", "LFCM"),
        "ResNet18_10_Natural":  ("Natural",      "method", "Natural"),
    },
}

@dataclass
class Experiment:
    """Holds all parsed data for one experiment."""
    dataset: str
    stem: str
    display_name: str
    group: str
    variant: str
    record_df: Optional[pd.DataFrame] = None
    test_metrics: Optional[Dict[str, float]] = None
    ood_metrics: Optional[Dict] = None          # {"corruptions": {name: acc}, "mean_acc": float, "mean_loss": float}
    has_test: bool = False
    has_ood: bool = False

# ==========================================================================
# 2. LOG PARSER
# ==========================================================================

class LogParser:
    """Parse HFDR training record logs, test evaluation logs and OOD logs."""

    # Regex for data rows: after "] - " the epoch number, then whitespace-separated floats
    _RE_DATA_ROW = re.compile(r"\]\s*-\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")

    # Regex for test log metrics
    _RE_NORMAL = re.compile(r"Normal Acc:\s*([\d.]+)")
    _RE_PGD = re.compile(r"PGD_attack:\[nb_iter:(\d+).*?\]->pgd_acc:\s*([\d.]+)")
    _RE_CW = re.compile(r"CW_attack:.*?->CW_acc:\s*([\d.]+)")
    _RE_AA = re.compile(r"Auto_attack:.*?->AA_acc:\s*([\d.]+)")
    _RE_BEST = re.compile(r"=======(?:Best|Last)_trained_model Performance=======")

    # Regex for OOD log
    _RE_OOD_LINE = re.compile(r"-\s+(\w+)\s+\|\s+Acc:\s*([\d.]+)\s+\|\s+Loss:\s*([\d.]+)")
    _RE_OOD_MEAN = re.compile(r"Mean Acc:\s*([\d.]+)\s+\|\s+Mean Loss:\s*([\d.]+)")

    @staticmethod
    def parse_record_log(path: str) -> pd.DataFrame:
        """Parse a training record log into a DataFrame.

        Columns: epoch, train_loss, train_acc, test_loss, test_acc, test_robust_acc
        """
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = LogParser._RE_DATA_ROW.search(line)
                if m:
                    rows.append([float(g) for g in m.groups()])

        if not rows:
            raise ValueError(f"No valid data rows found in {path}")

        df = pd.DataFrame(rows, columns=[
            "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "test_robust_acc"
        ])
        df["epoch"] = df["epoch"].astype(int)
        return df

    @staticmethod
    def parse_test_log(path: str) -> Dict[str, float]:
        """Parse a test evaluation log.

        Returns dict with keys: normal_acc, pgd1, pgd20, pgd100, cw, aa.
        If the log contains multiple evaluation runs (restarts), the *last*
        complete block is returned. Works for both "Best_trained_model" and
        "Last_trained_model" headers.
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into blocks delimited by the performance header
        blocks = LogParser._RE_BEST.split(text)

        metric_sets = []
        for block in blocks[1:]:  # skip pre-first-header noise
            metrics = {}
            m = LogParser._RE_NORMAL.search(block)
            if m:
                metrics["normal_acc"] = float(m.group(1))

            for pm in LogParser._RE_PGD.finditer(block):
                nb_iter = int(pm.group(1))
                val = float(pm.group(2))
                if nb_iter == 1:
                    metrics["pgd1"] = val
                elif nb_iter == 20:
                    metrics["pgd20"] = val
                elif nb_iter == 100:
                    metrics["pgd100"] = val

            m = LogParser._RE_CW.search(block)
            if m:
                metrics["cw"] = float(m.group(1))

            m = LogParser._RE_AA.search(block)
            if m:
                metrics["aa"] = float(m.group(1))

            # Consider complete if at least normal_acc + pgd20 present
            if "normal_acc" in metrics and "pgd20" in metrics:
                metric_sets.append(metrics)

        if not metric_sets:
            raise ValueError(f"No valid metric blocks found in {path}")

        last = metric_sets[-1]
        for key in ["pgd1", "pgd20", "pgd100", "cw", "aa"]:
            last.setdefault(key, None)

        return last

    @staticmethod
    def parse_ood_log(path: str) -> Dict:
        """Parse an OOD (corruption robustness) evaluation log.

        Returns {"corruptions": {name: acc}, "mean_acc": float, "mean_loss": float}
        taking the LAST complete summary block if multiple runs exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into blocks on mean-acc summary lines; each block before a
        # mean line contains that run's per-corruption metrics.
        raw_blocks = LogParser._RE_OOD_MEAN.split(text)
        # raw_blocks layout: [pre, mean_acc1, mean_loss1, mid, mean_acc2, mean_loss2, ...]
        # Group into (body, mean_acc, mean_loss) triples
        bodies = []
        cur = []
        for i, chunk in enumerate(raw_blocks):
            if i % 3 == 0:
                # body chunk
                cur = []
                bodies.append({"body": chunk})
            elif i % 3 == 1:
                bodies[-1]["mean_acc"] = float(chunk)
            else:
                bodies[-1]["mean_loss"] = float(chunk)

        complete = [b for b in bodies if "mean_acc" in b]
        if not complete:
            raise ValueError(f"No complete summary found in {path}")

        last = complete[-1]
        corruptions = {}
        for m in LogParser._RE_OOD_LINE.finditer(last["body"]):
            corruptions[m.group(1)] = float(m.group(2))

        return {
            "corruptions": corruptions,
            "mean_acc": last["mean_acc"],
            "mean_loss": last["mean_loss"],
        }

# ==========================================================================
# 3. SCAN RESULT DIRECTORY
# ==========================================================================

def scan_result_dir(result_dir: str) -> List[Experiment]:
    """Scan the result directory (and dataset subdirs) building Experiment objects."""
    result_path = Path(result_dir)
    experiments = []

    for dataset, subdir in DATASET_DIRS.items():
        base = result_path / subdir if subdir else result_path
        if not base.exists():
            continue

        meta = EXPERIMENT_META.get(dataset, {})
        for stem, (display_name, group, variant) in meta.items():
            exp = Experiment(dataset=dataset, stem=stem, display_name=display_name,
                             group=group, variant=variant)

            record_path = base / f"{stem}_record.log"
            if record_path.exists():
                try:
                    exp.record_df = LogParser.parse_record_log(str(record_path))
                except Exception as e:
                    print(f"  [WARN] Failed to parse {record_path}: {e}")

            test_path = base / f"{stem}_test.log"
            if test_path.exists():
                try:
                    exp.test_metrics = LogParser.parse_test_log(str(test_path))
                    exp.has_test = True
                except Exception as e:
                    print(f"  [WARN] Failed to parse {test_path}: {e}")

            ood_path = base / f"{stem}_ood_test.log"
            if ood_path.exists():
                try:
                    exp.ood_metrics = LogParser.parse_ood_log(str(ood_path))
                    exp.has_ood = True
                except Exception as e:
                    print(f"  [WARN] Failed to parse {ood_path}: {e}")

            if exp.record_df is not None or exp.test_metrics is not None:
                experiments.append(exp)
            else:
                print(f"  [INFO] No data found for {stem} ({dataset}), skipping.")

    return experiments

# ==========================================================================
# 4. MARKDOWN REPORT GENERATOR
# ==========================================================================

def _fmt(val: Optional[float], decimals: int = 2) -> str:
    """Format a float or return 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"

def _best(experiments: List[Experiment], key: str, higher_better: bool = True) -> Tuple[Optional[Experiment], float]:
    """Find the experiment with the best value for a given test metric."""
    best_exp, best_val = None, -float("inf") if higher_better else float("inf")
    for exp in experiments:
        if exp.test_metrics and exp.test_metrics.get(key) is not None:
            v = exp.test_metrics[key]
            if (higher_better and v > best_val) or (not higher_better and v < best_val):
                best_val = v
                best_exp = exp
    return best_exp, best_val

def _overall_table(exps: List[Experiment]) -> List[str]:
    """Build the standard 6-metric summary table for a list of experiments."""
    lines = []
    lines.append("| Experiment | Clean Acc | PGD-1 | PGD-20 | PGD-100 | CW-20 | AutoAttack |")
    lines.append("|:-----------|:---------:|:-----:|:------:|:-------:|:-----:|:----------:|")
    for exp in exps:
        tm = exp.test_metrics
        if tm:
            lines.append(f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd1'))} | "
                         f"{_fmt(tm.get('pgd20'))} | {_fmt(tm.get('pgd100'))} | {_fmt(tm.get('cw'))} | "
                         f"{_fmt(tm.get('aa'))} |")
        else:
            lines.append(f"| {exp.display_name} | ⚠️ no test log | — | — | — | — | — |")
    return lines

def _method_comparison_table(exps: List[Experiment]) -> List[str]:
    """Table comparing methods (Clean, PGD-20, CW, AA, ΔClean−AA)."""
    lines = []
    lines.append("| Method | Clean Acc | PGD-20 | CW-20 | AutoAttack | Δ Clean−AA |")
    lines.append("|:-------|:---------:|:------:|:-----:|:----------:|:----------:|")
    # Sort: LFCM family first (variant "LFCM" or "baseline"), then HFDR, AT, Natural
    order = {"LFCM": 0, "baseline": 0, "HFDR": 1, "AT": 2, "Natural": 3}
    exps = sorted(exps, key=lambda e: order.get(e.variant, 99))
    for exp in exps:
        tm = exp.test_metrics
        if not tm:
            lines.append(f"| {exp.display_name} | ⚠️ no test | — | — | — | — |")
            continue
        if tm.get("normal_acc") is not None and tm.get("aa") is not None:
            delta = tm["normal_acc"] - tm["aa"]
            lines.append(f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
                         f"{_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | {delta:.2f} |")
        else:
            lines.append(f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
                         f"{_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | — |")
    return lines

def _convergence_table(exps: List[Experiment]) -> List[str]:
    """Table of best-epoch robust accuracy and final metrics."""
    lines = []
    lines.append("| Experiment | Best Robust Acc | At Epoch | Final Train Acc | Final Test Acc |")
    lines.append("|:-----------|:---------------:|:--------:|:---------------:|:--------------:|")
    for exp in exps:
        df = exp.record_df
        if df is not None and "test_robust_acc" in df.columns:
            best_idx = df["test_robust_acc"].idxmax()
            best_row = df.iloc[best_idx]
            last_row = df.iloc[-1]
            lines.append(f"| {exp.display_name} | {best_row['test_robust_acc']:.2f} | "
                         f"{int(best_row['epoch'])} | {last_row['train_acc']:.2f} | {last_row['test_acc']:.2f} |")
        else:
            lines.append(f"| {exp.display_name} | N/A | — | — | — |")
    return lines

def generate_markdown_report(experiments: List[Experiment], figures_dir: str = "figures") -> str:
    """Generate a comprehensive multi-dataset markdown report."""
    lines = []

    def w(s: str = ""):
        lines.append(s)

    def dataset_exps(ds: str) -> List[Experiment]:
        return [e for e in experiments if e.dataset == ds]

    def fig_link(ds: str, name: str) -> str:
        """Markdown image link; ds='' means the figures root (cross-dataset charts)."""
        if not ds:
            return f"![{name}]({figures_dir}/{name})"
        return f"![{name}]({figures_dir}/{ds}/{name})"

    w("# HFDR / LFCM Experiment Analysis Report")
    w()
    w(f"**Experiments analysed:** {len(experiments)} "
      f"across {len(set(e.dataset for e in experiments))} datasets "
      f"({', '.join(DATASET_DIRS.keys())})")
    w()

    # ══════════════════════════════════════════════════════════════════
    # CIFAR10
    # ══════════════════════════════════════════════════════════════════
    cifar10 = dataset_exps("CIFAR10")
    if cifar10:
        w("---")
        w("## 1. CIFAR10 (WRN-34-10 backbone)")
        w()

        w("### 1.1 Overall Results")
        w()
        for line in _overall_table(cifar10):
            w(line)
        w()

        # Ablation: canonicalization strength
        w("### 1.2 Ablation: Canonicalization Strength (K=64, varying `w_canon`)")
        w()
        canon_order = ["nocanon", "weak", "baseline", "strong"]
        canon_exps = [e for e in cifar10 if e.group == "canon"]
        canon_exps.sort(key=lambda e: canon_order.index(e.variant) if e.variant in canon_order else 99)
        if canon_exps:
            for line in _method_comparison_table(canon_exps):
                w(line)
        w()
        w(fig_link("CIFAR10", "comparison_canon.png"))
        w()

        # Ablation: codebook size
        w("### 1.3 Ablation: Codebook Size (fixed `w_canon=0.5`)")
        w()
        cb_order = ["K32", "K64", "K128"]
        cb_exps = [e for e in cifar10 if e.group == "codebook"]
        cb_exps.sort(key=lambda e: cb_order.index(e.variant) if e.variant in cb_order else 99)
        baseline_lfcm = [e for e in cifar10 if e.stem == "WRN34_10_LFCM"]
        for be in baseline_lfcm:
            if be not in cb_exps:
                cb_exps.insert(1, be)
        if cb_exps:
            for line in _method_comparison_table(cb_exps):
                w(line)
        w()
        w(fig_link("CIFAR10", "comparison_codebook.png"))
        w()

        # Method comparison (include LFCM baseline even though it is in the "canon" group)
        w("### 1.4 Method Comparison (AT / HFDR / LFCM / Natural)")
        w()
        method_exps = [e for e in cifar10 if e.group == "method"]
        lfcm_baseline = [e for e in cifar10 if e.stem == "WRN34_10_LFCM"]
        for line in _method_comparison_table(method_exps + lfcm_baseline):
            w(line)
        w()
        w(fig_link("CIFAR10", "comparison_method.png"))
        w()

        # Convergence
        w("### 1.5 Training Convergence")
        w()
        for line in _convergence_table(cifar10):
            w(line)
        w()

    # ══════════════════════════════════════════════════════════════════
    # CIFAR100
    # ══════════════════════════════════════════════════════════════════
    cifar100 = dataset_exps("CIFAR100")
    if cifar100:
        w("---")
        w("## 2. CIFAR100 (WRN-34-100 backbone)")
        w()
        w("> Note: test logs evaluate the **Last** trained checkpoint (`Last_trained_model`), "
          "not the best model.")
        w()

        w("### 2.1 Overall Results")
        w()
        for line in _overall_table(cifar100):
            w(line)
        w()

        w("### 2.2 Method Comparison")
        w()
        for line in _method_comparison_table([e for e in cifar100 if e.group == "method"]):
            w(line)
        w()
        w(fig_link("CIFAR100", "comparison_method.png"))
        w()

        # OOD analysis (method group only; codebook sweep covered in 2.4)
        ood_exps = [e for e in cifar100 if e.has_ood and e.group == "method"]
        if ood_exps:
            w("### 2.3 OOD Robustness (CIFAR-100-C, 19 corruptions)")
            w()
            w("| Method | Mean Acc | Mean Loss |")
            w("|:-------|:--------:|:---------:|")
            for exp in sorted(ood_exps, key=lambda e: -(e.ood_metrics["mean_acc"] if e.ood_metrics else 0)):
                if exp.ood_metrics:
                    w(f"| {exp.display_name} | {exp.ood_metrics['mean_acc']:.2f} | {exp.ood_metrics['mean_loss']:.4f} |")
            w()
            w("**Per-corruption accuracy (%)** — best method per corruption in bold:")
            w()
            w("| Corruption | LFCM | HFDR | AT | Natural |")
            w("|:-----------|:----:|:----:|:--:|:-------:|")
            all_corruptions = []
            for exp in ood_exps:
                if exp.ood_metrics:
                    all_corruptions.extend(exp.ood_metrics["corruptions"].keys())
            all_corruptions = sorted(set(all_corruptions))
            for corr in all_corruptions:
                vals = {}
                for exp in ood_exps:
                    vals[exp.variant] = exp.ood_metrics["corruptions"].get(corr)
                best_val = max((v for v in vals.values() if v is not None), default=None)
                cells = []
                for method in ["LFCM", "HFDR", "AT", "Natural"]:
                    v = vals.get(method)
                    if v is None:
                        cells.append("—")
                    elif best_val is not None and v == best_val:
                        cells.append(f"**{v:.2f}**")
                    else:
                        cells.append(f"{v:.2f}")
                w(f"| {corr} | {' | '.join(cells)} |")
            w()
            w(fig_link("CIFAR100", "ood_heatmap.png"))
            w(fig_link("CIFAR100", "ood_per_corruption.png"))
            w()

        # Codebook size ablation (K=64/128/256/512)
        cb_exps = [e for e in cifar100 if e.group == "codebook"]
        cb_base = [e for e in cifar100 if e.stem == LFCM_BASELINE_STEMS.get("CIFAR100", "")]
        for be in cb_base:
            if be not in cb_exps:
                be2 = copy.copy(be)
                be2.variant = "K64"
                cb_exps.append(be2)
        cb_order = ["K64", "K128", "K256", "K512"]
        cb_exps.sort(key=lambda e: cb_order.index(e.variant) if e.variant in cb_order else 99)

        if cb_exps:
            w("### 2.4 Codebook Size Ablation (K=64 / 128 / 256 / 512)")
            w()
            w("> Hypothesis: K=64 < 100 classes limits LFCM capacity. "
              "K=128/256/512 experiments were run to test whether a larger codebook "
              "helps on CIFAR100. Note: these runs have OOD + training logs but "
              "**no `_test.log`** (no PGD-20/CW/AA evaluation yet).")
            w()
            w("| K | Clean Acc (final) | Robust Acc (final) | Best Robust | OOD Mean Acc | Δ OOD vs K64 |")
            w("|:-:|:-----------------:|:------------------:|:-----------:|:------------:|:-------------:|")
            base_ood = None
            for exp in cb_exps:
                df = exp.record_df
                clean = robust = best_robust = None
                if df is not None and not df.empty:
                    clean = df.iloc[-1]["test_acc"]
                    robust = df.iloc[-1]["test_robust_acc"]
                    best_robust = df["test_robust_acc"].max()
                ood = exp.ood_metrics["mean_acc"] if exp.ood_metrics else None
                if exp.variant == "K64":
                    base_ood = ood
                delta = f"{ood - base_ood:+.2f}" if (ood is not None and base_ood is not None) else "—"
                w(f"| {exp.variant} | {_fmt(clean)} | {_fmt(robust)} | {_fmt(best_robust)} | "
                  f"{_fmt(ood)} | {delta} |")
            w()
            w("**Per-corruption accuracy (%)** — best K per corruption in bold:")
            w()
            w("| Corruption | K64 | K128 | K256 | K512 |")
            w("|:-----------|:---:|:----:|:----:|:----:|")
            all_corr = sorted({c for e in cb_exps for c in (e.ood_metrics["corruptions"] if e.ood_metrics else {}).keys()})
            for corr in all_corr:
                vals = {e.variant: e.ood_metrics["corruptions"].get(corr)
                        for e in cb_exps if e.ood_metrics}
                best_val = max((v for v in vals.values() if v is not None), default=None)
                cells = []
                for k in cb_order:
                    v = vals.get(k)
                    if v is None:
                        cells.append("—")
                    elif best_val is not None and v == best_val:
                        cells.append(f"**{v:.2f}**")
                    else:
                        cells.append(f"{v:.2f}")
                w(f"| {corr} | {' | '.join(cells)} |")
            w()
            w(fig_link("CIFAR100", "comparison_codebook.png"))
            w(fig_link("CIFAR100", "ood_codebook_sweep.png"))
            w()

    # ══════════════════════════════════════════════════════════════════
    # TinyImageNet
    # ══════════════════════════════════════════════════════════════════
    tiny = dataset_exps("TinyImageNet")
    if tiny:
        w("---")
        w("## 3. TinyImageNet (WRN-34-200 backbone)")
        w()

        w("### 3.1 Overall Results")
        w()
        for line in _overall_table(tiny):
            w(line)
        w()
        w("> Only LFCM has completed test evaluation. AT training stopped at epoch 94/110 "
          "(no LR decay phase), Natural trained 110 epochs but was not evaluated.")
        w()
        w("### 3.2 Training Comparison")
        w()
        for line in _convergence_table(tiny):
            w(line)
        w()
        w(fig_link("TinyImageNet", "comparison_method.png"))
        w()

    # ══════════════════════════════════════════════════════════════════
    # Cross-dataset summary
    # ══════════════════════════════════════════════════════════════════
    w("---")
    # ══════════════════════════════════════════════════════════════════
    # ResNet18
    # ══════════════════════════════════════════════════════════════════
    res18 = dataset_exps("ResNet18")
    if res18:
        w("---")
        w("## 4. ResNet18 (ResNet-18 backbone, CIFAR-10)")
        w()
        w("> Training records only — no `_test.log` (PGD/CW/AA) and no OOD evaluation yet. "
          "`Test Robust Acc` is the adversarial evaluation inside the training loop.")
        w()

        w("### 4.1 Overall (training-derived)")
        w()
        for line in _convergence_table(res18):
            w(line)
        w()

        w("### 4.2 Method Comparison (epochs ≥ 100 zoom, Natural excluded)")
        w()
        w(fig_link("ResNet18", "comparison_method.png"))
        w()

        # Backbone comparison: ResNet-18 vs WRN-34-10 on CIFAR-10
        w("### 4.3 Backbone Comparison: ResNet-18 vs WRN-34-10 (both on CIFAR-10)")
        w()
        w("> All metrics below come from the training loop (final-epoch clean/robust acc and "
          "best-epoch robust acc) so both backbones are measured identically. Dedicated test-log "
          "metrics for WRN-34-10 are in Section 1; ResNet-18 has no `_test.log` yet.")
        w()
        wrn_exps = dataset_exps("CIFAR10")

        def _find_exp(exps: List[Experiment], stem: str) -> Optional[Experiment]:
            for e in exps:
                if e.stem == stem:
                    return e
            return None

        def _train_metrics(exp: Optional[Experiment]) -> Tuple[Optional[float], Optional[float]]:
            """(final clean acc, best robust acc) from the training record."""
            if exp is None or exp.record_df is None or exp.record_df.empty:
                return None, None
            df = exp.record_df
            return df.iloc[-1]["test_acc"], df["test_robust_acc"].max()

        backbone_pairs = [
            ("LFCM",    "ResNet18_10_LFCM_K64", "WRN34_10_LFCM"),
            ("HFDR",    "ResNet18_10_HFDR",     "WRN34_10_F_HFDR"),
            ("AT",      "ResNet18_10_AT",       "WRN34_10_F"),
            ("Natural", "ResNet18_10_Natural",  "WRN34_10_F_Natural"),
        ]
        w("| Method | ResNet-18 Clean (final) | WRN-34 Clean (final) | Δ Clean | ResNet-18 Best Robust | WRN-34 Best Robust | Δ Best Robust |")
        w("|:-------|:-----------------------:|:--------------------:|:-------:|:---------------------:|:------------------:|:-------------:|")
        for method, r18_stem, wrn_stem in backbone_pairs:
            def _pair_cells(a: Optional[float], b: Optional[float]):
                """Bold the better value of the pair; Δ = ResNet-18 − WRN-34."""
                if a is None and b is None:
                    return "—", "—", "—"
                ca, cb = _fmt(a), _fmt(b)
                if a is not None and b is not None:
                    if a > b:
                        ca = f"**{a:.2f}**"
                    elif b > a:
                        cb = f"**{b:.2f}**"
                d = f"{a - b:+.2f}" if (a is not None and b is not None) else "—"
                return ca, cb, d
            r_clean, r_rob = _train_metrics(_find_exp(res18, r18_stem))
            w_clean, w_rob = _train_metrics(_find_exp(wrn_exps, wrn_stem))
            rc, wc, dc = _pair_cells(r_clean, w_clean)
            rr, wr, dr = _pair_cells(r_rob, w_rob)
            w(f"| {method} | {rc} | {wc} | {dc} | {rr} | {wr} | {dr} |")
        w()
        w(fig_link("ResNet18", "backbone_comparison.png"))
        w()

    # ══════════════════════════════════════════════════════════════════
    # Cross-dataset summary
    # ══════════════════════════════════════════════════════════════════
    w("---")
    w("## 5. Cross-Dataset Summary")
    w()
    w("| Dataset | Method | Clean Acc | PGD-20 | AutoAttack |")
    w("|:--------|:-------|:---------:|:------:|:----------:|")
    for ds in ["CIFAR10", "CIFAR100", "TinyImageNet", "ResNet18"]:
        for exp in dataset_exps(ds):
            tm = exp.test_metrics
            if not tm:
                continue
            variant = ("LFCM" if (exp.stem in ("WRN34_10_LFCM", "WRN34_100_LFCM", "WRN34_200_LFCM")
                                  and exp.variant == "baseline") else exp.variant)
            if variant not in ("AT", "HFDR", "LFCM", "Natural"):
                continue
            w(f"| {ds} | {variant} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
              f"{_fmt(tm.get('aa'))} |")
    w()
    w(fig_link("", "cross_dataset_summary.png"))
    w()

    # ══════════════════════════════════════════════════════════════════
    # Highlights
    # ══════════════════════════════════════════════════════════════════
    w("---")
    w("## 6. Highlights")
    w()

    for ds in ["CIFAR10", "CIFAR100", "TinyImageNet", "ResNet18"]:
        ds_exps = dataset_exps(ds)
        if not ds_exps:
            continue
        with_test = [e for e in ds_exps if e.has_test]
        if not with_test:
            # Training-only dataset (ResNet18): report training-derived highlights
            rec = [e for e in ds_exps if e.record_df is not None and not e.record_df.empty]
            if not rec:
                continue
            best_rob = max(rec, key=lambda e: e.record_df["test_robust_acc"].max())
            best_clean = max(rec, key=lambda e: e.record_df.iloc[-1]["test_acc"])
            w(f"**{ds} (training metrics):**")
            w()
            w(f"- Best Clean Accuracy (final): **{best_clean.display_name}** — "
              f"{best_clean.record_df.iloc[-1]['test_acc']:.2f}%")
            br_series = best_rob.record_df["test_robust_acc"]
            w(f"- Best Robust Acc (training): **{best_rob.display_name}** — "
              f"{br_series.max():.2f}% (epoch {int(best_rob.record_df.iloc[br_series.idxmax()]['epoch'])})")
            w()
            continue
        best_clean, v_clean = _best(with_test, "normal_acc")
        best_aa, v_aa = _best(with_test, "aa")
        w(f"**{ds}:**")
        w()
        w(f"- Best Clean Accuracy: **{best_clean.display_name}** — {v_clean:.2f}%")
        if best_aa is not None:
            w(f"- Best AutoAttack (most robust): **{best_aa.display_name}** — {v_aa:.2f}%")
        # OOD highlight (best mean acc among experiments with OOD data)
        ood_have = [e for e in ds_exps if e.has_ood and e.ood_metrics]
        if ood_have:
            best_ood = max(ood_have, key=lambda e: e.ood_metrics["mean_acc"])
            w(f"- Best OOD Mean Acc: **{best_ood.display_name}** — {best_ood.ood_metrics['mean_acc']:.2f}%")
        w()

    # ══════════════════════════════════════════════════════════════════
    # Missing data
    # ══════════════════════════════════════════════════════════════════
    missing = [e for e in experiments if not e.has_test]
    if missing:
        w("## 7. Missing Test Data")
        w()
        w("Experiments with training logs but **no test evaluation logs**:")
        w()
        for exp in missing:
            w(f"- **{exp.display_name}** (`{exp.stem}`, {exp.dataset})")
        w()

    missing_aa = [e for e in experiments if e.has_test and e.test_metrics and e.test_metrics.get("aa") is None]
    if missing_aa:
        w("### Missing AutoAttack Evaluation")
        w()
        for exp in missing_aa:
            w(f"- **{exp.display_name}** (`{exp.stem}`, {exp.dataset}) — no AA result in test log")
        w()

    w("---")
    w("*Report generated by `analysis.py`*")
    w()

    return "\n".join(lines)

# ==========================================================================
# 5. VISUALIZATION
# ==========================================================================

class Visualization:
    """Generate all figures, organized per dataset."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, ds: str, name: str):
        """Save figure into the dataset subdirectory."""
        ds_dir = self.output_dir / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        path = ds_dir / name
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

    def _save_root(self, name: str):
        path = self.output_dir / name
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

    def _color_for(self, exp: Experiment) -> Optional[str]:
        """Pick a colour for an experiment based on its group/variant."""
        if exp.group == "canon":
            return PALETTE_CANON.get(exp.variant)
        if exp.group == "codebook":
            return PALETTE_CODEBOOK.get(exp.variant)
        if exp.group == "method":
            return PALETTE_METHOD.get(exp.variant)
        return None

    # ── 5a. Per-Experiment Training Curves ───────────────────────────

    def plot_training_curves(self, exp: Experiment):
        """4-subplot training curve figure for a single experiment."""
        df = exp.record_df
        if df is None:
            print(f"  [SKIP] No record data for {exp.display_name}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Curves — {exp.display_name} ({exp.dataset})",
                     fontsize=15, fontweight="bold", y=0.98)

        epochs = df["epoch"].values

        axes[0, 0].plot(epochs, df["train_loss"], color="#e74c3c", linewidth=0.8, alpha=0.9)
        axes[0, 0].set_ylabel("Train Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_title("Training Loss")

        axes[0, 1].plot(epochs, df["train_acc"], color="#2ecc71", linewidth=0.8)
        axes[0, 1].set_ylabel("Train Accuracy (%)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_title("Training Accuracy")

        axes[1, 0].plot(epochs, df["test_loss"], color="#3498db", linewidth=0.8)
        axes[1, 0].set_ylabel("Test Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_title("Test Loss")

        axes[1, 1].plot(epochs, df["test_acc"], color="#2ecc71", linewidth=1.0, label="Clean Acc", alpha=0.85)
        axes[1, 1].plot(epochs, df["test_robust_acc"], color="#e74c3c", linewidth=1.0, label="Robust Acc (PGD-10)", alpha=0.85)
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_title("Test Accuracy: Clean vs Robust")
        axes[1, 1].legend(loc="lower right")

        for ax in axes.flat:
            if epochs.max() >= 100:
                ax.axvline(x=100, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
                ax.axvline(x=105, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

        if "LFCM" in exp.stem:
            for ax in axes.flat:
                ax.axvspan(1, 30, alpha=0.06, color="blue")
                ax.axvspan(31, 60, alpha=0.06, color="orange")
                ax.axvspan(61, 110, alpha=0.06, color="green")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(exp.dataset, f"{exp.stem}_training.png")

    # ── 5b. Cross-Experiment Comparison Curves (Zoomed: epoch ≥ 100) ──

    def plot_comparison_curves(self, experiments: List[Experiment], ds: str, group: str, group_label: str):
        """Plot Test Acc and Test Robust Acc for a group of experiments, zoomed to epochs 100+."""
        group_exps = [e for e in experiments if e.dataset == ds and e.group == group and e.record_df is not None]
        if not group_exps:
            print(f"  [SKIP] No experiments in group '{group}' for {ds}")
            return

        if group == "codebook":
            base_stem = LFCM_BASELINE_STEMS.get(ds, "")
            baseline = [e for e in experiments if e.dataset == ds and e.stem == base_stem
                        and e.record_df is not None]
            for be in baseline:
                if be not in group_exps:
                    be2 = copy.copy(be)
                    be2.variant = "K64"
                    group_exps.append(be2)

        if group == "method":
            group_exps = [e for e in group_exps if "Natural" not in e.display_name]
            # Add LFCM baseline (stored in another group) into the method comparison
            lfcm_base = [e for e in experiments if e.dataset == ds
                         and e.stem in ("WRN34_10_LFCM", "WRN34_100_LFCM", "WRN34_200_LFCM")
                         and e.record_df is not None]
            for be in lfcm_base:
                if be not in group_exps:
                    be2 = copy.copy(be)
                    be2.variant = "LFCM"
                    group_exps.append(be2)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Comparison: {group_label} ({ds}) — Epochs 100–110, LR-decay zoom",
                     fontsize=15, fontweight="bold")

        all_test_acc = []
        all_robust_acc = []

        for exp in group_exps:
            df = exp.record_df
            zoom = df[df["epoch"] >= 100]
            if zoom.empty:
                continue
            color = self._color_for(exp) or "#7f8c8d"
            label = exp.display_name

            axes[0].plot(zoom["epoch"], zoom["test_acc"], "o-", linewidth=1.5, color=color,
                         label=label, alpha=0.9, markersize=4)
            axes[1].plot(zoom["epoch"], zoom["test_robust_acc"], "o-", linewidth=1.5, color=color,
                         label=label, alpha=0.9, markersize=4)
            all_test_acc.extend(zoom["test_acc"].tolist())
            all_robust_acc.extend(zoom["test_robust_acc"].tolist())

        if all_test_acc:
            ta_min, ta_max = min(all_test_acc), max(all_test_acc)
            ta_pad = max((ta_max - ta_min) * 0.5, 0.5)
            axes[0].set_ylim(ta_min - ta_pad, ta_max + ta_pad)

        if all_robust_acc:
            ra_min, ra_max = min(all_robust_acc), max(all_robust_acc)
            ra_pad = max((ra_max - ra_min) * 0.6, 0.3)
            axes[1].set_ylim(ra_min - ra_pad, ra_max + ra_pad)

        axes[0].set_xlim(99.5, max(e.record_df["epoch"].max() for e in group_exps) + 0.5)
        axes[1].set_xlim(99.5, max(e.record_df["epoch"].max() for e in group_exps) + 0.5)

        for ax in axes:
            ax.axvline(x=100, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label="LR ÷10")
            ax.axvline(x=105, color="red", linestyle=":", linewidth=1.0, alpha=0.5, label="LR ÷10 (2nd)")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))

        axes[0].set_ylabel("Test Accuracy (%)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Clean Test Accuracy  (zoomed LR-decay region)")
        axes[0].legend(loc="best", fontsize=8, ncol=2)

        axes[1].set_ylabel("Robust Accuracy (PGD-10, %)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Robust Test Accuracy  (zoomed LR-decay region)")
        axes[1].legend(loc="best", fontsize=8, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save(ds, f"comparison_{group}.png")

    # ── 5c. Final Test Metrics Bar Chart ─────────────────────────────

    def plot_test_metrics_bars(self, experiments: List[Experiment], ds: str):
        """Grouped bar chart of final test metrics."""
        exps_with_test = [e for e in experiments if e.dataset == ds and e.has_test and e.test_metrics]
        if not exps_with_test:
            print(f"  [SKIP] No experiments with test data for {ds}")
            return

        metrics = ["normal_acc", "pgd20", "cw", "aa"]
        metric_labels = ["Clean", "PGD-20", "CW-20", "AA"]
        colors = [PALETTE_METRIC[ml] for ml in metric_labels]

        x = range(len(exps_with_test))
        width = 0.18
        n_metrics = len(metrics)

        fig, ax = plt.subplots(figsize=(16, 7))

        for i, (metric_key, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            values = []
            for exp in exps_with_test:
                v = exp.test_metrics.get(metric_key)
                values.append(v if v is not None else 0)
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], values, width, label=label, color=color,
                          alpha=0.9, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_xticks(list(x))
        ax.set_xticklabels([e.display_name for e in exps_with_test], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Test Metrics Comparison ({ds})", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        self._save(ds, "test_metrics_bars.png")

    # ── 5d. Clean vs Robust Scatter ──────────────────────────────────

    def plot_clean_vs_robust_scatter(self, experiments: List[Experiment], ds: str):
        """Scatter plot: Clean Accuracy vs AutoAttack Accuracy."""
        exps_with_test = [e for e in experiments if e.dataset == ds and e.has_test and e.test_metrics
                          and e.test_metrics.get("normal_acc") and e.test_metrics.get("aa")]
        if not exps_with_test:
            print(f"  [SKIP] No experiments with clean+AA data for {ds}")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        for exp in exps_with_test:
            x = exp.test_metrics["normal_acc"]
            y = exp.test_metrics["aa"]
            color = self._color_for(exp) or "#7f8c8d"

            ax.scatter(x, y, c=color, s=120, edgecolors="white", linewidth=1.2, zorder=5)
            ax.annotate(exp.display_name, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, alpha=0.9)

        ax.set_xlabel("Clean Accuracy (%)")
        ax.set_ylabel("AutoAttack Accuracy (%)")
        ax.set_title(f"Clean vs Robust Accuracy Trade-off ({ds})", fontweight="bold")

        xs = [e.test_metrics["normal_acc"] for e in exps_with_test]
        ys = [e.test_metrics["aa"] for e in exps_with_test]
        if xs and ys:
            med_x = sorted(xs)[len(xs) // 2]
            med_y = sorted(ys)[len(ys) // 2]
            ax.axhline(y=med_y, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.axvline(x=med_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            x_margin, y_margin = 1.0, 1.0
            ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
            ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

        plt.tight_layout()
        self._save(ds, "clean_vs_robust_scatter.png")

    # ── 5e. Radar Chart (tight scale) ──────────────────────────────

    def plot_radar_comparison(self, experiments: List[Experiment], ds: str, group: str, group_label: str):
        """Radar/spider chart with tight axis scale to highlight small differences."""
        group_exps = [e for e in experiments if e.dataset == ds and e.group == group
                      and e.has_test and e.test_metrics]
        if group == "codebook":
            base_stem = LFCM_BASELINE_STEMS.get(ds, "")
            baseline = [e for e in experiments if e.dataset == ds and e.stem == base_stem
                        and e.has_test]
            for be in baseline:
                if be not in group_exps:
                    be2 = copy.copy(be)
                    be2.variant = "K64"
                    group_exps.append(be2)

        if group == "method":
            group_exps = [e for e in group_exps if "Natural" not in e.display_name]
            lfcm_base = [e for e in experiments if e.dataset == ds
                         and e.stem in ("WRN34_10_LFCM", "WRN34_100_LFCM", "WRN34_200_LFCM")
                         and e.has_test]
            for be in lfcm_base:
                if be not in group_exps:
                    be2 = copy.copy(be)
                    be2.variant = "LFCM"
                    group_exps.append(be2)

        if len(group_exps) < 2:
            print(f"  [SKIP] Not enough experiments in group '{group}' for {ds} radar chart (need ≥2)")
            return

        metric_keys = ["normal_acc", "pgd1", "pgd20", "pgd100", "cw", "aa"]
        metric_labels = ["Clean", "PGD-1", "PGD-20", "PGD-100", "CW-20", "AA"]
        n_metrics = len(metric_keys)

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the circle

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.set_title(f"Multi-Metric Radar — {group_label} ({ds}, tight scale)",
                     fontsize=14, fontweight="bold", pad=25)

        all_values = []
        for exp in group_exps:
            vals = [exp.test_metrics.get(k) for k in metric_keys]
            vals = [v for v in vals if v is not None]
            all_values.extend(vals)

        if all_values:
            data_min, data_max = min(all_values), max(all_values)
            y_bottom = max(0, data_min - 2)
            y_top = min(100, data_max + 2)
            ax.set_ylim(y_bottom, y_top)
            tick_start = int(np.ceil(y_bottom / 5) * 5)
            tick_end = int(np.floor(y_top / 5) * 5)
            yticks = list(range(tick_start, tick_end + 1, 5))
            if len(yticks) > 12:
                yticks = list(range(tick_start, tick_end + 1, 10))
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(t) for t in yticks], fontsize=8, color="gray")
            ax.set_rlabel_position(30)

        for exp in group_exps:
            values = [exp.test_metrics.get(k) or 0 for k in metric_keys]
            values += values[:1]
            color = self._color_for(exp) or "#7f8c8d"
            ax.fill(angles, values, alpha=0.08, color=color)
            ax.plot(angles, values, "o-", linewidth=2.0, color=color, label=exp.display_name, markersize=6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

        plt.tight_layout()
        self._save(ds, f"radar_{group}.png")

    # ── 5f. Bonus: Attack Degradation Curves ─────────────────────────

    def plot_attack_degradation(self, experiments: List[Experiment], ds: str):
        """Line plot: accuracy degradation as attack strength increases."""
        exps_with_test = [e for e in experiments if e.dataset == ds and e.has_test and e.test_metrics]
        if not exps_with_test:
            return

        attack_order = ["pgd1", "pgd20", "pgd100", "cw", "aa"]
        attack_labels = ["PGD-1", "PGD-20", "PGD-100", "CW-20", "AA"]
        x = range(len(attack_order))

        fig, ax = plt.subplots(figsize=(12, 7))

        for exp in exps_with_test:
            values = [exp.test_metrics.get(k) for k in attack_order]
            valid = [v for v in values if v is not None]
            if len(valid) < 3:
                continue
            color = self._color_for(exp) or "#7f8c8d"
            ax.plot(x, values, "o-", linewidth=1.5, color=color, label=exp.display_name, markersize=6)

        ax.set_xticks(list(x))
        ax.set_xticklabels(attack_labels, fontsize=11)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Attack Strength Degradation Curves ({ds})", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()
        self._save(ds, "attack_degradation.png")

    # ── 5g. Bonus: LR Decay Zoom ─────────────────────────────────────

    def plot_lr_decay_zoom(self, experiments: List[Experiment], ds: str):
        """Zoomed view of epochs 90+ to visualize LR decay impact."""
        ad_exps = [e for e in experiments if e.dataset == ds and e.record_df is not None
                   and e.group in ("canon", "codebook", "method", "schedule")
                   and "Natural" not in e.display_name]
        if not ad_exps:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"LR Decay Impact (Epochs 90+, {ds})", fontsize=14, fontweight="bold")

        for exp in ad_exps:
            df = exp.record_df
            zoom = df[df["epoch"] >= 90]
            if zoom.empty:
                continue
            color = self._color_for(exp)
            axes[0].plot(zoom["epoch"], zoom["test_acc"], linewidth=1.2, color=color,
                         label=exp.display_name, alpha=0.85)
            axes[1].plot(zoom["epoch"], zoom["test_robust_acc"], linewidth=1.2, color=color,
                         label=exp.display_name, alpha=0.85)

        for ax in axes:
            ax.axvline(x=100, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="LR ÷10")
            ax.axvline(x=105, color="red", linestyle=":", linewidth=1.0, alpha=0.6, label="LR ÷10 (2nd)")

        axes[0].set_ylabel("Test Accuracy (%)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Clean Test Accuracy")
        axes[0].legend(loc="lower right", fontsize=7)

        axes[1].set_ylabel("Robust Accuracy (PGD-10, %)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Robust Test Accuracy")
        axes[1].legend(loc="lower right", fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save(ds, "lr_decay_zoom.png")

    # ── 5h. Bonus: Robustness Stability ──────────────────────────────

    def plot_robustness_stability(self, experiments: List[Experiment], ds: str):
        """Rolling std of robust accuracy in the last 30 epochs."""
        ad_exps = [e for e in experiments if e.dataset == ds and e.record_df is not None
                   and e.group in ("canon", "codebook", "method")]

        fig, ax = plt.subplots(figsize=(12, 6))

        stability_data = []
        for exp in ad_exps:
            df = exp.record_df
            last30 = df[df["epoch"] >= df["epoch"].max() - 29]
            if len(last30) < 10:
                continue
            mean_ra = last30["test_robust_acc"].mean()
            std_ra = last30["test_robust_acc"].std()
            max_ra = last30["test_robust_acc"].max()
            stability_data.append((exp.display_name, mean_ra, std_ra, max_ra, self._color_for(exp) or "#7f8c8d"))

        if not stability_data:
            return

        stability_data.sort(key=lambda x: x[3], reverse=True)

        names = [d[0] for d in stability_data]
        means = [d[1] for d in stability_data]
        stds = [d[2] for d in stability_data]
        colors = [d[4] for d in stability_data]

        y_pos = range(len(names))
        ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, edgecolor="white",
                capsize=3, error_kw={"linewidth": 1.2})

        for i, (name, mean, std, max_val, _) in enumerate(stability_data):
            ax.text(mean + std + 0.3, i, f"max={max_val:.1f}", va="center", fontsize=8, alpha=0.8)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Robust Accuracy (PGD-10, %) — mean ± std (last 30 epochs)")
        ax.set_title(f"Training Stability: Late-Stage Robust Accuracy ({ds})", fontweight="bold")
        ax.invert_yaxis()

        plt.tight_layout()
        self._save(ds, "robustness_stability.png")

    # ── 5i. NEW: OOD Heatmap ─────────────────────────────────────────

    def plot_ood_heatmap(self, experiments: List[Experiment], ds: str):
        """Heatmap of per-corruption accuracy across methods."""
        ood_exps = [e for e in experiments if e.dataset == ds and e.has_ood
                    and e.group == "method"]
        if not ood_exps:
            print(f"  [SKIP] No OOD data for {ds}")
            return

        methods = ["LFCM", "HFDR", "AT", "Natural"]
        ood_exps = sorted(ood_exps, key=lambda e: methods.index(e.variant) if e.variant in methods else 99)

        all_corruptions = sorted({c for e in ood_exps for c in (e.ood_metrics["corruptions"] if e.ood_metrics else {}).keys()})

        data = np.zeros((len(all_corruptions), len(ood_exps)))
        for j, exp in enumerate(ood_exps):
            for i, corr in enumerate(all_corruptions):
                data[i, j] = exp.ood_metrics["corruptions"].get(corr, np.nan)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                    xticklabels=[e.display_name for e in ood_exps],
                    yticklabels=all_corruptions,
                    cbar_kws={"label": "Accuracy (%)"}, ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"OOD Robustness per Corruption ({ds})", fontweight="bold")
        ax.set_xlabel("Method")
        ax.set_ylabel("Corruption")

        plt.tight_layout()
        self._save(ds, "ood_heatmap.png")

    # ── 5j. NEW: OOD Per-Corruption Grouped Bars ─────────────────────

    def plot_ood_per_corruption(self, experiments: List[Experiment], ds: str):
        """Grouped bar chart: each corruption, bars per method."""
        ood_exps = [e for e in experiments if e.dataset == ds and e.has_ood
                    and e.group == "method"]
        if not ood_exps:
            return

        methods = ["LFCM", "HFDR", "AT", "Natural"]
        ood_exps = sorted(ood_exps, key=lambda e: methods.index(e.variant) if e.variant in methods else 99)

        all_corruptions = sorted({c for e in ood_exps for c in (e.ood_metrics["corruptions"] if e.ood_metrics else {}).keys()})

        fig, ax = plt.subplots(figsize=(18, 8))
        x = np.arange(len(all_corruptions))
        width = 0.2

        for j, exp in enumerate(ood_exps):
            vals = [exp.ood_metrics["corruptions"].get(c, 0) for c in all_corruptions]
            color = self._color_for(exp) or "#7f8c8d"
            ax.bar(x + (j - len(ood_exps) / 2 + 0.5) * width, vals, width,
                   label=exp.display_name, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(all_corruptions, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"OOD Accuracy by Corruption ({ds})", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        self._save(ds, "ood_per_corruption.png")

    # ── 5j2. OOD Codebook Sweep (K=64/128/256/512) ───────────────────

    def plot_ood_codebook_sweep(self, experiments: List[Experiment], ds: str):
        """Two-panel chart for codebook size sweep:
        1) OOD mean accuracy vs K;  2) per-corruption heatmap across K values.
        """
        base_stem = LFCM_BASELINE_STEMS.get(ds, "")
        cb_exps = [e for e in experiments if e.dataset == ds and e.group == "codebook"
                   and e.has_ood]
        base_exps = [e for e in experiments if e.dataset == ds and e.stem == base_stem
                     and e.has_ood]
        for be in base_exps:
            if be not in cb_exps:
                be2 = copy.copy(be)
                be2.variant = "K64"
                cb_exps.append(be2)
        if len(cb_exps) < 2:
            print(f"  [SKIP] Not enough codebook OOD experiments for {ds}")
            return

        cb_order = ["K64", "K128", "K256", "K512"]
        cb_exps.sort(key=lambda e: cb_order.index(e.variant) if e.variant in cb_order else 99)

        ks = [e.variant for e in cb_exps]
        means = [e.ood_metrics["mean_acc"] for e in cb_exps]

        fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                                 gridspec_kw={"width_ratios": [1, 2]})
        fig.suptitle(f"LFCM Codebook Size Sweep — OOD Robustness ({ds})",
                     fontsize=15, fontweight="bold")

        # Panel 1: mean OOD acc vs K
        ax = axes[0]
        colors = [PALETTE_CODEBOOK.get(k, "#7f8c8d") for k in ks]
        x = np.arange(len(ks))
        ax.plot(x, means, "-", color="#2c3e50", linewidth=2.0, alpha=0.7)
        ax.scatter(x, means, s=120, c=colors, edgecolors="white", linewidth=1.2, zorder=5)
        for xi, (m, k) in enumerate(zip(means, ks)):
            ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ks, fontsize=11)
        ax.set_ylabel("OOD Mean Accuracy (%)")
        ax.set_title("Mean OOD Accuracy vs Codebook Size")
        # Reference line: K64 baseline
        if means:
            ax.axhline(means[0], color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        y_lo, y_hi = min(means) - 0.5, max(means) + 0.5
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

        # Panel 2: per-corruption heatmap across K values
        ax2 = axes[1]
        all_corr = sorted({c for e in cb_exps for c in e.ood_metrics["corruptions"].keys()})
        data = np.zeros((len(all_corr), len(cb_exps)))
        for j, exp in enumerate(cb_exps):
            for i, corr in enumerate(all_corr):
                data[i, j] = exp.ood_metrics["corruptions"].get(corr, np.nan)

        sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn", vmin=20, vmax=80,
                    xticklabels=[e.variant for e in cb_exps],
                    yticklabels=all_corr,
                    cbar_kws={"label": "Accuracy (%)"}, ax=ax2,
                    linewidths=0.5, linecolor="white")
        ax2.set_title("Per-Corruption Accuracy by Codebook Size")
        ax2.set_xlabel("Codebook Size K")
        ax2.set_ylabel("Corruption")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save(ds, "ood_codebook_sweep.png")

    # ── 5k. NEW: Cross-Dataset Summary ───────────────────────────────

    def plot_cross_dataset_summary(self, experiments: List[Experiment]):
        """Grouped bar: Clean / AA accuracy for each method across datasets."""
        datasets = [ds for ds in DATASET_DIRS.keys() if any(e.dataset == ds and e.has_test for e in experiments)]
        methods = ["LFCM", "HFDR", "AT", "Natural"]

        # Collect (dataset, method) -> (clean, aa) with availability
        data = {}
        for ds in datasets:
            for exp in experiments:
                if exp.dataset != ds or exp.test_metrics is None:
                    continue
                # CIFAR10 LFCM baseline lives in the "canon" group with variant "baseline"
                variant = "LFCM" if (exp.stem in ("WRN34_10_LFCM", "WRN34_100_LFCM", "WRN34_200_LFCM")
                                     and exp.variant == "baseline") else exp.variant
                if variant in methods:
                    data.setdefault((ds, variant), exp.test_metrics)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Cross-Dataset Summary: Method Performance", fontsize=15, fontweight="bold")

        for ax, (key, title) in zip(axes, [("normal_acc", "Clean Accuracy (%)"), ("aa", "AutoAttack Accuracy (%)")]):
            x = np.arange(len(datasets))
            width = 0.2
            for i, method in enumerate(methods):
                vals = []
                for ds in datasets:
                    tm = data.get((ds, method), {})
                    v = tm.get(key)
                    vals.append(v if v is not None else np.nan)
                color = PALETTE_METHOD.get(method, "#7f8c8d")
                ax.bar(x + (i - len(methods) / 2 + 0.5) * width, vals, width,
                       label=method, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(datasets, fontsize=10)
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save_root("cross_dataset_summary.png")

    # ── 5l. NEW: Backbone Comparison (ResNet-18 vs WRN-34-10 on CIFAR-10) ──

    def plot_backbone_comparison(self, experiments: List[Experiment]):
        """Training curves of ResNet-18 vs WRN-34-10 for each method on CIFAR-10."""
        def get(ds: str, stem: str) -> Optional[Experiment]:
            for e in experiments:
                if e.dataset == ds and e.stem == stem and e.record_df is not None:
                    return e
            return None

        pairs = [
            ("LFCM",    "WRN34_10_LFCM",      "ResNet18_10_LFCM_K64"),
            ("HFDR",    "WRN34_10_F_HFDR",    "ResNet18_10_HFDR"),
            ("AT",      "WRN34_10_F",         "ResNet18_10_AT"),
            ("Natural", "WRN34_10_F_Natural", "ResNet18_10_Natural"),
        ]
        pairs = [p for p in pairs if get("CIFAR10", p[1]) is not None and get("ResNet18", p[2]) is not None]
        if not pairs:
            print("  [SKIP] Backbone comparison: no paired CIFAR10/ResNet18 experiments")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        fig.suptitle("Backbone Comparison on CIFAR-10: WRN-34-10 vs ResNet-18 (training metrics)",
                     fontsize=15, fontweight="bold")

        for ax, metric, mlabel in [(axes[0], "test_acc", "Clean Test Acc (%)"),
                                   (axes[1], "test_robust_acc", "Robust Test Acc (PGD-10, %)")]:
            for method, wrn_stem, r18_stem in pairs:
                wrn = get("CIFAR10", wrn_stem)
                r18 = get("ResNet18", r18_stem)
                color = PALETTE_METHOD.get(method, "#7f8c8d")
                ax.plot(wrn.record_df["epoch"], wrn.record_df[metric], "-", color=color,
                        linewidth=1.4, alpha=0.85, label=f"{method} (WRN-34-10)")
                ax.plot(r18.record_df["epoch"], r18.record_df[metric], "--", color=color,
                        linewidth=1.4, alpha=0.85, label=f"{method} (ResNet-18)")
            ax.axvline(x=100, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(mlabel)
            ax.legend(loc="lower right", fontsize=8, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save("ResNet18", "backbone_comparison.png")

# ==========================================================================
# 6. MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze HFDR/LFCM experiment logs")
    parser.add_argument("--result-dir", default="./result", help="Path to result directory (default: ./result)")
    parser.add_argument("--output-dir", default="./result/figures", help="Path for output figures (default: ./result/figures)")
    parser.add_argument("--report", default="./result/LFCM_analysis_report.md", help="Path for output markdown report")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation (report only)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"[ERROR] Result directory not found: {result_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  HFDR / LFCM Experiment Log Analyzer (multi-dataset)")
    print("=" * 60)
    print()

    # ── Scan ─────────────────────────────────────────────────────────
    print("[1/4] Scanning experiment logs...")
    experiments = scan_result_dir(str(result_dir))
    by_ds = defaultdict(list)
    for e in experiments:
        by_ds[e.dataset].append(e)
    for ds, exps in by_ds.items():
        n_test = sum(1 for e in exps if e.has_test)
        n_ood = sum(1 for e in exps if e.has_ood)
        print(f"  {ds:12s}: {len(exps)} experiments ({n_test} with test, {n_ood} with OOD)")
    print()

    # ── Report ───────────────────────────────────────────────────────
    print("[2/4] Generating markdown report...")
    report = generate_markdown_report(experiments, figures_dir=Path(args.output_dir).name)
    report_path = Path(args.report)
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved: {report_path}")
    print()

    # ── Visualizations ───────────────────────────────────────────────
    if args.skip_plots:
        print("[3/4] Skipping figure generation (--skip-plots)")
    else:
        print("[3/4] Generating figures...")
        viz = Visualization(args.output_dir)

        # 1. Individual training curves (all datasets)
        print("  ── Individual Training Curves ──")
        for exp in experiments:
            if exp.record_df is not None:
                viz.plot_training_curves(exp)

        # 2. Per-dataset comparison curves
        print("  ── Comparison Curves ──")
        for ds in DATASET_DIRS.keys():
            viz.plot_comparison_curves(experiments, ds, "canon", "Canonicalization Strength Ablation")
            viz.plot_comparison_curves(experiments, ds, "codebook", "Codebook Size Ablation")
            viz.plot_comparison_curves(experiments, ds, "method", "Training Method Comparison")

        # 3. Bar charts + scatter per dataset
        print("  ── Test Metrics Bar Charts ──")
        for ds in DATASET_DIRS.keys():
            viz.plot_test_metrics_bars(experiments, ds)

        print("  ── Clean vs Robust Scatter ──")
        for ds in DATASET_DIRS.keys():
            viz.plot_clean_vs_robust_scatter(experiments, ds)

        # 4. Radar charts per dataset
        print("  ── Radar Charts ──")
        for ds in DATASET_DIRS.keys():
            viz.plot_radar_comparison(experiments, ds, "canon", "Canonicalization Strength")
            viz.plot_radar_comparison(experiments, ds, "codebook", "Codebook Size")
            viz.plot_radar_comparison(experiments, ds, "method", "Training Method")

        # 5. Bonus charts per dataset
        print("  ── Bonus Charts ──")
        for ds in DATASET_DIRS.keys():
            viz.plot_attack_degradation(experiments, ds)
            viz.plot_lr_decay_zoom(experiments, ds)
            viz.plot_robustness_stability(experiments, ds)

        # 6. OOD charts (CIFAR100)
        print("  ── OOD Charts ──")
        for ds in DATASET_DIRS.keys():
            viz.plot_ood_heatmap(experiments, ds)
            viz.plot_ood_per_corruption(experiments, ds)
            viz.plot_ood_codebook_sweep(experiments, ds)

        # 7. Cross-dataset summary
        print("  ── Cross-Dataset Summary ──")
        viz.plot_cross_dataset_summary(experiments)

        # 8. Backbone comparison (ResNet-18 vs WRN-34-10 on CIFAR-10)
        print("  ── Backbone Comparison ──")
        viz.plot_backbone_comparison(experiments)

        print()

    # ── Summary ──────────────────────────────────────────────────────
    print("[4/4] Done!")
    print(f"  Report: {args.report}")
    if not args.skip_plots:
        print(f"  Figures: {args.output_dir}/")
        n_figs = sum(len(list(p.glob("*.png"))) for p in [Path(args.output_dir)]
                     + [Path(args.output_dir) / ds for ds in DATASET_DIRS.values() if ds])
        print(f"  Generated {n_figs} figure(s)")


if __name__ == "__main__":
    main()
