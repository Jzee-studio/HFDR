#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis & Visualization Script for HFDR / LFCM Experiment Logs
================================================================
Reads training record logs and test evaluation logs from the `result/` directory,
generates a comprehensive markdown report and publication-quality figures.

Usage:
    python analysis.py [--result-dir ./result] [--output-dir ./result/figures]

Dependencies: pandas, matplotlib, seaborn (all in environment.yaml)
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
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
PALETTE_CODEBOOK = {"K32": "#9b59b6", "K64": "#2ecc71", "K128": "#e67e22"}
PALETTE_METHOD = {"AT": "#95a5a6", "HFDR": "#3498db", "LFCM": "#2ecc71", "Natural": "#e74c3c"}
PALETTE_METRIC = {"Clean": "#2ecc71", "PGD-1": "#3498db", "PGD-20": "#f39c12", "PGD-100": "#e67e22", "CW-20": "#9b59b6", "AA": "#e74c3c"}

STAGE_COLORS = {"Stage1": "#ebf5fb", "Stage2": "#fef9e7", "Stage3": "#eafaf1"}

# ==========================================================================
# 1. LOG PARSER
# ==========================================================================

class LogParser:
    """Parse HFDR training record logs and test evaluation logs."""

    # Regex for data rows: after "] - " the epoch number, then whitespace-separated floats
    _RE_DATA_ROW = re.compile(r"\]\s*-\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")

    # Regex for test log metrics
    _RE_NORMAL = re.compile(r"Normal Acc:\s*([\d.]+)")
    _RE_PGD = re.compile(r"PGD_attack:\[nb_iter:(\d+).*?\]->pgd_acc:\s*([\d.]+)")
    _RE_CW = re.compile(r"CW_attack:.*?->CW_acc:\s*([\d.]+)")
    _RE_AA = re.compile(r"Auto_attack:.*?->AA_acc:\s*([\d.]+)")
    _RE_BEST = re.compile(r"=======Best_trained_model Performance=======")

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
        complete block is returned.
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into blocks delimited by "Best_trained_model Performance"
        blocks = LogParser._RE_BEST.split(text)
        # The first element is everything before the first "Best" header (header noise)
        # Each subsequent element starts with the metrics for one evaluation run

        # Collect all metric sets
        metric_sets = []
        for block in blocks[1:]:  # skip pre-first-header noise
            metrics = {}
            m = LogParser._RE_NORMAL.search(block)
            if m:
                metrics["normal_acc"] = float(m.group(1))

            # PGD lines with nb_iter
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

            # Consider complete if at least normal_acc + pgd20 + aa present
            if "normal_acc" in metrics and "pgd20" in metrics:
                metric_sets.append(metrics)

        if not metric_sets:
            raise ValueError(f"No valid metric blocks found in {path}")

        # Return the LAST complete set (latest run)
        last = metric_sets[-1]

        # Fill missing optional keys with None
        for key in ["pgd1", "pgd20", "pgd100", "cw", "aa"]:
            last.setdefault(key, None)

        return last

# ==========================================================================
# 2. EXPERIMENT METADATA
# ==========================================================================

# Experiment definitions: filename_stem -> (display_name, group, variant_label)
EXPERIMENT_META = {
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
}

@dataclass
class Experiment:
    """Holds all parsed data for one experiment."""
    stem: str
    display_name: str
    group: str
    variant: str
    record_df: Optional[pd.DataFrame] = None
    test_metrics: Optional[Dict[str, float]] = None
    has_test: bool = False

def scan_result_dir(result_dir: str) -> List[Experiment]:
    """Scan the result directory and build Experiment objects."""
    result_path = Path(result_dir)
    experiments = []

    for stem, (display_name, group, variant) in EXPERIMENT_META.items():
        exp = Experiment(stem=stem, display_name=display_name, group=group, variant=variant)

        # Try to load record log
        record_path = result_path / f"{stem}_record.log"
        if record_path.exists():
            try:
                exp.record_df = LogParser.parse_record_log(str(record_path))
            except Exception as e:
                print(f"  [WARN] Failed to parse {record_path}: {e}")

        # Try to load test log
        test_path = result_path / f"{stem}_test.log"
        if test_path.exists():
            try:
                exp.test_metrics = LogParser.parse_test_log(str(test_path))
                exp.has_test = True
            except Exception as e:
                print(f"  [WARN] Failed to parse {test_path}: {e}")

        # Only include if we have at least a record log
        if exp.record_df is not None or exp.test_metrics is not None:
            experiments.append(exp)
        else:
            print(f"  [INFO] No data found for {stem}, skipping.")

    return experiments

# ==========================================================================
# 3. MARKDOWN REPORT GENERATOR
# ==========================================================================

def _fmt(val: Optional[float], decimals: int = 2) -> str:
    """Format a float or return 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"

def _best(experiments: List[Experiment], key: str, higher_better: bool = True) -> Tuple[Experiment, float]:
    """Find the experiment with the best value for a given test metric."""
    best_exp, best_val = None, -float("inf") if higher_better else float("inf")
    for exp in experiments:
        if exp.test_metrics and exp.test_metrics.get(key) is not None:
            v = exp.test_metrics[key]
            if (higher_better and v > best_val) or (not higher_better and v < best_val):
                best_val = v
                best_exp = exp
    return best_exp, best_val

def generate_markdown_report(experiments: List[Experiment]) -> str:
    """Generate a comprehensive markdown report string."""
    lines = []

    def w(s: str = ""):
        lines.append(s)

    w("# LFCM Experiment Analysis Report")
    w()
    w(f"**Experiments analysed:** {len(experiments)}")
    w()

    # ── 1. Overall Summary Table ──────────────────────────────────────
    w("## 1. Overall Results Summary")
    w()
    w("| Experiment | Clean Acc | PGD-1 | PGD-20 | PGD-100 | CW-20 | AutoAttack |")
    w("|:-----------|:---------:|:-----:|:------:|:-------:|:-----:|:----------:|")

    # Sort: LFCM first, then baselines
    lfcm_exps = [e for e in experiments if "LFCM" in e.stem or e.group in ("canon", "codebook", "schedule")]
    baseline_exps = [e for e in experiments if e not in lfcm_exps]

    for exp in lfcm_exps + baseline_exps:
        tm = exp.test_metrics
        if tm:
            w(f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd1'))} | "
              f"{_fmt(tm.get('pgd20'))} | {_fmt(tm.get('pgd100'))} | {_fmt(tm.get('cw'))} | "
              f"{_fmt(tm.get('aa'))} |")
        else:
            w(f"| {exp.display_name} | ⚠️ no test log | — | — | — | — | — |")

    w()

    # ── 2. Best Results Highlights ────────────────────────────────────
    w("## 2. Highlights")
    w()

    best_clean, val_clean = _best(experiments, "normal_acc")
    best_aa, val_aa = _best(experiments, "aa")
    best_pgd20, val_pgd20 = _best(experiments, "pgd20")
    best_cw, val_cw = _best(experiments, "cw")

    w(f"- **Best Clean Accuracy:** {best_clean.display_name} — {val_clean:.2f}%")
    w(f"- **Best AutoAttack (most robust):** {best_aa.display_name} — {val_aa:.2f}%")
    w(f"- **Best PGD-20:** {best_pgd20.display_name} — {val_pgd20:.2f}%")
    w(f"- **Best CW-20:** {best_cw.display_name} — {val_cw:.2f}%")
    w()

    # ── 3. Ablation: Canonicalization Strength (K=64 fixed) ──────────
    w("## 3. Ablation: Canonicalization Strength  (K=64, varying `w_canon`)")
    w()
    canon_order = ["nocanon", "weak", "baseline", "strong"]
    canon_exps = [e for e in experiments if e.group == "canon"]
    canon_exps.sort(key=lambda e: canon_order.index(e.variant) if e.variant in canon_order else 99)

    if canon_exps:
        w("| w_canon | Clean Acc | PGD-20 | CW-20 | AutoAttack | Δ Clean−AA |")
        w("|:--------|:---------:|:------:|:-----:|:----------:|:----------:|")
        for exp in canon_exps:
            tm = exp.test_metrics
            if tm and tm.get("normal_acc") and tm.get("aa"):
                delta = tm["normal_acc"] - tm["aa"]
                w(f"| {exp.variant} (w_canon=...)" if "=" not in exp.display_name else
                  f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
                  f"{_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | {delta:.2f} |")
            elif tm:
                w(f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
                  f"{_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | — |")
            else:
                w(f"| {exp.display_name} | ⚠️ no test | — | — | — | — |")
        w()

    # ── 4. Ablation: Codebook Size ────────────────────────────────────
    w("## 4. Ablation: Codebook Size  (`K`, fixed `w_canon=0.5`)")
    w()
    cb_order = ["K32", "K64", "K128"]
    cb_exps = [e for e in experiments if e.group == "codebook"]
    cb_exps.sort(key=lambda e: cb_order.index(e.variant) if e.variant in cb_order else 99)
    # Also include baseline LFCM as K64 reference if not already in codebook group
    baseline_lfcm = [e for e in experiments if e.stem == "WRN34_10_LFCM"]
    for be in baseline_lfcm:
        if be not in cb_exps:
            cb_exps.insert(1, be)  # insert at K64 position

    if cb_exps:
        w("| K | Clean Acc | PGD-20 | CW-20 | AutoAttack | Δ Clean−AA |")
        w("|:-:|:---------:|:------:|:-----:|:----------:|:----------:|")
        for exp in cb_exps:
            tm = exp.test_metrics
            if tm and tm.get("normal_acc") and tm.get("aa"):
                delta = tm["normal_acc"] - tm["aa"]
                w(f"| {exp.variant} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
                  f"{_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | {delta:.2f} |")
            else:
                w(f"| {exp.display_name} | ⚠️ no test | — | — | — | — |")
        w()

    # ── 5. Comparison vs Baselines ────────────────────────────────────
    w("## 5. LFCM vs Baselines (AT, HFDR, Natural)")
    w()
    method_exps = [e for e in experiments if e.group == "method"]
    # Add best LFCM
    best_lfcm = best_aa  # most robust LFCM

    w("| Method | Clean Acc | PGD-20 | CW-20 | AutoAttack | Δ Clean−AA |")
    w("|:-------|:---------:|:------:|:-----:|:----------:|:----------:|")
    for exp in method_exps:
        tm = exp.test_metrics
        if tm and tm.get("normal_acc") and tm.get("aa"):
            delta = tm["normal_acc"] - tm["aa"]
            w(f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
              f"{_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | {delta:.2f} |")
        elif tm:
            w(f"| {exp.display_name} | {_fmt(tm.get('normal_acc'))} | {_fmt(tm.get('pgd20'))} | "
              f"{_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | — |")
    # Add the best LFCM row
    if best_lfcm and best_lfcm.test_metrics:
        tm = best_lfcm.test_metrics
        if tm.get("normal_acc") and tm.get("aa"):
            delta = tm["normal_acc"] - tm["aa"]
            w(f"| **Best LFCM** ({best_lfcm.display_name}) | {_fmt(tm.get('normal_acc'))} | "
              f"{_fmt(tm.get('pgd20'))} | {_fmt(tm.get('cw'))} | {_fmt(tm.get('aa'))} | {delta:.2f} |")
    w()

    # ── 6. Training Convergence Summary ───────────────────────────────
    w("## 6. Training Convergence (Best Epoch Robust Acc)")
    w()
    w("| Experiment | Best Robust Acc | At Epoch | Final Train Acc | Final Test Acc |")
    w("|:-----------|:---------------:|:--------:|:---------------:|:--------------:|")
    for exp in lfcm_exps + baseline_exps:
        df = exp.record_df
        if df is not None and "test_robust_acc" in df.columns:
            best_idx = df["test_robust_acc"].idxmax()
            best_row = df.iloc[best_idx]
            last_row = df.iloc[-1]
            w(f"| {exp.display_name} | {best_row['test_robust_acc']:.2f} | "
              f"{int(best_row['epoch'])} | {last_row['train_acc']:.2f} | {last_row['test_acc']:.2f} |")
        else:
            w(f"| {exp.display_name} | N/A | — | — | — |")
    w()

    # ── 7. Missing Data ───────────────────────────────────────────────
    missing = [e for e in experiments if not e.has_test]
    if missing:
        w("## 7. Missing Test Data")
        w()
        w("The following experiments have training logs but **no test evaluation logs**:")
        w()
        for exp in missing:
            w(f"- **{exp.display_name}** (`{exp.stem}`)")
        w()

    w("---")
    w(f"*Report generated by `analysis.py`*")
    w()

    return "\n".join(lines)

# ==========================================================================
# 4. VISUALIZATION
# ==========================================================================

class Visualization:
    """Generate all figures."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, name: str):
        path = self.output_dir / name
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

    # ── 4a. Per-Experiment Training Curves ──────────────────────────

    def plot_training_curves(self, exp: Experiment):
        """4-subplot training curve figure for a single experiment."""
        df = exp.record_df
        if df is None:
            print(f"  [SKIP] No record data for {exp.display_name}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Curves — {exp.display_name}", fontsize=15, fontweight="bold", y=0.98)

        epochs = df["epoch"].values

        # Subplot 1: Train Loss
        ax = axes[0, 0]
        ax.plot(epochs, df["train_loss"], color="#e74c3c", linewidth=0.8, alpha=0.9)
        ax.set_ylabel("Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_title("Training Loss")

        # Subplot 2: Train Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, df["train_acc"], color="#2ecc71", linewidth=0.8)
        ax.set_ylabel("Train Accuracy (%)")
        ax.set_xlabel("Epoch")
        ax.set_title("Training Accuracy")

        # Subplot 3: Test Loss
        ax = axes[1, 0]
        ax.plot(epochs, df["test_loss"], color="#3498db", linewidth=0.8)
        ax.set_ylabel("Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_title("Test Loss")

        # Subplot 4: Test Acc + Robust Acc overlay
        ax = axes[1, 1]
        ax.plot(epochs, df["test_acc"], color="#2ecc71", linewidth=1.0, label="Clean Acc", alpha=0.85)
        ax.plot(epochs, df["test_robust_acc"], color="#e74c3c", linewidth=1.0, label="Robust Acc (PGD-10)", alpha=0.85)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Epoch")
        ax.set_title("Test Accuracy: Clean vs Robust")
        ax.legend(loc="lower right")

        # Add LR decay lines on all subplots
        for ax in axes.flat:
            ax.axvline(x=100, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.axvline(x=105, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

        # Shade LFCM stages if applicable
        if "LFCM" in exp.stem:
            for ax in axes.flat:
                ax.axvspan(1, 30, alpha=0.06, color="blue", label="_Stage 1")
                ax.axvspan(31, 60, alpha=0.06, color="orange", label="_Stage 2")
                ax.axvspan(61, 110, alpha=0.06, color="green", label="_Stage 3")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(f"{exp.stem}_training.png")

    # ── 4b. Cross-Experiment Comparison Curves ───────────────────────

    def plot_comparison_curves(self, experiments: List[Experiment], group: str, group_label: str):
        """Plot Test Acc and Test Robust Acc for a group of experiments on the same axes."""
        group_exps = [e for e in experiments if e.group == group and e.record_df is not None]
        if not group_exps:
            print(f"  [SKIP] No experiments in group '{group}'")
            return

        # Also add baseline LFCM if this is the codebook group
        if group == "codebook":
            baseline = [e for e in experiments if e.stem == "WRN34_10_LFCM" and e.record_df is not None]
            for be in baseline:
                if be not in group_exps:
                    be.variant = "K64"
                    group_exps.append(be)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Comparison: {group_label}", fontsize=15, fontweight="bold")

        palette = {"canon": PALETTE_CANON, "codebook": PALETTE_CODEBOOK, "method": PALETTE_METHOD}.get(group, {})

        for exp in group_exps:
            df = exp.record_df
            color = palette.get(exp.variant, None)
            label = exp.display_name

            axes[0].plot(df["epoch"], df["test_acc"], linewidth=1.0, color=color, label=label, alpha=0.85)
            axes[1].plot(df["epoch"], df["test_robust_acc"], linewidth=1.0, color=color, label=label, alpha=0.85)

        axes[0].set_ylabel("Test Accuracy (%)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Clean Test Accuracy")
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].axvline(x=100, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
        axes[0].axvline(x=105, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)

        axes[1].set_ylabel("Robust Accuracy (PGD-10, %)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Robust Test Accuracy")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].axvline(x=100, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
        axes[1].axvline(x=105, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save(f"comparison_{group}.png")

    # ── 4c. Final Test Metrics Bar Chart ─────────────────────────────

    def plot_test_metrics_bars(self, experiments: List[Experiment]):
        """Grouped bar chart of final test metrics for all experiments with test data."""
        exps_with_test = [e for e in experiments if e.has_test and e.test_metrics]
        if not exps_with_test:
            print("  [SKIP] No experiments with test data")
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
            bars = ax.bar([xi + offset for xi in x], values, width, label=label, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)
            # Annotate values on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_xticks(list(x))
        ax.set_xticklabels([e.display_name for e in exps_with_test], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Test Metrics Comparison Across All Experiments", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, max(
            max((e.test_metrics.get("normal_acc") or 0) for e in exps_with_test) + 12,
            100
        ))

        plt.tight_layout()
        self._save("test_metrics_bars.png")

    # ── 4d. Clean vs Robust Scatter ──────────────────────────────────

    def plot_clean_vs_robust_scatter(self, experiments: List[Experiment]):
        """Scatter plot: Clean Accuracy vs AutoAttack Accuracy."""
        exps_with_test = [e for e in experiments if e.has_test and e.test_metrics
                          and e.test_metrics.get("normal_acc") and e.test_metrics.get("aa")]
        if not exps_with_test:
            print("  [SKIP] No experiments with clean+AA data")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine color by group
        for exp in exps_with_test:
            x = exp.test_metrics["normal_acc"]
            y = exp.test_metrics["aa"]

            if exp.group == "canon":
                color = PALETTE_CANON.get(exp.variant, "#7f8c8d")
            elif exp.group == "codebook":
                color = PALETTE_CODEBOOK.get(exp.variant, "#7f8c8d")
            elif exp.group == "method":
                color = PALETTE_METHOD.get(exp.variant, "#7f8c8d")
            else:
                color = "#7f8c8d"

            ax.scatter(x, y, c=color, s=120, edgecolors="white", linewidth=1.2, zorder=5)
            ax.annotate(exp.display_name, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, alpha=0.9)

        ax.set_xlabel("Clean Accuracy (%)")
        ax.set_ylabel("AutoAttack Accuracy (%)")
        ax.set_title("Clean vs Robust Accuracy Trade-off", fontweight="bold")

        # Add quadrant lines at median values
        xs = [e.test_metrics["normal_acc"] for e in exps_with_test]
        ys = [e.test_metrics["aa"] for e in exps_with_test]
        if xs and ys:
            med_x = sorted(xs)[len(xs) // 2]
            med_y = sorted(ys)[len(ys) // 2]
            ax.axhline(y=med_y, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.axvline(x=med_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        # Set axis limits with some padding
        x_margin = 1.0
        y_margin = 1.0
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

        plt.tight_layout()
        self._save("clean_vs_robust_scatter.png")

    # ── 4e. Radar Chart ──────────────────────────────────────────────

    def plot_radar_comparison(self, experiments: List[Experiment], group: str, group_label: str):
        """Radar/spider chart comparing multiple metrics across experiments in a group."""
        import numpy as np

        group_exps = [e for e in experiments if e.group == group and e.has_test and e.test_metrics]
        if group == "codebook":
            baseline = [e for e in experiments if e.stem == "WRN34_10_LFCM" and e.has_test]
            for be in baseline:
                if be not in group_exps:
                    be.variant = "K64"
                    group_exps.append(be)

        if len(group_exps) < 2:
            print(f"  [SKIP] Not enough experiments in group '{group}' for radar chart (need ≥2)")
            return

        metric_keys = ["normal_acc", "pgd1", "pgd20", "pgd100", "cw", "aa"]
        metric_labels = ["Clean", "PGD-1", "PGD-20", "PGD-100", "CW-20", "AA"]
        n_metrics = len(metric_keys)

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the circle

        palette = {"canon": PALETTE_CANON, "codebook": PALETTE_CODEBOOK, "method": PALETTE_METHOD}.get(group, {})

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.set_title(f"Multi-Metric Radar — {group_label}", fontsize=14, fontweight="bold", pad=25)

        for exp in group_exps:
            values = [exp.test_metrics.get(k) or 0 for k in metric_keys]
            values += values[:1]

            color = palette.get(exp.variant, "#7f8c8d")
            ax.fill(angles, values, alpha=0.08, color=color)
            ax.plot(angles, values, "o-", linewidth=1.8, color=color, label=exp.display_name, markersize=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80])
        ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="gray")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

        plt.tight_layout()
        self._save(f"radar_{group}.png")

    # ── 4f. Bonus: Attack Degradation Curves ─────────────────────────

    def plot_attack_degradation(self, experiments: List[Experiment]):
        """Line plot showing accuracy degradation as attack strength increases (PGD-1 → PGD-20 → PGD-100 → CW → AA)."""
        exps_with_test = [e for e in experiments if e.has_test and e.test_metrics]
        if not exps_with_test:
            return

        attack_order = ["pgd1", "pgd20", "pgd100", "cw", "aa"]
        attack_labels = ["PGD-1", "PGD-20", "PGD-100", "CW-20", "AA"]
        x = range(len(attack_order))

        fig, ax = plt.subplots(figsize=(12, 7))

        for exp in exps_with_test:
            values = [exp.test_metrics.get(k) for k in attack_order]
            # Only plot if we have at least 3 values
            valid = [v for v in values if v is not None]
            if len(valid) < 3:
                continue

            # Determine color
            if exp.group == "canon":
                color = PALETTE_CANON.get(exp.variant, "#7f8c8d")
            elif exp.group == "codebook":
                color = PALETTE_CODEBOOK.get(exp.variant, "#7f8c8d")
            elif exp.group == "method":
                color = PALETTE_METHOD.get(exp.variant, "#7f8c8d")
            else:
                color = "#7f8c8d"

            ax.plot(x, values, "o-", linewidth=1.5, color=color, label=exp.display_name, markersize=6)

        ax.set_xticks(list(x))
        ax.set_xticklabels(attack_labels, fontsize=11)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Attack Strength Degradation Curves", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_ylim(0, 90)

        plt.tight_layout()
        self._save("attack_degradation.png")

    # ── 4g. Bonus: LR Decay Zoom ─────────────────────────────────────

    def plot_lr_decay_zoom(self, experiments: List[Experiment]):
        """Zoomed view of epochs 90-110 to visualize LR decay impact."""
        ad_exps = [e for e in experiments if e.record_df is not None
                   and e.group in ("canon", "codebook", "method", "schedule")
                   and "Natural" not in e.display_name]
        if not ad_exps:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("LR Decay Impact (Epochs 90–110 Zoom)", fontsize=14, fontweight="bold")

        for exp in ad_exps:
            df = exp.record_df
            zoom = df[df["epoch"] >= 90]
            if zoom.empty:
                continue

            color = None
            if exp.group == "canon":
                color = PALETTE_CANON.get(exp.variant, None)
            elif exp.group == "codebook":
                color = PALETTE_CODEBOOK.get(exp.variant, None)

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
        self._save("lr_decay_zoom.png")

    # ── 4h. Bonus: Robustness Stability (rolling std) ─────────────────

    def plot_robustness_stability(self, experiments: List[Experiment]):
        """Plot rolling standard deviation of robust accuracy in the last 30 epochs."""
        ad_exps = [e for e in experiments if e.record_df is not None
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

            color = None
            if exp.group == "canon":
                color = PALETTE_CANON.get(exp.variant, "#7f8c8d")
            elif exp.group == "codebook":
                color = PALETTE_CODEBOOK.get(exp.variant, "#7f8c8d")
            else:
                color = PALETTE_METHOD.get(exp.variant, "#7f8c8d")

            stability_data.append((exp.display_name, mean_ra, std_ra, max_ra, color))

        if not stability_data:
            return

        stability_data.sort(key=lambda x: x[3], reverse=True)  # sort by max robust acc

        names = [d[0] for d in stability_data]
        means = [d[1] for d in stability_data]
        stds = [d[2] for d in stability_data]
        colors = [d[4] for d in stability_data]

        # Horizontal bar: mean with error bar
        y_pos = range(len(names))
        bars = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, edgecolor="white",
                       capsize=3, error_kw={"linewidth": 1.2})

        # Annotate max value
        for i, (name, mean, std, max_val, _) in enumerate(stability_data):
            ax.text(mean + std + 0.3, i, f"max={max_val:.1f}", va="center", fontsize=8, alpha=0.8)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Robust Accuracy (PGD-10, %) — mean ± std (last 30 epochs)")
        ax.set_title("Training Stability: Late-Stage Robust Accuracy", fontweight="bold")
        ax.invert_yaxis()

        plt.tight_layout()
        self._save("robustness_stability.png")


# ==========================================================================
# 5. MAIN
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
    print("  HFDR / LFCM Experiment Log Analyzer")
    print("=" * 60)
    print()

    # ── Scan ─────────────────────────────────────────────────────────
    print("[1/4] Scanning experiment logs...")
    experiments = scan_result_dir(str(result_dir))
    print(f"  Found {len(experiments)} experiments "
          f"({sum(1 for e in experiments if e.has_test)} with test data, "
          f"{sum(1 for e in experiments if e.record_df is not None)} with training data)")
    print()

    # ── Report ───────────────────────────────────────────────────────
    print("[2/4] Generating markdown report...")
    report = generate_markdown_report(experiments)
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

        # 1. Individual training curves
        print("  ── Individual Training Curves ──")
        for exp in experiments:
            if exp.record_df is not None:
                viz.plot_training_curves(exp)

        # 2. Comparison curves by group
        print("  ── Comparison Curves ──")
        viz.plot_comparison_curves(experiments, "canon", "Canonicalization Strength Ablation")
        viz.plot_comparison_curves(experiments, "codebook", "Codebook Size Ablation")
        viz.plot_comparison_curves(experiments, "method", "Training Method Comparison")

        # 3. Bar chart
        print("  ── Test Metrics Bar Chart ──")
        viz.plot_test_metrics_bars(experiments)

        # 4. Scatter
        print("  ── Clean vs Robust Scatter ──")
        viz.plot_clean_vs_robust_scatter(experiments)

        # 5. Radar charts by group
        print("  ── Radar Charts ──")
        for grp, grp_label in [("canon", "Canonicalization Strength"),
                               ("codebook", "Codebook Size"),
                               ("method", "Training Method")]:
            viz.plot_radar_comparison(experiments, grp, grp_label)

        # 6. Bonus: Attack degradation
        print("  ── Bonus: Attack Degradation ──")
        viz.plot_attack_degradation(experiments)

        # 7. Bonus: LR decay zoom
        print("  ── Bonus: LR Decay Zoom ──")
        viz.plot_lr_decay_zoom(experiments)

        # 8. Bonus: Robustness stability
        print("  ── Bonus: Robustness Stability ──")
        viz.plot_robustness_stability(experiments)

        print()

    # ── Summary ──────────────────────────────────────────────────────
    print("[4/4] Done!")
    print(f"  Report: {args.report}")
    if not args.skip_plots:
        print(f"  Figures: {args.output_dir}/")
        n_figs = len(list(Path(args.output_dir).glob("*.png")))
        print(f"  Generated {n_figs} figure(s)")


if __name__ == "__main__":
    main()
