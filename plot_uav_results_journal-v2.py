
"""
Publication-style plotting script for the UAV fault diagnosis paper.

Design principles:
- Minimal in-figure decoration
- No chart titles inside figures (captions should be used in the manuscript)
- Serif typography suitable for academic manuscripts
- Clean axes, outward ticks, restrained annotation density
- Vector-first export (PDF/SVG) plus high-resolution PNG

Usage:
    python plot_uav_results_journal.py
    python plot_uav_results_journal.py --outdir figures --formats pdf svg png --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SIX_CLASS_RESULTS = [
    {"Experiment": "G1", "Label": "MLP\n1 s\nEnhanced", "Model": "MLP", "Window": "1 s", "Feature": "Enhanced",
     "Accuracy": 42.78, "MacroPrecision": 45.85, "MacroRecall": 42.78, "MacroF1": 41.21, "StdF1": 9.73},
    {"Experiment": "G2", "Label": "MLP\n3 s\nEnhanced", "Model": "MLP", "Window": "3 s", "Feature": "Enhanced",
     "Accuracy": 43.89, "MacroPrecision": 47.70, "MacroRecall": 43.89, "MacroF1": 42.71, "StdF1": 7.74},
    {"Experiment": "G3", "Label": "MLP\n3 s\nMean/Std", "Model": "MLP", "Window": "3 s", "Feature": "Mean/Std",
     "Accuracy": 43.89, "MacroPrecision": 46.82, "MacroRecall": 43.89, "MacroF1": 42.29, "StdF1": 6.39},
    {"Experiment": "G4", "Label": "RF\n3 s\nEnhanced", "Model": "Random Forest", "Window": "3 s", "Feature": "Enhanced",
     "Accuracy": 55.56, "MacroPrecision": 56.33, "MacroRecall": 55.56, "MacroF1": 54.39, "StdF1": 9.74},
    {"Experiment": "G5", "Label": "XGBoost\n3 s\nEnhanced", "Model": "XGBoost", "Window": "3 s", "Feature": "Enhanced",
     "Accuracy": 75.00, "MacroPrecision": 79.50, "MacroRecall": 75.00, "MacroF1": 74.23, "StdF1": 10.18},
]

BINARY_RESULTS = [
    {"Experiment": "G0", "Label": "MLP\n3 s\nEnhanced", "Model": "MLP", "Window": "3 s", "Feature": "Enhanced",
     "Accuracy": 81.90, "F1": 89.20, "ROC_AUC": 0.665},
]


def set_journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    for fmt in formats:
        outpath = outdir / f"{stem}.{fmt.lower()}"
        if fmt.lower() == "png":
            fig.savefig(outpath, dpi=dpi)
        else:
            fig.savefig(outpath)
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, rects, offset: float = 2.2, fontsize: int = 8.5) -> None:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def apply_axis_format(ax: plt.Axes, ylabel: str, xlabel: str = "", ylim=None) -> None:
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def write_csv_summaries(outdir: Path) -> None:
    pd.DataFrame(SIX_CLASS_RESULTS).to_csv(outdir / "six_class_results.csv", index=False)
    pd.DataFrame(BINARY_RESULTS).to_csv(outdir / "binary_results.csv", index=False)


def plot_main_six_class_results(outdir: Path, formats: List[str], dpi: int) -> None:
    df = pd.DataFrame(SIX_CLASS_RESULTS)
    x = np.arange(len(df))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars1 = ax.bar(x - width / 2, df["Accuracy"], width, label="Accuracy", hatch="//", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, df["MacroF1"], width, label="Macro-F1", hatch="..", linewidth=0.8)

    apply_axis_format(ax, "Score (%)", "Experimental setting", ylim=(0, 85))
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"])
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.8, columnspacing=1.2)

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)

    save_figure(fig, outdir, "fig_main_six_class_results_journal", formats, dpi)


def plot_window_length_comparison(outdir: Path, formats: List[str], dpi: int) -> None:
    df = pd.DataFrame([r for r in SIX_CLASS_RESULTS if r["Model"] == "MLP" and r["Feature"] == "Enhanced"])
    x = np.arange(len(df))
    width = 0.32

    fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
    bars1 = ax1.bar(x - width / 2, df["Accuracy"], width, label="Accuracy", hatch="//", linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, df["MacroF1"], width, label="Macro-F1", hatch="..", linewidth=0.8)
    apply_axis_format(ax1, "Score (%)", "Window length", ylim=(0, 55))
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Window"])

    ax2 = ax1.twinx()
    line = ax2.plot(x, df["StdF1"].values, marker="o", linewidth=1.0, label="Std. of Macro-F1")
    ax2.set_ylabel("Standard deviation (%)")
    ax2.set_ylim(0, 14)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.9)
    ax2.tick_params(axis="y", direction="out", width=0.9, length=4)

    handles = [bars1, bars2, line[0]]
    labels = ["Accuracy", "Macro-F1", "Std. of Macro-F1"]
    ax1.legend(handles, labels, frameon=False, loc="upper left", ncol=1)

    add_bar_labels(ax1, bars1, fontsize=8)
    add_bar_labels(ax1, bars2, fontsize=8)
    for xi, yi in zip(x, df["StdF1"].values):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    save_figure(fig, outdir, "fig_window_length_comparison_journal", formats, dpi)


def plot_feature_ablation(outdir: Path, formats: List[str], dpi: int) -> None:
    df = pd.DataFrame([r for r in SIX_CLASS_RESULTS if r["Model"] == "MLP" and r["Window"] == "3 s"])
    df = df[df["Feature"].isin(["Mean/Std", "Enhanced"])].copy()
    df["Order"] = df["Feature"].map({"Mean/Std": 0, "Enhanced": 1})
    df = df.sort_values("Order")

    x = np.arange(len(df))
    width = 0.32

    fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
    bars1 = ax1.bar(x - width / 2, df["Accuracy"], width, label="Accuracy", hatch="//", linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, df["MacroF1"], width, label="Macro-F1", hatch="..", linewidth=0.8)
    apply_axis_format(ax1, "Score (%)", "Feature mode", ylim=(0, 55))
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Feature"])

    ax2 = ax1.twinx()
    line = ax2.plot(x, df["StdF1"].values, marker="o", linewidth=1.0, label="Std. of Macro-F1")
    ax2.set_ylabel("Standard deviation (%)")
    ax2.set_ylim(0, 10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.9)
    ax2.tick_params(axis="y", direction="out", width=0.9, length=4)

    handles = [bars1, bars2, line[0]]
    labels = ["Accuracy", "Macro-F1", "Std. of Macro-F1"]
    ax1.legend(handles, labels, frameon=False, loc="upper left", ncol=1)

    add_bar_labels(ax1, bars1, fontsize=8)
    add_bar_labels(ax1, bars2, fontsize=8)
    for xi, yi in zip(x, df["StdF1"].values):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    save_figure(fig, outdir, "fig_feature_ablation_journal", formats, dpi)


def plot_classifier_comparison(outdir: Path, formats: List[str], dpi: int) -> None:
    df = pd.DataFrame([r for r in SIX_CLASS_RESULTS if r["Window"] == "3 s" and r["Feature"] == "Enhanced"]).copy()
    order = {"MLP": 0, "Random Forest": 1, "XGBoost": 2}
    df["Order"] = df["Model"].map(order)
    df = df.sort_values("Order")

    x = np.arange(len(df))
    width = 0.32

    fig, ax1 = plt.subplots(figsize=(5.8, 3.8))
    bars1 = ax1.bar(x - width / 2, df["Accuracy"], width, label="Accuracy", hatch="//", linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, df["MacroF1"], width, label="Macro-F1", hatch="..", linewidth=0.8)
    apply_axis_format(ax1, "Score (%)", "Classifier", ylim=(0, 85))
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"])

    ax2 = ax1.twinx()
    line = ax2.plot(x, df["StdF1"].values, marker="o", linewidth=1.0, label="Std. of Macro-F1")
    ax2.set_ylabel("Standard deviation (%)")
    ax2.set_ylim(0, 14)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.9)
    ax2.tick_params(axis="y", direction="out", width=0.9, length=4)

    handles = [bars1, bars2, line[0]]
    labels = ["Accuracy", "Macro-F1", "Std. of Macro-F1"]
    ax1.legend(handles, labels, frameon=False, loc="upper left", ncol=1)

    add_bar_labels(ax1, bars1, fontsize=8)
    add_bar_labels(ax1, bars2, fontsize=8)
    for xi, yi in zip(x, df["StdF1"].values):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    save_figure(fig, outdir, "fig_classifier_comparison_journal", formats, dpi)


def plot_binary_detection_summary(outdir: Path, formats: List[str], dpi: int) -> None:
    df = pd.DataFrame(BINARY_RESULTS)
    labels = ["Accuracy", "F1", "ROC-AUC"]
    values = [df.loc[0, "Accuracy"], df.loc[0, "F1"], df.loc[0, "ROC_AUC"] * 100.0]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    bars = ax.bar(x, values, width=0.55, hatch="//", linewidth=0.8)

    apply_axis_format(ax, "Score (%)", "Metric", ylim=(0, 100))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    add_bar_labels(ax, bars, fontsize=8.5)
    ax.text(0.98, 0.04, "ROC-AUC is scaled to 0–100", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)

    save_figure(fig, outdir, "fig_binary_detection_summary_journal", formats, dpi)


def plot_compact_summary_panel(outdir: Path, formats: List[str], dpi: int) -> None:
    six_df = pd.DataFrame(SIX_CLASS_RESULTS)
    bin_df = pd.DataFrame(BINARY_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.15, 0.85]})

    x = np.arange(len(six_df))
    width = 0.32
    bars1 = axes[0].bar(x - width / 2, six_df["Accuracy"], width, label="Accuracy", hatch="//", linewidth=0.8)
    bars2 = axes[0].bar(x + width / 2, six_df["MacroF1"], width, label="Macro-F1", hatch="..", linewidth=0.8)
    apply_axis_format(axes[0], "Score (%)", "", ylim=(0, 85))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(six_df["Experiment"])
    axes[0].text(0.02, 0.97, "(a) Six-class attribution", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=9.5)
    axes[0].legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6)

    labels = ["Accuracy", "F1", "ROC-AUC"]
    values = [bin_df.loc[0, "Accuracy"], bin_df.loc[0, "F1"], bin_df.loc[0, "ROC_AUC"] * 100.0]
    xb = np.arange(len(labels))
    bars3 = axes[1].bar(xb, values, width=0.55, hatch="//", linewidth=0.8)
    apply_axis_format(axes[1], "Score (%)", "", ylim=(0, 100))
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(labels)
    axes[1].text(0.02, 0.97, "(b) Binary detection", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=9.5)

    add_bar_labels(axes[0], bars1, fontsize=7.5)
    add_bar_labels(axes[0], bars2, fontsize=7.5)
    add_bar_labels(axes[1], bars3, fontsize=7.5)

    save_figure(fig, outdir, "fig_summary_panel_journal", formats, dpi)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate journal-style figures for the UAV fault diagnosis paper.")
    parser.add_argument("--outdir", type=str, default="uav_paper_figures_journal")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    set_journal_style()
    write_csv_summaries(outdir)

    plot_main_six_class_results(outdir, args.formats, args.dpi)
    plot_window_length_comparison(outdir, args.formats, args.dpi)
    plot_feature_ablation(outdir, args.formats, args.dpi)
    plot_classifier_comparison(outdir, args.formats, args.dpi)
    plot_binary_detection_summary(outdir, args.formats, args.dpi)
    plot_compact_summary_panel(outdir, args.formats, args.dpi)

    print(f"Journal-style figures written to: {outdir.resolve()}")
    for p in sorted(outdir.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
