"""
Generate journal-style figures for the UAV fault diagnosis paper.

This script redraws the paper figures with a cleaner single-column layout,
minimal in-figure text, and a consistent academic theme suitable for journal
submission. By default, it exports the two main figures kept in the body of the
paper and three supplementary figures recommended for the appendix.

Outputs:
    - Vector figures: PDF and SVG
    - Raster preview figures: PNG (default 600 DPI)
    - CSV summaries copied into the output root

Example:
    python3 plot_uav_results.py
    python3 plot_uav_results.py --outdir outputs/paper_figures_journal
    python3 plot_uav_results.py --figures window_main classifier_comparison
    python3 plot_uav_results.py --figures all --formats pdf svg png --dpi 600
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator


SIX_CLASS_RESULTS = [
    {
        "Experiment": "G1",
        "Label": "MLP\n1 s\nEnhanced",
        "Model": "MLP",
        "Window": "1 s",
        "Feature": "Enhanced",
        "Accuracy": 42.78,
        "MacroPrecision": 45.85,
        "MacroRecall": 42.78,
        "MacroF1": 41.21,
        "StdF1": 9.73,
    },
    {
        "Experiment": "G2",
        "Label": "MLP\n3 s\nEnhanced",
        "Model": "MLP",
        "Window": "3 s",
        "Feature": "Enhanced",
        "Accuracy": 43.89,
        "MacroPrecision": 47.70,
        "MacroRecall": 43.89,
        "MacroF1": 42.71,
        "StdF1": 7.74,
    },
    {
        "Experiment": "G3",
        "Label": "MLP\n3 s\nMean/Std",
        "Model": "MLP",
        "Window": "3 s",
        "Feature": "Mean/Std",
        "Accuracy": 43.89,
        "MacroPrecision": 46.82,
        "MacroRecall": 43.89,
        "MacroF1": 42.29,
        "StdF1": 6.39,
    },
    {
        "Experiment": "G4",
        "Label": "RF\n3 s\nEnhanced",
        "Model": "Random Forest",
        "Window": "3 s",
        "Feature": "Enhanced",
        "Accuracy": 55.56,
        "MacroPrecision": 56.33,
        "MacroRecall": 55.56,
        "MacroF1": 54.39,
        "StdF1": 9.74,
    },
    {
        "Experiment": "G5",
        "Label": "XGBoost\n3 s\nEnhanced",
        "Model": "XGBoost",
        "Window": "3 s",
        "Feature": "Enhanced",
        "Accuracy": 75.00,
        "MacroPrecision": 79.50,
        "MacroRecall": 75.00,
        "MacroF1": 74.23,
        "StdF1": 10.18,
    },
]

BINARY_RESULTS = [
    {
        "Experiment": "G0",
        "Label": "MLP\n3 s\nEnhanced",
        "Model": "MLP",
        "Window": "3 s",
        "Feature": "Enhanced",
        "Accuracy": 81.90,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": 89.20,
        "ROC_AUC": 0.665,
    }
]

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class FigureSpec:
    key: str
    stem: str
    category: str
    plotter: Callable[[Path, list[str], int, str], None]


def mm_to_inches(value_mm: float) -> float:
    return value_mm / MM_PER_INCH


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_font_family() -> list[str]:
    return ["Times New Roman", "Liberation Serif", "DejaVu Serif"]


def set_paper_style(layout: str) -> None:
    if layout == "single":
        label_size = 8.5
        tick_size = 7.5
        legend_size = 7.5
        annotation_size = 7.0
    else:
        label_size = 9.0
        tick_size = 8.0
        legend_size = 8.0
        annotation_size = 7.5

    plt.rcParams.update(
        {
            "font.family": get_font_family(),
            "font.size": label_size,
            "axes.labelsize": label_size,
            "axes.linewidth": 0.8,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "hatch.linewidth": 0.7,
            "lines.markersize": 5.0,
            "lines.linewidth": 1.0,
        }
    )


def get_annotation_size(layout: str) -> float:
    return 7.0 if layout == "single" else 7.5


def journal_colors() -> dict[str, str]:
    return {
        "Accuracy": "#4C78A8",
        "Macro-F1": "#F58518",
        "Stability": "#4C4C4C",
        "Grid": "#D9DEE7",
    }


def style_axes(ax: plt.Axes, ylabel: str, ylim: tuple[float, float], ystep: float) -> None:
    colors = journal_colors()
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.grid(axis="y", color=colors["Grid"], linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=0.8, length=3)


def bar_edgecolor(fill_color: str) -> tuple[float, float, float]:
    r, g, b = to_rgb(fill_color)
    return (max(r - 0.18, 0.0), max(g - 0.18, 0.0), max(b - 0.18, 0.0))


def add_bar_labels(
    ax: plt.Axes,
    rects,
    offset: float = 0.45,
    fmt: str = "{:.1f}",
    fontsize: float = 7.0,
) -> None:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, offset * 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    ensure_dir(outdir)
    for fmt in formats:
        outpath = outdir / f"{stem}.{fmt.lower()}"
        if fmt.lower() == "png":
            fig.savefig(outpath, dpi=dpi)
        else:
            fig.savefig(outpath)
    plt.close(fig)


def write_csv_summaries(outdir: Path) -> None:
    pd.DataFrame(SIX_CLASS_RESULTS).to_csv(outdir / "six_class_results.csv", index=False)
    pd.DataFrame(BINARY_RESULTS).to_csv(outdir / "binary_results.csv", index=False)


def get_main_size(layout: str) -> tuple[float, float]:
    if layout == "single":
        return (mm_to_inches(85), mm_to_inches(60))
    return (mm_to_inches(170), mm_to_inches(70))


def get_supplementary_size(layout: str) -> tuple[float, float]:
    if layout == "single":
        return (mm_to_inches(85), mm_to_inches(56))
    return (mm_to_inches(170), mm_to_inches(66))


def prepare_window_df() -> pd.DataFrame:
    df = pd.DataFrame(
        [row for row in SIX_CLASS_RESULTS if row["Model"] == "MLP" and row["Feature"] == "Enhanced"]
    ).copy()
    order = {"1 s": 0, "3 s": 1}
    df["Order"] = df["Window"].map(order)
    return df.sort_values("Order")


def prepare_feature_df() -> pd.DataFrame:
    df = pd.DataFrame(
        [row for row in SIX_CLASS_RESULTS if row["Model"] == "MLP" and row["Window"] == "3 s"]
    ).copy()
    order = {"Mean/Std": 0, "Enhanced": 1}
    df["Order"] = df["Feature"].map(order)
    return df[df["Feature"].isin(order)].sort_values("Order")


def prepare_classifier_df() -> pd.DataFrame:
    df = pd.DataFrame(
        [row for row in SIX_CLASS_RESULTS if row["Window"] == "3 s" and row["Feature"] == "Enhanced"]
    ).copy()
    order = {"MLP": 0, "Random Forest": 1, "XGBoost": 2}
    label_map = {"MLP": "MLP", "Random Forest": "RF", "XGBoost": "XGB"}
    df["Order"] = df["Model"].map(order)
    df["Display"] = df["Model"].map(label_map)
    return df.sort_values("Order")


def plot_grouped_metric_bars(
    df: pd.DataFrame,
    xlabels: list[str],
    outdir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    layout: str,
    ylabel: str,
    ylim: tuple[float, float],
    ystep: float,
    annotate: bool,
    figure_size: tuple[float, float],
) -> None:
    colors = journal_colors()
    x = np.arange(len(df))
    width = 0.32
    annotation_size = get_annotation_size(layout)

    fig, ax = plt.subplots(figsize=figure_size)
    bars_accuracy = ax.bar(
        x - width / 2,
        df["Accuracy"],
        width,
        color=colors["Accuracy"],
        edgecolor=bar_edgecolor(colors["Accuracy"]),
        linewidth=0.75,
        hatch="//",
        label="Accuracy",
    )
    bars_f1 = ax.bar(
        x + width / 2,
        df["MacroF1"],
        width,
        color=colors["Macro-F1"],
        edgecolor=bar_edgecolor(colors["Macro-F1"]),
        linewidth=0.75,
        hatch="..",
        label="Macro-F1",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    style_axes(ax, ylabel=ylabel, ylim=ylim, ystep=ystep)
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0.2,
    )

    if annotate:
        add_bar_labels(ax, bars_accuracy, fontsize=annotation_size)
        add_bar_labels(ax, bars_f1, fontsize=annotation_size)

    save_figure(fig, outdir, stem, formats, dpi)


def plot_stability_points(
    df: pd.DataFrame,
    xlabels: list[str],
    outdir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    layout: str,
    ylim: tuple[float, float],
    ystep: float,
) -> None:
    colors = journal_colors()
    x = np.arange(len(df))
    means = df["MacroF1"].to_numpy()
    stds = df["StdF1"].to_numpy()
    annotation_size = get_annotation_size(layout)
    figure_size = get_supplementary_size(layout)

    fig, ax = plt.subplots(figsize=figure_size)
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        color=colors["Stability"],
        ecolor=colors["Stability"],
        elinewidth=0.9,
        capsize=4,
        capthick=0.9,
        markerfacecolor="white",
        markeredgewidth=1.0,
        markeredgecolor=colors["Stability"],
    )
    ax.plot(x, means, color=colors["Stability"], linewidth=0.9, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    style_axes(ax, ylabel="Macro-F1 (%)", ylim=ylim, ystep=ystep)

    for xi, mean_value, std_value in zip(x, means, stds):
        ax.annotate(
            f"{mean_value:.1f} ± {std_value:.1f}",
            xy=(xi, mean_value + std_value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annotation_size,
        )

    save_figure(fig, outdir, stem, formats, dpi)


def plot_window_main(outdir: Path, formats: list[str], dpi: int, layout: str) -> None:
    df = prepare_window_df()
    plot_grouped_metric_bars(
        df=df,
        xlabels=df["Window"].tolist(),
        outdir=outdir,
        stem="fig_5_1_window_length_main",
        formats=formats,
        dpi=dpi,
        layout=layout,
        ylabel="Score (%)",
        ylim=(35, 50),
        ystep=2.5,
        annotate=True,
        figure_size=get_main_size(layout),
    )


def plot_window_stability(outdir: Path, formats: list[str], dpi: int, layout: str) -> None:
    df = prepare_window_df()
    plot_stability_points(
        df=df,
        xlabels=df["Window"].tolist(),
        outdir=outdir,
        stem="fig_5_2_window_length_stability",
        formats=formats,
        dpi=dpi,
        layout=layout,
        ylim=(28, 56),
        ystep=4,
    )


def plot_feature_ablation(outdir: Path, formats: list[str], dpi: int, layout: str) -> None:
    df = prepare_feature_df()
    plot_grouped_metric_bars(
        df=df,
        xlabels=df["Feature"].tolist(),
        outdir=outdir,
        stem="fig_5_3_feature_representation_ablation",
        formats=formats,
        dpi=dpi,
        layout=layout,
        ylabel="Score (%)",
        ylim=(38, 48),
        ystep=2,
        annotate=False,
        figure_size=get_supplementary_size(layout),
    )


def plot_classifier_comparison(outdir: Path, formats: list[str], dpi: int, layout: str) -> None:
    df = prepare_classifier_df()
    plot_grouped_metric_bars(
        df=df,
        xlabels=df["Display"].tolist(),
        outdir=outdir,
        stem="fig_5_4_classifier_comparison",
        formats=formats,
        dpi=dpi,
        layout=layout,
        ylabel="Score (%)",
        ylim=(35, 80),
        ystep=5,
        annotate=True,
        figure_size=get_main_size(layout),
    )


def plot_classifier_stability(outdir: Path, formats: list[str], dpi: int, layout: str) -> None:
    df = prepare_classifier_df()
    plot_stability_points(
        df=df,
        xlabels=df["Display"].tolist(),
        outdir=outdir,
        stem="fig_5_5_classifier_stability",
        formats=formats,
        dpi=dpi,
        layout=layout,
        ylim=(35, 90),
        ystep=5,
    )


FIGURE_SPECS: dict[str, FigureSpec] = {
    "window_main": FigureSpec(
        key="window_main",
        stem="fig_5_1_window_length_main",
        category="main",
        plotter=plot_window_main,
    ),
    "window_stability": FigureSpec(
        key="window_stability",
        stem="fig_5_2_window_length_stability",
        category="supplementary",
        plotter=plot_window_stability,
    ),
    "feature_ablation": FigureSpec(
        key="feature_ablation",
        stem="fig_5_3_feature_representation_ablation",
        category="supplementary",
        plotter=plot_feature_ablation,
    ),
    "classifier_comparison": FigureSpec(
        key="classifier_comparison",
        stem="fig_5_4_classifier_comparison",
        category="main",
        plotter=plot_classifier_comparison,
    ),
    "classifier_stability": FigureSpec(
        key="classifier_stability",
        stem="fig_5_5_classifier_stability",
        category="supplementary",
        plotter=plot_classifier_stability,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate journal-style figures for the UAV fault diagnosis paper."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/paper_figures_journal",
        help="Root output directory for figures and CSV files.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="Output formats, e.g. pdf svg png",
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for PNG export.")
    parser.add_argument(
        "--layout",
        choices=["single", "double"],
        default="single",
        help="Target publication layout. 'single' is recommended for journal column width.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        choices=["all", *FIGURE_SPECS.keys()],
        help="Select specific figures to export, or use 'all'.",
    )
    return parser.parse_args()


def resolve_selected_figures(requested: list[str]) -> list[FigureSpec]:
    if "all" in requested:
        return [FIGURE_SPECS[key] for key in FIGURE_SPECS]
    seen: set[str] = set()
    resolved: list[FigureSpec] = []
    for key in requested:
        if key not in seen:
            resolved.append(FIGURE_SPECS[key])
            seen.add(key)
    return resolved


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    main_dir = outdir / "main"
    supplementary_dir = outdir / "supplementary"

    ensure_dir(outdir)
    ensure_dir(main_dir)
    ensure_dir(supplementary_dir)

    set_paper_style(args.layout)
    write_csv_summaries(outdir)

    selected_specs = resolve_selected_figures(args.figures)
    generated_paths: list[Path] = []

    for spec in selected_specs:
        target_dir = main_dir if spec.category == "main" else supplementary_dir
        spec.plotter(target_dir, args.formats, args.dpi, args.layout)
        for fmt in args.formats:
            generated_paths.append(target_dir / f"{spec.stem}.{fmt.lower()}")

    print(f"Figures written to: {outdir.resolve()}")
    print("Generated files:")
    for path in sorted(generated_paths):
        print(f" - {path.relative_to(outdir)}")
    print(" - six_class_results.csv")
    print(" - binary_results.csv")


if __name__ == "__main__":
    main()
