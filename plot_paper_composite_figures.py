"""
Generate three high-information composite figures for the UAV fault diagnosis paper.

This script reads experiment outputs from the existing `outputs/*` directories and
rebuilds the manuscript figures using SciencePlots plus a stronger, structured
visual style inspired by the provided examples.

Required dependency:
    python3 -m pip install --user SciencePlots

Examples:
    python3 plot_paper_composite_figures.py
    python3 plot_paper_composite_figures.py --figures overall tradeoff
    python3 plot_paper_composite_figures.py --outdir outputs/paper_composite_figures
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

try:
    import scienceplots  # noqa: F401
except ImportError as exc:  # pragma: no cover - explicit runtime guidance
    raise SystemExit(
        "SciencePlots is required. Install it with: python3 -m pip install --user SciencePlots"
    ) from exc

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


MM_PER_INCH = 25.4
ACC_COLOR = "#8BB9F0"
F1_COLOR = "#F4C36A"
AUC_COLOR = "#98D98E"


@dataclass(frozen=True)
class ExperimentDef:
    key: str
    directory: str
    label: str
    short_label: str
    model: str
    window: str
    feature: str
    task: str


SIX_CLASS_EXPERIMENTS = [
    ExperimentDef("mlp_1s_enhanced", "esf_csmlp_v2", "MLP 1 s Enhanced", "MLP-1s-E", "MLP", "1 s", "Enhanced", "six_class"),
    ExperimentDef("mlp_3s_enhanced", "esf_csmlp_v2_3sec", "MLP 3 s Enhanced", "MLP-3s-E", "MLP", "3 s", "Enhanced", "six_class"),
    ExperimentDef("mlp_3s_mean_std", "mlp_mean_std_3sec", "MLP 3 s Mean/Std", "MLP-3s-MS", "MLP", "3 s", "Mean/Std", "six_class"),
    ExperimentDef("rf_3s_enhanced", "enhanced_rf_3sec", "RF 3 s Enhanced", "RF-3s-E", "RF", "3 s", "Enhanced", "six_class"),
    ExperimentDef("xgb_3s_enhanced", "enhanced_xgb_3sec", "XGB 3 s Enhanced", "XGB-3s-E", "XGB", "3 s", "Enhanced", "six_class"),
]

BINARY_EXPERIMENTS = [
    ExperimentDef("binary_mlp_3s", "failure_vs_nofailure_mlp_3sec", "MLP 3 s", "MLP", "MLP", "3 s", "Enhanced", "binary"),
    ExperimentDef("binary_rf_3s", "failure_vs_nofailure_rf_3sec", "RF 3 s", "RF", "RF", "3 s", "Enhanced", "binary"),
    ExperimentDef("binary_xgb_3s", "failure_vs_nofailure_xgb_3sec", "XGB 3 s", "XGB", "XGB", "3 s", "Enhanced", "binary"),
]

BEST_MODEL_KEY = "xgb_3s_enhanced"
BEST_MODEL_DIR = "enhanced_xgb_3sec"
CLASS_ORDER = ["GPS", "RC", "Accelerometer", "Gyroscope", "Compass", "Barometer"]
CLASS_DISPLAY = {
    "GPS": "GPS",
    "RC": "RC",
    "Accelerometer": "Accel.",
    "Gyroscope": "Gyro.",
    "Compass": "Compass",
    "Barometer": "Baro.",
}


def mm_to_inches(value_mm: float) -> float:
    return value_mm / MM_PER_INCH


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    ensure_dir(outdir)
    for fmt in formats:
        outpath = outdir / f"{stem}.{fmt.lower()}"
        if fmt.lower() == "png":
            fig.savefig(outpath, dpi=dpi)
        else:
            fig.savefig(outpath)
    plt.close(fig)


def font_family() -> list[str]:
    return ["Times New Roman", "Liberation Serif", "DejaVu Serif"]


def apply_scienceplots_theme(style_name: str) -> dict[str, object]:
    if style_name != "science_strong":
        raise ValueError(f"Unsupported style preset: {style_name}")

    plt.style.use(["science", "grid", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": font_family(),
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.18,
            "grid.color": "#D4DCE8",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )
    return {
        "metric_colors": {
            "Accuracy": ACC_COLOR,
            "Macro-F1": F1_COLOR,
            "ROC-AUC": AUC_COLOR,
            "Precision": "#9CBCE5",
            "Recall": "#E7A09A",
            "F1": "#88C8BD",
        },
        "model_colors": {
            "MLP": "#D97A9A",
            "RF": "#7C83D6",
            "XGB": "#48AFC6",
        },
        "group_band_colors": {
            "Window": "#EEF5FD",
            "Feature": "#FFF3DE",
            "Binary": "#EEF7F3",
        },
        "annotation_size": 7.3,
        "label_box": dict(boxstyle="round,pad=0.16", fc="white", ec="#C9D4E3", alpha=0.94),
    }


def experiment_files(root: Path, directory: str) -> dict[str, Path]:
    exp_root = root / directory
    return {
        "root": exp_root,
        "summary": exp_root / "summary.json",
        "split_metrics": exp_root / "split_metrics.csv",
        "confusion": exp_root / "confusion_matrix_mean.csv",
    }


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment_results(outputs_root: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for exp in [*SIX_CLASS_EXPERIMENTS, *BINARY_EXPERIMENTS]:
        files = experiment_files(outputs_root, exp.directory)
        if not files["summary"].exists():
            raise FileNotFoundError(f"Missing summary file: {files['summary']}")
        if not files["split_metrics"].exists():
            raise FileNotFoundError(f"Missing split metrics file: {files['split_metrics']}")

        summary = load_json(files["summary"])
        split_metrics = pd.read_csv(files["split_metrics"])
        payload = {
            "definition": exp,
            "summary": summary,
            "split_metrics": split_metrics,
        }
        if files["confusion"].exists():
            payload["confusion"] = pd.read_csv(files["confusion"], index_col=0)
        results[exp.key] = payload
    return results


def build_plot_tables(results: dict[str, dict]) -> dict[str, pd.DataFrame]:
    six_rows = []
    for exp in SIX_CLASS_EXPERIMENTS:
        summary = results[exp.key]["summary"]
        six_rows.append(
            {
                "key": exp.key,
                "label": exp.label,
                "short_label": exp.short_label,
                "model": exp.model,
                "window": exp.window,
                "feature": exp.feature,
                "accuracy": summary["metrics_mean"]["accuracy"] * 100.0,
                "macro_f1": summary["metrics_mean"]["macro_f1"] * 100.0,
                "std_f1": summary["metrics_std"]["macro_f1"] * 100.0,
            }
        )
    six_df = pd.DataFrame(six_rows)

    binary_rows = []
    for exp in BINARY_EXPERIMENTS:
        summary = results[exp.key]["summary"]
        metrics_mean = summary["metrics_mean"]
        binary_rows.append(
            {
                "key": exp.key,
                "label": exp.label,
                "model": exp.model,
                "accuracy": metrics_mean["accuracy"] * 100.0,
                "f1": metrics_mean["f1"] * 100.0,
                "roc_auc": metrics_mean["roc_auc"] * 100.0,
            }
        )
    binary_df = pd.DataFrame(binary_rows)

    ablation_rows = [
        {"group": "Window", "item": "1 s", "accuracy": six_df.loc[six_df["key"] == "mlp_1s_enhanced", "accuracy"].iloc[0], "macro_f1": six_df.loc[six_df["key"] == "mlp_1s_enhanced", "macro_f1"].iloc[0]},
        {"group": "Window", "item": "3 s", "accuracy": six_df.loc[six_df["key"] == "mlp_3s_enhanced", "accuracy"].iloc[0], "macro_f1": six_df.loc[six_df["key"] == "mlp_3s_enhanced", "macro_f1"].iloc[0]},
        {"group": "Feature", "item": "Mean/Std", "accuracy": six_df.loc[six_df["key"] == "mlp_3s_mean_std", "accuracy"].iloc[0], "macro_f1": six_df.loc[six_df["key"] == "mlp_3s_mean_std", "macro_f1"].iloc[0]},
        {"group": "Feature", "item": "Enhanced", "accuracy": six_df.loc[six_df["key"] == "mlp_3s_enhanced", "accuracy"].iloc[0], "macro_f1": six_df.loc[six_df["key"] == "mlp_3s_enhanced", "macro_f1"].iloc[0]},
    ]
    ablation_df = pd.DataFrame(ablation_rows)

    return {"six_class": six_df, "binary": binary_df, "ablation": ablation_df}


def add_panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        0.0,
        1.04,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="semibold",
        color="#334155",
    )


def add_group_band(ax: plt.Axes, x_start: float, x_end: float, label: str, color: str) -> None:
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_patch(
        plt.Rectangle(
            (x_start, -0.22),
            x_end - x_start,
            0.12,
            transform=trans,
            facecolor=color,
            edgecolor="none",
            alpha=0.95,
            clip_on=False,
            zorder=0,
        )
    )
    ax.text(
        (x_start + x_end) / 2.0,
        -0.16,
        label,
        transform=trans,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#334155",
    )


def add_bar_value_labels(ax: plt.Axes, rects, fontsize: float, y_offset: float = 1.0, top_padding: float = 1.2) -> None:
    y_min, y_max = ax.get_ylim()
    for rect in rects:
        height = rect.get_height()
        target_y = min(height + y_offset, y_max - top_padding)
        va = "bottom" if target_y > height else "top"
        offset_points = 0 if va == "bottom" else -1
        ax.annotate(
            f"{height:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2.0, target_y),
            xytext=(0, offset_points),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=fontsize,
            color="#374151",
        )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def bar_kwargs(color: str, edgecolor: str | None = None, linewidth: float = 0.6) -> dict[str, object]:
    return {
        "color": color,
        "edgecolor": edgecolor if edgecolor is not None else color,
        "linewidth": linewidth,
        "alpha": 0.96,
    }


def plot_overall_figure(
    tables: dict[str, pd.DataFrame],
    outdir: Path,
    formats: list[str],
    dpi: int,
    palette: dict[str, object],
) -> None:
    metric_colors = palette["metric_colors"]
    band_colors = palette["group_band_colors"]
    annotation_size = palette["annotation_size"]

    fig = plt.figure(figsize=(mm_to_inches(178), mm_to_inches(116)))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.02, 0.98], hspace=0.32, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :2])
    ax_c = fig.add_subplot(gs[1, 2:])

    six_df = tables["six_class"]
    x = np.arange(len(six_df))
    width = 0.34
    bars_acc = ax_a.bar(
        x - width / 2,
        six_df["accuracy"],
        width,
        **bar_kwargs(metric_colors["Accuracy"], edgecolor="#739DCD", linewidth=0.55),
        label="Accuracy",
    )
    bars_f1 = ax_a.bar(
        x + width / 2,
        six_df["macro_f1"],
        width,
        **bar_kwargs(metric_colors["Macro-F1"], edgecolor="#D7AA55", linewidth=0.55),
        label="Macro-F1",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["MLP-1s-E", "MLP-3s-E", "MLP-3s-MS", "RF-3s-E", "XGB-3s-E"])
    ax_a.tick_params(axis="x", pad=0.8)
    ax_a.set_ylabel("Score (%)")
    ax_a.set_ylim(30, 82)
    ax_a.legend(loc="upper left", ncol=2, handlelength=1.3, columnspacing=0.9, borderaxespad=0.2)
    add_bar_value_labels(ax_a, bars_acc, fontsize=annotation_size)
    add_bar_value_labels(ax_a, bars_f1, fontsize=annotation_size)
    style_axis(ax_a)
    add_panel_tag(ax_a, "(a) Six-class main results")

    abl_df = tables["ablation"]
    x2 = np.array([0.0, 1.15, 3.7, 5.05], dtype=float)
    bars_acc_b = ax_b.bar(
        x2 - width / 2,
        abl_df["accuracy"],
        width,
        **bar_kwargs(metric_colors["Accuracy"], edgecolor="#739DCD", linewidth=0.55),
    )
    bars_f1_b = ax_b.bar(
        x2 + width / 2,
        abl_df["macro_f1"],
        width,
        **bar_kwargs(metric_colors["Macro-F1"], edgecolor="#D7AA55", linewidth=0.55),
    )
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(abl_df["item"].tolist())
    ax_b.set_xlim(-0.55, 5.6)
    ax_b.tick_params(axis="x", pad=1.0)
    ax_b.set_ylabel("Score (%)")
    ax_b.set_ylim(37, 48)
    style_axis(ax_b)
    add_group_band(ax_b, -0.55, 1.7, "Window", band_colors["Window"])
    add_group_band(ax_b, 3.15, 5.6, "Feature", band_colors["Feature"])
    ax_b.text(
        0.98,
        0.92,
        "Feature comparison under 3 s window",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#6B7280",
    )
    add_panel_tag(ax_b, "(b) Window and feature ablation")

    binary_df = tables["binary"]
    x3 = np.arange(len(binary_df))
    width_c = 0.24
    bars_acc_c = ax_c.bar(
        x3 - width_c,
        binary_df["accuracy"],
        width_c,
        **bar_kwargs(metric_colors["Accuracy"], edgecolor="#739DCD", linewidth=0.55),
        label="Accuracy",
    )
    bars_f1_c = ax_c.bar(
        x3,
        binary_df["f1"],
        width_c,
        **bar_kwargs(metric_colors["F1"], edgecolor="#D7AA55", linewidth=0.55),
        label="F1",
    )
    bars_auc_c = ax_c.bar(
        x3 + width_c,
        binary_df["roc_auc"],
        width_c,
        **bar_kwargs(metric_colors["ROC-AUC"], edgecolor="#7EB57A", linewidth=0.55),
        label="ROC-AUC",
    )
    ax_c.set_xticks(x3)
    ax_c.set_xticklabels(binary_df["model"].tolist())
    ax_c.tick_params(axis="x", pad=1.0)
    ax_c.set_ylabel("Score (%)")
    ax_c.set_ylim(60, 100)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=3, handlelength=1.1, columnspacing=0.8, borderaxespad=0.15)
    add_bar_value_labels(ax_c, bars_acc_c, fontsize=annotation_size - 0.1, y_offset=1.0, top_padding=1.8)
    add_bar_value_labels(ax_c, bars_f1_c, fontsize=annotation_size - 0.1, y_offset=1.0, top_padding=1.8)
    add_bar_value_labels(ax_c, bars_auc_c, fontsize=annotation_size - 0.1, y_offset=1.0, top_padding=1.8)
    style_axis(ax_c)
    ax_c.text(
        0.5,
        -0.15,
        "ROC-AUC scaled to 100",
        transform=ax_c.transAxes,
        ha="center",
        va="top",
        fontsize=6.9,
        color="#6B7280",
        clip_on=False,
    )
    add_panel_tag(ax_c, "(c) Binary overview")

    save_figure(fig, outdir, "fig_1_overall_experimental_comparison", formats, dpi)


def plot_tradeoff_figure(
    tables: dict[str, pd.DataFrame],
    outdir: Path,
    formats: list[str],
    dpi: int,
    palette: dict[str, object],
) -> None:
    model_colors = palette["model_colors"]
    label_box = palette["label_box"]

    df = tables["six_class"].copy()
    marker_map = {
        ("MLP", "1 s", "Enhanced"): "o",
        ("MLP", "3 s", "Enhanced"): "s",
        ("MLP", "3 s", "Mean/Std"): "D",
        ("RF", "3 s", "Enhanced"): "^",
        ("XGB", "3 s", "Enhanced"): "P",
    }
    fig, ax = plt.subplots(figsize=(mm_to_inches(168), mm_to_inches(102)))

    x_min = max(df["std_f1"].min() - 0.55, 0)
    x_max = df["std_f1"].max() + 0.45
    y_min = df["macro_f1"].min() - 3.2
    y_max = df["macro_f1"].max() + 2.8

    for _, row in df.iterrows():
        marker = marker_map[(row["model"], row["window"], row["feature"])]
        ax.scatter(
            row["std_f1"],
            row["macro_f1"],
            s=220,
            color=model_colors[row["model"]],
            marker=marker,
            edgecolor="#233142",
            linewidth=1.2,
            alpha=0.95,
            zorder=3,
        )
        ax.annotate(
            row["short_label"],
            xy=(row["std_f1"], row["macro_f1"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.6,
            bbox=label_box,
        )

    x_mean = df["std_f1"].mean()
    y_mean = df["macro_f1"].mean()
    ax.axvline(x_mean, color="#606C7A", linestyle="--", linewidth=1.0)
    ax.axhline(y_mean, color="#606C7A", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Std(Macro-F1) (%)")
    ax.set_ylabel("Mean Macro-F1 (%)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    style_axis(ax)
    ax.minorticks_on()
    ax.text(
        0.965,
        0.94,
        "Higher is better; lower dispersion is preferred",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#6B7280",
    )

    handles = []
    labels = []
    for model, color in model_colors.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="#233142", markersize=8))
        labels.append(model)
    ax.legend(handles, labels, loc="upper left", title="Model", frameon=False, borderaxespad=0.2)

    save_figure(fig, outdir, "fig_2_performance_stability_tradeoff", formats, dpi)


def aggregate_class_metrics(summary: dict) -> pd.DataFrame:
    split_results = summary["split_results"]
    rows = []
    for class_name in CLASS_ORDER:
        precision_vals = []
        recall_vals = []
        f1_vals = []
        support_vals = []
        for split in split_results:
            report = split["classification_report"][class_name]
            precision_vals.append(report["precision"] * 100.0)
            recall_vals.append(report["recall"] * 100.0)
            f1_vals.append(report["f1-score"] * 100.0)
            support_vals.append(report["support"])
        rows.append(
            {
                "class_name": class_name,
                "precision": float(np.mean(precision_vals)),
                "recall": float(np.mean(recall_vals)),
                "f1": float(np.mean(f1_vals)),
                "support_std": float(np.std(support_vals)),
                "support_mean": float(np.mean(support_vals)),
            }
        )
    return pd.DataFrame(rows)


def build_heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_heat",
        ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"],
    )


def plot_best_model_figure(
    results: dict[str, dict],
    outdir: Path,
    formats: list[str],
    dpi: int,
    palette: dict[str, object],
) -> None:
    annotation_size = palette["annotation_size"]
    class_metric_styles = {
        "Precision": {"fill": "#6F8FB8", "edge": "#58739A"},
        "Recall": {"fill": "#D98C6C", "edge": "#B47257"},
        "F1": {"fill": "#8FB7A3", "edge": "#739684"},
    }

    best = results[BEST_MODEL_KEY]
    summary = best["summary"]
    cm_df = best["confusion"].loc[CLASS_ORDER, CLASS_ORDER]
    class_df = aggregate_class_metrics(summary)

    fig = plt.figure(figsize=(mm_to_inches(178), mm_to_inches(116)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.22, 1.28], wspace=0.28)
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_cls = fig.add_subplot(gs[0, 1])

    im = ax_cm.imshow(cm_df.values, cmap=build_heatmap_cmap(), aspect="equal")
    ax_cm.set_xticks(np.arange(len(CLASS_ORDER)))
    ax_cm.set_yticks(np.arange(len(CLASS_ORDER)))
    ax_cm.set_xticklabels([CLASS_DISPLAY[name] for name in CLASS_ORDER], rotation=28, ha="right")
    ax_cm.set_yticklabels([CLASS_DISPLAY[name] for name in CLASS_ORDER])
    ax_cm.set_xlabel("Predicted class")
    ax_cm.set_ylabel("True class")
    add_panel_tag(ax_cm, "(a) Mean confusion matrix")

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            value = cm_df.iloc[i, j]
            ax_cm.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8.1,
                color="white" if value > cm_df.values.max() * 0.52 else "#1F2937",
            )

    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.048, pad=0.03)
    cbar.ax.tick_params(labelsize=8.0)

    x = np.arange(len(class_df))
    width = 0.22
    bars_p = ax_cls.bar(
        x - width,
        class_df["precision"],
        width,
        **bar_kwargs(class_metric_styles["Precision"]["fill"], edgecolor=class_metric_styles["Precision"]["edge"], linewidth=0.55),
        label="Precision",
    )
    bars_r = ax_cls.bar(
        x,
        class_df["recall"],
        width,
        **bar_kwargs(class_metric_styles["Recall"]["fill"], edgecolor=class_metric_styles["Recall"]["edge"], linewidth=0.55),
        label="Recall",
    )
    bars_f = ax_cls.bar(
        x + width,
        class_df["f1"],
        width,
        **bar_kwargs(class_metric_styles["F1"]["fill"], edgecolor=class_metric_styles["F1"]["edge"], linewidth=0.55),
        label="F1",
    )
    ax_cls.set_xticks(x)
    ax_cls.set_xticklabels([CLASS_DISPLAY[name] for name in class_df["class_name"]], rotation=16, ha="right")
    ax_cls.set_ylabel("Score (%)")
    ax_cls.set_ylim(0, 108)
    ax_cls.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.995),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.7,
        borderaxespad=0.1,
    )
    style_axis(ax_cls)
    add_panel_tag(ax_cls, "(b) Class-level metrics")

    add_bar_value_labels(ax_cls, bars_f, fontsize=annotation_size - 0.3, y_offset=0.8, top_padding=4.0)

    ax_cls.axhline(
        class_df["f1"].mean(),
        color="#97A5B6",
        linestyle="--",
        linewidth=0.9,
        alpha=0.9,
    )
    save_figure(fig, outdir, "fig_3_best_model_category_analysis", formats, dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate composite paper figures from existing UAV experiment outputs.")
    parser.add_argument("--outputs-root", type=str, default="outputs", help="Root directory containing experiment outputs.")
    parser.add_argument("--outdir", type=str, default="outputs/paper_composite_figures", help="Output directory for exported figures.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], help="Export formats, e.g. pdf svg png")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for PNG export.")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["overall", "tradeoff", "best_model", "all"],
        default=["all"],
        help="Choose which figures to export.",
    )
    parser.add_argument("--style", choices=["science_strong"], default="science_strong", help="Composite figure style preset.")
    return parser.parse_args()


def selected_figures(requested: list[str]) -> list[str]:
    if "all" in requested:
        return ["overall", "tradeoff", "best_model"]
    seen: list[str] = []
    for item in requested:
        if item not in seen:
            seen.append(item)
    return seen


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    palette = apply_scienceplots_theme(args.style)
    results = load_experiment_results(outputs_root)
    tables = build_plot_tables(results)

    chosen = selected_figures(args.figures)
    generated: list[str] = []

    if "overall" in chosen:
        plot_overall_figure(tables, outdir, args.formats, args.dpi, palette)
        generated.extend([f"fig_1_overall_experimental_comparison.{fmt}" for fmt in args.formats])

    if "tradeoff" in chosen:
        plot_tradeoff_figure(tables, outdir, args.formats, args.dpi, palette)
        generated.extend([f"fig_2_performance_stability_tradeoff.{fmt}" for fmt in args.formats])

    if "best_model" in chosen:
        plot_best_model_figure(results, outdir, args.formats, args.dpi, palette)
        generated.extend([f"fig_3_best_model_category_analysis.{fmt}" for fmt in args.formats])

    print(f"Figures written to: {outdir.resolve()}")
    print("Generated files:")
    for name in generated:
        print(f" - {name}")


if __name__ == "__main__":
    main()
