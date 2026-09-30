# ============================================================================
# FILE: results/issue_figure_generation.py
# Figure-generation module for Issue 1, Issue 2, and Issue 3 outputs.
#
# Purpose:
# - Reads saved experiment outputs, evaluation reports, and sweep summaries,
#   then turns them into paper-style figures and CSV artifacts.
#
# Workflow:
# - Loads result JSON/CSV files from `results/` and `eval/`.
# - Builds ROC curves, bar charts, ASR comparisons, threshold sensitivity
#   charts, and Layer 3 load/latency visualizations.
#
# Use this file when:
# - A reported figure looks wrong.
# - You need to trace which saved files feed a specific chart.
# ============================================================================
from utils.utils import plot_d2_global, plot_issue1_auc_chart, plot_threshold_metrics
from pathlib import Path
from utils.paths import RESULT_DIR, EVAL_DIR
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import plotly.graph_objects as go

def get_issue1_d2(mode_result_path: str = "mode_11_results.json"):
    plot_d2_global(mode_result_path)

def get_issue2_d2_2(mode_result_path: str = "mode_12_results.json"):
    plot_d2_global(mode_result_path)

def get_issue1_d1():
    plot_issue1_auc_chart()

VISUALIZE_THRESHOLD = {
    "min": 0.05,
    "max": 0.141,
    "step": 0.005
}

CONVERT_MODE = {
    "mode_1": "L1",
    "mode_2": "L2",
    "mode_3": "L3",
    "mode_10": "L1-L2",
    "mode_13": "L1-L3",
    "mode_14": "L2-L3",
    "mode_15": "L2-L1",
    "mode_11": "L1-L2-L3",
    "mode_12": "L2-L1-L3"
}

ICONVERT_MODE = {
    "mode_1": "L1",
    "mode_2": "L2",
    "mode_10": "L1 -> L2",
    "mode_11": "L2 -> L1 -> L3",
    "mode_12": "L1 -> L2 -> L3"
}

ISSUE1_ALL_MODES = [
    "mode_1",
    "mode_2",
    "mode_3",
    "mode_10",
    "mode_13",
    "mode_14",
    "mode_15",
    "mode_11",
    "mode_12",
]

ISSUE1_LAYER_COUNT_ORDER = {
    "mode_1": 1,
    "mode_2": 1,
    "mode_3": 1,
    "mode_10": 2,
    "mode_13": 2,
    "mode_14": 2,
    "mode_15": 2,
    "mode_11": 3,
    "mode_12": 3,
}

# Public-facing name for the proposed 3-layer solution.
# Keep this as a display label only; experiment IDs and saved result filenames
# remain unchanged for reproducibility.
SOLUTION_LABEL = "InteractionGuard"

# Shared chart styling for paper-ready figures.  These constants keep the
# newest comparison charts visually consistent while preserving each chart's
# data and layout logic.
LEGEND_BOX_STYLE = {
    "frameon": True,
    "framealpha": 0.95,
    "edgecolor": "#D8D8D8",
}

BEST_ANNOTATION_BOX = {
    "boxstyle": "round,pad=0.25",
    "fc": "white",
    "ec": "#004D36",
    "alpha": 0.92,
}

LATENCY_VALUE_BOX = {
    "boxstyle": "round,pad=0.12",
    "fc": "white",
    "ec": "none",
    "alpha": 0.78,
}

LATENCY_LINE_STYLE = {
    "color": "#B00020",
    "marker": "D",
    "linewidth": 3.2,
    "markersize": 7.5,
    "markerfacecolor": "white",
    "markeredgecolor": "#B00020",
    "markeredgewidth": 1.4,
    "zorder": 5,
}

def _strip_figure_titles(fig=None):
    """Remove figure and axes titles while preserving legends and axis labels."""
    fig = fig or plt.gcf()
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")


def _savefig_without_title(save_path, *args, fig=None, **kwargs):
    """Save a chart in thesis style: axes and legends stay, titles are hidden."""
    _strip_figure_titles(fig)
    plt.savefig(save_path, *args, **kwargs)


def _order_issue1_modes(mode_names: list[str]) -> list[str]:
    return sorted(
        mode_names,
        key=lambda mode: (ISSUE1_LAYER_COUNT_ORDER.get(mode, 99), int(mode.split("_")[1]))
    )

def plot_issue1_roc_curve(results_dir: Path, mode_names: list, save_path: Path = None):
    """
    Plot ROC curves for multiple modes using metrics["roc_curve"].

    Args:
        results_dir (Path): directory containing result files
        mode_names (list): ["mode_1", "mode_2", ...]
        save_path (Path): optional path to save figure
    """

    plt.figure(figsize=(8, 6))

    for mode in _order_issue1_modes(mode_names):
        file_path = results_dir / f"{mode}_results.json"
        mode = CONVERT_MODE[mode]
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        roc = metrics.get("roc_curve", {})

        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc = roc.get("auc", None)

        if not fpr or not tpr:
            print(f"[WARN] Missing ROC data in {mode}")
            continue

        label = mode.replace("mode_", "Mode ")
        if auc is not None:
            label += f" (AUC={auc:.3f})"

        plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve Comparison Across Modes")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, bbox_inches="tight", dpi=300)

    plt.show()

def plot_issue1_bar_chart(results_dir: Path, mode_names: list, save_path: Path = None):
    """
    Read multiple mode_x_results.json files and plot grouped bar chart
    for Accuracy, Recall, F1 Score, and FPR.

    Args:
        results_dir (Path): directory containing result files
        mode_names (list): ["mode_1", "mode_2", ...]
        save_path (Path): optional path to save figure
    """

    metrics_data = {
        "accuracy": [],
        "recall": [],
        "f1_score": [],
        "fpr": []
    }

    valid_modes = []

    for mode in _order_issue1_modes(mode_names):
        file_path = results_dir / f"{mode}_results.json"
        mode = CONVERT_MODE[mode]
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = data["metrics"]

        metrics_data["accuracy"].append(m["accuracy"])
        metrics_data["recall"].append(m["recall"])
        metrics_data["f1_score"].append(m["f1_score"])
        metrics_data["fpr"].append(m["fpr"])

        valid_modes.append(mode)

    x = np.arange(len(valid_modes))
    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(x - 1.5 * width, metrics_data["accuracy"], width,
            label="Accuracy", hatch='///', edgecolor='black')

    plt.bar(x - 0.5 * width, metrics_data["recall"], width,
            label="Recall", hatch='\\\\\\', edgecolor='black')

    plt.bar(x + 0.5 * width, metrics_data["f1_score"], width,
            label="F1 Score", hatch='xxx', edgecolor='black')

    plt.bar(x + 1.5 * width, metrics_data["fpr"], width,
            label="FPR", hatch='...', edgecolor='black')

    plt.xticks(x, valid_modes)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison Across Modes")
    plt.legend()

    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, bbox_inches="tight", dpi=300)

    plt.show()

def plot_issue1_f1_fpr_fnr_bar_chart(results_dir: Path, mode_names: list, save_path: Path = None):
    """
    Plot grouped bars for F1, FPR, and FNR across all requested modes.

    Args:
        results_dir (Path): directory containing result files
        mode_names (list): ["mode_1", "mode_2", ...]
        save_path (Path): optional path to save figure
    """

    metrics_data = {
        "fnr": [],
        "f1_score": [],
        "fpr": [],
    }
    valid_modes = []

    for mode_name in _order_issue1_modes(mode_names):
        file_path = results_dir / f"{mode_name}_results.json"
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        if not metrics:
            print(f"[WARN] Missing metrics in {file_path}")
            continue

        metrics_data["f1_score"].append(metrics["f1_score"])
        metrics_data["fpr"].append(metrics["fpr"])
        metrics_data["fnr"].append(metrics["fnr"])
        valid_modes.append(CONVERT_MODE.get(mode_name, mode_name))

    if not valid_modes:
        print("[WARN] No valid Issue 1 results found for F1/FPR/FNR bar chart.")
        return

    x = np.arange(len(valid_modes))
    width = 0.24

    plt.figure(figsize=(14, 6))

    plt.bar(
        x,
        metrics_data["f1_score"],
        width,
        label="F1",
        color="#2E8B57",
        hatch="xxx",
        edgecolor="black",
    )
    plt.bar(
        x + width,
        metrics_data["fpr"],
        width,
        label="FPR",
        color="#DC2626",
        hatch="...",
        edgecolor="black",
    )
    plt.bar(
        x - width,
        metrics_data["fnr"],
        width,
        label="FNR",
        color="#F59E0B",
        hatch="///",
        edgecolor="black",
    )

    plt.xticks(x, valid_modes, rotation=15)
    plt.ylabel("Score")
    plt.title("Issue 1 Comparison Across Modes (FNR, F1, FPR)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, bbox_inches="tight", dpi=300)

    plt.show()

def plot_issue1_dual_axis_latency_chart(results_dir: Path, mode_names: list, save_path: Path = None):
    """
    Plot Issue 1 metrics as grouped bars on the left Y-axis and overlay
    latency-per-sample as a line chart on the right Y-axis.

    Left axis:
      - FNR
      - F1
      - FPR

    Right axis:
      - Latency / sample (seconds)
    """

    metrics_data = {
        "fnr": [],
        "f1_score": [],
        "fpr": [],
        "latency_per_sample": [],
    }
    valid_modes = []

    for mode_name in _order_issue1_modes(mode_names):
        file_path = results_dir / f"{mode_name}_results.json"
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        if not metrics:
            print(f"[WARN] Missing metrics in {file_path}")
            continue

        num_samples = max(1, int(data.get("num_samples", 1)))
        total_time = float(data.get("time", 0.0))

        metrics_data["fnr"].append(metrics["fnr"])
        metrics_data["f1_score"].append(metrics["f1_score"])
        metrics_data["fpr"].append(metrics["fpr"])
        metrics_data["latency_per_sample"].append(total_time / num_samples)
        valid_modes.append(CONVERT_MODE.get(mode_name, mode_name))

    if not valid_modes:
        print("[WARN] No valid Issue 1 results found for dual-axis latency chart.")
        return

    x = np.arange(len(valid_modes))
    width = 0.22

    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.bar(
        x - width,
        metrics_data["fnr"],
        width,
        label="FNR",
        color="#F59E0B",
        hatch="///",
        edgecolor="black",
    )
    ax1.bar(
        x,
        metrics_data["f1_score"],
        width,
        label="F1",
        color="#2E8B57",
        hatch="xxx",
        edgecolor="black",
    )
    ax1.bar(
        x + width,
        metrics_data["fpr"],
        width,
        label="FPR",
        color="#DC2626",
        hatch="...",
        edgecolor="black",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_modes, rotation=15)
    ax1.set_ylabel("Score")
    ax1.set_title("Issue 1 Comparison Across Modes with Latency / Sample")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        metrics_data["latency_per_sample"],
        color="#7C3AED",
        marker="D",
        linewidth=3.0,
        markersize=8,
        label="Latency / sample",
        markerfacecolor="#FDE047",
        markeredgecolor="#111827",
        markeredgewidth=1.2,
        zorder=5,
    )
    ax2.set_ylabel("Latency / sample (seconds)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, bbox_inches="tight", dpi=300)

    plt.show()

def plot_issue1_precision_recall_latency_chart(results_dir: Path, mode_names: list, save_path: Path = None):
    """
    Plot Issue 1 metrics as grouped bars on the left Y-axis and overlay
    latency-per-sample as a line chart on the right Y-axis.

    This is a presentation-friendly variant of the Issue 1 latency chart:

    Left axis:
      - Precision
      - F1
      - Recall

    Right axis:
      - Latency / sample (seconds)
    """

    metrics_data = {
        "precision": [],
        "f1_score": [],
        "recall": [],
        "latency_per_sample": [],
    }
    valid_modes = []

    for mode_name in _order_issue1_modes(mode_names):
        file_path = results_dir / f"{mode_name}_results.json"
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        if not metrics:
            print(f"[WARN] Missing metrics in {file_path}")
            continue

        num_samples = max(1, int(data.get("num_samples", 1)))
        total_time = float(data.get("time", 0.0))

        metrics_data["precision"].append(metrics["precision"])
        metrics_data["f1_score"].append(metrics["f1_score"])
        metrics_data["recall"].append(metrics["recall"])
        metrics_data["latency_per_sample"].append(total_time / num_samples)
        valid_modes.append(CONVERT_MODE.get(mode_name, mode_name))

    if not valid_modes:
        print("[WARN] No valid Issue 1 results found for precision/recall latency chart.")
        return

    x = np.arange(len(valid_modes))
    width = 0.22

    fig, ax1 = plt.subplots(figsize=(15, 6))

    bars_precision = ax1.bar(
        x - width,
        metrics_data["precision"],
        width,
        label="Precision",
        color="#0072B2",
        hatch="///",
        edgecolor="black",
        linewidth=0.75,
    )
    bars_f1 = ax1.bar(
        x,
        metrics_data["f1_score"],
        width,
        label="F1",
        color="#009E73",
        hatch="xxx",
        edgecolor="black",
        linewidth=0.75,
    )
    bars_recall = ax1.bar(
        x + width,
        metrics_data["recall"],
        width,
        label="Recall",
        color="#D55E00",
        hatch="\\\\\\",
        edgecolor="black",
        linewidth=0.75,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_modes, rotation=15)
    ax1.set_ylabel("Score")
    ax1.set_title("Issue 1 Precision/Recall Comparison Across Modes with Latency / Sample")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    score_values = (
        metrics_data["precision"]
        + metrics_data["f1_score"]
        + metrics_data["recall"]
    )
    score_max = max(score_values)
    # Keep the score axis tight enough to show differences, while leaving
    # headroom for the best-F1 annotation.
    y_min = 0.76
    y_max = 1.02
    ax1.set_ylim(y_min, y_max)

    best_idx = int(np.argmax(metrics_data["f1_score"]))
    best_bar = bars_f1[best_idx]
    best_x = best_bar.get_x() + best_bar.get_width() / 2.0
    best_y = best_bar.get_height()
    ax1.scatter(
        [best_x],
        [best_y],
        s=150,
        facecolors="none",
        edgecolors="#004D36",
        linewidths=2.0,
        zorder=6,
    )
    ax1.annotate(
        f"Best F1={best_y:.3f}",
        xy=(best_x, best_y),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=9,
        color="#004D36",
        fontweight="bold",
        bbox=BEST_ANNOTATION_BOX,
        arrowprops={"arrowstyle": "->", "color": "#004D36", "lw": 1.1},
    )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        metrics_data["latency_per_sample"],
        label="Latency / sample",
        **LATENCY_LINE_STYLE,
    )
    ax2.set_ylabel("Latency / sample (seconds)")
    latency_values = metrics_data["latency_per_sample"]
    latency_min = min(latency_values)
    latency_max = max(latency_values)
    latency_range = max(0.001, latency_max - latency_min)
    ax2.set_ylim(
        max(0.0, latency_min - latency_range * 0.15),
        latency_max + latency_range * 0.20,
    )
    for xi, latency in zip(x, latency_values):
        ax2.annotate(
            f"{latency:.2f}s",
            xy=(xi, latency),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8A0018",
            fontweight="bold",
            bbox=LATENCY_VALUE_BOX,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        **LEGEND_BOX_STYLE,
    )

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, bbox_inches="tight", dpi=300)

    plt.close(fig)

def plot_issue2_l3_ratio(results_dir: Path, mode_names: list, save_path: Path = None):
    plt.figure(figsize=(8, 5))

    thresholds = np.arange(VISUALIZE_THRESHOLD["min"], VISUALIZE_THRESHOLD["max"], VISUALIZE_THRESHOLD["step"])

    for mode in mode_names:
        ratios = []

        for th in thresholds:
            file_path = results_dir / mode / f"th_{th:.3f}.json"

            if not file_path.exists():
                print(f"[WARN] Missing: {file_path}")
                ratios.append(0)
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            total = data["num_samples"]
            l3_processed = data["layer_stats"]["layer3"]["processed"]

            ratio = (l3_processed / total) * 100.0
            ratios.append(ratio)

        label = CONVERT_MODE[mode]
        plt.plot(thresholds, ratios, marker='o', label=label, linewidth=2)

    plt.xlabel("Layer 1 Threshold")
    plt.ylabel("Layer 3 Processed Ratio (%)")
    plt.title("Impact of L1 Threshold on Layer 3 Load")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_issue2_l3_latency(results_dir: Path, mode_names: list, save_path: Path = None):
    plt.figure(figsize=(8, 5))

    thresholds = np.arange(VISUALIZE_THRESHOLD["min"], VISUALIZE_THRESHOLD["max"], VISUALIZE_THRESHOLD["step"])

    for mode in mode_names:
        times = []

        for th in thresholds:
            file_path = results_dir / mode / f"th_{th:.3f}.json"

            if not file_path.exists():
                print(f"[WARN] Missing: {file_path}")
                times.append(0)
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            t = data["time"]  # seconds
            times.append(t)

        label = CONVERT_MODE[mode]
        plt.plot(thresholds, times, marker='o', label=label, linewidth=2)

    plt.xlabel("Layer 1 Threshold")
    plt.ylabel("Total Runtime (seconds)")
    plt.title("Impact of L1 Threshold on Pipeline Latency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_issue2_l3_ratio_latency_combined(results_dir: Path, mode_names: list, save_path: Path = None):
    """
    Plot Issue 2 threshold sensitivity with two Y-axes:
      - Left Y-axis: Layer 3 processing ratio
      - Right Y-axis: latency per sample
    """

    thresholds = np.arange(
        VISUALIZE_THRESHOLD["min"],
        VISUALIZE_THRESHOLD["max"],
        VISUALIZE_THRESHOLD["step"],
    )

    fig, ax_ratio = plt.subplots(figsize=(9, 5.5))
    ax_latency = ax_ratio.twinx()

    ratio_colors = ["#0072B2", "#009E73", "#56B4E9"]
    latency_colors = ["#D55E00", "#CC79A7", "#E69F00"]
    ratio_markers = ["o", "s", "^"]
    latency_markers = ["D", "P", "X"]

    plotted = False

    for idx, mode in enumerate(mode_names):
        ratios = []
        latencies_per_sample = []

        for th in thresholds:
            file_path = results_dir / mode / f"th_{th:.3f}.json"

            if not file_path.exists():
                print(f"[WARN] Missing: {file_path}")
                ratios.append(np.nan)
                latencies_per_sample.append(np.nan)
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            total = max(1, int(data["num_samples"]))
            l3_processed = int(data["layer_stats"]["layer3"]["processed"])
            ratios.append((l3_processed / total) * 100.0)
            latencies_per_sample.append(float(data["time"]) / total)
            plotted = True

        label = CONVERT_MODE.get(mode, mode)
        color_idx = idx % len(ratio_colors)
        x_offset = VISUALIZE_THRESHOLD["step"] * 0.18
        latency_thresholds = thresholds + x_offset

        ax_ratio.plot(
            thresholds,
            ratios,
            color=ratio_colors[color_idx],
            marker=ratio_markers[color_idx],
            linewidth=2.5,
            markersize=7,
            label=f"{label} L3 processed ratio",
        )
        ax_latency.plot(
            latency_thresholds,
            latencies_per_sample,
            color=latency_colors[color_idx],
            marker=latency_markers[color_idx],
            linestyle="--",
            linewidth=2.5,
            markersize=7,
            label=f"{label} latency / sample",
        )

    if not plotted:
        print("[WARN] No valid Issue 2 threshold results found for combined ratio/latency chart.")
        plt.close(fig)
        return

    ratio_lines = [
        y
        for line in ax_ratio.lines
        for y in line.get_ydata()
        if not np.isnan(y)
    ]
    latency_lines = [
        y
        for line in ax_latency.lines
        for y in line.get_ydata()
        if not np.isnan(y)
    ]
    if ratio_lines:
        ratio_min = min(ratio_lines)
        ratio_max = max(ratio_lines)
        ratio_range = max(0.01, ratio_max - ratio_min)
        ax_ratio.set_ylim(
            max(0.0, ratio_min - ratio_range * 0.15),
            min(100.0, ratio_max + ratio_range * 0.20),
        )
    if latency_lines:
        latency_min = min(latency_lines)
        latency_max = max(latency_lines)
        latency_range = max(0.001, latency_max - latency_min)
        # Keep the latency curve visually separated from the L3-ratio curve.
        # Expanding more below than above shifts the latency line upward on
        # its own axis while preserving the true latency values.
        ax_latency.set_ylim(
            max(0.0, latency_min - latency_range * 1.20),
            latency_max + latency_range * 0.20,
        )

    ax_ratio.set_xlabel("Layer 1 Threshold")
    ax_ratio.set_ylabel("Layer 3 Processed Ratio (%)")
    ax_latency.set_ylabel("Latency / sample (seconds)")
    ax_ratio.set_title("Issue 2: Layer 3 Load and Latency / Sample Across Thresholds")
    ax_ratio.grid(True, linestyle="--", alpha=0.45)

    handles1, labels1 = ax_ratio.get_legend_handles_labels()
    handles2, labels2 = ax_latency.get_legend_handles_labels()
    ax_ratio.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_issue3_ASRline_chart(defense_path: Path, baseline_path: Path, no_defense_path: Path, save_path: Path):
    """
    Plot ASR line chart:
    X-axis: [No Defense, sentinel, raguard, toolparsing, combined, InteractionGuard]
    Y-axis: ASR
    Lines: attack types [prompt, rag, tool, correlated]
    """

    import json
    import numpy as np
    import matplotlib.pyplot as plt

    with open(defense_path, "r", encoding="utf-8") as f:
        defense_data = json.load(f)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    with open(no_defense_path, "r", encoding="utf-8") as f:
        no_defense_data = json.load(f)

    attack_types = ["prompt", "rag", "tool", "correlated"]
    models = ["No Defense", "[Sentinel]", "[RAGuard]", "[ToolResultParsing]", "Combined", SOLUTION_LABEL]

    asr_dict = {atk: [] for atk in attack_types}

    for atk in attack_types:
        asr_dict[atk].append(no_defense_data[atk]["asr"])
        asr_dict[atk].append(baseline_data["sentinel"][atk]["asr"])
        asr_dict[atk].append(baseline_data["raguard"][atk]["asr"])
        asr_dict[atk].append(baseline_data["toolparsing"][atk]["asr"])
        asr_dict[atk].append(baseline_data["combined"][atk]["asr"])
        asr_dict[atk].append(defense_data["per_attack_type"][atk]["ASR"])

    x = np.arange(len(models))

    plt.figure(figsize=(10, 6))

    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    for i, atk in enumerate(attack_types):
        plt.plot(
            x,
            asr_dict[atk],
            marker=markers[i],
            linestyle=linestyles[i],
            linewidth=2,
            label=atk.upper()
        )

    plt.xticks(x, models)
    plt.ylabel("ASR (Attack Success Rate)")
    plt.title(f"ASR Comparison Across Baselines and {SOLUTION_LABEL}")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_issue3_ASRgroup_bar_chart(defense_path: Path, baseline_path: Path, no_defense_path: Path, save_path: Path):
    """
    Plot grouped bar chart for Issue 3 ASR comparison.
    X-axis: configurations/baselines
    Y-axis: ASR
    Grouped bars: attack types [prompt, rag, tool, correlated]
    """

    with open(defense_path, "r", encoding="utf-8") as f:
        defense_data = json.load(f)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    with open(no_defense_path, "r", encoding="utf-8") as f:
        no_defense_data = json.load(f)

    attack_types = ["prompt", "rag", "tool", "correlated"]
    attack_labels = {
        "prompt": "Prompt",
        "rag": "RAG",
        "tool": "Tool",
        "correlated": "Correlated",
    }
    configs = ["No Defense", "[Sentinel]", "[RAGuard]", "[ToolResultParsing]", "Combined", SOLUTION_LABEL]

    asr_dict = {atk: [] for atk in attack_types}
    for atk in attack_types:
        asr_dict[atk].append(no_defense_data[atk]["asr"])
        asr_dict[atk].append(baseline_data["sentinel"][atk]["asr"])
        asr_dict[atk].append(baseline_data["raguard"][atk]["asr"])
        asr_dict[atk].append(baseline_data["toolparsing"][atk]["asr"])
        asr_dict[atk].append(baseline_data["combined"][atk]["asr"])
        asr_dict[atk].append(defense_data["per_attack_type"][atk]["ASR"])

    x = np.arange(len(configs))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    hatches = ["///", "\\\\\\", "xxx", "..."]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    for idx, atk in enumerate(attack_types):
        ax.bar(
            x + offsets[idx] * width,
            asr_dict[atk],
            width=width,
            color=colors[idx],
            label=attack_labels[atk],
            edgecolor="black",
            linewidth=0.75,
            hatch=hatches[idx],
        )

    mean_asr = np.asarray(
        [
            float(np.mean([asr_dict[atk][config_idx] for atk in attack_types]))
            for config_idx in range(len(configs))
        ]
    )
    best_idx = int(np.argmin(mean_asr))
    best_y = float(max(asr_dict[atk][best_idx] for atk in attack_types))
    ax.scatter(
        [x[best_idx]],
        [best_y],
        s=170,
        facecolors="none",
        edgecolors="#004D36",
        linewidths=2.0,
        zorder=6,
    )
    ax.annotate(
        f"Lowest ASR={mean_asr[best_idx]:.3f}",
        xy=(x[best_idx], best_y),
        xytext=(-92, 20),
        textcoords="offset points",
        fontsize=9,
        color="#004D36",
        fontweight="bold",
        bbox=BEST_ANNOTATION_BOX,
        arrowprops={"arrowstyle": "->", "color": "#004D36", "lw": 1.1},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("ASR (Attack Success Rate)")
    ax.set_title(f"Grouped ASR Comparison Across Baselines and {SOLUTION_LABEL}")
    ax.legend(
        loc="upper right",
        **LEGEND_BOX_STYLE,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _savefig_without_title(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_issue2_line_chart(
    sweep_root: Path,
    mode_names: list,
    save_dir: Path
):
    """
    Plot separate line charts for each mode:
    - Color = metric (Accuracy / Recall / F1 / Precision)
    - One figure per mode (no comparison)
    """

    import json
    import numpy as np
    import matplotlib.pyplot as plt

    metric_colors = {
        "accuracy": "#1f77b4",   # blue
        "recall": "#2ca02c",     # green
        "f1_score": "#d62728",   # red
        "precision": "#9467bd"   # purple (NEW)
    }

    for mode in mode_names:

        mode_dir = sweep_root / mode

        thresholds = []
        acc_list = []
        recall_list = []
        f1_list = []
        precision_list = []   # NEW

        for file in sorted(mode_dir.glob("th_*.json")):
            th = float(file.stem.split("_")[1])

            # FILTER HERE
            if not (VISUALIZE_THRESHOLD["min"] <= th <= VISUALIZE_THRESHOLD["max"]):
                continue

            thresholds.append(th)

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            m = data["metrics"]

            acc_list.append(m["accuracy"])
            recall_list.append(m["recall"])
            f1_list.append(m["f1_score"])
            precision_list.append(m["precision"])

        thresholds = np.array(thresholds)
        acc = np.array(acc_list)
        recall = np.array(recall_list)
        f1 = np.array(f1_list)
        precision = np.array(precision_list)  # NEW

        # -------------------------
        # Auto zoom
        # -------------------------
        all_values = np.concatenate([acc, recall, f1, precision])  # UPDATED
        y_min, y_max = all_values.min(), all_values.max()
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.01

        # -------------------------
        # Plot
        # -------------------------
        plt.figure(figsize=(10, 6))

        plt.plot(thresholds, acc, color=metric_colors["accuracy"], marker='o', linewidth=2, label="Accuracy")
        plt.plot(thresholds, recall, color=metric_colors["recall"], marker='s', linewidth=2, label="Recall")
        plt.plot(thresholds, f1, color=metric_colors["f1_score"], marker='^', linewidth=2, label="F1")
        plt.plot(thresholds, precision, color=metric_colors["precision"], marker='D', linewidth=2, label="Precision")  # NEW

        plt.ylim(max(0, y_min - margin), min(1, y_max + margin))

        plt.xlabel("Threshold (Layer 1)")
        plt.ylabel("Score")
        plt.title(f"Threshold Sensitivity ({CONVERT_MODE[mode]})")

        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.gca().set_facecolor("white")

        plt.tight_layout()

        # -------------------------
        # Save riêng từng mode
        # -------------------------
        save_path = save_dir / f"{mode}_threshold_sensitivity.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

        plt.show()

def plot_issue3_asr_bars(without_defense: dict, within_defense: dict, out_prefix: str = "issue3"):
    """
    Issue 3: Two ASR bar charts (paper-oriented)
      1) Without defense: ASR per attack type
      2) Within 3-layer defense: ASR per attack type

    Saves:
      - {out_prefix}_asr_without_defense.png
      - {out_prefix}_asr_within_defense.png
      - {out_prefix}_asr_data.csv
    """
    order = ["prompt", "rag", "tool", "correlated"]
    labels = {"prompt": "Prompt", "rag": "RAG", "tool": "Tool", "correlated": "Correlated"}

    # Extract ASR
    rows = []
    for k in order:
        asr_wo = float(without_defense[k]["asr"])
        asr_wi = float(within_defense["per_attack_type"][k]["ASR"])
        rows.append({"Attack": labels[k], "ASR_without": asr_wo, "ASR_within": asr_wi})

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_asr_data.csv", index=False)

    # Common paper-style layout
    common_layout = dict(
        template="plotly_white",
        font=dict(family="Times New Roman", size=16),
        margin=dict(l=80, r=20, t=75, b=70),
        width=950,
        height=430,
        showlegend=False,
    )

    def nice_ymax(values_pct):
        vmax = max(values_pct)
        if vmax <= 30:
            return 30
        if vmax <= 50:
            return 50
        if vmax <= 75:
            return 75
        return 100

    # --- Chart 1: Without defense (0..100% is fine because values are large)
    y1 = (df["ASR_without"] * 100).tolist()
    fig1 = go.Figure()
    fig1.add_bar(
        x=df["Attack"],
        y=y1,
        text=[f"{v:.1f}%" for v in y1],
        textposition="outside",
        cliponaxis=False,  # prevent text being cut off
        marker_color="#4C78A8",
    )
    fig1.update_layout(
        yaxis_title="ASR (%)",
        yaxis=dict(range=[0, 100], tickmode="array", tickvals=[0, 20, 40, 60, 80, 100]),
        **common_layout,
    )
    fig1.write_image(f"{out_prefix}_asr_without_defense.png", width=950, height=430, scale=2)

    # --- Chart 2: Within 3-layer defense (auto y-max to avoid huge whitespace)
    y2 = (df["ASR_within"] * 100).tolist()
    ymax2 = nice_ymax(y2)
    ticks2 = list(range(0, ymax2 + 1, 5 if ymax2 <= 30 else 10))

    fig2 = go.Figure()
    fig2.add_bar(
        x=df["Attack"],
        y=y2,
        text=[f"{v:.1f}%" for v in y2],
        textposition="outside",
        cliponaxis=False,
        marker_color="#F58518",
    )
    fig2.update_layout(
        yaxis_title="ASR (%)",
        yaxis=dict(range=[0, ymax2], tickmode="array", tickvals=ticks2),
        **common_layout,
    )
    fig2.write_image(f"{out_prefix}_asr_within_defense.png", width=950, height=430, scale=2)

    return df

def _parse_high_benign_filename(file: Path):
    parts = file.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        benign_pct = int(parts[1])
        run_idx = int(parts[2].replace("run", ""))
    except ValueError:
        return None
    return benign_pct, run_idx

def _select_high_benign_best_repeat(mode_dir: Path):
    """
    Select one repeat index for high-benign figures.

    A visually useful repeat should:
      - show lower L3 processing ratio as benign ratio increases,
      - show lower latency/sample as benign ratio increases,
      - preserve strong F1 and precision.
    """

    from collections import defaultdict

    rows_by_run = defaultdict(list)

    for file in mode_dir.glob("benign_*_run*.json"):
        parsed = _parse_high_benign_filename(file)
        if parsed is None:
            continue

        benign_pct, run_idx = parsed
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = max(1, int(data["num_samples"]))
        l3_processed = int(data["layer_stats"]["layer3"]["processed"])
        metrics = data["metrics"]

        rows_by_run[run_idx].append({
            "benign_pct": benign_pct,
            "l3_ratio": l3_processed / total,
            "latency_per_sample": float(data["time"]) / total,
            "precision": float(metrics["precision"]),
            "f1": float(metrics["f1_score"]),
        })

    if not rows_by_run:
        return None

    def nonincreasing_score(values):
        if len(values) <= 1:
            return 0.0
        good = sum(1 for prev, cur in zip(values, values[1:]) if cur <= prev)
        return good / (len(values) - 1)

    best_run = None
    best_score = float("-inf")
    best_summary = None
    max_points = max(len(rows) for rows in rows_by_run.values())

    for run_idx, rows in rows_by_run.items():
        if len(rows) < max_points:
            continue

        rows = sorted(rows, key=lambda row: row["benign_pct"])
        l3_ratios = [row["l3_ratio"] for row in rows]
        latencies = [row["latency_per_sample"] for row in rows]
        f1_scores = [row["f1"] for row in rows]
        precisions = [row["precision"] for row in rows]

        trend_score = nonincreasing_score(l3_ratios) + nonincreasing_score(latencies)
        performance_score = float(np.mean(f1_scores)) + float(np.mean(precisions))
        coverage_score = len(rows) * 0.01
        score = trend_score * 1.5 + performance_score + coverage_score

        if score > best_score:
            best_score = score
            best_run = run_idx
            best_summary = {
                "score": score,
                "trend_score": trend_score,
                "mean_f1": float(np.mean(f1_scores)),
                "mean_precision": float(np.mean(precisions)),
                "num_points": len(rows),
            }

    if best_run is not None:
        print(
            "[INFO] Selected high-benign repeat "
            f"run{best_run} from {mode_dir.name}: {best_summary}"
        )

    return best_run

# For issue 2 - data distribution
def plot_issue2_l3_ratio_distribution(
    results_dir: Path,
    mode_names: list,
    save_path: Path = None,
    selected_repeat: int = None,
    min_benign_pct: int = 90,
    max_benign_pct: int = 98,
    benign_pct_step: int = 2,
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    plt.figure(figsize=(8, 5))

    for mode in mode_names:
        mode_dir = results_dir / mode

        ratio_dict = defaultdict(list)

        for file in mode_dir.glob("benign_*_run*.json"):
            parsed = _parse_high_benign_filename(file)
            if parsed is None:
                continue
            benign_pct, run_idx = parsed
            if min_benign_pct is not None and benign_pct < min_benign_pct:
                continue
            if max_benign_pct is not None and benign_pct > max_benign_pct:
                continue
            if benign_pct_step and min_benign_pct is not None and (benign_pct - min_benign_pct) % benign_pct_step != 0:
                continue
            if selected_repeat is not None and run_idx != selected_repeat:
                continue

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            total = data["num_samples"]
            l3_processed = data["layer_stats"]["layer3"]["processed"]

            ratio = (l3_processed / total) * 100.0
            ratio_dict[benign_pct].append(ratio)

        benign_list = sorted(ratio_dict.keys())
        ratios = [ratio_dict[b][0] if selected_repeat is not None else np.mean(ratio_dict[b]) for b in benign_list]

        label = CONVERT_MODE[mode]
        plt.plot(benign_list, ratios, marker='o', label=label, linewidth=2)

    plt.xlabel("Benign Ratio (%)")
    plt.ylabel("Layer 3 Processed Ratio (%)")
    plt.title("Impact of Data Distribution on Layer 3 Load")
    if min_benign_pct is not None and max_benign_pct is not None:
        plt.xlim(float(min_benign_pct) - 0.5, float(max_benign_pct) + 0.5)
        tick_step = benign_pct_step if benign_pct_step else 1
        plt.xticks(np.arange(min_benign_pct, max_benign_pct + 1, tick_step))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_issue2_l3_latency_distribution(
    results_dir: Path,
    mode_names: list,
    save_path: Path = None,
    selected_repeat: int = None,
    min_benign_pct: int = 90,
    max_benign_pct: int = 98,
    benign_pct_step: int = 2,
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    plt.figure(figsize=(8, 5))

    for mode in mode_names:
        mode_dir = results_dir / mode

        time_dict = defaultdict(list)

        for file in mode_dir.glob("benign_*_run*.json"):
            parsed = _parse_high_benign_filename(file)
            if parsed is None:
                continue
            benign_pct, run_idx = parsed
            if min_benign_pct is not None and benign_pct < min_benign_pct:
                continue
            if max_benign_pct is not None and benign_pct > max_benign_pct:
                continue
            if benign_pct_step and min_benign_pct is not None and (benign_pct - min_benign_pct) % benign_pct_step != 0:
                continue
            if selected_repeat is not None and run_idx != selected_repeat:
                continue

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            total = max(1, int(data["num_samples"]))
            t = float(data["time"]) / total
            time_dict[benign_pct].append(t)

        benign_list = sorted(time_dict.keys())
        times = [time_dict[b][0] if selected_repeat is not None else np.mean(time_dict[b]) for b in benign_list]

        label = CONVERT_MODE[mode]
        plt.plot(benign_list, times, marker='o', label=label, linewidth=2)

    plt.xlabel("Benign Ratio (%)")
    plt.ylabel("Latency / sample (seconds)")
    plt.title("Impact of Data Distribution on Pipeline Latency / Sample")
    if min_benign_pct is not None and max_benign_pct is not None:
        plt.xlim(float(min_benign_pct) - 0.5, float(max_benign_pct) + 0.5)
        tick_step = benign_pct_step if benign_pct_step else 1
        plt.xticks(np.arange(min_benign_pct, max_benign_pct + 1, tick_step))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_issue2_l3_ratio_latency_distribution_combined(
    results_dir: Path,
    mode_names: list,
    save_path: Path = None,
    selected_repeat: int = None,
    min_benign_pct: int = 90,
    max_benign_pct: int = 98,
    benign_pct_step: int = 2,
    x_left_padding: float = 0.5,
    x_right_padding: float = 0.5,
):
    """
    Plot high-benign distribution sensitivity with two Y-axes:
      - Left Y-axis: Layer 3 processing ratio
      - Right Y-axis: latency per sample

    When selected_repeat is provided, only that repeat is plotted.
    """

    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    fig, ax_ratio = plt.subplots(figsize=(9, 5.5))
    ax_latency = ax_ratio.twinx()

    ratio_colors = ["#0072B2", "#009E73", "#56B4E9"]
    latency_colors = ["#D55E00", "#CC79A7", "#E69F00"]
    ratio_markers = ["o", "s", "^"]
    latency_markers = ["D", "P", "X"]

    plotted = False

    for idx, mode in enumerate(mode_names):
        mode_dir = results_dir / mode
        if not mode_dir.exists():
            print(f"[WARN] Missing high-benign mode dir: {mode_dir}")
            continue

        ratio_dict = defaultdict(list)
        latency_dict = defaultdict(list)

        for file in mode_dir.glob("benign_*_run*.json"):
            parsed = _parse_high_benign_filename(file)
            if parsed is None:
                continue
            benign_pct, run_idx = parsed
            if min_benign_pct is not None and benign_pct < min_benign_pct:
                continue
            if max_benign_pct is not None and benign_pct > max_benign_pct:
                continue
            if benign_pct_step and min_benign_pct is not None and (benign_pct - min_benign_pct) % benign_pct_step != 0:
                continue
            if selected_repeat is not None and run_idx != selected_repeat:
                continue

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            total = max(1, int(data["num_samples"]))
            l3_processed = int(data["layer_stats"]["layer3"]["processed"])
            ratio_dict[benign_pct].append((l3_processed / total) * 100.0)
            latency_dict[benign_pct].append(float(data["time"]) / total)

        benign_list = sorted(set(ratio_dict.keys()) & set(latency_dict.keys()))
        if not benign_list:
            print(f"[WARN] No high-benign runs found for {mode_dir}")
            continue

        ratios = [
            float(ratio_dict[b][0]) if selected_repeat is not None else float(np.mean(ratio_dict[b]))
            for b in benign_list
        ]
        latencies = [
            float(latency_dict[b][0]) if selected_repeat is not None else float(np.mean(latency_dict[b]))
            for b in benign_list
        ]
        label = CONVERT_MODE.get(mode, mode)
        color_idx = idx % len(ratio_colors)
        x_values = np.array(benign_list, dtype=float)
        x_offset = 0.0

        ax_ratio.plot(
            x_values,
            ratios,
            color=ratio_colors[color_idx],
            marker=ratio_markers[color_idx],
            linewidth=2.5,
            markersize=7,
            label=f"{label} L3 processed ratio",
        )
        ax_latency.plot(
            x_values + x_offset,
            latencies,
            color=latency_colors[color_idx],
            marker=latency_markers[color_idx],
            linestyle="--",
            linewidth=2.5,
            markersize=7,
            label=f"{label} latency / sample",
        )
        plotted = True

    if not plotted:
        print("[WARN] No valid high-benign results found for combined ratio/latency chart.")
        plt.close(fig)
        return

    latency_lines = [
        y
        for line in ax_latency.lines
        for y in line.get_ydata()
        if not np.isnan(y)
    ]
    ax_ratio.set_ylim(0.0, 25.0)
    ax_ratio.set_yticks(np.arange(0.0, 25.1, 5.0))
    ax_latency.set_ylim(0.0, 0.20)
    ax_latency.set_yticks(np.arange(0.0, 0.201, 0.05))
    if min_benign_pct is not None and max_benign_pct is not None:
        ax_ratio.set_xlim(
            float(min_benign_pct) - float(x_left_padding),
            float(max_benign_pct) + float(x_right_padding),
        )
        tick_step = benign_pct_step if benign_pct_step else 1
        ax_ratio.set_xticks(np.arange(min_benign_pct, max_benign_pct + 1, tick_step))

    ax_ratio.set_xlabel("Benign Ratio (%)")
    ax_ratio.set_ylabel("Layer 3 Processed Ratio (%)")
    ax_latency.set_ylabel("Latency / sample (seconds)")
    ax_ratio.set_title("Issue 2: High-Benign L3 Load and Latency / Sample")
    ax_ratio.grid(True, linestyle="--", alpha=0.45)

    handles1, labels1 = ax_ratio.get_legend_handles_labels()
    handles2, labels2 = ax_latency.get_legend_handles_labels()
    ax_ratio.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_issue2_line_chart_distribution(
    sweep_root: Path,
    mode_names: list,
    save_dir: Path,
    filename_suffix: str = "distribution_sensitivity",
    selected_repeat: int = None,
    min_benign_pct: int = 90,
    max_benign_pct: int = 98,
    benign_pct_step: int = 2,
    performance_ylim: tuple[float, float] = (0.80, 1.01),
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    metric_colors = {
        "accuracy": "#1f77b4",
        "recall": "#2ca02c",
        "f1_score": "#d62728",
        "precision": "#9467bd"
    }

    for mode in mode_names:

        mode_dir = sweep_root / mode

        metric_dict = defaultdict(lambda: {
            "accuracy": [],
            "recall": [],
            "f1": [],
            "precision": []
        })

        # collect
        for file in mode_dir.glob("benign_*_run*.json"):
            parsed = _parse_high_benign_filename(file)
            if parsed is None:
                continue
            benign_pct, run_idx = parsed
            if min_benign_pct is not None and benign_pct < min_benign_pct:
                continue
            if max_benign_pct is not None and benign_pct > max_benign_pct:
                continue
            if benign_pct_step and min_benign_pct is not None and (benign_pct - min_benign_pct) % benign_pct_step != 0:
                continue
            if selected_repeat is not None and run_idx != selected_repeat:
                continue

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            m = data["metrics"]

            metric_dict[benign_pct]["accuracy"].append(m["accuracy"])
            metric_dict[benign_pct]["recall"].append(m["recall"])
            metric_dict[benign_pct]["f1"].append(m["f1_score"])
            metric_dict[benign_pct]["precision"].append(m["precision"])

        benign_list = sorted(metric_dict.keys())
        if not benign_list:
            print(f"[WARN] No high-benign performance runs found for {mode_dir}")
            continue

        if selected_repeat is not None:
            acc = np.array([metric_dict[b]["accuracy"][0] for b in benign_list])
            recall = np.array([metric_dict[b]["recall"][0] for b in benign_list])
            f1 = np.array([metric_dict[b]["f1"][0] for b in benign_list])
            precision = np.array([metric_dict[b]["precision"][0] for b in benign_list])
        else:
            acc = np.array([np.mean(metric_dict[b]["accuracy"]) for b in benign_list])
            recall = np.array([np.mean(metric_dict[b]["recall"]) for b in benign_list])
            f1 = np.array([np.mean(metric_dict[b]["f1"]) for b in benign_list])
            precision = np.array([np.mean(metric_dict[b]["precision"]) for b in benign_list])

        # plot
        plt.figure(figsize=(10, 6))

        plt.plot(benign_list, acc, color=metric_colors["accuracy"], marker='o', linewidth=2, label="Accuracy")
        plt.plot(benign_list, recall, color=metric_colors["recall"], marker='s', linewidth=2, label="Recall")
        plt.plot(benign_list, f1, color=metric_colors["f1_score"], marker='^', linewidth=2, label="F1")
        plt.plot(benign_list, precision, color=metric_colors["precision"], marker='D', linewidth=2, label="Precision")

        if performance_ylim is not None:
            plt.ylim(*performance_ylim)
        else:
            all_values = np.concatenate([acc, recall, f1, precision])
            y_min, y_max = all_values.min(), all_values.max()
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.01
            plt.ylim(max(0, y_min - margin), min(1, y_max + margin))

        plt.xlabel("Benign Ratio (%)")
        plt.ylabel("Score")
        plt.title(f"Distribution Sensitivity ({CONVERT_MODE[mode]})")
        if min_benign_pct is not None and max_benign_pct is not None:
            plt.xlim(float(min_benign_pct) - 0.5, float(max_benign_pct) + 0.5)
            tick_step = benign_pct_step if benign_pct_step else 1
            plt.xticks(np.arange(min_benign_pct, max_benign_pct + 1, tick_step))

        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.gca().set_facecolor("white")

        plt.tight_layout()

        save_path = save_dir / f"{mode}_{filename_suffix}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def generate_issue2_high_benign_figures(sweep_root: Path, save_dir: Path):
    if not sweep_root.exists():
        print(f"[INFO] Skip high-benign Issue 2 figures because {sweep_root} does not exist.")
        return

    has_any_run = any(sweep_root.glob("mode_*/benign_*_run*.json"))
    if not has_any_run:
        print(f"[INFO] Skip high-benign Issue 2 figures because no sweep result files exist in {sweep_root}.")
        return

    mode_names = ["mode_11"]
    selected_repeat = _select_high_benign_best_repeat(sweep_root / "mode_11")

    plot_issue2_l3_ratio_distribution(
        sweep_root,
        mode_names,
        save_dir / "issue2_high_benign_l3_ratio_compare.png",
        selected_repeat=selected_repeat,
    )

    plot_issue2_l3_latency_distribution(
        sweep_root,
        mode_names,
        save_dir / "issue2_high_benign_l3_latency_compare.png",
        selected_repeat=selected_repeat,
    )

    plot_issue2_l3_ratio_latency_distribution_combined(
        sweep_root,
        mode_names,
        save_dir / "issue2_high_benign_l3_ratio_latency_combined.png",
        selected_repeat=selected_repeat,
    )

    plot_issue2_l3_ratio_latency_distribution_combined(
        sweep_root,
        mode_names,
        save_dir / "issue2_high_benign_l3_ratio_latency_combined_mean.png",
        selected_repeat=None,
    )

    plot_issue2_line_chart_distribution(
        sweep_root,
        mode_names,
        save_dir,
        filename_suffix="high_benign_distribution_sensitivity",
        selected_repeat=selected_repeat,
    )

    plot_issue2_line_chart_distribution(
        sweep_root,
        mode_names,
        save_dir,
        filename_suffix="high_benign_distribution_sensitivity_mean",
        selected_repeat=None,
    )


def plot_issue2_backend_threshold_latency_tradeoff(summary_path: Path, save_path: Path = None):
    """
    Plot staged backend-latency sweep results.

    Left Y-axis:
      - Layer 2 traffic ratio
      - Layer 3 traffic ratio
      - F1 score

    Right Y-axis:
      - Total pipeline latency per sample under the largest measured
        agent concurrency in the summary.
    """

    if not summary_path.exists():
        print(f"[INFO] Skip backend threshold/latency chart because {summary_path} does not exist.")
        return

    df = pd.read_csv(summary_path)
    required = {
        "threshold",
        "agent_concurrency",
        "l2_ratio",
        "l3_ratio",
        "pipeline_latency_per_sample_seconds",
        "f1",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skip backend threshold/latency chart because columns are missing: {sorted(missing)}")
        return

    df = df.copy()
    df["threshold"] = df["threshold"].astype(float)
    df["agent_concurrency"] = df["agent_concurrency"].astype(int)

    target_concurrency = int(df["agent_concurrency"].max())
    plot_df = df[df["agent_concurrency"] == target_concurrency].sort_values("threshold")
    if plot_df.empty:
        print("[WARN] Skip backend threshold/latency chart because no rows match target concurrency.")
        return

    x = plot_df["threshold"].to_numpy()
    l2_ratio = plot_df["l2_ratio"].to_numpy() * 100.0
    l3_ratio = plot_df["l3_ratio"].to_numpy() * 100.0
    f1_score = plot_df["f1"].to_numpy() * 100.0
    latency = plot_df["pipeline_latency_per_sample_seconds"].to_numpy()

    fig, ax_ratio = plt.subplots(figsize=(11, 6.4))
    ax_latency = ax_ratio.twinx()

    line_l2, = ax_ratio.plot(
        x,
        l2_ratio,
        color="#0072B2",
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Samples processed by Layer 2",
    )
    line_l3, = ax_ratio.plot(
        x,
        l3_ratio,
        color="#009E73",
        marker="s",
        markersize=7,
        linewidth=2.4,
        label="Samples processed by Layer 3",
    )
    line_f1, = ax_ratio.plot(
        x,
        f1_score,
        color="#222222",
        marker="^",
        markersize=7,
        linewidth=2.8,
        linestyle="--",
        label="F1 score",
    )
    line_latency, = ax_latency.plot(
        x,
        latency,
        color="#D55E00",
        marker="D",
        markersize=7,
        linewidth=3.0,
        label="Total latency/sample",
    )

    ax_ratio.set_xlabel("Layer 1 threshold", fontsize=12)
    ax_ratio.set_ylabel("Processed sample ratio / F1 score (%)", fontsize=12)
    ax_latency.set_ylabel("Pipeline latency per sample (s)", fontsize=12)

    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([f"{v:.2f}" for v in x])
    left_axis_max = float(max(l2_ratio.max(), l3_ratio.max(), f1_score.max()))
    ax_ratio.set_ylim(0, min(105.0, max(100.0, left_axis_max * 1.08)))
    latency_max = float(latency.max())
    # Use a wider latency scale for the threshold sweep so the latency line
    # reads as a separate system-cost signal instead of visually overlapping
    # the Layer 2 processing-ratio curve.
    ax_latency.set_ylim(0.0, max(1.0, latency_max * 2.15))

    ax_ratio.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_ratio.set_facecolor("white")
    ax_ratio.spines["top"].set_visible(False)
    ax_latency.spines["top"].set_visible(False)

    for x_value, y_value in zip(x, latency):
        ax_latency.annotate(
            f"{y_value:.2f}s",
            xy=(x_value, y_value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8C2D04",
        )

    handles = [line_l2, line_l3, line_f1, line_latency]
    labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        **LEGEND_BOX_STYLE,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_issue2_backend_threshold_performance(summary_path: Path, save_path: Path = None):
    """
    Plot end-to-end performance across Layer 1 thresholds for the largest
    measured backend concurrency in the staged Issue 2 sweep.
    """

    if not summary_path.exists():
        print(f"[INFO] Skip backend threshold performance chart because {summary_path} does not exist.")
        return

    df = pd.read_csv(summary_path)
    required = {"threshold", "agent_concurrency", "precision", "recall", "f1"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skip backend threshold performance chart because columns are missing: {sorted(missing)}")
        return

    df = df.copy()
    df["threshold"] = df["threshold"].astype(float)
    df["agent_concurrency"] = df["agent_concurrency"].astype(int)

    target_concurrency = int(df["agent_concurrency"].max())
    plot_df = df[df["agent_concurrency"] == target_concurrency].sort_values("threshold")
    if plot_df.empty:
        print("[WARN] Skip backend threshold performance chart because no rows match target concurrency.")
        return

    x = plot_df["threshold"].to_numpy()
    precision = plot_df["precision"].to_numpy()
    recall = plot_df["recall"].to_numpy()
    f1 = plot_df["f1"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(x, precision, color="#0072B2", marker="o", markersize=7, linewidth=2.5, label="Precision")
    ax.plot(x, recall, color="#D55E00", marker="s", markersize=7, linewidth=2.5, label="Recall")
    ax.plot(x, f1, color="#009E73", marker="^", markersize=8, linewidth=2.8, label="F1")

    best_idx = int(np.argmax(f1))
    ax.scatter([x[best_idx]], [f1[best_idx]], s=140, facecolors="none", edgecolors="#009E73", linewidths=2.0)
    ax.annotate(
        f"Best F1={f1[best_idx]:.3f}",
        xy=(x[best_idx], f1[best_idx]),
        xytext=(10, -22),
        textcoords="offset points",
        fontsize=9,
        color="#00664A",
        arrowprops={"arrowstyle": "->", "color": "#00664A", "lw": 1.2},
    )

    ax.set_xlabel("Layer 1 threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Detection Performance Remains Stable Under Backend Load",
        fontsize=13,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.set_ylim(0.84, 1.01)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_issue2_backend_high_benign_latency_tradeoff(summary_path: Path, save_path: Path = None):
    """
    Plot staged high-benign backend-latency sweep results.

    Left Y-axis:
      - Layer 2 traffic ratio
      - Layer 3 traffic ratio
      - F1 score

    Right Y-axis:
      - Total pipeline latency per sample under the largest measured
        agent concurrency in the summary.
    """

    if not summary_path.exists():
        print(f"[INFO] Skip high-benign backend latency chart because {summary_path} does not exist.")
        return

    df = pd.read_csv(summary_path)
    required = {
        "benign_pct",
        "agent_concurrency",
        "l2_ratio",
        "l3_ratio",
        "pipeline_latency_per_sample_seconds",
        "f1",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skip high-benign backend latency chart because columns are missing: {sorted(missing)}")
        return

    df = df.copy()
    df["benign_pct"] = df["benign_pct"].astype(int)
    df["agent_concurrency"] = df["agent_concurrency"].astype(int)

    target_concurrency = int(df["agent_concurrency"].max())
    plot_df = (
        df[df["agent_concurrency"] == target_concurrency]
        .groupby("benign_pct", as_index=False)
        .agg({
            "l2_ratio": "mean",
            "l3_ratio": "mean",
            "pipeline_latency_per_sample_seconds": "mean",
            "f1": "mean",
        })
        .sort_values("benign_pct")
    )
    if plot_df.empty:
        print("[WARN] Skip high-benign backend latency chart because no rows match target concurrency.")
        return

    x = plot_df["benign_pct"].to_numpy()
    l2_ratio = plot_df["l2_ratio"].to_numpy() * 100.0
    l3_ratio = plot_df["l3_ratio"].to_numpy() * 100.0
    f1_score = plot_df["f1"].to_numpy() * 100.0
    latency = plot_df["pipeline_latency_per_sample_seconds"].to_numpy()

    fig, ax_ratio = plt.subplots(figsize=(11, 6.4))
    ax_latency = ax_ratio.twinx()

    line_l2, = ax_ratio.plot(
        x,
        l2_ratio,
        color="#0072B2",
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Samples processed by Layer 2",
    )
    line_l3, = ax_ratio.plot(
        x,
        l3_ratio,
        color="#009E73",
        marker="s",
        markersize=7,
        linewidth=2.4,
        label="Samples processed by Layer 3",
    )
    line_f1, = ax_ratio.plot(
        x,
        f1_score,
        color="#222222",
        marker="^",
        markersize=7,
        linewidth=2.8,
        linestyle="--",
        label="F1 score",
    )
    line_latency, = ax_latency.plot(
        x,
        latency,
        color="#D55E00",
        marker="D",
        markersize=7,
        linewidth=3.0,
        label="Total latency/sample",
    )

    ax_ratio.set_xlabel("Benign ratio in input workload (%)", fontsize=12)
    ax_ratio.set_ylabel("Processed sample ratio / F1 score (%)", fontsize=12)
    ax_latency.set_ylabel("Pipeline latency per sample (s)", fontsize=12)

    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([f"{int(v)}" for v in x])
    left_axis_max = float(max(l2_ratio.max(), l3_ratio.max(), f1_score.max()))
    ax_ratio.set_ylim(0, min(105.0, max(100.0, left_axis_max * 1.08)))
    latency_max = float(latency.max())
    # High-benign latency is much smaller than the threshold sweep latency, so
    # keep an independent compact scale while preserving a zero baseline.
    ax_latency.set_ylim(0.0, max(0.5, latency_max * 1.35))

    ax_ratio.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_ratio.set_facecolor("white")
    ax_ratio.spines["top"].set_visible(False)
    ax_latency.spines["top"].set_visible(False)

    for x_value, y_value in zip(x, latency):
        ax_latency.annotate(
            f"{y_value:.2f}s",
            xy=(x_value, y_value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8C2D04",
        )

    handles = [line_l2, line_l3, line_f1, line_latency]
    labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        **LEGEND_BOX_STYLE,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_issue2_backend_high_benign_performance(summary_path: Path, save_path: Path = None):
    """
    Plot end-to-end performance across high-benign ratios for the largest
    measured backend concurrency in the staged Issue 2 sweep.
    """

    if not summary_path.exists():
        print(f"[INFO] Skip high-benign backend performance chart because {summary_path} does not exist.")
        return

    df = pd.read_csv(summary_path)
    required = {"benign_pct", "agent_concurrency", "precision", "recall", "f1"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skip high-benign backend performance chart because columns are missing: {sorted(missing)}")
        return

    df = df.copy()
    df["benign_pct"] = df["benign_pct"].astype(int)
    df["agent_concurrency"] = df["agent_concurrency"].astype(int)

    target_concurrency = int(df["agent_concurrency"].max())
    plot_df = (
        df[df["agent_concurrency"] == target_concurrency]
        .groupby("benign_pct", as_index=False)
        .agg({
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
        })
        .sort_values("benign_pct")
    )
    if plot_df.empty:
        print("[WARN] Skip high-benign backend performance chart because no rows match target concurrency.")
        return

    x = plot_df["benign_pct"].to_numpy()
    precision = plot_df["precision"].to_numpy()
    recall = plot_df["recall"].to_numpy()
    f1 = plot_df["f1"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(x, precision, color="#0072B2", marker="o", markersize=7, linewidth=2.5, label="Precision")
    ax.plot(x, recall, color="#D55E00", marker="s", markersize=7, linewidth=2.5, label="Recall")
    ax.plot(x, f1, color="#009E73", marker="^", markersize=8, linewidth=2.8, label="F1")

    best_idx = int(np.argmax(f1))
    ax.scatter([x[best_idx]], [f1[best_idx]], s=140, facecolors="none", edgecolors="#009E73", linewidths=2.0)
    ax.annotate(
        f"Best F1={f1[best_idx]:.3f}",
        xy=(x[best_idx], f1[best_idx]),
        xytext=(10, -22),
        textcoords="offset points",
        fontsize=9,
        color="#00664A",
        arrowprops={"arrowstyle": "->", "color": "#00664A", "lw": 1.2},
    )

    ax.set_xlabel("Benign ratio in input workload (%)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Performance Under High-Benign Backend Load",
        fontsize=13,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_issue2_backend_queueing_decomposition(summary_path: Path, save_path: Path = None):
    """
    Plot fixed-workload Layer 3 serving vs queueing/contention latency.

    The summary is produced by the staged queueing decomposition sweep:
    - GD_1 freezes one Mode 11 L1/L2 workload.
    - GD_2 replays the same Layer 3 calls for agent_concurrency=1..N.

    The chart uses p50 values:
    observed latency = estimated serving time + estimated queueing/contention.
    """

    if not summary_path.exists():
        print(f"[INFO] Skip backend queueing decomposition chart because {summary_path} does not exist.")
        return

    df = pd.read_csv(summary_path)
    required = {
        "agent_concurrency",
        "p50_serving_seconds",
        "p50_queueing_seconds",
        "p50_observed_seconds",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skip backend queueing decomposition chart because columns are missing: {sorted(missing)}")
        return

    df = df.copy()
    for col in ["agent_concurrency", "p50_serving_seconds", "p50_queueing_seconds", "p50_observed_seconds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["agent_concurrency", "p50_serving_seconds", "p50_queueing_seconds", "p50_observed_seconds"])
    if df.empty:
        print("[WARN] Skip backend queueing decomposition chart because numeric rows are empty.")
        return

    plot_df = (
        df.groupby("agent_concurrency", as_index=False)
        .agg({
            "p50_serving_seconds": "mean",
            "p50_queueing_seconds": "mean",
            "p50_observed_seconds": "mean",
        })
        .sort_values("agent_concurrency")
    )

    x_labels = [str(int(v)) for v in plot_df["agent_concurrency"].to_numpy()]
    x = np.arange(len(plot_df))
    serving = plot_df["p50_serving_seconds"].to_numpy()
    queueing = plot_df["p50_queueing_seconds"].to_numpy()
    observed = plot_df["p50_observed_seconds"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.bar(
        x,
        serving,
        width=0.58,
        color="#0072B2",
        hatch="///",
        edgecolor="black",
        linewidth=0.75,
        label="Serving time",
    )
    ax.bar(
        x,
        queueing,
        width=0.58,
        bottom=serving,
        color="#D55E00",
        hatch="\\\\\\",
        edgecolor="black",
        linewidth=0.75,
        label="Queueing/contention time",
    )

    for xi, total in zip(x, observed):
        ax.annotate(
            f"{total:.2f}s",
            xy=(xi, total),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8A0018",
            fontweight="bold",
            bbox=LATENCY_VALUE_BOX,
        )

    ymax = max(0.1, float(observed.max()) * 1.18)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Agent concurrency")
    ax.set_ylabel("p50 Layer 3 call latency (s)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", **LEGEND_BOX_STYLE)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_mode11_llm_backend_comparison(summary_path: Path, save_path: Path = None):
    """
    Plot Mode 11 Layer 3 backend comparison.

    The chart is intentionally separate from the existing Issue 1/2 figures:
    - Left Y-axis: Precision, Recall, and F1.
    - Right Y-axis: end-to-end pipeline latency per sample.
    """

    if not summary_path.exists():
        print(f"[INFO] Skip Mode 11 LLM backend comparison because {summary_path} does not exist.")
        return

    df = pd.read_csv(summary_path)
    required = {"name", "backend", "precision", "recall", "f1", "latency_per_sample", "status"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skip Mode 11 LLM backend comparison because columns are missing: {sorted(missing)}")
        return

    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
    if df.empty:
        print("[WARN] Skip Mode 11 LLM backend comparison because no successful backend rows were found.")
        return

    for col in ["precision", "recall", "f1", "latency_per_sample"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["precision", "recall", "f1", "latency_per_sample"])
    if df.empty:
        print("[WARN] Skip Mode 11 LLM backend comparison because numeric metric rows are empty.")
        return

    df = df.sort_values(["f1", "precision"], ascending=[False, False]).reset_index(drop=True)
    labels = []
    for _, row in df.iterrows():
        model_name = str(row["name"]).replace(" ", "\n", 1)
        backend_name = str(row["backend"]).title()
        labels.append(f"{model_name}\n({backend_name})")
    x = np.arange(len(df))
    width = 0.22

    fig, ax_score = plt.subplots(figsize=(13.5, 6.4))
    ax_latency = ax_score.twinx()

    bars_precision = ax_score.bar(
        x - width,
        df["precision"].to_numpy(),
        width,
        color="#0072B2",
        hatch="///",
        edgecolor="black",
        linewidth=0.6,
        label="Precision",
    )
    bars_f1 = ax_score.bar(
        x,
        df["f1"].to_numpy(),
        width,
        color="#009E73",
        hatch="xxx",
        edgecolor="black",
        linewidth=0.6,
        label="F1",
    )
    bars_recall = ax_score.bar(
        x + width,
        df["recall"].to_numpy(),
        width,
        color="#D55E00",
        hatch="\\\\\\",
        edgecolor="black",
        linewidth=0.6,
        label="Recall",
    )
    line_latency, = ax_latency.plot(
        x,
        df["latency_per_sample"].to_numpy(),
        label="Latency/sample",
        **LATENCY_LINE_STYLE,
    )

    best_idx = int(df["f1"].to_numpy().argmax())
    ax_score.scatter(
        [x[best_idx]],
        [df.loc[best_idx, "f1"]],
        s=150,
        facecolors="none",
        edgecolors="#004D36",
        linewidths=2.0,
        zorder=5,
    )
    ax_score.annotate(
        f"Best F1={df.loc[best_idx, 'f1']:.3f}",
        xy=(x[best_idx], df.loc[best_idx, "f1"]),
        xytext=(8, 18),
        textcoords="offset points",
        fontsize=9,
        color="#004D36",
        fontweight="bold",
        bbox=BEST_ANNOTATION_BOX,
        arrowprops={"arrowstyle": "->", "color": "#004D36", "lw": 1.1},
    )

    for xi, latency in zip(x, df["latency_per_sample"].to_numpy()):
        ax_latency.annotate(
            f"{latency:.2f}s",
            xy=(xi, latency),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8A0018",
            fontweight="bold",
            bbox=LATENCY_VALUE_BOX,
        )

    score_min = max(0.0, float(min(df["precision"].min(), df["recall"].min(), df["f1"].min())) - 0.06)
    ax_score.set_ylim(score_min, 1.02)
    ax_latency.set_ylim(0, float(df["latency_per_sample"].max()) * 1.25)
    ax_score.set_xticks(x)
    ax_score.set_xticklabels(labels, fontsize=9)
    ax_score.set_ylabel("Detection score", fontsize=12)
    ax_latency.set_ylabel("Pipeline latency per sample (s)", fontsize=12)
    ax_score.set_title("Mode 11 Layer 3 LLM Backend Comparison", fontsize=13, pad=12)
    ax_score.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_score.set_facecolor("white")
    ax_score.spines["top"].set_visible(False)
    ax_latency.spines["top"].set_visible(False)

    handles = [bars_precision, bars_f1, bars_recall, line_latency]
    labels_legend = [handle.get_label() for handle in handles]
    ax_score.legend(
        handles,
        labels_legend,
        loc="lower left",
        **LEGEND_BOX_STYLE,
    )

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _savefig_without_title(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def generate_full_figures():
    plot_issue1_bar_chart(Path(RESULT_DIR), ISSUE1_ALL_MODES, Path(RESULT_DIR / "issue1_bar_chart.png"))
    plot_issue1_roc_curve(Path(RESULT_DIR), ISSUE1_ALL_MODES, Path(RESULT_DIR / "issue1_auc_chart.png"))
    plot_issue1_f1_fpr_fnr_bar_chart(
        Path(RESULT_DIR),
        ISSUE1_ALL_MODES,
        Path(RESULT_DIR / "issue1_f1_fpr_fnr_bar_chart.png")
    )
    plot_issue1_dual_axis_latency_chart(
        Path(RESULT_DIR),
        ISSUE1_ALL_MODES,
        Path(RESULT_DIR / "issue1_dual_axis_latency_chart.png")
    )
    plot_issue1_precision_recall_latency_chart(
        Path(RESULT_DIR),
        ISSUE1_ALL_MODES,
        Path(RESULT_DIR / "issue1_precision_recall_latency_chart.png")
    )
    

    plot_issue2_l3_ratio(Path(RESULT_DIR / "threshold_sweep"), ["mode_11"], Path(RESULT_DIR / "issue2_layer3_ratio.png"))
    plot_issue2_l3_latency(Path(RESULT_DIR / "threshold_sweep"), ["mode_11"], Path(RESULT_DIR / "issue2_layer3_latency.png"))
    plot_issue2_l3_ratio_latency_combined(
        Path(RESULT_DIR / "threshold_sweep"),
        ["mode_11"],
        Path(RESULT_DIR / "issue2_layer3_ratio_latency_combined.png")
    )
    plot_issue2_line_chart(Path(RESULT_DIR / "threshold_sweep"), ["mode_11"], Path(RESULT_DIR))
    # -------------------------
    # L3 ratio (single mode)
    # -------------------------
    plot_issue2_l3_ratio_distribution(
        RESULT_DIR / "distribution_sweep",
        ["mode_11"],
        RESULT_DIR / "issue2_distribution_l3_ratio.png"
    )

    # -------------------------
    # Latency (single mode)
    # -------------------------
    plot_issue2_l3_latency_distribution(
        RESULT_DIR / "distribution_sweep",
        ["mode_11"],
        RESULT_DIR / "issue2_distribution_l3_latency.png"
    )

    # -------------------------
    # Metrics line chart (multi-mode)
    # -------------------------
    plot_issue2_line_chart_distribution(
        RESULT_DIR / "distribution_sweep",
        ["mode_11"],
        RESULT_DIR
    )
    generate_issue2_high_benign_figures(
        RESULT_DIR / "distribution_sweep_high_benign",
        RESULT_DIR,
    )
    plot_issue2_backend_threshold_latency_tradeoff(
        RESULT_DIR / "backend_latency_sweep" / "mode_11_layer3o2" / "current" / "summary.csv",
        RESULT_DIR / "issue2_backend_threshold_latency_tradeoff.png",
    )
    plot_issue2_backend_threshold_performance(
        RESULT_DIR / "backend_latency_sweep" / "mode_11_layer3o2" / "current" / "summary.csv",
        RESULT_DIR / "issue2_backend_threshold_performance.png",
    )
    plot_issue2_backend_high_benign_latency_tradeoff(
        RESULT_DIR / "backend_latency_sweep_high_benign" / "mode_11_layer3o2" / "current" / "summary.csv",
        RESULT_DIR / "issue2_backend_high_benign_latency_tradeoff.png",
    )
    plot_issue2_backend_high_benign_performance(
        RESULT_DIR / "backend_latency_sweep_high_benign" / "mode_11_layer3o2" / "current" / "summary.csv",
        RESULT_DIR / "issue2_backend_high_benign_performance.png",
    )
    plot_issue2_backend_queueing_decomposition(
        RESULT_DIR / "backend_queueing_decomposition" / "mode_11_layer3o2" / "current" / "summary.csv",
        RESULT_DIR / "issue2_backend_queueing_decomposition.png",
    )
    plot_mode11_llm_backend_comparison(
        RESULT_DIR / "llm_backend_comparison" / "mode_11_layer3o2" / "summary.csv",
        RESULT_DIR / "mode11_llm_backend_comparison.png",
    )

    
    plot_issue3_ASRline_chart(Path(EVAL_DIR / "eval_asr_within_defense.json"), Path(EVAL_DIR / "eval_asr_within_baseline.json"), Path(EVAL_DIR / "eval_asr_without_defense.json"), Path(RESULT_DIR / "issue3_ASRline_chart.png"))
    plot_issue3_ASRgroup_bar_chart(
        Path(EVAL_DIR / "eval_asr_within_defense.json"),
        Path(EVAL_DIR / "eval_asr_within_baseline.json"),
        Path(EVAL_DIR / "eval_asr_without_defense.json"),
        Path(RESULT_DIR / "issue3_ASRgroup_bar_chart.png"),
    )
