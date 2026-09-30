"""
Std Testing -- figure generation.

Redraws the existing "Issue 1 Precision/Recall/Latency" chart
(results/issue_figure_generation.py:plot_issue1_precision_recall_latency_chart)
but sourced from results/std_testing/summary.json (mean +/- std across the
5 retrained-seed trials) instead of a single results/mode_X_results.json run
per mode -- i.e. the same visual layout, with error bars added.

This is deliberately a separate function writing to a separate output file
(results/std_testing/issue1_precision_recall_latency_chart_with_error_bars.png)
-- it never overwrites the original single-run chart
(results/issue1_precision_recall_latency_chart.png), which stays as the
single-run reference figure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from results.issue_figure_generation import (
    CONVERT_MODE,
    LATENCY_LINE_STYLE,
    LATENCY_VALUE_BOX,
    LEGEND_BOX_STYLE,
    _order_issue1_modes,
    _savefig_without_title,
)
from experiments.std_testing.common import STD_TESTING_RESULT_DIR

METRIC_KEYS = ["precision", "f1_score", "recall"]


def plot_issue1_precision_recall_latency_chart_with_error_bars(
    summary_path: Path,
    save_path: Optional[Path] = None,
) -> None:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary: Dict[str, Any] = json.load(f)

    mode_names = [f"mode_{mode_id}" for mode_id in summary.keys()]
    ordered = _order_issue1_modes(mode_names)

    means: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["latency_per_sample"]}
    stds: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["latency_per_sample"]}
    trial_counts: List[int] = []
    valid_modes: List[str] = []

    for mode_name in ordered:
        mode_id = mode_name.split("_", 1)[1]
        mode_summary = summary.get(mode_id)
        if not mode_summary:
            print(f"[WARN] Missing mode {mode_id} in {summary_path}")
            continue

        # Validate first, into a scratch dict -- only commit to the parallel
        # means/stds arrays once every required key is confirmed present, so
        # a mode missing one metric never leaves the per-metric lists at
        # mismatched lengths.
        mode_means: Dict[str, float] = {}
        mode_stds: Dict[str, float] = {}
        missing_key = False
        for key in METRIC_KEYS + ["latency_per_sample"]:
            entry = mode_summary.get(key)
            if not entry:
                print(f"[WARN] Missing '{key}' for mode {mode_id} in {summary_path}")
                missing_key = True
                break
            mode_means[key] = entry["mean"]
            mode_stds[key] = entry["std"]
        if missing_key:
            continue

        for key in METRIC_KEYS + ["latency_per_sample"]:
            means[key].append(mode_means[key])
            stds[key].append(mode_stds[key])

        trial_counts.append(mode_summary.get("num_trials", 0))
        valid_modes.append(CONVERT_MODE.get(mode_name, mode_name))

    if not valid_modes:
        print("[WARN] No valid Std Testing summary entries found for precision/recall latency chart.")
        return

    n_trials = max(trial_counts) if trial_counts else 0

    x = np.arange(len(valid_modes))
    width = 0.22
    error_kw = {"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2}

    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.bar(
        x - width, means["precision"], width,
        yerr=stds["precision"], capsize=4, error_kw=error_kw,
        label=f"Precision (mean±std, n={n_trials})", color="#0072B2",
        hatch="///", edgecolor="black", linewidth=0.75,
    )
    ax1.bar(
        x, means["f1_score"], width,
        yerr=stds["f1_score"], capsize=4, error_kw=error_kw,
        label=f"F1 (mean±std, n={n_trials})", color="#009E73",
        hatch="xxx", edgecolor="black", linewidth=0.75,
    )
    ax1.bar(
        x + width, means["recall"], width,
        yerr=stds["recall"], capsize=4, error_kw=error_kw,
        label=f"Recall (mean±std, n={n_trials})", color="#D55E00",
        hatch="\\\\\\", edgecolor="black", linewidth=0.75,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_modes, rotation=15)
    ax1.set_ylabel("Score")
    ax1.set_title(
        f"Issue 1 Precision/Recall Comparison Across Modes with Latency / Sample "
        f"(mean±std, n={n_trials} retrained seeds)"
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    lower_bounds = [
        v - s for key in METRIC_KEYS for v, s in zip(means[key], stds[key])
    ]
    upper_bounds = [
        v + s for key in METRIC_KEYS for v, s in zip(means[key], stds[key])
    ]
    y_min = max(0.0, min(lower_bounds) - 0.03)
    y_max = min(1.05, max(upper_bounds) + 0.06)
    ax1.set_ylim(y_min, y_max)

    # Deliberately no "Best F1" marker: with overlapping error bars across
    # several modes (see STD_RESULTS.md 4.4), no single mode can be claimed
    # as statistically best from n=5 trials -- annotating one as "best" would
    # misrepresent the data.

    ax2 = ax1.twinx()
    ax2.errorbar(
        x, means["latency_per_sample"], yerr=stds["latency_per_sample"],
        capsize=4, label=f"Latency / sample (mean±std, n={n_trials})",
        **LATENCY_LINE_STYLE,
    )
    ax2.set_ylabel("Latency / sample (seconds)")
    lat_mean = means["latency_per_sample"]
    lat_std = stds["latency_per_sample"]
    latency_min = min(m - s for m, s in zip(lat_mean, lat_std))
    latency_max = max(m + s for m, s in zip(lat_mean, lat_std))
    latency_range = max(0.001, latency_max - latency_min)
    ax2.set_ylim(
        max(0.0, latency_min - latency_range * 0.15),
        latency_max + latency_range * 0.25,
    )
    for xi, m, s in zip(x, lat_mean, lat_std):
        ax2.annotate(
            f"{m:.2f}±{s:.2f}s",
            xy=(xi, m + s),
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
        print(f"Saved: {save_path}")

    plt.close(fig)


def run_std_testing_generate_figures() -> None:
    print("\n" + "=" * 72)
    print("STD TESTING -- GENERATE FIGURES (mean±std)")
    print("=" * 72)

    summary_path = STD_TESTING_RESULT_DIR / "summary.json"
    if not summary_path.exists():
        print(
            f"{summary_path} not found. Run 'Aggregate' first (Std Testing menu) "
            "to produce it from the per-trial results."
        )
        return

    save_path = STD_TESTING_RESULT_DIR / "issue1_precision_recall_latency_chart_with_error_bars.png"
    plot_issue1_precision_recall_latency_chart_with_error_bars(summary_path, save_path)
    print("\nFigure generation complete.")


if __name__ == "__main__":
    run_std_testing_generate_figures()
