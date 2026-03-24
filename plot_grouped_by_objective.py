#!/usr/bin/env python3
"""
Grouped bar charts: bars grouped by training objective (simple, hybrid, vlb),
with one bar per noise schedule (linear, cosine, ours) within each group.

Reuses the same results.txt parsing as plot_nll_tv_bars.py.

Usage:
  python3 plot_grouped_by_objective.py --in_file results.txt --out_dir plots
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# Import the parser from the existing script.
from plot_nll_tv_bars import parse_results


SCHEDULES = ["linear", "cosine", "ours", "ours_v2"]
OBJECTIVES = ["simple", "hybrid", "vlb"]

SCHEDULE_COLORS = {
    "linear": "#4C78A8",  # blue
    "cosine": "#F58518",  # orange
    "ours": "#54A24B",    # green
    "ours_v2": "#E45756", # red (optimal schedule)
}

SCHEDULE_LABELS = {
    "linear": "Linear",
    "cosine": "Cosine",
    "ours": "Ours",
    "ours_v2": "Ours v2",
}

OBJECTIVE_LABELS = {
    "simple": r"$L_{\mathrm{simple}}$",
    "hybrid": r"$L_{\mathrm{hybrid}}$",
    "vlb": r"$L_{\mathrm{vlb}}$",
}


def plot_grouped(
    *,
    title: str,
    ylabel: str,
    values: dict[str, float],
    ds_key: str,
    out_path: Path,
):
    import matplotlib.pyplot as plt

    n_objectives = len(OBJECTIVES)
    n_schedules = len(SCHEDULES)
    bar_width = 0.18
    group_gap = 0.15

    fig, ax = plt.subplots(figsize=(8, 5))

    group_centers = np.arange(n_objectives)
    offsets = np.arange(n_schedules) - (n_schedules - 1) / 2

    for i, sched in enumerate(SCHEDULES):
        vals = []
        for obj in OBJECTIVES:
            key = f"{ds_key}_{sched}_{obj}"
            v = values.get(key, float("nan"))
            vals.append(v if math.isfinite(v) else 0.0)

        positions = group_centers + offsets[i] * (bar_width + 0.02)
        bars = ax.bar(
            positions,
            vals,
            width=bar_width,
            color=SCHEDULE_COLORS[sched],
            edgecolor="black",
            linewidth=0.4,
            label=SCHEDULE_LABELS[sched],
        )
        # Add value labels on top of bars.
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [OBJECTIVE_LABELS[obj] for obj in OBJECTIVES], fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="results.txt")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    text = Path(args.in_file).read_text()
    nll, fid, tv, exp_to_dataset = parse_results(text)

    datasets = [
        ("cifar10", "CIFAR-10"),
        ("fashionmnist", "Fashion-MNIST"),
        ("imagenet64", "ImageNet-64"),
    ]

    metrics = [
        ("nll", "NLL (bits/dim)", nll),
        ("fid", "FID", fid),
        ("tv", "TV distance", tv),
    ]

    out_dir = Path(args.out_dir)

    for ds_key, ds_label in datasets:
        for metric_key, metric_ylabel, metric_dict in metrics:
            ds_vals = {
                k: v
                for k, v in metric_dict.items()
                if exp_to_dataset.get(k) == ds_key
            }
            if not ds_vals:
                continue

            out_path = out_dir / f"{ds_key}_{metric_key}_by_objective.png"
            plot_grouped(
                title=f"{ds_label}: {metric_ylabel} by objective",
                ylabel=metric_ylabel,
                values=ds_vals,
                ds_key=ds_key,
                out_path=out_path,
            )
            print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
