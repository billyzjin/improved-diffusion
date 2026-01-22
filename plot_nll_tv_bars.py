#!/usr/bin/env python3
"""
Parse results_summary-style text (like results.txt) and plot:
  1) NLL (bits/dim) bar chart
  2) TV distance bar chart

One bar per experiment (e.g. cifar10_cosine_hybrid), with bars colored by dataset.
FID is ignored.

Usage:
  python3 plot_nll_tv_bars.py --in_file results.txt
  python3 plot_nll_tv_bars.py --in_file results.txt --out_dir plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_results(text: str):
    # Dataset headers in your file.
    dataset = None  # "cifar10" | "fashionmnist" | "mnist"
    mode = None  # "nll" | "tv" | None

    # Keep insertion order (Python 3.7+ dict preserves order).
    nll = {}
    tv = {}
    exp_to_dataset = {}

    # Matches lines like:
    #   cifar10_cosine_hybrid    : 3.206410 bits/dimension
    #   mnist_linear_simple      : 0.272224
    line_re = re.compile(r"^\s*([a-z0-9_]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)")

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("COMPREHENSIVE MODEL EVALUATION RESULTS"):
            dataset = "cifar10"
            mode = None
            continue
        if line.startswith("FASHION-MNIST MODEL EVALUATION RESULTS"):
            dataset = "fashionmnist"
            mode = None
            continue
        if line.startswith("MNIST MODEL EVALUATION RESULTS"):
            dataset = "mnist"
            mode = None
            continue

        if line.startswith("NLL RESULTS"):
            mode = "nll"
            continue
        if line.startswith("TOTAL VARIATION (TV) RESULTS"):
            mode = "tv"
            continue
        if line.startswith("FID RESULTS"):
            mode = None  # ignore
            continue
        if line.startswith("SAMPLE GENERATION STATUS"):
            mode = None
            continue
        if line.startswith("PAPER BASELINE COMPARISON"):
            mode = None
            continue

        m = line_re.match(raw)
        if not m or mode not in ("nll", "tv") or dataset is None:
            continue

        exp = m.group(1).strip()
        val = float(m.group(2))

        exp_to_dataset[exp] = dataset
        if mode == "nll":
            nll[exp] = val
        elif mode == "tv":
            tv[exp] = val

    return nll, tv, exp_to_dataset


def plot_bars(
    *,
    title: str,
    ylabel: str,
    values: dict,
    exp_to_dataset: dict,
    out_path: Path,
    dataset_prefix_to_strip: str | None = None,
):
    import matplotlib.pyplot as plt

    items = list(values.items())
    # Sort bars in decreasing order of value.
    items.sort(key=lambda kv: kv[1], reverse=True)

    labels = []
    raw_labels = []
    for k, _ in items:
        raw_labels.append(k)
        if dataset_prefix_to_strip and k.startswith(dataset_prefix_to_strip):
            labels.append(k[len(dataset_prefix_to_strip) :])
        else:
            labels.append(k)
    y = [v for _, v in items]

    colors = []
    # Color by schedule family (linear / cosine / ours).
    # This is meaningful within each dataset plot.
    for short in labels:
        if short.startswith("linear_"):
            colors.append("#4C78A8")  # blue
        elif short.startswith("cosine_"):
            colors.append("#F58518")  # orange
        elif short.startswith("ours_"):
            colors.append("#54A24B")  # green
        else:
            colors.append("#9D9DA0")  # gray fallback

    fig_w = max(12, 0.45 * len(labels))
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.bar(range(len(labels)), y, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("Experiment", fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="results.txt", help="Path to results.txt")
    ap.add_argument("--out_dir", default=".", help="Output directory for PNGs")
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_dir = Path(args.out_dir)

    text = in_path.read_text()
    nll, tv, exp_to_dataset = parse_results(text)

    if not nll:
        raise SystemExit(f"No NLL entries found in {in_path}")
    if not tv:
        raise SystemExit(f"No TV entries found in {in_path}")

    datasets = [("cifar10", "CIFAR-10"), ("fashionmnist", "Fashion-MNIST"), ("mnist", "MNIST")]

    for ds_key, ds_label in datasets:
        nll_ds = {k: v for k, v in nll.items() if exp_to_dataset.get(k) == ds_key}
        tv_ds = {k: v for k, v in tv.items() if exp_to_dataset.get(k) == ds_key}

        if not nll_ds and not tv_ds:
            continue

        if nll_ds:
            out_path = out_dir / f"{ds_key}_nll_bar_chart.png"
            plot_bars(
                title=f"{ds_label}: NLL (bits/dim) by experiment",
                ylabel="NLL (bits/dim)",
                values=nll_ds,
                exp_to_dataset=exp_to_dataset,
                out_path=out_path,
                dataset_prefix_to_strip=f"{ds_key}_",
            )
            print(f"Wrote: {out_path}")

        if tv_ds:
            out_path = out_dir / f"{ds_key}_tv_bar_chart.png"
            plot_bars(
                title=f"{ds_label}: Total Variation (TV) distance by experiment",
                ylabel="TV distance",
                values=tv_ds,
                exp_to_dataset=exp_to_dataset,
                out_path=out_path,
                dataset_prefix_to_strip=f"{ds_key}_",
            )
            print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

