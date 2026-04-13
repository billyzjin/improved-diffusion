"""
Plot training convergence curves across noise schedules.

Reads progress.csv files from training log directories and plots loss vs. step
for each schedule. Produces one plot per (dataset, objective) combination,
with all schedules overlaid.

Usage:
    python3 plot_training_convergence.py
    python3 plot_training_convergence.py --checkpoint_base /project_gpfs/bata0/bjin0/bjin0
    python3 plot_training_convergence.py --datasets cifar10 --objectives simple hybrid
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEDULE_COLORS = {
    "linear": "#1f77b4",
    "cosine": "#ff7f0e",
    "ours": "#2ca02c",
    "ours_v2": "#d62728",
    "geometric_linear": "#9467bd",
    "geometric_cosine": "#e377c2",
}

SCHEDULE_ORDER = ["linear", "cosine", "ours", "geometric_linear", "geometric_cosine"]

OBJECTIVES = ["simple", "hybrid", "vlb"]

DATASETS = ["cifar10", "fashionmnist", "mnist", "imagenet64"]

# Log interval per dataset (steps between CSV rows)
LOG_INTERVALS = {
    "cifar10": 1000,
    "fashionmnist": 1000,
    "mnist": 1000,
    "imagenet64": 500,
}


def extract_schedule(exp_name):
    """Extract schedule name from experiment name like 'cifar10_geometric_linear_simple'."""
    for sched in sorted(SCHEDULE_COLORS.keys(), key=len, reverse=True):
        if sched in exp_name:
            return sched
    return None


def extract_objective(exp_name):
    """Extract objective from experiment name."""
    for obj in OBJECTIVES:
        if exp_name.endswith("_" + obj):
            return obj
    return None


def extract_dataset(exp_name):
    """Extract dataset from experiment name."""
    for ds in DATASETS:
        if exp_name.startswith(ds + "_"):
            return ds
    return None


def find_progress_files(checkpoint_base):
    """Find all progress.csv files in training directories.

    The structure is: {checkpoint_base}/{job_id}/logs/{exp_name}/progress.csv
    When multiple jobs exist for the same experiment, pick the one with the
    most rows (most training progress).

    Returns dict: {exp_name: path_to_progress.csv}
    """
    results = {}
    if not os.path.isdir(checkpoint_base):
        print(f"WARNING: checkpoint_base {checkpoint_base} does not exist")
        return results

    for job_id in os.listdir(checkpoint_base):
        logs_dir = os.path.join(checkpoint_base, job_id, "logs")
        if not os.path.isdir(logs_dir):
            continue
        for exp_name in os.listdir(logs_dir):
            csv_path = os.path.join(logs_dir, exp_name, "progress.csv")
            if not os.path.isfile(csv_path):
                continue
            # Count rows to pick the most complete run
            try:
                with open(csv_path) as f:
                    n_rows = sum(1 for _ in f) - 1  # subtract header
            except OSError:
                continue
            if n_rows <= 0:
                continue
            if exp_name not in results or n_rows > results[exp_name][1]:
                results[exp_name] = (csv_path, n_rows)

    return {name: path for name, (path, _) in results.items()}


def load_progress_csv(csv_path, metric="loss"):
    """Load a metric column from progress.csv.

    Returns numpy array of metric values, one per logged step.
    """
    values = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            return np.array([])
        for row in reader:
            try:
                values.append(float(row[metric]))
            except (ValueError, KeyError):
                values.append(np.nan)
    return np.array(values)


def smooth(values, window=10):
    """Apply simple moving average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Use 'valid' mode and pad to keep length
    smoothed = np.convolve(values, kernel, mode="same")
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Plot training convergence curves")
    parser.add_argument(
        "--checkpoint_base", default="/project_gpfs/bata0/bjin0/bjin0",
        help="Base directory containing {job_id}/logs/{exp_name}/progress.csv",
    )
    parser.add_argument(
        "--output_dir", default="plots",
        help="Directory to save plots (default: plots)",
    )
    parser.add_argument(
        "--metric", default="loss",
        help="Metric to plot from progress.csv (default: loss)",
    )
    parser.add_argument(
        "--smoothing", type=int, default=10,
        help="Moving average window size (default: 10). Set to 1 for no smoothing.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to plot (default: all found)",
    )
    parser.add_argument(
        "--objectives", nargs="*", default=None,
        help="Objectives to plot (default: all found)",
    )
    args = parser.parse_args()

    progress_files = find_progress_files(args.checkpoint_base)
    if not progress_files:
        print("ERROR: No progress.csv files found.")
        print(f"Searched in: {args.checkpoint_base}/*/logs/*/progress.csv")
        return

    print(f"Found {len(progress_files)} experiments with training logs")

    # Group by (dataset, objective)
    groups = {}
    for exp_name, csv_path in progress_files.items():
        ds = extract_dataset(exp_name)
        obj = extract_objective(exp_name)
        sched = extract_schedule(exp_name)
        if not ds or not obj or not sched:
            continue
        if args.datasets and ds not in args.datasets:
            continue
        if args.objectives and obj not in args.objectives:
            continue
        key = (ds, obj)
        if key not in groups:
            groups[key] = {}
        groups[key][sched] = csv_path

    if not groups:
        print("ERROR: No matching experiments found after filtering.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot one figure per (dataset, objective)
    for (ds, obj), sched_dict in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(10, 5))
        log_interval = LOG_INTERVALS.get(ds, 1000)

        for sched in SCHEDULE_ORDER:
            if sched not in sched_dict:
                continue
            values = load_progress_csv(sched_dict[sched], metric=args.metric)
            if len(values) == 0:
                print(f"  WARNING: No '{args.metric}' data for {ds}_{sched}_{obj}")
                continue

            steps = np.arange(len(values)) * log_interval
            values_smooth = smooth(values, window=args.smoothing)

            color = SCHEDULE_COLORS.get(sched, "gray")
            ax.plot(steps, values_smooth, label=sched, color=color, linewidth=1.0, alpha=0.85)

        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel(args.metric, fontsize=12)
        ax.set_title(f"Training convergence: {ds} / {obj}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"convergence_{ds}_{obj}_{args.metric}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Combined plot per dataset
    for ds in DATASETS:
        ds_groups = {obj: sd for (d, obj), sd in groups.items() if d == ds}
        if not ds_groups:
            continue

        present_objectives = [obj for obj in OBJECTIVES if obj in ds_groups]
        n_panels = len(present_objectives)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
        log_interval = LOG_INTERVALS.get(ds, 1000)

        for idx, obj in enumerate(present_objectives):
            ax = axes[0][idx]
            sched_dict = ds_groups[obj]

            for sched in SCHEDULE_ORDER:
                if sched not in sched_dict:
                    continue
                values = load_progress_csv(sched_dict[sched], metric=args.metric)
                if len(values) == 0:
                    continue

                steps = np.arange(len(values)) * log_interval
                values_smooth = smooth(values, window=args.smoothing)

                color = SCHEDULE_COLORS.get(sched, "gray")
                ax.plot(steps, values_smooth, label=sched, color=color, linewidth=1.0, alpha=0.85)

            ax.set_xlabel("Training step", fontsize=11)
            ax.set_ylabel(args.metric, fontsize=11)
            ax.set_title(f"{obj}", fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Training convergence: {ds}", fontsize=14)
        plt.tight_layout()
        fname = f"convergence_{ds}_combined_{args.metric}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
