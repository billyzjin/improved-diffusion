"""
Plot per-step VLB terms across noise schedules.

Loads vb_terms.npz files from evaluation directories and plots the per-timestep
variational lower-bound loss L_t for each schedule on the same axes. The theory
predicts that the geometric schedule should produce more uniform per-step costs
compared to linear/cosine, which concentrate cost at the endpoints.

Usage:
    python3 plot_per_step_vlb.py --eval_dirs DIR1 DIR2 ...
    python3 plot_per_step_vlb.py  # auto-discover from /project_gpfs/bata0/bjin0
"""

import argparse
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


def find_vb_terms(eval_dirs):
    """Find all vb_terms.npz files across evaluation directories.

    Returns dict: {exp_name: path_to_vb_terms.npz}
    """
    results = {}
    for eval_dir in eval_dirs:
        if not os.path.isdir(eval_dir):
            continue
        for sub in os.listdir(eval_dir):
            vb_path = os.path.join(eval_dir, sub, "logger", "vb_terms.npz")
            if os.path.isfile(vb_path):
                results[sub] = vb_path
    return results


def auto_discover_eval_dirs(base="/project_gpfs/bata0/bjin0"):
    """Find all evaluation_parallel directories."""
    dirs = []
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if os.path.isdir(full) and "evaluation_parallel" in entry:
            dirs.append(full)
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(description="Plot per-step VLB across schedules")
    parser.add_argument(
        "--eval_dirs", nargs="*", default=None,
        help="Evaluation directories to scan. Auto-discovers if omitted.",
    )
    parser.add_argument(
        "--output_dir", default="plots",
        help="Directory to save plots (default: plots)",
    )
    parser.add_argument(
        "--log_scale", action="store_true",
        help="Use log scale for y-axis",
    )
    parser.add_argument(
        "--smoothing", type=int, default=20,
        help="Window size for moving average smoothing (default: 20). Set to 1 for no smoothing.",
    )
    args = parser.parse_args()

    if args.eval_dirs:
        eval_dirs = args.eval_dirs
    else:
        eval_dirs = auto_discover_eval_dirs()
        print(f"Auto-discovered {len(eval_dirs)} evaluation directories")

    vb_files = find_vb_terms(eval_dirs)
    if not vb_files:
        print("ERROR: No vb_terms.npz files found.")
        print("Make sure evaluation has been run and eval_dirs are correct.")
        return

    print(f"Found {len(vb_files)} experiments with VLB terms")

    # Group by (dataset, objective)
    groups = {}
    for exp_name, vb_path in vb_files.items():
        ds = extract_dataset(exp_name)
        obj = extract_objective(exp_name)
        sched = extract_schedule(exp_name)
        if ds and obj and sched:
            key = (ds, obj)
            if key not in groups:
                groups[key] = {}
            groups[key][sched] = vb_path

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot one figure per (dataset, objective)
    for (ds, obj), sched_dict in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(12, 5))

        for sched in SCHEDULE_ORDER:
            if sched not in sched_dict:
                continue
            data = np.load(sched_dict[sched])
            vb = data["arr_0"][::-1]  # reverse: calc_bpd_loop stores t=T-1 first
            T = len(vb)
            timesteps = np.arange(T)

            # Smooth for readability (use 'valid' + pad to avoid boundary artifacts)
            if args.smoothing > 1 and len(vb) > args.smoothing:
                kernel = np.ones(args.smoothing) / args.smoothing
                valid = np.convolve(vb, kernel, mode="valid")
                pad_left = args.smoothing // 2
                pad_right = len(vb) - len(valid) - pad_left
                vb_smooth = np.concatenate([vb[:pad_left], valid, vb[-pad_right:]]) if pad_right > 0 else np.concatenate([vb[:pad_left], valid])
            else:
                vb_smooth = vb

            color = SCHEDULE_COLORS.get(sched, "gray")
            ax.plot(timesteps, vb_smooth, label=sched, color=color, linewidth=1.0, alpha=0.85)

        ax.set_xlabel("Timestep $t$", fontsize=12)
        ax.set_ylabel("$L_t$ (bits/dim)", fontsize=12)
        ax.set_title(f"Per-step VLB: {ds} / {obj}", fontsize=13)
        if args.log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"per_step_vlb_{ds}_{obj}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Also make a combined plot per dataset (all objectives on subplots)
    for ds in DATASETS:
        ds_groups = {obj: sched_dict for (d, obj), sched_dict in groups.items() if d == ds}
        if not ds_groups:
            continue

        present_objectives = [obj for obj in OBJECTIVES if obj in ds_groups]
        fig, axes = plt.subplots(1, len(present_objectives), figsize=(6 * len(present_objectives), 5), squeeze=False)
        for idx, obj in enumerate(present_objectives):
            ax = axes[0][idx]
            sched_dict = ds_groups[obj]

            for sched in SCHEDULE_ORDER:
                if sched not in sched_dict:
                    continue
                data = np.load(sched_dict[sched])
                vb = data["arr_0"][::-1]  # reverse: calc_bpd_loop stores t=T-1 first
                T = len(vb)
                timesteps = np.arange(T)

                if args.smoothing > 1 and len(vb) > args.smoothing:
                    kernel = np.ones(args.smoothing) / args.smoothing
                    valid = np.convolve(vb, kernel, mode="valid")
                    pad_left = args.smoothing // 2
                    pad_right = len(vb) - len(valid) - pad_left
                    vb_smooth = np.concatenate([vb[:pad_left], valid, vb[-pad_right:]]) if pad_right > 0 else np.concatenate([vb[:pad_left], valid])
                else:
                    vb_smooth = vb

                color = SCHEDULE_COLORS.get(sched, "gray")
                ax.plot(timesteps, vb_smooth, label=sched, color=color, linewidth=1.0, alpha=0.85)

            ax.set_xlabel("Timestep $t$", fontsize=11)
            ax.set_ylabel("$L_t$ (bits/dim)", fontsize=11)
            ax.set_title(f"{obj}", fontsize=12)
            if args.log_scale:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Per-step VLB decomposition: {ds}", fontsize=14)
        plt.tight_layout()
        fname = f"per_step_vlb_{ds}_combined.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
