"""
Generate sample visualization grids comparing noise schedules.

Loads generated sample .npz files from evaluation directories and creates
side-by-side image grids for visual comparison across schedules.

Usage:
    python3 plot_sample_grids.py --eval_dirs DIR1 DIR2 ...
    python3 plot_sample_grids.py  # auto-discover
    python3 plot_sample_grids.py --n_rows 4 --n_cols 8 --datasets cifar10
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEDULE_ORDER = ["linear", "cosine", "ours", "geometric_linear", "geometric_cosine"]

OBJECTIVES = ["simple", "hybrid", "vlb"]

DATASETS = ["cifar10", "fashionmnist", "mnist", "imagenet64"]

# Sample npz file name patterns
SAMPLE_PATTERNS = [
    "samples_50000x32x32x3.npz",
    "samples_50000x64x64x3.npz",
    "samples_10000x64x64x3.npz",
    "samples.npz",
]


def extract_schedule(exp_name):
    """Extract schedule name from experiment name."""
    schedules = ["geometric_linear", "geometric_cosine", "ours_v2", "ours", "cosine", "linear"]
    for sched in schedules:
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


def find_sample_files(eval_dirs):
    """Find all sample .npz files across evaluation directories.

    Returns dict: {exp_name: path_to_samples.npz}
    """
    results = {}
    for eval_dir in eval_dirs:
        if not os.path.isdir(eval_dir):
            continue
        for sub in os.listdir(eval_dir):
            sub_path = os.path.join(eval_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            for pattern in SAMPLE_PATTERNS:
                sample_path = os.path.join(sub_path, pattern)
                if os.path.isfile(sample_path):
                    results[sub] = sample_path
                    break
    return results


def auto_discover_eval_dirs(base="/project_gpfs/bata0/bjin0"):
    """Find all evaluation_parallel directories."""
    dirs = []
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if os.path.isdir(full) and "evaluation_parallel" in entry:
            dirs.append(full)
    return sorted(dirs)


def load_samples(npz_path, n_samples=None):
    """Load sample images from npz file.

    Returns array of shape (N, H, W, C) with dtype uint8.
    """
    data = np.load(npz_path)
    images = data["arr_0"]
    if n_samples is not None and n_samples < len(images):
        images = images[:n_samples]
    return images


def make_grid(images, n_rows, n_cols, padding=2, pad_value=255):
    """Arrange images into a grid.

    Args:
        images: (N, H, W, C) uint8 array
        n_rows, n_cols: grid dimensions
        padding: pixels between images
        pad_value: padding color (255 = white)

    Returns: (grid_H, grid_W, C) uint8 array
    """
    n = min(n_rows * n_cols, len(images))
    H, W, C = images.shape[1], images.shape[2], images.shape[3]
    grid_H = n_rows * H + (n_rows + 1) * padding
    grid_W = n_cols * W + (n_cols + 1) * padding
    grid = np.full((grid_H, grid_W, C), pad_value, dtype=np.uint8)

    for idx in range(n):
        row = idx // n_cols
        col = idx % n_cols
        y = padding + row * (H + padding)
        x = padding + col * (W + padding)
        grid[y : y + H, x : x + W] = images[idx]

    return grid


def main():
    parser = argparse.ArgumentParser(description="Generate sample comparison grids")
    parser.add_argument(
        "--eval_dirs", nargs="*", default=None,
        help="Evaluation directories to scan. Auto-discovers if omitted.",
    )
    parser.add_argument(
        "--output_dir", default="plots",
        help="Directory to save grid images (default: plots)",
    )
    parser.add_argument(
        "--n_rows", type=int, default=4,
        help="Number of rows in each per-experiment grid (default: 4)",
    )
    parser.add_argument(
        "--n_cols", type=int, default=8,
        help="Number of columns in each per-experiment grid (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Filter by dataset (default: all found)",
    )
    parser.add_argument(
        "--objectives", nargs="*", default=None,
        help="Filter by objective (default: all found)",
    )
    args = parser.parse_args()

    if args.eval_dirs:
        eval_dirs = args.eval_dirs
    else:
        eval_dirs = auto_discover_eval_dirs()
        print(f"Auto-discovered {len(eval_dirs)} evaluation directories")

    sample_files = find_sample_files(eval_dirs)
    if not sample_files:
        print("ERROR: No sample .npz files found.")
        return

    print(f"Found {len(sample_files)} experiments with samples")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Group by (dataset, objective)
    groups = {}
    for exp_name, sample_path in sample_files.items():
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
        groups[key][sched] = sample_path

    # 1. Per-experiment grids (one image per experiment)
    for exp_name, sample_path in sorted(sample_files.items()):
        ds = extract_dataset(exp_name)
        if args.datasets and ds not in args.datasets:
            continue
        obj = extract_objective(exp_name)
        if args.objectives and obj not in args.objectives:
            continue

        n_needed = args.n_rows * args.n_cols
        images = load_samples(sample_path, n_samples=max(n_needed * 10, 1000))
        indices = rng.choice(len(images), size=n_needed, replace=False)
        selected = images[indices]
        grid = make_grid(selected, args.n_rows, args.n_cols)

        fig, ax = plt.subplots(figsize=(args.n_cols * 1.2, args.n_rows * 1.2))
        ax.imshow(grid)
        ax.set_title(exp_name, fontsize=11)
        ax.axis("off")
        plt.tight_layout()

        fname = f"samples_{exp_name}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    # 2. Comparison grids: one figure per (dataset, objective), all schedules side by side
    for (ds, obj), sched_dict in sorted(groups.items()):
        available = [s for s in SCHEDULE_ORDER if s in sched_dict]
        if not available:
            continue

        # Use the same random indices for fair comparison
        n_per_schedule = args.n_rows * args.n_cols
        # Check total sample count without loading all data
        first_path = list(sched_dict.values())[0]
        with np.load(first_path) as data:
            total_samples = data["arr_0"].shape[0]
        indices = rng.choice(total_samples, size=n_per_schedule, replace=False)

        n_schedules = len(available)
        fig, axes = plt.subplots(
            1, n_schedules,
            figsize=(args.n_cols * 1.2 * n_schedules, args.n_rows * 1.2 + 0.6),
            squeeze=False,
        )

        for col, sched in enumerate(available):
            images = load_samples(sched_dict[sched])
            safe_indices = indices[indices < len(images)]
            selected = images[safe_indices[:n_per_schedule]]
            grid = make_grid(selected, args.n_rows, args.n_cols)

            ax = axes[0][col]
            ax.imshow(grid)
            ax.set_title(sched, fontsize=11)
            ax.axis("off")

        fig.suptitle(f"{ds} / {obj}", fontsize=13)
        plt.tight_layout()

        fname = f"sample_comparison_{ds}_{obj}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
