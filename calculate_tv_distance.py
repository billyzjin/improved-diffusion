#!/usr/bin/env python3
"""
Compute a simple Total Variation (TV) distance between generated samples and the
real CIFAR-10 test set. One TV score is produced per experiment directory.

Definition (discrete):
    TV(P, Q) = 0.5 * sum_i |p_i - q_i|

Approximation used here:
    - Treat each color channel independently.
    - Build 256-bin histograms over raw uint8 pixel values (0..255).
    - Compute TV per channel and average over channels.

Inputs:
    1) eval_dir: parent evaluation directory containing subdirs like
       cifar10_cosine_simple/, each with samples_*.npz.
    2) real_dir (optional): directory with real CIFAR-10 images (default ./cifar_test).

Output:
    - Writes tv_results.txt inside eval_dir with one line per experiment.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_real_hist(real_dir: Path) -> np.ndarray:
    """
    Load real images and compute per-channel histograms (256 bins).
    Returns hist shape (3, 256) normalized to sum to 1 per channel.
    """
    imgs = []
    for p in real_dir.rglob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            imgs.append(p)
    if not imgs:
        raise FileNotFoundError(f"No images found in real_dir={real_dir}")

    hist = np.zeros((3, 256), dtype=np.float64)
    for p in imgs:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)  # (H, W, 3)
        # accumulate hist per channel
        for c in range(3):
            hist[c] += np.bincount(arr[..., c].ravel(), minlength=256)

    # normalize
    hist_sum = hist.sum(axis=1, keepdims=True)
    hist = hist / np.maximum(hist_sum, 1e-12)
    return hist


def load_samples_hist(npz_path: Path) -> np.ndarray:
    """
    Load samples_*.npz and compute per-channel histograms (256 bins).
    Expects array shape (N, H, W, 3) or (N, 3, H, W) uint8.
    Returns hist shape (3, 256) normalized to sum to 1 per channel.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"Sample file not found: {npz_path}")

    data = np.load(npz_path)
    # handle both arr_0 only or with labels
    arr = data["arr_0"]
    # ensure (N, H, W, 3)
    if arr.ndim == 4 and arr.shape[1] == 3:  # (N, 3, H, W)
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected sample array shape: {arr.shape}")
    if arr.dtype != np.uint8:
        # if stored as float, clip/convert
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    hist = np.zeros((3, 256), dtype=np.float64)
    for c in range(3):
        hist[c] = np.bincount(arr[..., c].ravel(), minlength=256)
    hist_sum = hist.sum(axis=1, keepdims=True)
    hist = hist / np.maximum(hist_sum, 1e-12)
    return hist


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    p, q: histograms shape (3, 256), each channel sums to 1.
    Returns average TV over channels.
    """
    tv_per_c = 0.5 * np.abs(p - q).sum(axis=1)
    return float(tv_per_c.mean())


def find_sample_file(exp_dir: Path) -> Path:
    """
    Find samples_*.npz in the experiment directory.
    Picks the newest if multiple exist.
    """
    candidates = sorted(exp_dir.glob("samples_*x32x32x3.npz"))
    if not candidates:
        # fallback: any samples_*.npz
        candidates = sorted(exp_dir.glob("samples_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No samples_*.npz found in {exp_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Compute TV distance between generated samples and real CIFAR-10 test set.")
    parser.add_argument("eval_dir", type=str, help="Parent evaluation directory (contains cifar10_* subdirs).")
    parser.add_argument("--real_dir", type=str, default="./cifar_test", help="Directory with real CIFAR-10 test images.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    real_dir = Path(args.real_dir)

    if not eval_dir.is_dir():
        raise FileNotFoundError(f"eval_dir not found: {eval_dir}")
    if not real_dir.is_dir():
        raise FileNotFoundError(f"real_dir not found: {real_dir}")

    print(f"Loading real data from {real_dir} ...")
    real_hist = load_real_hist(real_dir)

    out_path = eval_dir / "tv_results.txt"
    lines = []
    lines.append("TOTAL VARIATION (TV) RESULTS (lower is better)")
    lines.append("===============================================")

    exp_dirs = sorted([d for d in eval_dir.glob("cifar10_*") if d.is_dir()])
    if not exp_dirs:
        raise FileNotFoundError(f"No cifar10_* subdirectories found in {eval_dir}")

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        try:
            sample_file = find_sample_file(exp_dir)
            print(f"[{exp_name}] Using sample file: {sample_file.name}")
            sample_hist = load_samples_hist(sample_file)
            tv = tv_distance(real_hist, sample_hist)
            lines.append(f"{exp_name:<25}: {tv:.6f}")
        except Exception as e:
            print(f"[{exp_name}] ERROR: {e}")
            lines.append(f"{exp_name:<25}: ERROR ({e})")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("==========================================")
    print("TV calculation complete.")
    print(f"Results saved to: {out_path}")
    print("==========================================")


if __name__ == "__main__":
    main()

