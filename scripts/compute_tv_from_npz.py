#!/usr/bin/env python3
"""
Compute a simple Total Variation (TV) distance between generated samples and a
directory of real images.

Definition (discrete):
    TV(P, Q) = 0.5 * sum_i |p_i - q_i|

Approximation used here (matches calculate_tv_distance.py):
  - Treat each color channel independently.
  - Build 256-bin histograms over raw uint8 pixel values (0..255).
  - Compute TV per channel and average over channels.

Intended usage: run inside evaluation SLURM jobs to produce one numeric TV score
per experiment directory, without heavy feature extractors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_real_hist(real_dir: Path) -> np.ndarray:
    imgs = []
    for p in real_dir.rglob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            imgs.append(p)
    if not imgs:
        raise FileNotFoundError(f"No images found in real_dir={real_dir}")

    hist = np.zeros((3, 256), dtype=np.float64)
    for p in imgs:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)  # (H, W, 3)
        for c in range(3):
            hist[c] += np.bincount(arr[..., c].ravel(), minlength=256)

    hist_sum = hist.sum(axis=1, keepdims=True)
    return hist / np.maximum(hist_sum, 1e-12)


def load_samples_hist(npz_path: Path) -> np.ndarray:
    if not npz_path.is_file():
        raise FileNotFoundError(f"Sample file not found: {npz_path}")

    data = np.load(npz_path)
    arr = data["arr_0"]
    data.close()

    # Ensure (N, H, W, 3)
    if arr.ndim == 4 and arr.shape[1] == 3:  # (N, 3, H, W)
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected sample array shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    hist = np.zeros((3, 256), dtype=np.float64)
    for c in range(3):
        hist[c] = np.bincount(arr[..., c].ravel(), minlength=256)
    hist_sum = hist.sum(axis=1, keepdims=True)
    return hist / np.maximum(hist_sum, 1e-12)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    tv_per_c = 0.5 * np.abs(p - q).sum(axis=1)
    return float(tv_per_c.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_npz", required=True, help="Path to samples_*.npz")
    ap.add_argument("--real_dir", required=True, help="Directory of real images")
    ap.add_argument("--output_txt", default=None, help="Write numeric TV here")
    args = ap.parse_args()

    samples_npz = Path(args.samples_npz)
    real_dir = Path(args.real_dir)
    if not real_dir.is_dir():
        raise FileNotFoundError(real_dir)

    real_hist = load_real_hist(real_dir)
    samp_hist = load_samples_hist(samples_npz)
    tv = tv_distance(real_hist, samp_hist)

    if args.output_txt:
        out = Path(args.output_txt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"{tv:.6f}\n")

    print(f"TV: {tv:.6f}")


if __name__ == "__main__":
    main()


