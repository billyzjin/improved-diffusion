#!/usr/bin/env python3
"""
Compute FID reference statistics (mu, sigma) from a directory of real images,
recursively (i.e. includes nested subdirectories).

Why this exists:
  - `pytorch-fid --save-stats` typically only scans the top-level directory for images.
  - Our ImageNet-64 prep stores images under class subdirectories (train/class####/*.png).
  - This helper matches the paper's protocol (stats over the full training set)
    while supporting nested directory layouts.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="Directory containing real images (scanned recursively).")
    ap.add_argument("--stats_npz", required=True, help="Output stats file (.npz) path.")
    ap.add_argument("--device", default="cuda", help="Device for Inception (cuda or cpu).")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for Inception forward passes.")
    ap.add_argument("--num_workers", type=int, default=4, help="PyTorch DataLoader workers.")
    ap.add_argument("--dims", type=int, default=2048, help="Inception feature dims (default 2048).")
    args = ap.parse_args()

    real_dir = Path(args.real_dir)
    stats_npz = Path(args.stats_npz)
    if not real_dir.is_dir():
        raise FileNotFoundError(real_dir)
    stats_npz.parent.mkdir(parents=True, exist_ok=True)

    # Collect images recursively.
    exts = {".png", ".jpg", ".jpeg"}
    files = [p for p in real_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"No images found under real_dir={real_dir}")

    # Import pytorch_fid internals.
    from pytorch_fid.fid_score import calculate_activation_statistics
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM.get(args.dims)
    if block_idx is None:
        raise ValueError(f"Unsupported dims={args.dims}. Available: {sorted(InceptionV3.BLOCK_INDEX_BY_DIM.keys())}")

    model = InceptionV3([block_idx]).to(args.device)
    model.eval()

    # pytorch-fid expects a list of file paths (strings).
    m, s = calculate_activation_statistics(
        [str(p) for p in files],
        model,
        args.batch_size,
        args.dims,
        args.device,
        args.num_workers,
    )

    np.savez(stats_npz, mu=m, sigma=s)
    print(f"Saved stats to {stats_npz} (n_images={len(files)})")


if __name__ == "__main__":
    main()

