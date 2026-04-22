"""
Compute Inception Score (IS) and Precision/Recall (PRC) from generated samples.

Uses torch-fidelity to compute metrics from npz sample files.
Samples are expected in (N, H, W, 3) uint8 format (as saved by our sampling scripts).

Usage:
    python3 scripts/compute_is_prc.py \
        --samples_npz path/to/samples_50000x32x32x3.npz \
        --real_dir ./cifar_test \
        --output_txt results.txt
"""

import argparse
import os
import tempfile

import numpy as np
from PIL import Image
import torch_fidelity


def npz_to_image_dir(npz_path, out_dir):
    """Extract npz samples to a directory of PNG images."""
    data = np.load(npz_path)
    arr = data["arr_0"]
    data.close()

    # Handle (N, C, H, W) -> (N, H, W, C)
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i])
        img.save(os.path.join(out_dir, f"{i:06d}.png"))
    return arr.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_npz", required=True)
    parser.add_argument("--real_dir", required=True,
                        help="Directory of real images (PNG/JPG)")
    parser.add_argument("--output_txt", default=None)
    parser.add_argument("--keep_images", action="store_true",
                        help="Don't delete extracted images after computation")
    parser.add_argument("--cpu", action="store_true",
                        help="Run on CPU instead of GPU")
    args = parser.parse_args()

    # Extract samples to temp directory
    tmpdir = tempfile.mkdtemp(prefix="is_prc_samples_")
    print(f"Extracting samples to {tmpdir}...")
    n = npz_to_image_dir(args.samples_npz, tmpdir)
    print(f"  Extracted {n} images")

    try:
        print("Computing IS and Precision/Recall...")
        metrics = torch_fidelity.calculate_metrics(
            input1=tmpdir,
            input2=args.real_dir,
            isc=True,           # Inception Score
            fid=False,          # Skip FID (we compute it separately with pytorch-fid)
            prc=True,           # Precision and Recall
            verbose=True,
            cuda=not args.cpu,
        )

        print(f"\nResults:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")

        if args.output_txt:
            with open(args.output_txt, "w") as f:
                for k, v in sorted(metrics.items()):
                    f.write(f"{k}: {v}\n")
            print(f"Saved to {args.output_txt}")

    finally:
        if not args.keep_images:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
