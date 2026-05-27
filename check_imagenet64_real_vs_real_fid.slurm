#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=imagenet64_real_vs_real_fid
#SBATCH --output=imagenet64_real_vs_real_fid_%j.out
#SBATCH --error=imagenet64_real_vs_real_fid_%j.err

set -euo pipefail

echo "=========================================="
echo "ImageNet-64 real-vs-real FID sanity check"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

module load python/booth/3.12 || { echo "ERROR: Failed to load python/booth/3.12"; exit 1; }

if module load cuda/12.8 2>/dev/null; then
  echo "Loaded cuda/12.8"
elif module load cuda/12.4 2>/dev/null; then
  echo "Loaded cuda/12.4"
elif module load cuda/12.2 2>/dev/null; then
  echo "Loaded cuda/12.2"
elif module load cuda/11.8 2>/dev/null; then
  echo "Loaded cuda/11.8"
else
  echo "ERROR: No CUDA module found"
  exit 1
fi

echo "Testing Python + PyTorch..."
python3 - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

cd /home/bjin0/improved-diffusion || exit 1

export IMAGENET_DATA_ROOT="${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}"
export IMAGENET_TRAIN_DIR="${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}"
export IMAGENET_VAL_DIR="${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}"
export IMAGENET_TRAIN_STATS="${IMAGENET_TRAIN_STATS:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505_train_stats.npz}"
export OUT_DIR="${OUT_DIR:-/project_gpfs/bata0/bjin0/imagenet64_real_vs_real_fid_${SLURM_JOB_ID}}"

mkdir -p "$OUT_DIR"

echo "Train dir : $IMAGENET_TRAIN_DIR"
echo "Val dir   : $IMAGENET_VAL_DIR"
echo "TrainStats: $IMAGENET_TRAIN_STATS"
echo "Out dir   : $OUT_DIR"

python3 -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('pytorch_fid') else 1)" || {
  echo "ERROR: pytorch_fid is not installed in this environment."
  echo "Install it (once) in your cluster python env: pip install pytorch-fid"
  exit 1
}

# Ensure train stats exist (computed over full train set, recursively).
if [ ! -f "$IMAGENET_TRAIN_STATS" ]; then
  echo "Train stats missing; building..."
  python3 scripts/compute_fid_stats_from_dir_recursive.py \
    --real_dir "$IMAGENET_TRAIN_DIR" \
    --stats_npz "$IMAGENET_TRAIN_STATS" \
    --device cuda \
    --batch_size 64 \
    --num_workers 4 \
    2>&1 | tee "$OUT_DIR/train_stats_build.txt"
fi

# Compute FID(val_images vs train_stats) using pytorch-fid internals, with recursive val scan.
python3 - <<'PY' 2>&1 | tee "$OUT_DIR/real_vs_real_fid_log.txt"
import os
from pathlib import Path

import numpy as np

val_dir = Path(os.environ["IMAGENET_VAL_DIR"])
stats_npz = Path(os.environ["IMAGENET_TRAIN_STATS"])
out_dir = Path(os.environ["OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
out_txt = out_dir / "real_vs_real_fid.txt"

exts = {".png", ".jpg", ".jpeg"}
val_files = [p for p in val_dir.rglob("*") if p.suffix.lower() in exts]
if not val_files:
    raise RuntimeError(f"No images found under IMAGENET_VAL_DIR={val_dir}")
print(f"Val images found: {len(val_files)}")

z = np.load(stats_npz)
mu_train = z["mu"]
sigma_train = z["sigma"]
z.close()
print(f"Loaded train stats: mu.shape={mu_train.shape} sigma.shape={sigma_train.shape}")

from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

device = "cuda"
batch_size = 64
num_workers = 4
dims = 2048

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM.get(dims)
if block_idx is None:
    raise ValueError(
        f"Unsupported dims={dims}. Available: {sorted(InceptionV3.BLOCK_INDEX_BY_DIM.keys())}"
    )

model = InceptionV3([block_idx]).to(device)
model.eval()

mu_val, sigma_val = calculate_activation_statistics(
    [str(p) for p in val_files],
    model,
    batch_size,
    dims,
    device,
    num_workers,
)

fid = float(calculate_frechet_distance(mu_val, sigma_val, mu_train, sigma_train))
print(f"FID(val vs train_stats): {fid:.6f}")
out_txt.write_text(f"{fid:.6f}\n")
print(f"Wrote: {out_txt}")
PY

echo "=========================================="
echo "Done: $(date)"
echo "Output dir: $OUT_DIR"
echo "FID result:"
cat "$OUT_DIR/real_vs_real_fid.txt"
