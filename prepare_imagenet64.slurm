#!/bin/bash

#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested (CPU-only; this is I/O heavy, not GPU heavy)
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

#---------------------------------------------------------------------------------
# Job specific name
#SBATCH --job-name=imagenet64_prepare
#SBATCH --output=imagenet64_prepare_%j.out
#SBATCH --error=imagenet64_prepare_%j.err

set -euo pipefail

echo "=========================================="
echo "SLURM_JOB_ID      = $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR  = $SLURM_SUBMIT_DIR"
echo "SLURM_NODELIST    = $SLURM_NODELIST"
echo "=========================================="

module load python/booth/3.12

cd /home/bjin0/improved-diffusion || exit 1

#---------------------------------------------------------------------------------
# Inputs (override via sbatch --export or environment)
#---------------------------------------------------------------------------------
TRAIN_NPZ_DIR_PART1=${TRAIN_NPZ_DIR_PART1:-/project_gpfs/bjin0/imagenet64_downloads/unzipped/Imagenet64_train_part1_npz}
TRAIN_NPZ_DIR_PART2=${TRAIN_NPZ_DIR_PART2:-/project_gpfs/bjin0/imagenet64_downloads/unzipped/Imagenet64_train_part2_npz}
VAL_NPZ=${VAL_NPZ:-/project_gpfs/bjin0/imagenet64_downloads/unzipped/Imagenet64_val_npz/val_data.npz}

# Output root directory (will create train/ and val/ under this)
OUT_ROOT=${OUT_ROOT:-/project_gpfs/bjin0/imagenet64}

# Optional: set MAX_IMAGES_PER_SPLIT for a quick smoke test (0 = full dataset).
MAX_IMAGES_PER_SPLIT=${MAX_IMAGES_PER_SPLIT:-0}

echo "TRAIN_NPZ_DIR_PART1  = $TRAIN_NPZ_DIR_PART1"
echo "TRAIN_NPZ_DIR_PART2  = $TRAIN_NPZ_DIR_PART2"
echo "VAL_NPZ              = $VAL_NPZ"
echo "OUT_ROOT             = $OUT_ROOT"
echo "MAX_IMAGES_PER_SPLIT = $MAX_IMAGES_PER_SPLIT"

if [ ! -d "$TRAIN_NPZ_DIR_PART1" ]; then
  echo "ERROR: TRAIN_NPZ_DIR_PART1 not found or not a directory: $TRAIN_NPZ_DIR_PART1"
  exit 1
fi
if [ ! -d "$TRAIN_NPZ_DIR_PART2" ]; then
  echo "ERROR: TRAIN_NPZ_DIR_PART2 not found or not a directory: $TRAIN_NPZ_DIR_PART2"
  exit 1
fi
if [ ! -f "$VAL_NPZ" ]; then
  echo "ERROR: VAL_NPZ not found: $VAL_NPZ"
  exit 1
fi

echo "Checking required Python packages..."
python3 -c "import numpy, PIL; print('numpy+PIL OK')" || {
  echo "ERROR: Missing Python packages. In this environment you need numpy + pillow."
  echo "Try: pip install --user numpy pillow"
  exit 1
}

mkdir -p "$OUT_ROOT"

echo "Starting ImageNet64 conversion..."
python3 datasets/imagenet64.py \
  --train_npz_dir_part1 "$TRAIN_NPZ_DIR_PART1" \
  --train_npz_dir_part2 "$TRAIN_NPZ_DIR_PART2" \
  --val_npz "$VAL_NPZ" \
  --out_root "$OUT_ROOT" \
  --max_images_per_split "$MAX_IMAGES_PER_SPLIT"

echo "=========================================="
echo "ImageNet64 prep job complete."
echo "Train dir: $OUT_ROOT/train"
echo "Val dir:   $OUT_ROOT/val"
echo "=========================================="

