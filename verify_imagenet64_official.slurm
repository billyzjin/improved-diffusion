#!/bin/bash

#---------------------------------------------------------------------------------
# Account information
#---------------------------------------------------------------------------------
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested (CPU-only; conversion and validation are I/O heavy)
#---------------------------------------------------------------------------------
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

#---------------------------------------------------------------------------------
# Job specific name
#---------------------------------------------------------------------------------
#SBATCH --job-name=imagenet64_verify
#SBATCH --output=imagenet64_verify_%j.out
#SBATCH --error=imagenet64_verify_%j.err

set -euo pipefail

echo "=========================================="
echo "ImageNet-64 official source verification"
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID:-}"
echo "SLURM_SUBMIT_DIR  = ${SLURM_SUBMIT_DIR:-}"
echo "SLURM_NODELIST    = ${SLURM_NODELIST:-}"
echo "Start             = $(date)"
echo "=========================================="

module load python/booth/3.12

cd /home/bjin0/improved-diffusion || exit 1

#---------------------------------------------------------------------------------
# Inputs. Override these via sbatch --export or the submit wrapper.
#---------------------------------------------------------------------------------
SOURCE_ROOT=${SOURCE_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_downloads/unzipped}
TRAIN_NPZ_DIR_PART1=${TRAIN_NPZ_DIR_PART1:-${SOURCE_ROOT}/Imagenet64_train_part1_npz}
TRAIN_NPZ_DIR_PART2=${TRAIN_NPZ_DIR_PART2:-${SOURCE_ROOT}/Imagenet64_train_part2_npz}
VAL_NPZ=${VAL_NPZ:-${SOURCE_ROOT}/Imagenet64_val_npz/val_data.npz}

# Fresh verified output. Do not use the old imagenet64_fixed_20260423 tree here.
OUT_ROOT=${OUT_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}

# Modes: audit-sources, convert, audit-tree, verify, all.
MODE=${MODE:-all}

# Optional smoke test: set to e.g. 2000. 0 means full official counts.
MAX_IMAGES_PER_SPLIT=${MAX_IMAGES_PER_SPLIT:-0}

WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
SAMPLE_COUNT=${SAMPLE_COUNT:-64}
SPOT_CHECK_COUNT=${SPOT_CHECK_COUNT:-128}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-50000}
PNG_COMPRESS_LEVEL=${PNG_COMPRESS_LEVEL:-0}

# Set to 1 only for fast smoke tests. Full validation should keep SHA256 enabled.
SKIP_SOURCE_SHA256=${SKIP_SOURCE_SHA256:-0}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-0}

echo "SOURCE_ROOT          = $SOURCE_ROOT"
echo "TRAIN_NPZ_DIR_PART1  = $TRAIN_NPZ_DIR_PART1"
echo "TRAIN_NPZ_DIR_PART2  = $TRAIN_NPZ_DIR_PART2"
echo "VAL_NPZ              = $VAL_NPZ"
echo "OUT_ROOT             = $OUT_ROOT"
echo "MODE                 = $MODE"
echo "MAX_IMAGES_PER_SPLIT = $MAX_IMAGES_PER_SPLIT"
echo "WORKERS              = $WORKERS"
echo "SAMPLE_COUNT         = $SAMPLE_COUNT"
echo "SPOT_CHECK_COUNT     = $SPOT_CHECK_COUNT"
echo "SKIP_SOURCE_SHA256   = $SKIP_SOURCE_SHA256"
echo "ALLOW_PARTIAL        = $ALLOW_PARTIAL"
echo "OVERWRITE            = $OVERWRITE"
echo "RESUME               = $RESUME"

echo "Checking required Python packages..."
python3 -c "import numpy, PIL; print('numpy+PIL OK')"

ARGS=(
  --mode "$MODE"
  --train-npz-dir-part1 "$TRAIN_NPZ_DIR_PART1"
  --train-npz-dir-part2 "$TRAIN_NPZ_DIR_PART2"
  --val-npz "$VAL_NPZ"
  --out-root "$OUT_ROOT"
  --workers "$WORKERS"
  --sample-count "$SAMPLE_COUNT"
  --spot-check-count "$SPOT_CHECK_COUNT"
  --progress-interval "$PROGRESS_INTERVAL"
  --png-compress-level "$PNG_COMPRESS_LEVEL"
)

if [[ "$MAX_IMAGES_PER_SPLIT" != "0" ]]; then
  ARGS+=(--max-images-per-split "$MAX_IMAGES_PER_SPLIT")
fi
if [[ "$SKIP_SOURCE_SHA256" == "1" ]]; then
  ARGS+=(--skip-source-sha256)
fi
if [[ "$ALLOW_PARTIAL" == "1" ]]; then
  ARGS+=(--allow-partial)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi

python3 datasets/verify_imagenet64_official.py "${ARGS[@]}"

echo "=========================================="
echo "ImageNet-64 official verification complete"
echo "End      = $(date)"
echo "Manifest = $OUT_ROOT/verification/manifest.json"
echo "Train dir after acceptance: $OUT_ROOT/train"
echo "Val dir after acceptance:   $OUT_ROOT/val"
echo "=========================================="
