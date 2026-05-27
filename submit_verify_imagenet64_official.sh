#!/bin/bash

# Submit the CPU-side official ImageNet-64 rebuild and verifier.
# Override paths before running if your source shards live elsewhere.

set -euo pipefail

mkdir -p slurm_logs

SOURCE_ROOT=${SOURCE_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_downloads/unzipped}
TRAIN_NPZ_DIR_PART1=${TRAIN_NPZ_DIR_PART1:-${SOURCE_ROOT}/Imagenet64_train_part1_npz}
TRAIN_NPZ_DIR_PART2=${TRAIN_NPZ_DIR_PART2:-${SOURCE_ROOT}/Imagenet64_train_part2_npz}
VAL_NPZ=${VAL_NPZ:-${SOURCE_ROOT}/Imagenet64_val_npz/val_data.npz}
OUT_ROOT=${OUT_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}

MODE=${MODE:-all}
MAX_IMAGES_PER_SPLIT=${MAX_IMAGES_PER_SPLIT:-0}
WORKERS=${WORKERS:-8}
SAMPLE_COUNT=${SAMPLE_COUNT:-64}
SPOT_CHECK_COUNT=${SPOT_CHECK_COUNT:-128}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-50000}
PNG_COMPRESS_LEVEL=${PNG_COMPRESS_LEVEL:-0}
SKIP_SOURCE_SHA256=${SKIP_SOURCE_SHA256:-0}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-0}

echo "Submitting ImageNet-64 official verification job with:"
echo "  SOURCE_ROOT=$SOURCE_ROOT"
echo "  TRAIN_NPZ_DIR_PART1=$TRAIN_NPZ_DIR_PART1"
echo "  TRAIN_NPZ_DIR_PART2=$TRAIN_NPZ_DIR_PART2"
echo "  VAL_NPZ=$VAL_NPZ"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  MODE=$MODE"
echo "  MAX_IMAGES_PER_SPLIT=$MAX_IMAGES_PER_SPLIT"
echo "  WORKERS=$WORKERS"
echo "  SAMPLE_COUNT=$SAMPLE_COUNT"
echo "  SPOT_CHECK_COUNT=$SPOT_CHECK_COUNT"
echo "  SKIP_SOURCE_SHA256=$SKIP_SOURCE_SHA256"
echo "  ALLOW_PARTIAL=$ALLOW_PARTIAL"
echo "  OVERWRITE=$OVERWRITE"
echo "  RESUME=$RESUME"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" && "$MODE" != "audit-tree" ]]; then
  if [[ ! -d "$TRAIN_NPZ_DIR_PART1" ]]; then
    echo "ERROR: TRAIN_NPZ_DIR_PART1 not found: $TRAIN_NPZ_DIR_PART1"
    echo "Set TRAIN_NPZ_DIR_PART1 or SOURCE_ROOT to the official ImageNet-64 NPZ download location."
    exit 1
  fi
  if [[ ! -d "$TRAIN_NPZ_DIR_PART2" ]]; then
    echo "ERROR: TRAIN_NPZ_DIR_PART2 not found: $TRAIN_NPZ_DIR_PART2"
    echo "Set TRAIN_NPZ_DIR_PART2 or SOURCE_ROOT to the official ImageNet-64 NPZ download location."
    exit 1
  fi
  if [[ ! -f "$VAL_NPZ" ]]; then
    echo "ERROR: VAL_NPZ not found: $VAL_NPZ"
    echo "Set VAL_NPZ or SOURCE_ROOT to the official ImageNet-64 NPZ download location."
    exit 1
  fi
fi

sbatch \
  --export=ALL,SOURCE_ROOT="$SOURCE_ROOT",TRAIN_NPZ_DIR_PART1="$TRAIN_NPZ_DIR_PART1",TRAIN_NPZ_DIR_PART2="$TRAIN_NPZ_DIR_PART2",VAL_NPZ="$VAL_NPZ",OUT_ROOT="$OUT_ROOT",MODE="$MODE",MAX_IMAGES_PER_SPLIT="$MAX_IMAGES_PER_SPLIT",WORKERS="$WORKERS",SAMPLE_COUNT="$SAMPLE_COUNT",SPOT_CHECK_COUNT="$SPOT_CHECK_COUNT",PROGRESS_INTERVAL="$PROGRESS_INTERVAL",PNG_COMPRESS_LEVEL="$PNG_COMPRESS_LEVEL",SKIP_SOURCE_SHA256="$SKIP_SOURCE_SHA256",ALLOW_PARTIAL="$ALLOW_PARTIAL",OVERWRITE="$OVERWRITE",RESUME="$RESUME" \
  --job-name=im64_verify \
  --output=slurm_logs/im64_verify_%j.out \
  --error=slurm_logs/im64_verify_%j.err \
  verify_imagenet64_official.slurm
