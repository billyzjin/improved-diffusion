#!/bin/bash

# Submits a single SLURM job that converts ImageNet64 npz shards into
# a directory-of-images dataset compatible with this repo's loader and SLURM scripts.

mkdir -p slurm_logs

# Override these before running if your paths differ:
TRAIN_NPZ_DIR_PART1=${TRAIN_NPZ_DIR_PART1:-/project_gpfs/bjin0/imagenet64_downloads/unzipped/Imagenet64_train_part1_npz}
TRAIN_NPZ_DIR_PART2=${TRAIN_NPZ_DIR_PART2:-/project_gpfs/bjin0/imagenet64_downloads/unzipped/Imagenet64_train_part2_npz}
VAL_NPZ=${VAL_NPZ:-/project_gpfs/bjin0/imagenet64_downloads/unzipped/Imagenet64_val_npz/val_data.npz}
OUT_ROOT=${OUT_ROOT:-/project_gpfs/bjin0/imagenet64}

# Optional smoke test: set to e.g. 2000 (0 = full dataset).
MAX_IMAGES_PER_SPLIT=${MAX_IMAGES_PER_SPLIT:-0}

echo "Submitting ImageNet64 prep job with:"
echo "  TRAIN_NPZ_DIR_PART1=$TRAIN_NPZ_DIR_PART1"
echo "  TRAIN_NPZ_DIR_PART2=$TRAIN_NPZ_DIR_PART2"
echo "  VAL_NPZ=$VAL_NPZ"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  MAX_IMAGES_PER_SPLIT=$MAX_IMAGES_PER_SPLIT"

sbatch \
  --export=ALL,TRAIN_NPZ_DIR_PART1="$TRAIN_NPZ_DIR_PART1",TRAIN_NPZ_DIR_PART2="$TRAIN_NPZ_DIR_PART2",VAL_NPZ="$VAL_NPZ",OUT_ROOT="$OUT_ROOT",MAX_IMAGES_PER_SPLIT="$MAX_IMAGES_PER_SPLIT" \
  --job-name=im64_prepare \
  --output=slurm_logs/im64_prepare_%j.out \
  --error=slurm_logs/im64_prepare_%j.err \
  prepare_imagenet64.slurm

