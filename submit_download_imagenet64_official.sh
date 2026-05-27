#!/bin/bash

set -euo pipefail

mkdir -p slurm_logs

DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_downloads}
PART1_URL=${PART1_URL:-https://www.image-net.org/data/downsample/Imagenet64_train_part1_npz.zip}
PART2_URL=${PART2_URL:-https://www.image-net.org/data/downsample/Imagenet64_train_part2_npz.zip}
VAL_URL=${VAL_URL:-https://www.image-net.org/data/downsample/Imagenet64_val_npz.zip}
COOKIE_FILE=${COOKIE_FILE:-}

echo "Submitting ImageNet-64 official NPZ download job with:"
echo "  DOWNLOAD_ROOT=$DOWNLOAD_ROOT"
echo "  PART1_URL=$PART1_URL"
echo "  PART2_URL=$PART2_URL"
echo "  VAL_URL=$VAL_URL"
if [[ -n "$COOKIE_FILE" ]]; then
  echo "  COOKIE_FILE=$COOKIE_FILE"
fi

sbatch \
  --export=ALL,DOWNLOAD_ROOT="$DOWNLOAD_ROOT",PART1_URL="$PART1_URL",PART2_URL="$PART2_URL",VAL_URL="$VAL_URL",COOKIE_FILE="$COOKIE_FILE" \
  --job-name=im64_download \
  --output=slurm_logs/im64_download_%j.out \
  --error=slurm_logs/im64_download_%j.err \
  download_imagenet64_official.slurm
