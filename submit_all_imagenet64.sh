#!/bin/bash

echo "=========================================="
echo "SUBMITTING ALL 9 IMAGENET-64 EXPERIMENTS"
echo "=========================================="
echo "Unconditional ImageNet-64, T=4000, 200K iterations:"
echo "  1. linear_simple"
echo "  2. linear_hybrid"
echo "  3. linear_vlb"
echo "  4. cosine_simple"
echo "  5. cosine_hybrid"
echo "  6. cosine_vlb"
echo "  7. ours_simple"
echo "  8. ours_hybrid"
echo "  9. ours_vlb"
echo "=========================================="

IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}

if [ ! -d "$IMAGENET_TRAIN_DIR" ] || [ ! -d "$IMAGENET_VAL_DIR" ]; then
  echo "ERROR: Default ImageNet dirs not found."
  echo "IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
  echo "IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
  echo "Set IMAGENET_TRAIN_DIR/IMAGENET_VAL_DIR if your dataset lives elsewhere."
  exit 1
fi

mkdir -p "slurm_logs"
echo "SLURM logs will be saved in the 'slurm_logs/' directory."
echo ""

submit_one () {
  local exp="$1"
  sbatch \
    --export=ALL,EXPERIMENT="$exp",IMAGENET_DATA_ROOT="$IMAGENET_DATA_ROOT",IMAGENET_TRAIN_DIR="$IMAGENET_TRAIN_DIR",IMAGENET_VAL_DIR="$IMAGENET_VAL_DIR" \
    --job-name="im64_train_${exp}" \
    --output="slurm_logs/im64_train_${exp}_%j.out" \
    --error="slurm_logs/im64_train_${exp}_%j.err" \
    train_imagenet64_no_mpi.slurm
}

submit_one linear_simple
submit_one linear_hybrid
submit_one linear_vlb
submit_one cosine_simple
submit_one cosine_hybrid
submit_one cosine_vlb
submit_one ours_simple
submit_one ours_hybrid
submit_one ours_vlb

echo ""
echo "All ImageNet-64 training jobs submitted."
