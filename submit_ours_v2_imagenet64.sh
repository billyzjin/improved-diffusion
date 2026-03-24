#!/bin/bash

echo "=========================================="
echo "SUBMITTING 3 IMAGENET-64 ours_v2 EXPERIMENTS"
echo "=========================================="
echo "Optimal schedule beta_t = 1/(T-t+1):"
echo "  1. ours_v2_simple"
echo "  2. ours_v2_hybrid"
echo "  3. ours_v2_vlb"
echo "=========================================="

IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-/project_gpfs/bjin0/imagenet64/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-/project_gpfs/bjin0/imagenet64/val}

# NOTE: /project_gpfs may not be mounted on the login node. Skip the check
# here; the training SLURM script validates the paths on the compute node.
if [ ! -d "$IMAGENET_TRAIN_DIR" ] || [ ! -d "$IMAGENET_VAL_DIR" ]; then
  echo "WARNING: ImageNet dirs not visible from this node (may be compute-only)."
  echo "IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
  echo "IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
  echo "Submitting anyway — the compute node will validate."
fi

mkdir -p "slurm_logs"

submit_one () {
  local exp="$1"
  sbatch \
    --export=ALL,EXPERIMENT="$exp",IMAGENET_TRAIN_DIR="$IMAGENET_TRAIN_DIR",IMAGENET_VAL_DIR="$IMAGENET_VAL_DIR" \
    --job-name="im64_train_${exp}" \
    --output="slurm_logs/im64_train_${exp}_%j.out" \
    --error="slurm_logs/im64_train_${exp}_%j.err" \
    train_imagenet64_no_mpi.slurm
}

submit_one ours_v2_simple
submit_one ours_v2_hybrid
submit_one ours_v2_vlb

echo ""
echo "All 3 ImageNet-64 ours_v2 training jobs submitted."
echo "Monitor with: squeue -u \$USER"
