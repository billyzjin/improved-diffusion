#!/bin/bash

# Submits 3 parallel Slurm jobs to evaluate ours_v2 ImageNet-64 models.

IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}
IMAGENET_TRAIN_STATS=${IMAGENET_TRAIN_STATS:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505_train_stats.npz}

# NOTE: /project_gpfs may not be mounted on the login node. Skip the check
# here; the evaluation SLURM script validates the paths on the compute node.
if [ ! -d "$IMAGENET_TRAIN_DIR" ] || [ ! -d "$IMAGENET_VAL_DIR" ]; then
  echo "WARNING: ImageNet dirs not visible from this node (may be compute-only)."
  echo "IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
  echo "IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
  echo "Submitting anyway — the compute node will validate."
fi

PARENT_EVAL_DIR="/project_gpfs/bata0/bjin0/imagenet64_evaluation_parallel_ours_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 3 parallel ImageNet-64 ours_v2 evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

mkdir -p "slurm_logs"

EXPERIMENTS=(
    "imagenet64_ours_v2_simple"
    "imagenet64_ours_v2_hybrid"
    "imagenet64_ours_v2_vlb"
)

for exp_name in "${EXPERIMENTS[@]}"; do
  echo "--> Submitting job for: $exp_name"
    sbatch \
    --export=ALL,EVAL_MODEL_NAME="$exp_name",PARENT_EVAL_DIR="$PARENT_EVAL_DIR",IMAGENET_DATA_ROOT="$IMAGENET_DATA_ROOT",IMAGENET_TRAIN_DIR="$IMAGENET_TRAIN_DIR",IMAGENET_VAL_DIR="$IMAGENET_VAL_DIR",IMAGENET_TRAIN_STATS="$IMAGENET_TRAIN_STATS" \
    --job-name="im64_eval_$exp_name" \
    --output="slurm_logs/im64_eval_${exp_name}_%j.out" \
    --error="slurm_logs/im64_eval_${exp_name}_%j.err" \
    evaluate_imagenet64_final.slurm
done

echo ""
echo "All 3 ImageNet-64 ours_v2 evaluation jobs submitted."
echo "======================================================================="
echo "After ALL jobs have completed, run:"
echo "  bash aggregate_imagenet64_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="
