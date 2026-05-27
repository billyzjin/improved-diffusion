#!/bin/bash

# Submits 9 parallel Slurm jobs to evaluate each ImageNet-64 model.
# Each job computes NLL + FID + TV and writes results under a shared parent directory.

IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}
IMAGENET_TRAIN_STATS=${IMAGENET_TRAIN_STATS:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505_train_stats.npz}

if [ ! -d "$IMAGENET_TRAIN_DIR" ] || [ ! -d "$IMAGENET_VAL_DIR" ]; then
  echo "ERROR: Default ImageNet dirs not found."
  echo "IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
  echo "IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
  echo "Set IMAGENET_TRAIN_DIR/IMAGENET_VAL_DIR if your dataset lives elsewhere."
  exit 1
fi

PARENT_EVAL_DIR="/project_gpfs/bata0/bjin0/imagenet64_evaluation_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 9 parallel ImageNet-64 evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

mkdir -p "slurm_logs"
echo "SLURM logs will be saved in the 'slurm_logs/' directory."

EXPERIMENTS=(
  "imagenet64_linear_simple"
  "imagenet64_linear_hybrid"
  "imagenet64_linear_vlb"
  "imagenet64_cosine_simple"
  "imagenet64_cosine_hybrid"
  "imagenet64_cosine_vlb"
  "imagenet64_ours_simple"
  "imagenet64_ours_hybrid"
  "imagenet64_ours_vlb"
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
echo "All 9 ImageNet-64 evaluation jobs have been submitted."
echo "======================================================================="
echo "After ALL jobs have completed successfully, run:"
echo "  bash aggregate_imagenet64_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="
