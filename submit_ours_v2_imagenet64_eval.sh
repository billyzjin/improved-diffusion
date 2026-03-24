#!/bin/bash

# Submits 3 parallel Slurm jobs to evaluate ours_v2 ImageNet-64 models.

IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-/project_gpfs/bjin0/imagenet64/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-/project_gpfs/bjin0/imagenet64/val}

if [ ! -d "$IMAGENET_TRAIN_DIR" ] || [ ! -d "$IMAGENET_VAL_DIR" ]; then
  echo "ERROR: Default ImageNet dirs not found."
  echo "IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
  echo "IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
  exit 1
fi

PARENT_EVAL_DIR="/project_gpfs/bjin0/imagenet64_evaluation_parallel_ours_v2_$(date +%Y%m%d_%H%M%S)"
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
    --export=ALL,EVAL_MODEL_NAME="$exp_name",PARENT_EVAL_DIR="$PARENT_EVAL_DIR",IMAGENET_TRAIN_DIR="$IMAGENET_TRAIN_DIR",IMAGENET_VAL_DIR="$IMAGENET_VAL_DIR" \
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
