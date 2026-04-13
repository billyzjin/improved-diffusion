#!/bin/bash

# Submits 6 parallel Slurm jobs to evaluate geometric CIFAR-10 models.

PARENT_EVAL_DIR="/project_gpfs/bata0/bjin0/evaluation_parallel_geometric_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 6 parallel CIFAR-10 geometric evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

mkdir -p "slurm_logs"

EXPERIMENTS=(
    "cifar10_geometric_linear_simple"
    "cifar10_geometric_linear_hybrid"
    "cifar10_geometric_linear_vlb"
    "cifar10_geometric_cosine_simple"
    "cifar10_geometric_cosine_hybrid"
    "cifar10_geometric_cosine_vlb"
)

for exp_name in "${EXPERIMENTS[@]}"; do
    echo "--> Submitting job for: $exp_name"
    sbatch \
        --export=ALL,EVAL_MODEL_NAME="$exp_name",PARENT_EVAL_DIR="$PARENT_EVAL_DIR" \
        --job-name="eval_$exp_name" \
        --output="slurm_logs/eval_${exp_name}_%j.out" \
        --error="slurm_logs/eval_${exp_name}_%j.err" \
        evaluate_models_final.slurm
done

echo ""
echo "All 6 CIFAR-10 geometric evaluation jobs submitted."
echo "======================================================================="
echo "After ALL jobs have completed, run:"
echo "  bash aggregate_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="
