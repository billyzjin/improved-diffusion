#!/bin/bash

# Submits 6 parallel Slurm jobs to evaluate geometric Fashion-MNIST models.

PARENT_EVAL_DIR="/project_gpfs/bata0/bjin0/fashionmnist_evaluation_parallel_geometric_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 6 parallel Fashion-MNIST geometric evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

mkdir -p "slurm_logs"

EXPERIMENTS=(
    "fashionmnist_geometric_linear_simple"
    "fashionmnist_geometric_linear_hybrid"
    "fashionmnist_geometric_linear_vlb"
    "fashionmnist_geometric_cosine_simple"
    "fashionmnist_geometric_cosine_hybrid"
    "fashionmnist_geometric_cosine_vlb"
)

for exp_name in "${EXPERIMENTS[@]}"; do
    echo "--> Submitting job for: $exp_name"
    sbatch \
        --export=ALL,EVAL_MODEL_NAME="$exp_name",PARENT_EVAL_DIR="$PARENT_EVAL_DIR" \
        --job-name="eval_$exp_name" \
        --output="slurm_logs/eval_${exp_name}_%j.out" \
        --error="slurm_logs/eval_${exp_name}_%j.err" \
        evaluate_fashionmnist_final.slurm
done

echo ""
echo "All 6 Fashion-MNIST geometric evaluation jobs submitted."
echo "======================================================================="
echo "After ALL jobs have completed, run:"
echo "  bash aggregate_fashionmnist_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="
