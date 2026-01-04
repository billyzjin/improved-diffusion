#!/bin/bash

# Submits 8 parallel Slurm jobs to evaluate each of the 8 MNIST models.
# Each job runs on its own dedicated GPU and computes NLL + FID + TV.

PARENT_EVAL_DIR="/project_gpfs/bjin0/mnist_evaluation_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 8 parallel MNIST evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

mkdir -p "slurm_logs"
echo "SLURM logs will be saved in the 'slurm_logs/' directory."

EXPERIMENTS=(
    "mnist_ours_simple"
    "mnist_linear_simple"
    "mnist_cosine_simple"
    "mnist_linear_hybrid"
    "mnist_cosine_hybrid"
    "mnist_cosine_vlb"
    "mnist_ours_hybrid"
    "mnist_ours_vlb"
)

for exp_name in "${EXPERIMENTS[@]}"; do
    echo "--> Submitting job for: $exp_name"
    sbatch \
        --export=ALL,EVAL_MODEL_NAME="$exp_name",PARENT_EVAL_DIR="$PARENT_EVAL_DIR" \
        --job-name="mn_eval_$exp_name" \
        --output="slurm_logs/mn_eval_${exp_name}_%j.out" \
        --error="slurm_logs/mn_eval_${exp_name}_%j.err" \
        evaluate_mnist_final.slurm
done

echo ""
echo "All 8 MNIST evaluation jobs have been submitted."
echo "======================================================================="
echo "After ALL jobs have completed successfully, run:"
echo "  bash aggregate_mnist_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="


