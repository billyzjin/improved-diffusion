#!/bin/bash

# This script submits 7 parallel Slurm jobs to evaluate each of the 7 models.
# Each job will run on its own dedicated H100 GPU.

# 1. Create a single parent directory for all evaluation results from this run.
PARENT_EVAL_DIR="/scratch/bjin0/evaluation_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 7 parallel evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

# 2. Create a directory for the SLURM logs for tidiness.
mkdir -p "slurm_logs"
echo "SLURM logs will be saved in the 'slurm_logs/' directory."

# 3. List of all experiments to evaluate.
EXPERIMENTS=(
    "cifar10_ours_simple"
    "cifar10_linear_simple" 
    "cifar10_cosine_simple"
    "cifar10_linear_hybrid"
    "cifar10_cosine_hybrid"
    "cifar10_cosine_vlb"
    "cifar10_ours_hybrid"
)

# 4. Loop through the experiments and submit a separate job for each one.
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
echo "All 7 evaluation jobs have been submitted."
echo "======================================================================="
echo "To monitor job progress, run:"
echo "  squeue -u $USER"
echo ""
echo "After ALL jobs have completed, run the following command to gather results:"
echo "  bash aggregate_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="
