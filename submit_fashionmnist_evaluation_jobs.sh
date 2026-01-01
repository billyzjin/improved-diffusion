#!/bin/bash

# Submits 8 parallel Slurm jobs to evaluate each of the 8 Fashion-MNIST models.
# Each job runs on its own dedicated GPU and computes NLL + FID.

PARENT_EVAL_DIR="/project_gpfs/bjin0/fashion_evaluation_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 8 parallel Fashion-MNIST evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

mkdir -p "slurm_logs"
echo "SLURM logs will be saved in the 'slurm_logs/' directory."

EXPERIMENTS=(
    "fashionmnist_ours_simple"
    "fashionmnist_linear_simple"
    "fashionmnist_cosine_simple"
    "fashionmnist_linear_hybrid"
    "fashionmnist_cosine_hybrid"
    "fashionmnist_cosine_vlb"
    "fashionmnist_ours_hybrid"
    "fashionmnist_ours_vlb"
)

for exp_name in "${EXPERIMENTS[@]}"; do
    echo "--> Submitting job for: $exp_name"
    sbatch \
        --export=ALL,EVAL_MODEL_NAME="$exp_name",PARENT_EVAL_DIR="$PARENT_EVAL_DIR" \
        --job-name="fm_eval_$exp_name" \
        --output="slurm_logs/fm_eval_${exp_name}_%j.out" \
        --error="slurm_logs/fm_eval_${exp_name}_%j.err" \
        evaluate_fashionmnist_final.slurm
done

echo ""
echo "All 8 Fashion-MNIST evaluation jobs have been submitted."
echo "======================================================================="
echo "After ALL jobs have completed successfully, run:"
echo "  bash aggregate_fashionmnist_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="


