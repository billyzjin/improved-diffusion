#!/bin/bash

# Submits 3 parallel Slurm jobs to evaluate ours_v2 Fashion-MNIST models.

PARENT_EVAL_DIR="/project_gpfs/bjin0/fashion_evaluation_parallel_ours_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_EVAL_DIR"
echo "Submitting 3 parallel Fashion-MNIST ours_v2 evaluation jobs."
echo "Results will be saved in: $PARENT_EVAL_DIR"

mkdir -p "slurm_logs"

EXPERIMENTS=(
    "fashionmnist_ours_v2_simple"
    "fashionmnist_ours_v2_hybrid"
    "fashionmnist_ours_v2_vlb"
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
echo "All 3 Fashion-MNIST ours_v2 evaluation jobs submitted."
echo "======================================================================="
echo "After ALL jobs have completed, run:"
echo "  bash aggregate_fashionmnist_evaluation_results.sh $PARENT_EVAL_DIR"
echo "======================================================================="
