#!/bin/bash

echo "=========================================="
echo "SUBMITTING 3 FASHION-MNIST ours_v2 EXPERIMENTS"
echo "=========================================="
echo "Optimal schedule beta_t = 1/(T-t+1):"
echo "  1. ours_v2_simple"
echo "  2. ours_v2_hybrid"
echo "  3. ours_v2_vlb"
echo "=========================================="

mkdir -p "slurm_logs"

JOB1=$(sbatch --export=EXPERIMENT=ours_v2_simple --job-name="fm_train_ours_v2_simple" --output="slurm_logs/fm_train_ours_v2_simple_%j.out" --error="slurm_logs/fm_train_ours_v2_simple_%j.err" train_fashionmnist_no_mpi.slurm | awk '{print $4}')
echo "  ours_v2_simple: $JOB1"

JOB2=$(sbatch --export=EXPERIMENT=ours_v2_hybrid --job-name="fm_train_ours_v2_hybrid" --output="slurm_logs/fm_train_ours_v2_hybrid_%j.out" --error="slurm_logs/fm_train_ours_v2_hybrid_%j.err" train_fashionmnist_no_mpi.slurm | awk '{print $4}')
echo "  ours_v2_hybrid: $JOB2"

JOB3=$(sbatch --export=EXPERIMENT=ours_v2_vlb --job-name="fm_train_ours_v2_vlb" --output="slurm_logs/fm_train_ours_v2_vlb_%j.out" --error="slurm_logs/fm_train_ours_v2_vlb_%j.err" train_fashionmnist_no_mpi.slurm | awk '{print $4}')
echo "  ours_v2_vlb:    $JOB3"

echo ""
echo "All 3 Fashion-MNIST ours_v2 training jobs submitted."
echo "Monitor with: squeue -u \$USER"
