#!/bin/bash

echo "=========================================="
echo "SUBMITTING ALL 9 MNIST EXPERIMENTS"
echo "=========================================="
echo "This mirrors the CIFAR-10/Fashion-MNIST experiment matrix:"
echo "  1. linear_simple"
echo "  2. linear_hybrid"
echo "  2b. linear_vlb"
echo "  3. cosine_simple"
echo "  4. cosine_hybrid"
echo "  5. cosine_vlb"
echo "  6. ours_simple"
echo "  7. ours_hybrid"
echo "  8. ours_vlb"
echo "=========================================="

mkdir -p "slurm_logs"
echo "SLURM logs will be saved in the 'slurm_logs/' directory."
echo ""

JOB1=$(sbatch --export=EXPERIMENT=linear_simple --job-name="mn_train_linear_simple" --output="slurm_logs/mn_train_linear_simple_%j.out" --error="slurm_logs/mn_train_linear_simple_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB2=$(sbatch --export=EXPERIMENT=linear_hybrid --job-name="mn_train_linear_hybrid" --output="slurm_logs/mn_train_linear_hybrid_%j.out" --error="slurm_logs/mn_train_linear_hybrid_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB2B=$(sbatch --export=EXPERIMENT=linear_vlb --job-name="mn_train_linear_vlb" --output="slurm_logs/mn_train_linear_vlb_%j.out" --error="slurm_logs/mn_train_linear_vlb_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB3=$(sbatch --export=EXPERIMENT=cosine_simple --job-name="mn_train_cosine_simple" --output="slurm_logs/mn_train_cosine_simple_%j.out" --error="slurm_logs/mn_train_cosine_simple_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB4=$(sbatch --export=EXPERIMENT=cosine_hybrid --job-name="mn_train_cosine_hybrid" --output="slurm_logs/mn_train_cosine_hybrid_%j.out" --error="slurm_logs/mn_train_cosine_hybrid_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB5=$(sbatch --export=EXPERIMENT=cosine_vlb --job-name="mn_train_cosine_vlb" --output="slurm_logs/mn_train_cosine_vlb_%j.out" --error="slurm_logs/mn_train_cosine_vlb_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB6=$(sbatch --export=EXPERIMENT=ours_simple --job-name="mn_train_ours_simple" --output="slurm_logs/mn_train_ours_simple_%j.out" --error="slurm_logs/mn_train_ours_simple_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB7=$(sbatch --export=EXPERIMENT=ours_hybrid --job-name="mn_train_ours_hybrid" --output="slurm_logs/mn_train_ours_hybrid_%j.out" --error="slurm_logs/mn_train_ours_hybrid_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')
JOB8=$(sbatch --export=EXPERIMENT=ours_vlb --job-name="mn_train_ours_vlb" --output="slurm_logs/mn_train_ours_vlb_%j.out" --error="slurm_logs/mn_train_ours_vlb_%j.err" train_mnist_no_mpi.slurm | awk '{print $4}')

echo "=========================================="
echo "ALL MNIST EXPERIMENTS SUBMITTED!"
echo "=========================================="
echo "Job IDs:"
echo "  linear_simple:  $JOB1"
echo "  linear_hybrid:  $JOB2"
echo "  linear_vlb:     $JOB2B"
echo "  cosine_simple:  $JOB3"
echo "  cosine_hybrid:  $JOB4"
echo "  cosine_vlb:     $JOB5"
echo "  ours_simple:    $JOB6"
echo "  ours_hybrid:    $JOB7"
echo "  ours_vlb:       $JOB8"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"


