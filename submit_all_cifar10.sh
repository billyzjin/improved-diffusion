#!/bin/bash

echo "=========================================="
echo "SUBMITTING ALL 5 CIFAR-10 EXPERIMENTS"
echo "=========================================="
echo "This will submit all experiments to reproduce the paper results:"
echo "  1. linear_simple    - Target: FID = 2.90 (best FID)"
echo "  2. linear_hybrid    - Baseline comparison"
echo "  3. cosine_simple    - Cosine schedule test"
echo "  4. cosine_hybrid    - Cosine + learn_sigma"
echo "  5. cosine_vlb       - Target: NLL = 2.94 (best NLL)"
echo "=========================================="

# Submit all 5 experiments
echo "Submitting linear_simple experiment..."
JOB1=$(sbatch --export=EXPERIMENT=linear_simple train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting linear_hybrid experiment..."
JOB2=$(sbatch --export=EXPERIMENT=linear_hybrid train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting cosine_simple experiment..."
JOB3=$(sbatch --export=EXPERIMENT=cosine_simple train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting cosine_hybrid experiment..."
JOB4=$(sbatch --export=EXPERIMENT=cosine_hybrid train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB4"

echo "Submitting cosine_vlb experiment..."
JOB5=$(sbatch --export=EXPERIMENT=cosine_vlb train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB5"

echo "=========================================="
echo "ALL EXPERIMENTS SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo "Job IDs:"
echo "  linear_simple:  $JOB1"
echo "  linear_hybrid:  $JOB2"
echo "  cosine_simple:  $JOB3"
echo "  cosine_hybrid:  $JOB4"
echo "  cosine_vlb:     $JOB5"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job with:"
echo "  squeue -j $JOB1"
echo ""
echo "View output files:"
echo "  tail -f train_cifar10_no_mpi_${JOB1}.out"
echo "  tail -f train_cifar10_no_mpi_${JOB2}.out"
echo "  tail -f train_cifar10_no_mpi_${JOB3}.out"
echo "  tail -f train_cifar10_no_mpi_${JOB4}.out"
echo "  tail -f train_cifar10_no_mpi_${JOB5}.out"
echo ""
echo "Expected runtime: ~10-12 hours per experiment"
echo "All experiments will run in parallel"
echo "=========================================="
