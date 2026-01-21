#!/bin/bash

echo "=========================================="
echo "SUBMITTING ALL 9 CIFAR-10 EXPERIMENTS"
echo "=========================================="
echo "This will submit all experiments to reproduce the paper results and test custom schedules:"
echo "  1. linear_simple    - Target: FID = 2.90 (best FID)"
echo "  2. linear_hybrid    - Baseline comparison"
echo "  2b. linear_vlb      - Extra (not in paper Table 2)"
echo "  3. cosine_simple    - Cosine schedule test"
echo "  4. cosine_hybrid    - Cosine + learn_sigma"
echo "  5. cosine_vlb       - Target: NLL = 2.94 (best NLL)"
echo "  6. ours_simple      - Custom noise schedule"
echo "  7. ours_hybrid      - Custom noise schedule"
echo "  8. ours_vlb         - Custom noise schedule with VLB"
echo "=========================================="

# Create a directory for the SLURM logs for tidiness.
mkdir -p "slurm_logs"
echo "SLURM logs will be saved in the 'slurm_logs/' directory."
echo ""

# Submit all 8 experiments
echo "Submitting linear_simple experiment..."
JOB1=$(sbatch --export=EXPERIMENT=linear_simple --job-name="train_linear_simple" --output="slurm_logs/train_linear_simple_%j.out" --error="slurm_logs/train_linear_simple_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting linear_hybrid experiment..."
JOB2=$(sbatch --export=EXPERIMENT=linear_hybrid --job-name="train_linear_hybrid" --output="slurm_logs/train_linear_hybrid_%j.out" --error="slurm_logs/train_linear_hybrid_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting linear_vlb experiment..."
JOB2B=$(sbatch --export=EXPERIMENT=linear_vlb --job-name="train_linear_vlb" --output="slurm_logs/train_linear_vlb_%j.out" --error="slurm_logs/train_linear_vlb_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB2B"

echo "Submitting cosine_simple experiment..."
JOB3=$(sbatch --export=EXPERIMENT=cosine_simple --job-name="train_cosine_simple" --output="slurm_logs/train_cosine_simple_%j.out" --error="slurm_logs/train_cosine_simple_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting cosine_hybrid experiment..."
JOB4=$(sbatch --export=EXPERIMENT=cosine_hybrid --job-name="train_cosine_hybrid" --output="slurm_logs/train_cosine_hybrid_%j.out" --error="slurm_logs/train_cosine_hybrid_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB4"

echo "Submitting cosine_vlb experiment..."
JOB5=$(sbatch --export=EXPERIMENT=cosine_vlb --job-name="train_cosine_vlb" --output="slurm_logs/train_cosine_vlb_%j.out" --error="slurm_logs/train_cosine_vlb_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB5"

echo "Submitting ours_simple experiment..."
JOB6=$(sbatch --export=EXPERIMENT=ours_simple --job-name="train_ours_simple" --output="slurm_logs/train_ours_simple_%j.out" --error="slurm_logs/train_ours_simple_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB6"

echo "Submitting ours_hybrid experiment..."
JOB7=$(sbatch --export=EXPERIMENT=ours_hybrid --job-name="train_ours_hybrid" --output="slurm_logs/train_ours_hybrid_%j.out" --error="slurm_logs/train_ours_hybrid_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB7"

echo "Submitting ours_vlb experiment..."
JOB8=$(sbatch --export=EXPERIMENT=ours_vlb --job-name="train_ours_vlb" --output="slurm_logs/train_ours_vlb_%j.out" --error="slurm_logs/train_ours_vlb_%j.err" train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "  Job ID: $JOB8"

echo "=========================================="
echo "ALL EXPERIMENTS SUBMITTED SUCCESSFULLY!"
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
echo ""
echo "View output files in the 'slurm_logs/' directory, for example:"
echo "  tail -f slurm_logs/train_linear_simple_${JOB1}.out"
echo ""
echo "Expected runtime: ~12-24 hours per experiment"
echo "All experiments will run in parallel"
echo "=========================================="
