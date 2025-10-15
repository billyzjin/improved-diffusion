#!/bin/bash

echo "=========================================="
echo "SUBMITTING OUR CUSTOM NOISE SCHEDULE EXPERIMENTS"
echo "=========================================="

# Submit ours_simple experiment
echo "Submitting ours_simple experiment..."
JOB_ID_1=$(sbatch --export=EXPERIMENT=ours_simple train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "Job ID: $JOB_ID_1"

# Submit ours_hybrid experiment  
echo "Submitting ours_hybrid experiment..."
JOB_ID_2=$(sbatch --export=EXPERIMENT=ours_hybrid train_cifar10_no_mpi.slurm | awk '{print $4}')
echo "Job ID: $JOB_ID_2"

echo "=========================================="
echo "OUR CUSTOM EXPERIMENTS SUBMITTED!"
echo "=========================================="
echo "Ours Simple: Job $JOB_ID_1"
echo "Ours Hybrid: Job $JOB_ID_2"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm-*.out"
echo ""
echo "Check results in:"
echo "  /scratch/bjin0/\${SLURM_JOB_USER}/\${SLURM_JOB_ID}/logs/"
