#!/bin/bash

echo "=========================================="
echo "RESUME TRAINING FROM CHECKPOINT"
echo "=========================================="

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [experiment_name]"
    echo ""
    echo "Examples:"
    echo "  $0 /scratch/bjin0/job123/logs/cifar10_ours_simple/model100000.pt ours_simple"
    echo "  $0 /scratch/bjin0/job123/logs/cifar10_ours_hybrid/model150000.pt ours_hybrid"
    echo ""
    echo "To find checkpoints:"
    echo "  find /scratch/bjin0 -name 'model*.pt' -type f"
    exit 1
fi

CHECKPOINT_PATH="$1"
EXPERIMENT_NAME="${2:-ours_simple}"

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo ""

# Submit job with resume checkpoint
echo "Submitting resume job..."
JOB_ID=$(sbatch --export=EXPERIMENT=$EXPERIMENT_NAME,RESUME_CHECKPOINT="$CHECKPOINT_PATH" train_cifar10_no_mpi.slurm | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "Resume job submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f slurm-$JOB_ID.out"
    echo ""
    echo "Check logs in:"
    echo "  /scratch/bjin0/\${SLURM_JOB_USER}/\${SLURM_JOB_ID}/logs/"
else
    echo "ERROR: Failed to submit resume job"
    exit 1
fi
