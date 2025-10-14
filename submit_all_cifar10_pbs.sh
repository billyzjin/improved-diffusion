#!/bin/bash

echo "=========================================="
echo "SUBMITTING ALL 5 CIFAR-10 EXPERIMENTS (PBS)"
echo "=========================================="

# Define the base PBS script
PBS_SCRIPT="train_cifar10_no_mpi.pbs"

# Define the experiments to run
EXPERIMENTS=("linear_simple" "linear_hybrid" "cosine_simple" "cosine_hybrid" "cosine_vlb")

# Loop through each experiment and submit a job
for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    echo "Submitting ${EXPERIMENT_NAME} experiment..."
    JOB_ID=$(qsub -v EXPERIMENT=${EXPERIMENT_NAME} "$PBS_SCRIPT" | awk '{print $1}')
    if [ -n "$JOB_ID" ]; then
        echo "  Job ID: $JOB_ID"
    else
        echo "  Failed to submit ${EXPERIMENT_NAME} experiment."
    fi
done

echo "=========================================="
echo "ALL EXPERIMENTS SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo "Use 'qstat -u \$USER' to check job status."
echo "Use 'tail -f *.o*' to monitor output."
