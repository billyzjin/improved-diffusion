#!/bin/bash

echo "=========================================="
echo "SUBMITTING 6 IMAGENET-64 GEOMETRIC EXPERIMENTS"
echo "=========================================="
echo "Geometric schedule (linear endpoints):"
echo "  1. geometric_linear_simple"
echo "  2. geometric_linear_hybrid"
echo "  3. geometric_linear_vlb"
echo "Geometric schedule (cosine endpoints):"
echo "  4. geometric_cosine_simple"
echo "  5. geometric_cosine_hybrid"
echo "  6. geometric_cosine_vlb"
echo "=========================================="

IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}

mkdir -p "slurm_logs"

for obj in simple hybrid vlb; do
    for endpt in linear cosine; do
        exp="geometric_${endpt}_${obj}"
        JOB=$(sbatch --export=ALL,EXPERIMENT="$exp",IMAGENET_DATA_ROOT="$IMAGENET_DATA_ROOT",IMAGENET_TRAIN_DIR="$IMAGENET_TRAIN_DIR",IMAGENET_VAL_DIR="$IMAGENET_VAL_DIR" \
            --job-name="train_${exp}" \
            --output="slurm_logs/train_${exp}_%j.out" \
            --error="slurm_logs/train_${exp}_%j.err" \
            train_imagenet64_no_mpi.slurm | awk '{print $4}')
        echo "  ${exp}: $JOB"
    done
done

echo ""
echo "All 6 ImageNet-64 geometric training jobs submitted."
echo "Monitor with: squeue -u \$USER"
