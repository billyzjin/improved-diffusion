#!/bin/bash

# Submit evaluation jobs with timestep respacing for fast-sampling FID comparison.
#
# Tests how each schedule degrades under reduced sampling steps.
# Only runs sampling + FID + TV (skips NLL, which is independent of respacing).
#
# Usage:
#   bash submit_respaced_eval.sh                           # all datasets, default respacing steps
#   bash submit_respaced_eval.sh --datasets cifar10        # one dataset
#   bash submit_respaced_eval.sh --steps 250 100 50        # custom respacing steps
#   bash submit_respaced_eval.sh --schedules geometric_linear geometric_cosine cosine  # subset

set -e

# Defaults
RESPACING_STEPS=(250 100 50)
TARGET_DATASETS=(cifar10 fashionmnist mnist imagenet64)
TARGET_SCHEDULES=(linear cosine geometric_linear geometric_cosine)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            shift
            RESPACING_STEPS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                RESPACING_STEPS+=("$1")
                shift
            done
            ;;
        --datasets)
            shift
            TARGET_DATASETS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TARGET_DATASETS+=("$1")
                shift
            done
            ;;
        --schedules)
            shift
            TARGET_SCHEDULES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TARGET_SCHEDULES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "RESPACED SAMPLING EVALUATION"
echo "=========================================="
echo "Datasets:   ${TARGET_DATASETS[*]}"
echo "Schedules:  ${TARGET_SCHEDULES[*]}"
echo "Respacing:  ${RESPACING_STEPS[*]}"
echo "=========================================="

# Map datasets to eval SLURM scripts
declare -A EVAL_SCRIPTS
EVAL_SCRIPTS[cifar10]="evaluate_models_final.slurm"
EVAL_SCRIPTS[fashionmnist]="evaluate_fashionmnist_final.slurm"
EVAL_SCRIPTS[mnist]="evaluate_mnist_final.slurm"
EVAL_SCRIPTS[imagenet64]="evaluate_imagenet64_final.slurm"

mkdir -p "slurm_logs"

TOTAL_JOBS=0

for RESPACE in "${RESPACING_STEPS[@]}"; do
    for DS in "${TARGET_DATASETS[@]}"; do
        EVAL_SCRIPT="${EVAL_SCRIPTS[$DS]}"
        if [ -z "$EVAL_SCRIPT" ]; then
            echo "ERROR: Unknown dataset: $DS"
            continue
        fi

        PARENT_EVAL_DIR="/project_gpfs/bata0/bjin0/${DS}_respaced_${RESPACE}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$PARENT_EVAL_DIR"

        for SCHED in "${TARGET_SCHEDULES[@]}"; do
            for OBJ in simple hybrid vlb; do
                EXP_NAME="${DS}_${SCHED}_${OBJ}"

                echo "--> Submitting: ${EXP_NAME} (respacing=${RESPACE})"
                sbatch \
                    --export=ALL,EVAL_MODEL_NAME="$EXP_NAME",PARENT_EVAL_DIR="$PARENT_EVAL_DIR",EVAL_TIMESTEP_RESPACING="$RESPACE",SKIP_NLL=1 \
                    --job-name="eval_r${RESPACE}_${SCHED}_${OBJ}" \
                    --output="slurm_logs/eval_r${RESPACE}_${EXP_NAME}_%j.out" \
                    --error="slurm_logs/eval_r${RESPACE}_${EXP_NAME}_%j.err" \
                    "$EVAL_SCRIPT"

                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            done
        done

        echo "Results for ${DS} (respacing=${RESPACE}) will be in: $PARENT_EVAL_DIR"
        echo ""
    done
done

echo "=========================================="
echo "Submitted $TOTAL_JOBS respaced evaluation jobs."
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
