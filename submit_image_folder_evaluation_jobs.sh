#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

DATASET=${DATASET:-${1:-}}
if [ -z "$DATASET" ]; then
    echo "ERROR: set DATASET or pass it as the first argument (cifar100, celeba64, or lsun_bedroom64)"
    exit 1
fi

EVAL_SLURM=${EVAL_SLURM:-evaluate_image_folder_final.slurm}
TRAIN_SUBMISSION_TSV=${TRAIN_SUBMISSION_TSV:-}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/${DATASET}_evaluation_nll_fid_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$PARENT_EVAL_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
FORCE=${FORCE:-0}
SKIP_NLL=${SKIP_NLL:-0}
SKIP_FID=${SKIP_FID:-0}
EVAL_AFTER_TRAIN=${EVAL_AFTER_TRAIN:-0}
TRAIN_DEPENDENCY_TYPE=${TRAIN_DEPENDENCY_TYPE:-afterok}

case "$DATASET" in
    cifar100)
        DATA_ROOT=${DATA_ROOT:-/project_gpfs/bata0/bjin0/cifar100_32x32}
        TRAIN_DIR=${TRAIN_DIR:-$DATA_ROOT/train}
        TEST_DIR=${TEST_DIR:-$DATA_ROOT/test}
        IMAGE_SIZE=${IMAGE_SIZE:-32}
        EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT:-50000}
        EXPECTED_TEST_COUNT=${EXPECTED_TEST_COUNT:-10000}
        STATS_FILE=${STATS_FILE:-/project_gpfs/bata0/bjin0/cifar100_train_stats.npz}
        SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-256}
        NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES:-10000}
        FID_NUM_SAMPLES=${FID_NUM_SAMPLES:-50000}
        NLL_BATCH_SIZE=${NLL_BATCH_SIZE:-128}
        ;;
    celeba64)
        DATA_ROOT=${DATA_ROOT:-/project_gpfs/bata0/bjin0/celeba_64x64}
        TRAIN_DIR=${TRAIN_DIR:-$DATA_ROOT/train}
        TEST_DIR=${TEST_DIR:-$DATA_ROOT/test}
        IMAGE_SIZE=${IMAGE_SIZE:-64}
        EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT:-162770}
        EXPECTED_TEST_COUNT=${EXPECTED_TEST_COUNT:-19962}
        STATS_FILE=${STATS_FILE:-/project_gpfs/bata0/bjin0/celeba64_train_stats.npz}
        SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-64}
        NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES:-10000}
        FID_NUM_SAMPLES=${FID_NUM_SAMPLES:-50000}
        NLL_BATCH_SIZE=${NLL_BATCH_SIZE:-128}
        ;;
    lsun_bedroom64)
        DATA_ROOT=${DATA_ROOT:-/project_gpfs/bata0/bjin0/lsun_bedroom_64x64}
        LSUN_SOURCE_ROOT=${LSUN_SOURCE_ROOT:-$DATA_ROOT/source}
        TRAIN_DIR=${TRAIN_DIR:-$LSUN_SOURCE_ROOT/bedroom_train_lmdb}
        TEST_DIR=${TEST_DIR:-$LSUN_SOURCE_ROOT/bedroom_val_lmdb}
        IMAGE_SIZE=${IMAGE_SIZE:-64}
        EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT:-1}
        EXPECTED_TEST_COUNT=${EXPECTED_TEST_COUNT:-1}
        STATS_FILE=${STATS_FILE:-/project_gpfs/bata0/bjin0/lsun_bedroom64_train_stats.npz}
        SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-64}
        NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES:-300}
        FID_NUM_SAMPLES=${FID_NUM_SAMPLES:-50000}
        NLL_BATCH_SIZE=${NLL_BATCH_SIZE:-100}
        FID_STATS_BATCH_SIZE=${FID_STATS_BATCH_SIZE:-64}
        FID_STATS_NUM_WORKERS=${FID_STATS_NUM_WORKERS:-4}
        ;;
    *)
        echo "ERROR: unsupported DATASET=$DATASET; expected cifar100, celeba64, or lsun_bedroom64"
        exit 1
        ;;
esac

if [ -z "$TRAIN_SUBMISSION_TSV" ]; then
    echo "ERROR: TRAIN_SUBMISSION_TSV must point to the training submission.tsv"
    exit 1
fi
if [ ! -f "$TRAIN_SUBMISSION_TSV" ]; then
    echo "ERROR: training manifest not found: $TRAIN_SUBMISSION_TSV"
    exit 1
fi
if [ ! -f "$EVAL_SLURM" ]; then
    echo "ERROR: evaluation slurm not found: $EVAL_SLURM"
    exit 1
fi
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: train dir missing: $TRAIN_DIR"
    exit 1
fi
if [ ! -d "$TEST_DIR" ]; then
    echo "ERROR: test dir missing: $TEST_DIR"
    exit 1
fi

mkdir -p "$SLURM_LOG_DIR" "$PARENT_EVAL_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tdataset\tschedule_name\tobjective\trun_name\teval_name\tmodel_path\teval_dir\tnll_num_samples\tfid_num_samples\tforce\tskip_nll\tskip_fid\tdependency\ttrain_job_id\n" > "$SUBMISSION_TSV"
fi

echo "=========================================="
echo "Submitting image-folder NLL/FID evaluations"
echo "Dataset: $DATASET"
echo "Training manifest: $TRAIN_SUBMISSION_TSV"
echo "Eval dir: $PARENT_EVAL_DIR"
echo "Dry run: $DRY_RUN"
echo "Eval after train: $EVAL_AFTER_TRAIN"
echo "=========================================="

submitted=0
while IFS=$'\t' read -r submitted_at status job_id dataset schedule_name objective run_name train_dir image_size train_steps time_limit dependency prep_job_id logdir; do
    if [ "$submitted_at" = "submitted_at" ]; then
        continue
    fi
    if [ "$dataset" != "$DATASET" ]; then
        continue
    fi
    if [ "$status" != "submitted" ]; then
        continue
    fi
    if [ "$schedule_name" = "prepare" ]; then
        continue
    fi

    model_path="$logdir/ema_0.9999_${train_steps}.pt"
    dependency_arg=()
    dependency="none"
    if [ "$EVAL_AFTER_TRAIN" = "1" ] && [ ! -f "$model_path" ]; then
        dependency="${TRAIN_DEPENDENCY_TYPE}:${job_id}"
        dependency_arg=(--dependency="$dependency")
    fi

    if [ ! -f "$model_path" ] && [ "$EVAL_AFTER_TRAIN" != "1" ]; then
        echo "ERROR: missing final EMA checkpoint for $run_name: $model_path"
        echo "Training may still be running or failed. Re-run after the checkpoint exists."
        exit 1
    fi

    eval_name="${DATASET}_${run_name}"
    job_name="eval_${DATASET}_${schedule_name}_${objective}"
    export_arg="ALL,DATASET=${DATASET},EVAL_NAME=${eval_name},RUN_NAME=${run_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},MODEL_PATH=${model_path},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},TRAIN_DIR=${TRAIN_DIR},TEST_DIR=${TEST_DIR},STATS_FILE=${STATS_FILE},IMAGE_SIZE=${IMAGE_SIZE},NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES},NUM_SAMPLES=${FID_NUM_SAMPLES},NLL_BATCH_SIZE=${NLL_BATCH_SIZE},SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE},FID_STATS_BATCH_SIZE=${FID_STATS_BATCH_SIZE:-$SAMPLE_BATCH_SIZE},FID_STATS_NUM_WORKERS=${FID_STATS_NUM_WORKERS:-4},FORCE=${FORCE},SKIP_NLL=${SKIP_NLL},SKIP_FID=${SKIP_FID}"

    echo "SUBMIT eval $eval_name model=$model_path dependency=$dependency"
    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                "${dependency_arg[@]}" \
                --export="$export_arg" \
                --job-name="$job_name" \
                --output="$SLURM_LOG_DIR/${job_name}_%j.out" \
                --error="$SLURM_LOG_DIR/${job_name}_%j.err" \
                "$EVAL_SLURM"
        )
        echo "$sbatch_output"
        eval_job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
        if [ -z "$eval_job_id" ]; then
            eval_job_id="UNKNOWN"
        fi
        eval_status="submitted"
    else
        eval_job_id="DRY_RUN"
        eval_status="dry_run"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Is)" "$eval_status" "$eval_job_id" "$DATASET" "$schedule_name" "$objective" "$run_name" "$eval_name" "$model_path" "$PARENT_EVAL_DIR/$eval_name" "$NLL_NUM_SAMPLES" "$FID_NUM_SAMPLES" "$FORCE" "$SKIP_NLL" "$SKIP_FID" "$dependency" "$job_id" >> "$SUBMISSION_TSV"

    submitted=$((submitted + 1))
    if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
        echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
        break
    fi
done < "$TRAIN_SUBMISSION_TSV"

echo "=========================================="
echo "Submission complete"
echo "Submitted: $submitted"
echo "Submission manifest: $SUBMISSION_TSV"
echo "Aggregate after jobs finish:"
echo "  python3 scripts/aggregate_image_folder_evaluation_results.py \"$PARENT_EVAL_DIR\" --output \"results/${DATASET}_evaluation_nll_fid_$(basename "$PARENT_EVAL_DIR").tsv\""
echo "=========================================="
