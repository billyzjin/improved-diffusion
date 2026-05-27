#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SUBMISSION_TSV=${TRAIN_SUBMISSION_TSV:-/project_gpfs/bata0/bjin0/tuned_geometric_full_slate_20260518_004340/submission.tsv}
EVAL_SLURM=${EVAL_SLURM:-evaluate_tuned_geometric_final.slurm}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/tuned_geometric_evaluation_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$PARENT_EVAL_DIR/submission.tsv}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
READY_ONLY=${READY_ONLY:-1}
FORCE=${FORCE:-0}
SKIP_NLL=${SKIP_NLL:-0}
SKIP_FID=${SKIP_FID:-0}
DATASETS=${DATASETS:-all}
SKIP_ALREADY_SUBMITTED=${SKIP_ALREADY_SUBMITTED:-1}

if [ ! -f "$TRAIN_SUBMISSION_TSV" ]; then
    echo "ERROR: training submission manifest not found: $TRAIN_SUBMISSION_TSV"
    exit 1
fi
if [ ! -f "$EVAL_SLURM" ]; then
    echo "ERROR: eval slurm not found: $EVAL_SLURM"
    exit 1
fi

mkdir -p "$PARENT_EVAL_DIR" "$SLURM_LOG_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\ttrain_job_id\tdataset\tschedule_name\tobjective\trun_name\teval_name\tbeta_1\talpha_bar_T\tmodel_path\treason\n" > "$SUBMISSION_TSV"
fi

dataset_selected() {
    local dataset="$1"
    if [ "$DATASETS" = "all" ]; then
        return 0
    fi
    for item in $DATASETS; do
        if [ "$item" = "$dataset" ]; then
            return 0
        fi
    done
    return 1
}

expected_ema_for_dataset() {
    case "$1" in
        imagenet64) echo "ema_0.9999_200000.pt" ;;
        *) echo "ema_0.9999_500000.pt" ;;
    esac
}

find_model_path() {
    local train_job_id="$1"
    local dataset="$2"
    local run_name="$3"
    local ema_name="$4"

    find "/project_gpfs/bata0/bjin0/bjin0/$train_job_id/logs" \
        -maxdepth 3 \
        -type f \
        -name "$ema_name" \
        -path "*${dataset}*${run_name}*" \
        -print 2>/dev/null | sort | tail -1
}

eval_already_submitted() {
    local eval_name="$1"
    if [ ! -f "$SUBMISSION_TSV" ]; then
        return 1
    fi
    awk -F '\t' -v eval_name="$eval_name" 'NR > 1 && $2 == "submitted" && $9 == eval_name {found=1} END {exit found ? 0 : 1}' "$SUBMISSION_TSV"
}

time_for_dataset() {
    case "$1" in
        imagenet64) echo "24:00:00" ;;
        *) echo "24:00:00" ;;
    esac
}

echo "=========================================="
echo "Submitting tuned geometric NLL+FID evaluations"
echo "Training manifest: $TRAIN_SUBMISSION_TSV"
echo "Eval dir: $PARENT_EVAL_DIR"
echo "Datasets: $DATASETS"
echo "Ready only: $READY_ONLY"
echo "Dry run: $DRY_RUN"
echo "Skip NLL: $SKIP_NLL"
echo "Skip FID: $SKIP_FID"
echo "Skip already submitted: $SKIP_ALREADY_SUBMITTED"
echo "=========================================="

submitted=0
skipped=0

while IFS=$'\t' read -r submitted_at train_status train_job_id dataset schedule_name objective run_name beta alpha reason; do
    if [ "$submitted_at" = "submitted_at" ]; then
        continue
    fi
    if ! dataset_selected "$dataset"; then
        continue
    fi

    ema_name=$(expected_ema_for_dataset "$dataset")
    model_path=$(find_model_path "$train_job_id" "$dataset" "$run_name" "$ema_name")
    eval_name="${dataset}_${run_name}"
    job_name="eval_tg_${dataset}_${schedule_name}_${objective}"

    if [ "$SKIP_ALREADY_SUBMITTED" = "1" ] && eval_already_submitted "$eval_name"; then
        echo "SKIP $dataset $run_name: already submitted in $SUBMISSION_TSV"
        skipped=$((skipped + 1))
        continue
    fi

    if [ -z "$model_path" ]; then
        message="missing $ema_name"
        echo "SKIP $dataset $run_name: $message"
        if [ "$DRY_RUN" != "1" ]; then
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$(date -Is)" "skipped" "" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "$beta" "$alpha" "" "$message" >> "$SUBMISSION_TSV"
        fi
        skipped=$((skipped + 1))
        if [ "$READY_ONLY" = "1" ]; then
            continue
        fi
    fi

    export_arg="ALL,DATASET=${dataset},RUN_NAME=${run_name},EVAL_NAME=${eval_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},GEOMETRIC_BETA1=${beta},GEOMETRIC_ALPHA_BAR_T=${alpha},MODEL_PATH=${model_path},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},FORCE=${FORCE},SKIP_NLL=${SKIP_NLL},SKIP_FID=${SKIP_FID}"
    if [ -n "${EVAL_TIMESTEP_RESPACING:-}" ]; then
        export_arg="${export_arg},EVAL_TIMESTEP_RESPACING=${EVAL_TIMESTEP_RESPACING}"
    fi
    if [ -n "${NLL_NUM_SAMPLES:-}" ]; then
        export_arg="${export_arg},NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES}"
    fi
    if [ -n "${NUM_SAMPLES:-}" ]; then
        export_arg="${export_arg},NUM_SAMPLES=${NUM_SAMPLES}"
    fi
    if [ -n "${NLL_BATCH_SIZE:-}" ]; then
        export_arg="${export_arg},NLL_BATCH_SIZE=${NLL_BATCH_SIZE}"
    fi
    if [ -n "${SAMPLE_BATCH_SIZE:-}" ]; then
        export_arg="${export_arg},SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE}"
    fi

    slurm_time=$(time_for_dataset "$dataset")
    echo "SUBMIT $dataset $run_name model=$model_path"
    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --time="$slurm_time" \
                --export="$export_arg" \
                --job-name="$job_name" \
                --output="$SLURM_LOG_DIR/${job_name}_%j.out" \
                --error="$SLURM_LOG_DIR/${job_name}_%j.err" \
                "$EVAL_SLURM"
        )
        echo "$sbatch_output"
        job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
        if [ -z "$job_id" ]; then
            job_id="UNKNOWN"
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "submitted" "$job_id" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "$beta" "$alpha" "$model_path" "$reason" >> "$SUBMISSION_TSV"
    fi

    submitted=$((submitted + 1))
    if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
        echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
        break
    fi
done < "$TRAIN_SUBMISSION_TSV"

echo "=========================================="
echo "Submission complete"
echo "Submitted: $submitted"
echo "Skipped: $skipped"
echo "Eval dir: $PARENT_EVAL_DIR"
echo "Manifest: $SUBMISSION_TSV"
echo "Aggregate after jobs finish:"
echo "  bash aggregate_tuned_geometric_evaluation_results.sh $PARENT_EVAL_DIR"
echo "=========================================="
