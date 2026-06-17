#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SUBMISSION_TSV=${TRAIN_SUBMISSION_TSV:-/project_gpfs/bata0/bjin0/svhn_full_slate_20260604_031305/submission.tsv}
EVAL_SLURM=${EVAL_SLURM:-evaluate_svhn_final.slurm}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/svhn_evaluation_nll_fid_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$PARENT_EVAL_DIR/submission.tsv}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
FORCE=${FORCE:-0}
SKIP_NLL=${SKIP_NLL:-0}
SKIP_FID=${SKIP_FID:-0}
SKIP_ALREADY_SUBMITTED=${SKIP_ALREADY_SUBMITTED:-1}
EXCLUDE_RUNS=${EXCLUDE_RUNS:-linabar_cosine_hybrid}
SUSPICIOUS_RUNS=${SUSPICIOUS_RUNS:-linabar_cosine_vlb}
INCLUDE_SUSPICIOUS=${INCLUDE_SUSPICIOUS:-1}

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
    printf "submitted_at\tstatus\tjob_id\ttrain_job_id\tdataset\tschedule_name\tobjective\trun_name\teval_name\tmodel_path\treason\n" > "$SUBMISSION_TSV"
fi

contains_word() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

read_words() {
    local value="$1"
    # shellcheck disable=SC2086
    printf "%s\n" $value
}

find_model_path() {
    local train_job_id="$1"
    local run_name="$2"

    find "/project_gpfs/bata0/bjin0/bjin0/$train_job_id/logs" \
        -maxdepth 3 \
        -type f \
        -name "ema_0.9999_500000.pt" \
        -path "*svhn_${run_name}*" \
        -print 2>/dev/null | sort | tail -1
}

find_log_dir() {
    local train_job_id="$1"
    local run_name="$2"

    find "/project_gpfs/bata0/bjin0/bjin0/$train_job_id/logs" \
        -maxdepth 2 \
        -type d \
        -name "svhn_${run_name}" \
        -print 2>/dev/null | sort | tail -1
}

log_has_nan() {
    local log_dir="$1"
    [ -f "$log_dir/log.txt" ] && grep -qi "nan" "$log_dir/log.txt"
}

eval_already_submitted() {
    local eval_name="$1"
    if [ ! -f "$SUBMISSION_TSV" ]; then
        return 1
    fi
    awk -F '\t' -v eval_name="$eval_name" 'NR > 1 && $2 == "submitted" && $9 == eval_name {found=1} END {exit found ? 0 : 1}' "$SUBMISSION_TSV"
}

record_row() {
    local status="$1"
    local job_id="$2"
    local train_job_id="$3"
    local dataset="$4"
    local schedule_name="$5"
    local objective="$6"
    local run_name="$7"
    local eval_name="$8"
    local model_path="$9"
    local reason="${10}"

    if [ "$DRY_RUN" != "1" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "$status" "$job_id" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "$model_path" "$reason" >> "$SUBMISSION_TSV"
    fi
}

mapfile -t excluded < <(read_words "$EXCLUDE_RUNS")
mapfile -t suspicious < <(read_words "$SUSPICIOUS_RUNS")

echo "=========================================="
echo "Submitting SVHN NLL+FID evaluations"
echo "Training manifest: $TRAIN_SUBMISSION_TSV"
echo "Eval dir: $PARENT_EVAL_DIR"
echo "Dry run: $DRY_RUN"
echo "Skip NLL: $SKIP_NLL"
echo "Skip FID: $SKIP_FID"
echo "Exclude runs: $EXCLUDE_RUNS"
echo "Suspicious runs: $SUSPICIOUS_RUNS"
echo "Include suspicious: $INCLUDE_SUSPICIOUS"
echo "Skip already submitted: $SKIP_ALREADY_SUBMITTED"
echo "=========================================="

submitted=0
skipped=0

while IFS=$'\t' read -r submitted_at train_status train_job_id dataset schedule_name objective run_name data_dir time_limit dependency prep_job_id; do
    if [ "$submitted_at" = "submitted_at" ]; then
        continue
    fi
    if [ "$dataset" != "svhn" ] || [ "$schedule_name" = "prepare" ]; then
        continue
    fi

    eval_name="svhn_${run_name}"
    reason=""

    if contains_word "$run_name" "${excluded[@]}"; then
        reason="excluded_by_default"
        echo "SKIP $run_name: $reason"
        record_row "skipped" "" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "" "$reason"
        skipped=$((skipped + 1))
        continue
    fi

    if contains_word "$run_name" "${suspicious[@]}"; then
        reason="warning_large_final_grad_norm"
        if [ "$INCLUDE_SUSPICIOUS" != "1" ]; then
            echo "SKIP $run_name: $reason"
            record_row "skipped" "" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "" "$reason"
            skipped=$((skipped + 1))
            continue
        fi
    fi

    if [ "$SKIP_ALREADY_SUBMITTED" = "1" ] && eval_already_submitted "$eval_name"; then
        echo "SKIP $run_name: already submitted in $SUBMISSION_TSV"
        skipped=$((skipped + 1))
        continue
    fi

    log_dir=$(find_log_dir "$train_job_id" "$run_name")
    if [ -z "$log_dir" ]; then
        reason="missing_training_log_dir"
        echo "SKIP $run_name: $reason"
        record_row "skipped" "" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "" "$reason"
        skipped=$((skipped + 1))
        continue
    fi

    if log_has_nan "$log_dir"; then
        reason="training_log_contains_nan"
        echo "SKIP $run_name: $reason"
        record_row "skipped" "" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "" "$reason"
        skipped=$((skipped + 1))
        continue
    fi

    model_path=$(find_model_path "$train_job_id" "$run_name")
    if [ -z "$model_path" ]; then
        reason="missing_ema_0.9999_500000.pt"
        echo "SKIP $run_name: $reason"
        record_row "skipped" "" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "" "$reason"
        skipped=$((skipped + 1))
        continue
    fi

    job_name="eval_svhn_${schedule_name}_${objective}"
    export_arg="ALL,EVAL_NAME=${eval_name},RUN_NAME=${run_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},MODEL_PATH=${model_path},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},FORCE=${FORCE},SKIP_NLL=${SKIP_NLL},SKIP_FID=${SKIP_FID}"
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
    if [ -n "${STATS_FILE:-}" ]; then
        export_arg="${export_arg},STATS_FILE=${STATS_FILE}"
    fi

    echo "SUBMIT $run_name model=$model_path ${reason:+($reason)}"
    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --time=24:00:00 \
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
        record_row "submitted" "$job_id" "$train_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$eval_name" "$model_path" "$reason"
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
echo "=========================================="
