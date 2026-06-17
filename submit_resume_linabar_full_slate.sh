#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SLURM=${TRAIN_SLURM:-train_linabar_no_mpi.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/linabar_resume_full_slate_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$SUBMISSION_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY:-1}
MIN_CHECKPOINT_SIZE=${MIN_CHECKPOINT_SIZE:-100000000}

mkdir -p "$SLURM_LOG_DIR" "$SUBMISSION_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tmode\told_job_id\tdataset\tschedule_name\tobjective\trun_name\tresume_step\tresume_checkpoint\told_logdir\ttime_limit\tskip_dataset_verify\n" > "$SUBMISSION_TSV"
fi

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: training slurm not found: $TRAIN_SLURM"
    exit 1
fi

slurm_time_for_dataset() {
    case "$1" in
        imagenet64) echo "3-00:00:00" ;;
        *) echo "4-00:00:00" ;;
    esac
}

find_latest_complete_checkpoint() {
    local logdir="$1"
    local ckpt base step ema opt

    while IFS= read -r ckpt; do
        base=$(basename "$ckpt")
        step=${base#model}
        step=${step%.pt}
        ema="$logdir/ema_0.9999_${step}.pt"
        opt="$logdir/opt${step}.pt"
        if [ -s "$ema" ] && [ -s "$opt" ]; then
            if [ "$(stat -c%s "$ema")" -ge "$MIN_CHECKPOINT_SIZE" ] && [ "$(stat -c%s "$opt")" -ge "$MIN_CHECKPOINT_SIZE" ]; then
                printf "%s\t%s\n" "$step" "$ckpt"
                return 0
            fi
        fi
    done < <(find "$logdir" -maxdepth 1 -type f -name 'model*.pt' -size +"$((MIN_CHECKPOINT_SIZE - 1))"c -printf '%p\n' | sort -r)

    return 1
}

entries=(
    "125615|mnist|linabar_linear|simple"
    "125616|mnist|linabar_linear|hybrid"
    "125617|mnist|linabar_linear|vlb"
    "125618|mnist|linabar_cosine|simple"
    "125619|mnist|linabar_cosine|hybrid"
    "125620|mnist|linabar_cosine|vlb"
    "125621|fashionmnist|linabar_linear|simple"
    "125622|fashionmnist|linabar_linear|hybrid"
    "125623|fashionmnist|linabar_linear|vlb"
    "125624|fashionmnist|linabar_cosine|simple"
    "125625|fashionmnist|linabar_cosine|hybrid"
    "125626|fashionmnist|linabar_cosine|vlb"
    "125627|cifar10|linabar_linear|simple"
    "125628|cifar10|linabar_linear|hybrid"
    "125629|cifar10|linabar_linear|vlb"
    "125630|cifar10|linabar_cosine|simple"
    "125631|cifar10|linabar_cosine|hybrid"
    "125632|cifar10|linabar_cosine|vlb"
    "125633|imagenet64|linabar_linear|simple"
    "125634|imagenet64|linabar_linear|hybrid"
    "125635|imagenet64|linabar_linear|vlb"
    "125636|imagenet64|linabar_cosine|simple"
    "125637|imagenet64|linabar_cosine|hybrid"
    "125638|imagenet64|linabar_cosine|vlb"
)

echo "=========================================="
echo "Submitting linear-in-alpha_bar resume slate"
echo "Submission dir: $SUBMISSION_DIR"
echo "Dry run: $DRY_RUN"
echo "Skip dataset verify: $SKIP_DATASET_VERIFY"
echo "=========================================="

submitted=0
for entry in "${entries[@]}"; do
    IFS='|' read -r old_job_id dataset schedule_name objective <<< "$entry"
    run_name="${schedule_name}_${objective}"
    old_logdir="/project_gpfs/bata0/bjin0/bjin0/${old_job_id}/logs/${dataset}_${run_name}"
    slurm_time=$(slurm_time_for_dataset "$dataset")
    job_name="lab_resume_${dataset}_${schedule_name#linabar_}_${objective}"
    mode="restart"
    resume_step="0"
    resume_checkpoint=""

    if [ -d "$old_logdir" ]; then
        if latest_info=$(find_latest_complete_checkpoint "$old_logdir"); then
            resume_step=${latest_info%%$'\t'*}
            resume_checkpoint=${latest_info#*$'\t'}
            mode="resume"
        fi
    fi

    export_arg="ALL,DATASET=${dataset},RUN_NAME=${run_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY}"
    if [ "$mode" = "resume" ]; then
        export_arg="${export_arg},RESUME_CHECKPOINT=${resume_checkpoint}"
    fi

    echo "SUBMIT mode=$mode old=$old_job_id dataset=$dataset schedule=$schedule_name objective=$objective step=$resume_step time=$slurm_time"
    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --time="$slurm_time" \
                --export="$export_arg" \
                --job-name="$job_name" \
                --output="$SLURM_LOG_DIR/${job_name}_%j.out" \
                --error="$SLURM_LOG_DIR/${job_name}_%j.err" \
                "$TRAIN_SLURM"
        )
        echo "$sbatch_output"
        job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
        if [ -z "$job_id" ]; then
            job_id="UNKNOWN"
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "submitted" "$job_id" "$mode" "$old_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$resume_step" "${resume_checkpoint:-none}" "$old_logdir" "$slurm_time" "$SKIP_DATASET_VERIFY" >> "$SUBMISSION_TSV"
    else
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "dry_run" "DRY_RUN" "$mode" "$old_job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$resume_step" "${resume_checkpoint:-none}" "$old_logdir" "$slurm_time" "$SKIP_DATASET_VERIFY" >> "$SUBMISSION_TSV"
    fi

    submitted=$((submitted + 1))
    if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
        echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
        echo "Manifest: $SUBMISSION_TSV"
        exit 0
    fi
done

echo "=========================================="
echo "Submission complete"
echo "Jobs considered: $submitted"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
