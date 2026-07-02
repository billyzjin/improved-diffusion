#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SLURM=${TRAIN_SLURM:-train_linabar_no_mpi.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/linabar_full_slate_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$SUBMISSION_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY:-1}
HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT:-0.001}

mkdir -p "$SLURM_LOG_DIR" "$SUBMISSION_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tdataset\tschedule_name\tobjective\trun_name\ttime_limit\tskip_dataset_verify\n" > "$SUBMISSION_TSV"
fi

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: training slurm not found: $TRAIN_SLURM"
    exit 1
fi

datasets=(mnist fashionmnist cifar10 imagenet64)
schedules=(linabar_linear linabar_cosine)
objectives=(simple hybrid vlb)

slurm_time_for_dataset() {
    case "$1" in
        imagenet64) echo "3-00:00:00" ;;
        *) echo "4-00:00:00" ;;
    esac
}

echo "=========================================="
echo "Submitting linear-in-alpha_bar full slate"
echo "Datasets: ${datasets[*]}"
echo "Schedules: ${schedules[*]}"
echo "Objectives: ${objectives[*]}"
echo "Submission dir: $SUBMISSION_DIR"
echo "Dry run: $DRY_RUN"
echo "Skip dataset verify: $SKIP_DATASET_VERIFY"
echo "Hybrid VB weight: $HYBRID_VB_WEIGHT"
echo "=========================================="

submitted=0
for dataset in "${datasets[@]}"; do
    for schedule_name in "${schedules[@]}"; do
        for objective in "${objectives[@]}"; do
            run_name="${schedule_name}_${objective}"
            job_name="lab_${dataset}_${schedule_name#linabar_}_${objective}"
            slurm_time=$(slurm_time_for_dataset "$dataset")
            export_arg="ALL,DATASET=${dataset},RUN_NAME=${run_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT},SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY}"

            echo "SUBMIT dataset=$dataset schedule=$schedule_name objective=$objective time=$slurm_time"
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
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$(date -Is)" "submitted" "$job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$slurm_time" "$SKIP_DATASET_VERIFY" >> "$SUBMISSION_TSV"
            else
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$(date -Is)" "dry_run" "DRY_RUN" "$dataset" "$schedule_name" "$objective" "$run_name" "$slurm_time" "$SKIP_DATASET_VERIFY" >> "$SUBMISSION_TSV"
            fi

            submitted=$((submitted + 1))
            if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
                echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
                echo "Manifest: $SUBMISSION_TSV"
                exit 0
            fi
        done
    done
done

echo "=========================================="
echo "Submission complete"
echo "Jobs considered: $submitted"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
