#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SLURM=${TRAIN_SLURM:-train_tuned_geometric_no_mpi.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/tuned_geometric_full_slate_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$SUBMISSION_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
SKIP_ALREADY_SUBMITTED=${SKIP_ALREADY_SUBMITTED:-1}
HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT:-0.001}

mkdir -p "$SLURM_LOG_DIR" "$SUBMISSION_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tdataset\tschedule_name\tobjective\trun_name\tbeta_1\talpha_bar_T\treason\n" > "$SUBMISSION_TSV"
fi

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: training slurm not found: $TRAIN_SLURM"
    exit 1
fi

datasets=(mnist fashionmnist cifar10 imagenet64)
objectives=(simple hybrid vlb)

# Format: schedule_name|beta_1|alpha_bar_T|reason
schedules=(
    "fid|3e-3|1e-2|best CIFAR-10 50k probe FID"
    "nll|1e-5|3e-3|best CIFAR-10 50k probe NLL"
    "balanced|3e-5|1e-3|best measured overlap point"
)

declare -A existing_jobs=(
    ["cifar10|fid|hybrid"]="113346"
    ["cifar10|fid|simple"]="113347"
    ["cifar10|nll|hybrid"]="113348"
    ["cifar10|balanced|hybrid"]="113349"
)

slurm_time_for_dataset() {
    case "$1" in
        cifar10) echo "4-00:00:00" ;;
        imagenet64) echo "3-00:00:00" ;;
        *) echo "4-00:00:00" ;;
    esac
}

echo "=========================================="
echo "Submitting full tuned geometric training slate"
echo "Submission dir: $SUBMISSION_DIR"
echo "Dry run: $DRY_RUN"
echo "Skip already submitted: $SKIP_ALREADY_SUBMITTED"
echo "Hybrid VB weight: $HYBRID_VB_WEIGHT"
echo "=========================================="

submitted=0
recorded_existing=0

for dataset in "${datasets[@]}"; do
    for schedule_entry in "${schedules[@]}"; do
        IFS='|' read -r schedule_name beta alpha reason <<< "$schedule_entry"
        for objective in "${objectives[@]}"; do
            key="${dataset}|${schedule_name}|${objective}"
            run_name="${schedule_name}_b${beta}_a${alpha}_${objective}"
            run_name=${run_name//./}
            job_name="tg_${dataset}_${schedule_name}_${objective}"

            if [ "$SKIP_ALREADY_SUBMITTED" = "1" ] && [ -n "${existing_jobs[$key]:-}" ]; then
                job_id="${existing_jobs[$key]}"
                echo "EXISTING $key job_id=$job_id"
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$(date -Is)" "existing" "$job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$beta" "$alpha" "$reason" >> "$SUBMISSION_TSV"
                recorded_existing=$((recorded_existing + 1))
                continue
            fi

            export_arg="ALL,DATASET=${dataset},RUN_NAME=${run_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT},GEOMETRIC_BETA1=${beta},GEOMETRIC_ALPHA_BAR_T=${alpha}"
            slurm_time=$(slurm_time_for_dataset "$dataset")

            echo "SUBMIT $key beta=$beta alpha=$alpha time=$slurm_time"
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
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$(date -Is)" "submitted" "$job_id" "$dataset" "$schedule_name" "$objective" "$run_name" "$beta" "$alpha" "$reason" >> "$SUBMISSION_TSV"
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
echo "Submitted new jobs: $submitted"
echo "Recorded existing jobs: $recorded_existing"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
