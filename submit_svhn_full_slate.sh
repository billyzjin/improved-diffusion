#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SLURM=${TRAIN_SLURM:-train_svhn_no_mpi.slurm}
PREP_SLURM=${PREP_SLURM:-prepare_svhn.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/svhn_full_slate_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$SUBMISSION_DIR/submission.tsv}
SVHN_ROOT=${SVHN_ROOT:-/project_gpfs/bata0/bjin0/svhn_32x32}
SVHN_TRAIN_DIR=${SVHN_TRAIN_DIR:-$SVHN_ROOT/train}
EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT:-73257}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
FORCE_PREP=${FORCE_PREP:-0}
SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY:-1}
SLURM_TIME=${SLURM_TIME:-4-00:00:00}
HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT:-0.001}

mkdir -p "$SLURM_LOG_DIR" "$SUBMISSION_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tdataset\tschedule_name\tobjective\trun_name\tdata_dir\ttime_limit\tdependency\tprep_job_id\n" > "$SUBMISSION_TSV"
fi

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: training slurm not found: $TRAIN_SLURM"
    exit 1
fi
if [ ! -f "$PREP_SLURM" ]; then
    echo "ERROR: prepare slurm not found: $PREP_SLURM"
    exit 1
fi

schedules=(linear cosine geometric_linear geometric_cosine linabar_linear linabar_cosine)
objectives=(simple hybrid vlb)

schedule_short() {
    case "$1" in
        linear) echo "lin" ;;
        cosine) echo "cos" ;;
        geometric_linear) echo "glin" ;;
        geometric_cosine) echo "gcos" ;;
        linabar_linear) echo "lablin" ;;
        linabar_cosine) echo "labcos" ;;
        *) echo "$1" ;;
    esac
}

dataset_ready=0
if [ -d "$SVHN_TRAIN_DIR" ] && [ "$FORCE_PREP" != "1" ]; then
    train_count=$(find "$SVHN_TRAIN_DIR" -name "*.png" | wc -l)
    if [ "$train_count" -ge "$EXPECTED_TRAIN_COUNT" ]; then
        dataset_ready=1
    else
        echo "ERROR: $SVHN_TRAIN_DIR exists but has $train_count PNGs; expected $EXPECTED_TRAIN_COUNT"
        echo "Set FORCE_PREP=1 and SVHN_OVERWRITE=1 if you want to rebuild it."
        exit 1
    fi
fi

prep_job_id=""
dependency_arg=()
if [ "$dataset_ready" = "1" ]; then
    echo "SVHN dataset ready: $SVHN_TRAIN_DIR"
else
    echo "SVHN dataset is missing; submitting prepare job first."
    if [ "$DRY_RUN" = "1" ]; then
        prep_job_id="DRY_RUN_PREP"
        echo "DRY RUN prepare: $PREP_SLURM"
    else
        prep_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --export="ALL,SVHN_ROOT=${SVHN_ROOT}" \
                --output="$SLURM_LOG_DIR/svhn_prepare_%j.out" \
                --error="$SLURM_LOG_DIR/svhn_prepare_%j.err" \
                "$PREP_SLURM"
        )
        echo "$prep_output"
        prep_job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$prep_output")
        if [ -z "$prep_job_id" ]; then
            echo "ERROR: could not parse prepare job id"
            exit 1
        fi
    fi
    dependency_arg=(--dependency="afterok:${prep_job_id}")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Is)" "prep_submitted" "$prep_job_id" "svhn" "prepare" "prepare" "prepare_svhn" "$SVHN_TRAIN_DIR" "2:00:00" "none" "$prep_job_id" >> "$SUBMISSION_TSV"
fi

echo "=========================================="
echo "Submitting SVHN training slate"
echo "Schedules: ${schedules[*]}"
echo "Objectives: ${objectives[*]}"
echo "Objective note: using simple/hybrid/vlb; simple corresponds to L_simple."
echo "Hybrid VB weight: $HYBRID_VB_WEIGHT"
echo "SVHN train dir: $SVHN_TRAIN_DIR"
echo "Submission dir: $SUBMISSION_DIR"
echo "Dry run: $DRY_RUN"
echo "Training dependency: ${dependency_arg[*]:-none}"
echo "=========================================="

submitted=0
for schedule_name in "${schedules[@]}"; do
    short_schedule=$(schedule_short "$schedule_name")
    for objective in "${objectives[@]}"; do
        run_name="${schedule_name}_${objective}"
        job_name="svhn_${short_schedule}_${objective}"
        export_arg="ALL,RUN_NAME=${run_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT},SVHN_TRAIN_DIR=${SVHN_TRAIN_DIR},EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT},SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY}"

        echo "SUBMIT schedule=$schedule_name objective=$objective time=$SLURM_TIME"
        if [ "$DRY_RUN" != "1" ]; then
            sbatch_output=$(
                env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                    sbatch \
                    --time="$SLURM_TIME" \
                    "${dependency_arg[@]}" \
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
            status="submitted"
        else
            job_id="DRY_RUN"
            status="dry_run"
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "$status" "$job_id" "svhn" "$schedule_name" "$objective" "$run_name" "$SVHN_TRAIN_DIR" "$SLURM_TIME" "${dependency_arg[*]:-none}" "${prep_job_id:-none}" >> "$SUBMISSION_TSV"

        submitted=$((submitted + 1))
        if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
            echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
            echo "Manifest: $SUBMISSION_TSV"
            exit 0
        fi
    done
done

echo "=========================================="
echo "Submission complete"
echo "Jobs considered: $submitted"
echo "Prepare job: ${prep_job_id:-none}"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
