#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

DATASET=${DATASET:-${1:-}}
if [ -z "$DATASET" ]; then
    echo "ERROR: set DATASET or pass it as the first argument (cifar100, celeba64, or lsun_bedroom64)"
    exit 1
fi

TRAIN_SLURM=${TRAIN_SLURM:-train_image_folder_no_mpi.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/${DATASET}_full_slate_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$SUBMISSION_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
FORCE_PREP=${FORCE_PREP:-0}
SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY:-1}

case "$DATASET" in
    cifar100)
        PREP_SLURM=${PREP_SLURM:-prepare_cifar100.slurm}
        DATA_ROOT=${DATA_ROOT:-/project_gpfs/bata0/bjin0/cifar100_32x32}
        TRAIN_DIR=${TRAIN_DIR:-$DATA_ROOT/train}
        EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT:-50000}
        EXPECTED_TEST_COUNT=${EXPECTED_TEST_COUNT:-10000}
        IMAGE_SIZE=${IMAGE_SIZE:-32}
        LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-500000}
        BATCH_SIZE=${BATCH_SIZE:-128}
        LOG_INTERVAL=${LOG_INTERVAL:-1000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
        USE_FP16=${USE_FP16:-False}
        MICRO_BATCH=${MICRO_BATCH:-}
        SLURM_TIME=${SLURM_TIME:-4-00:00:00}
        ;;
    celeba64)
        PREP_SLURM=${PREP_SLURM:-prepare_celeba64.slurm}
        DATA_ROOT=${DATA_ROOT:-/project_gpfs/bata0/bjin0/celeba_64x64}
        TRAIN_DIR=${TRAIN_DIR:-$DATA_ROOT/train}
        EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT:-162770}
        EXPECTED_TEST_COUNT=${EXPECTED_TEST_COUNT:-19962}
        IMAGE_SIZE=${IMAGE_SIZE:-64}
        LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-200000}
        BATCH_SIZE=${BATCH_SIZE:-128}
        LOG_INTERVAL=${LOG_INTERVAL:-500}
        SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
        USE_FP16=${USE_FP16:-True}
        MICRO_BATCH=${MICRO_BATCH:-16}
        SLURM_TIME=${SLURM_TIME:-4-00:00:00}
        ;;
    lsun_bedroom64)
        PREP_SLURM=${PREP_SLURM:-prepare_lsun_bedroom64.slurm}
        DATA_ROOT=${DATA_ROOT:-/project_gpfs/bata0/bjin0/lsun_bedroom_64x64}
        LSUN_SOURCE_ROOT=${LSUN_SOURCE_ROOT:-$DATA_ROOT/source}
        TRAIN_DIR=${TRAIN_DIR:-$LSUN_SOURCE_ROOT/bedroom_train_lmdb}
        DATASET_READY_MARKER=${DATASET_READY_MARKER:-$TRAIN_DIR/data.mdb}
        EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT:-1}
        EXPECTED_TEST_COUNT=${EXPECTED_TEST_COUNT:-1}
        IMAGE_SIZE=${IMAGE_SIZE:-64}
        LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-200000}
        BATCH_SIZE=${BATCH_SIZE:-128}
        LOG_INTERVAL=${LOG_INTERVAL:-500}
        SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
        USE_FP16=${USE_FP16:-True}
        MICRO_BATCH=${MICRO_BATCH:-16}
        SLURM_TIME=${SLURM_TIME:-4-00:00:00}
        ;;
    *)
        echo "ERROR: unsupported DATASET=$DATASET; expected cifar100, celeba64, or lsun_bedroom64"
        exit 1
        ;;
esac

mkdir -p "$SLURM_LOG_DIR" "$SUBMISSION_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tdataset\tschedule_name\tobjective\trun_name\ttrain_dir\timage_size\ttrain_steps\ttime_limit\tdependency\tprep_job_id\tlogdir\n" > "$SUBMISSION_TSV"
fi

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: training slurm not found: $TRAIN_SLURM"
    exit 1
fi
if [ ! -f "$PREP_SLURM" ]; then
    echo "ERROR: prepare slurm not found: $PREP_SLURM"
    exit 1
fi

SCHEDULES=${SCHEDULES:-linear,cosine,geometric_linear,geometric_cosine}
OBJECTIVES=${OBJECTIVES:-simple,hybrid,vlb}
read -r -a schedules <<< "${SCHEDULES//,/ }"
read -r -a objectives <<< "${OBJECTIVES//,/ }"

if [ "${#schedules[@]}" -eq 0 ]; then
    echo "ERROR: no schedules selected"
    exit 1
fi
if [ "${#objectives[@]}" -eq 0 ]; then
    echo "ERROR: no objectives selected"
    exit 1
fi

for schedule_name in "${schedules[@]}"; do
    case "$schedule_name" in
        linear|cosine|geometric_linear|geometric_cosine) ;;
        *)
            echo "ERROR: unknown schedule in SCHEDULES: $schedule_name"
            echo "Expected: linear, cosine, geometric_linear, geometric_cosine"
            exit 1
            ;;
    esac
done
for objective in "${objectives[@]}"; do
    case "$objective" in
        simple|hybrid|vlb) ;;
        *)
            echo "ERROR: unknown objective in OBJECTIVES: $objective"
            echo "Expected: simple, hybrid, vlb"
            exit 1
            ;;
    esac
done

schedule_short() {
    case "$1" in
        linear) echo "lin" ;;
        cosine) echo "cos" ;;
        geometric_linear) echo "glin" ;;
        geometric_cosine) echo "gcos" ;;
        *) echo "$1" ;;
    esac
}

dataset_ready=0
if [ -n "${DATASET_READY_MARKER:-}" ] && [ "$FORCE_PREP" != "1" ]; then
    if [ -f "$DATASET_READY_MARKER" ]; then
        dataset_ready=1
    elif [ -d "$TRAIN_DIR" ]; then
        echo "ERROR: $TRAIN_DIR exists but ready marker is missing: $DATASET_READY_MARKER"
        echo "Set FORCE_PREP=1 and the dataset overwrite env var if you want to rebuild it."
        exit 1
    fi
elif [ -d "$TRAIN_DIR" ] && [ "$FORCE_PREP" != "1" ]; then
    train_count=$(find "$TRAIN_DIR" -name "*.png" | wc -l)
    if [ "$train_count" -ge "$EXPECTED_TRAIN_COUNT" ]; then
        dataset_ready=1
    else
        echo "ERROR: $TRAIN_DIR exists but has $train_count PNGs; expected $EXPECTED_TRAIN_COUNT"
        echo "Set FORCE_PREP=1 and the dataset overwrite env var if you want to rebuild it."
        exit 1
    fi
fi

prep_job_id=""
dependency_arg=()
if [ "$dataset_ready" = "1" ]; then
    echo "$DATASET dataset ready: $TRAIN_DIR"
else
    echo "$DATASET dataset is missing; submitting prepare job first."
    if [ "$DRY_RUN" = "1" ]; then
        prep_job_id="DRY_RUN_PREP"
        echo "DRY RUN prepare: $PREP_SLURM"
    else
        prep_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --output="$SLURM_LOG_DIR/${DATASET}_prepare_%j.out" \
                --error="$SLURM_LOG_DIR/${DATASET}_prepare_%j.err" \
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
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Is)" "prep_submitted" "$prep_job_id" "$DATASET" "prepare" "prepare" "prepare_${DATASET}" "$TRAIN_DIR" "$IMAGE_SIZE" "$LR_ANNEAL_STEPS" "prepare" "none" "$prep_job_id" "none" >> "$SUBMISSION_TSV"
fi

echo "=========================================="
echo "Submitting image-folder training slate"
echo "Dataset: $DATASET"
echo "Schedules: ${schedules[*]}"
echo "Objectives: ${objectives[*]}"
echo "Train dir: $TRAIN_DIR"
echo "Image size: $IMAGE_SIZE"
echo "Train steps: $LR_ANNEAL_STEPS"
echo "Submission dir: $SUBMISSION_DIR"
echo "Dry run: $DRY_RUN"
echo "Training dependency: ${dependency_arg[*]:-none}"
echo "=========================================="

submitted=0
for schedule_name in "${schedules[@]}"; do
    short_schedule=$(schedule_short "$schedule_name")
    for objective in "${objectives[@]}"; do
        run_name="${schedule_name}_${objective}"
        job_name="${DATASET}_${short_schedule}_${objective}"
        export_arg="ALL,DATASET=${DATASET},RUN_NAME=${run_name},SCHEDULE_NAME=${schedule_name},OBJECTIVE=${objective},TRAIN_DIR=${TRAIN_DIR},IMAGE_SIZE=${IMAGE_SIZE},EXPECTED_TRAIN_COUNT=${EXPECTED_TRAIN_COUNT},SKIP_DATASET_VERIFY=${SKIP_DATASET_VERIFY},LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS},BATCH_SIZE=${BATCH_SIZE},LOG_INTERVAL=${LOG_INTERVAL},SAVE_INTERVAL=${SAVE_INTERVAL},USE_FP16=${USE_FP16},MICRO_BATCH=${MICRO_BATCH}"

        echo "SUBMIT dataset=$DATASET schedule=$schedule_name objective=$objective time=$SLURM_TIME"
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
            logdir="/project_gpfs/bata0/bjin0/${USER}/${job_id}/logs/${DATASET}_${run_name}"
        else
            job_id="DRY_RUN"
            status="dry_run"
            logdir="DRY_RUN"
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "$status" "$job_id" "$DATASET" "$schedule_name" "$objective" "$run_name" "$TRAIN_DIR" "$IMAGE_SIZE" "$LR_ANNEAL_STEPS" "$SLURM_TIME" "${dependency_arg[*]:-none}" "${prep_job_id:-none}" "$logdir" >> "$SUBMISSION_TSV"

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
