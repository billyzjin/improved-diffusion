#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/linabar_evaluation_nll_fid_${TIMESTAMP}}
SUBMISSION_TSV=${SUBMISSION_TSV:-${PARENT_EVAL_DIR}/submission.tsv}
RICHER_RESULTS_DIR=${RICHER_RESULTS_DIR:-results/linabar_richer_metrics_${TIMESTAMP}}
RICHER_MANIFEST=${RICHER_MANIFEST:-${RICHER_RESULTS_DIR}/manifest.tsv}
RICHER_SUBMISSION_TSV=${RICHER_SUBMISSION_TSV:-${RICHER_RESULTS_DIR}/submission.tsv}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-/project_gpfs/bata0/bjin0/richer_metrics_features}
TRAIN_SUBMISSION_TSV=${TRAIN_SUBMISSION_TSV:-/project_gpfs/bata0/bjin0/linabar_resume_full_slate_20260531_131307/submission.tsv}
METRICS=${METRICS:-cmmd,kid,density_coverage}
CLIP_MODEL=${CLIP_MODEL:-ViT-B-32}
CLIP_PRETRAINED=${CLIP_PRETRAINED:-/project_gpfs/bata0/bjin0/model_cache/openclip/open_clip_pytorch_model.bin}
MAX_RICHER_CONCURRENT=${MAX_RICHER_CONCURRENT:-8}
RICHER_DEPENDENCY_TYPE=${RICHER_DEPENDENCY_TYPE:-afterany}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
SKIP_NLL=${SKIP_NLL:-0}
SKIP_TV=${SKIP_TV:-1}
SUBMIT_RICHER=${SUBMIT_RICHER:-1}

mkdir -p "$SLURM_LOG_DIR" "$PARENT_EVAL_DIR" "$RICHER_RESULTS_DIR"

if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tdataset\tschedule\tobjective\texperiment\tmodel_path\teval_dir\tskip_nll\tskip_tv\n" > "$SUBMISSION_TSV"
fi
if [ ! -f "$RICHER_SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\tarray_spec\tmanifest\tresults_dir\tdependency\tmetrics\n" > "$RICHER_SUBMISSION_TSV"
fi

datasets=(mnist fashionmnist cifar10 imagenet64)
schedules=(linabar_linear linabar_cosine)
objectives=(simple hybrid vlb)

eval_slurm_for_dataset() {
    case "$1" in
        mnist) echo "evaluate_mnist_final.slurm" ;;
        fashionmnist) echo "evaluate_fashionmnist_final.slurm" ;;
        cifar10) echo "evaluate_models_final.slurm" ;;
        imagenet64) echo "evaluate_imagenet64_final.slurm" ;;
        *) echo "ERROR: unknown dataset $1" >&2; return 1 ;;
    esac
}

ckpt_name_for_dataset() {
    case "$1" in
        imagenet64) echo "ema_0.9999_200000.pt" ;;
        *) echo "ema_0.9999_500000.pt" ;;
    esac
}

image_size_for_dataset() {
    case "$1" in
        imagenet64) echo "64" ;;
        *) echo "32" ;;
    esac
}

sample_count_for_dataset() {
    case "$1" in
        imagenet64) echo "10000" ;;
        *) echo "50000" ;;
    esac
}

sample_path_for_dataset() {
    local dataset="$1"
    local exp_dir="$2"
    case "$dataset" in
        imagenet64) echo "${exp_dir}/samples.npz" ;;
        *) echo "${exp_dir}/samples_50000x32x32x3.npz" ;;
    esac
}

real_dir_for_dataset() {
    case "$1" in
        mnist) echo "mnist_train" ;;
        fashionmnist) echo "fashion_train" ;;
        cifar10) echo "cifar_train" ;;
        imagenet64) echo "/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505/train" ;;
        *) echo "ERROR: unknown dataset $1" >&2; return 1 ;;
    esac
}

short_schedule() {
    case "$1" in
        linabar_linear) echo "lablin" ;;
        linabar_cosine) echo "labcos" ;;
        *) echo "$1" ;;
    esac
}

for file in evaluate_models_final.slurm evaluate_mnist_final.slurm evaluate_fashionmnist_final.slurm evaluate_imagenet64_final.slurm richer_metrics.slurm; do
    bash -n "$file"
done
for file in scripts/run_richer_metrics.py scripts/aggregate_richer_metrics.py; do
    test -f "$file" || { echo "ERROR: missing $file"; exit 1; }
done

if [ "$SUBMIT_RICHER" = "1" ] && [[ ",$METRICS," == *",cmmd,"* ]] && [ ! -f "$CLIP_PRETRAINED" ]; then
    echo "ERROR: CLIP_PRETRAINED does not exist: $CLIP_PRETRAINED"
    exit 1
fi
if [ ! -f "$TRAIN_SUBMISSION_TSV" ]; then
    echo "ERROR: TRAIN_SUBMISSION_TSV does not exist: $TRAIN_SUBMISSION_TSV"
    exit 1
fi

model_path_for_run() {
    local dataset="$1"
    local schedule="$2"
    local objective="$3"
    local exp_name="$4"
    local ckpt_name="$5"
    local job_id

    job_id=$(
        awk -F'\t' -v d="$dataset" -v s="$schedule" -v o="$objective" '
            NR > 1 && $2 != "dry_run" && (($4 == d && $5 == s && $6 == o) || ($6 == d && $7 == s && $8 == o)) {
                id = $3
            }
            END {
                if (id != "") print id
            }
        ' "$TRAIN_SUBMISSION_TSV"
    )
    if [ -z "$job_id" ]; then
        return 1
    fi
    echo "/project_gpfs/bata0/bjin0/bjin0/${job_id}/logs/${exp_name}/${ckpt_name}"
}

printf "dataset\tobjective\tschedule\tsamples_npz\treal_dir\tn_samples\timage_size\tnll_bpd\tfid\tnll_samples\tfid_samples\tsampling_steps\tsource_eval_dir\n" > "$RICHER_MANIFEST"

echo "=========================================="
echo "Submitting linabar NLL/FID evaluations"
echo "Evaluation dir:       $PARENT_EVAL_DIR"
echo "Richer results dir:   $RICHER_RESULTS_DIR"
echo "Richer manifest:      $RICHER_MANIFEST"
echo "Training manifest:    $TRAIN_SUBMISSION_TSV"
echo "Dry run:              $DRY_RUN"
echo "Skip NLL:             $SKIP_NLL"
echo "Skip TV:              $SKIP_TV"
echo "Submit richer:        $SUBMIT_RICHER"
echo "=========================================="

eval_job_ids=()
row_count=0
for dataset in "${datasets[@]}"; do
    eval_slurm=$(eval_slurm_for_dataset "$dataset")
    ckpt_name=$(ckpt_name_for_dataset "$dataset")
    image_size=$(image_size_for_dataset "$dataset")
    sample_count=$(sample_count_for_dataset "$dataset")
    real_dir=$(real_dir_for_dataset "$dataset")

    for schedule in "${schedules[@]}"; do
        schedule_short=$(short_schedule "$schedule")
        for objective in "${objectives[@]}"; do
            exp_name="${dataset}_${schedule}_${objective}"
            exp_dir="${PARENT_EVAL_DIR}/${exp_name}"
            samples_npz=$(sample_path_for_dataset "$dataset" "$exp_dir")
            model_path=$(model_path_for_run "$dataset" "$schedule" "$objective" "$exp_name" "$ckpt_name" || true)
            if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
                echo "ERROR: missing final EMA checkpoint for $exp_name ($ckpt_name)"
                echo "Looked for: $model_path"
                exit 1
            fi

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$dataset" "$objective" "$schedule" "$samples_npz" "$real_dir" "$sample_count" "$image_size" "" "" "10000" "$sample_count" "4000" "$exp_dir" >> "$RICHER_MANIFEST"

            job_name="eval_${dataset}_${schedule_short}_${objective}"
            echo "SUBMIT eval dataset=$dataset schedule=$schedule objective=$objective model=$model_path"
            if [ "$DRY_RUN" != "1" ]; then
                sbatch_output=$(
                    env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                    sbatch \
                        --account=bata0-external \
                        --partition=long_hopper \
                        --gres=gpu:h100:1 \
                        --export="ALL,EVAL_MODEL_NAME=${exp_name},MODEL_PATH=${model_path},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},SKIP_NLL=${SKIP_NLL},SKIP_TV=${SKIP_TV},FORCE=${FORCE}" \
                        --job-name="$job_name" \
                        --output="$SLURM_LOG_DIR/${job_name}_%j.out" \
                        --error="$SLURM_LOG_DIR/${job_name}_%j.err" \
                        "$eval_slurm"
                )
                echo "$sbatch_output"
                job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
                if [ -z "$job_id" ]; then
                    echo "ERROR: could not parse sbatch job id for $exp_name"
                    exit 1
                fi
                eval_job_ids+=("$job_id")
                status="submitted"
            else
                job_id="DRY_RUN"
                status="dry_run"
            fi

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$(date -Is)" "$status" "$job_id" "$dataset" "$schedule" "$objective" "$exp_name" "$model_path" "$exp_dir" "$SKIP_NLL" "$SKIP_TV" >> "$SUBMISSION_TSV"
            row_count=$((row_count + 1))
        done
    done
done

echo "Wrote richer metrics manifest rows: $row_count"

if [ "$SUBMIT_RICHER" != "1" ]; then
    echo "Skipping richer metrics submission (SUBMIT_RICHER=0)"
    exit 0
fi

if [ "$row_count" -le 0 ]; then
    echo "ERROR: no richer metric rows"
    exit 1
fi

last_index=$((row_count - 1))
array_spec="0-${last_index}"
if [ "$MAX_RICHER_CONCURRENT" -gt 0 ]; then
    array_spec="${array_spec}%${MAX_RICHER_CONCURRENT}"
fi

dependency="none"
dependency_arg=()
if [ "$DRY_RUN" != "1" ]; then
    IFS=:
    dependency_ids="${eval_job_ids[*]}"
    unset IFS
    dependency="${RICHER_DEPENDENCY_TYPE}:${dependency_ids}"
    dependency_arg=(--dependency="$dependency")
fi

METRICS_EXPORT=${METRICS//,/;}
echo "SUBMIT richer metrics array=$array_spec dependency=$dependency metrics=$METRICS"
if [ "$DRY_RUN" != "1" ]; then
    richer_output=$(
        env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
        sbatch \
            --array="$array_spec" \
            "${dependency_arg[@]}" \
            --export="ALL,MANIFEST=${RICHER_MANIFEST},OUTPUT_DIR=${RICHER_RESULTS_DIR},FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT},METRICS=${METRICS_EXPORT},CLIP_MODEL=${CLIP_MODEL},CLIP_PRETRAINED=${CLIP_PRETRAINED}" \
            --output="$SLURM_LOG_DIR/linabar_richer_%A_%a.out" \
            --error="$SLURM_LOG_DIR/linabar_richer_%A_%a.err" \
            richer_metrics.slurm
    )
    echo "$richer_output"
    richer_job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$richer_output")
    if [ -z "$richer_job_id" ]; then
        echo "ERROR: could not parse richer metrics job id"
        exit 1
    fi
    richer_status="submitted"
else
    richer_job_id="DRY_RUN"
    richer_status="dry_run"
fi

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$richer_status" "$richer_job_id" "$array_spec" "$RICHER_MANIFEST" "$RICHER_RESULTS_DIR" "$dependency" "$METRICS" >> "$RICHER_SUBMISSION_TSV"

echo "=========================================="
echo "Submission complete"
echo "Eval manifest:   $SUBMISSION_TSV"
echo "Richer manifest: $RICHER_MANIFEST"
echo "Richer submit:   $RICHER_SUBMISSION_TSV"
echo "Aggregate richer metrics after completion:"
echo "  module load python/booth/3.12"
echo "  python3 scripts/aggregate_richer_metrics.py --manifest \"$RICHER_MANIFEST\" --results_dir \"$RICHER_RESULTS_DIR\" --output_tsv \"$RICHER_RESULTS_DIR/richer_metrics_summary.tsv\""
echo "=========================================="
