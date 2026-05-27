#!/bin/bash
set -euo pipefail

# Submit evaluations for the 12 ImageNet-64 models from the completed May 2026
# standard + geometric training batch. This intentionally excludes old ours_ jobs.

cd /home/bjin0/improved-diffusion

IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}
IMAGENET_TRAIN_STATS=${IMAGENET_TRAIN_STATS:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505_train_stats.npz}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/imagenet64_evaluation_parallel_finished12_$(date +%Y%m%d_%H%M%S)}
NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES:-10000}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-64}
FORCE=${FORCE:-0}
SKIP_TV=${SKIP_TV:-1}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
SUBMISSION_TSV=${SUBMISSION_TSV:-$PARENT_EVAL_DIR/submission.tsv}

if [ ! -d "$IMAGENET_TRAIN_DIR" ] || [ ! -d "$IMAGENET_VAL_DIR" ]; then
  echo "ERROR: ImageNet dirs not found."
  echo "IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
  echo "IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
  exit 1
fi

mkdir -p "$PARENT_EVAL_DIR" slurm_logs
if [ ! -f "$SUBMISSION_TSV" ]; then
  printf "submitted_at\tjob_id\texperiment\teval_dir\tnll_num_samples\tnum_samples\tskip_tv\n" > "$SUBMISSION_TSV"
fi

EXPERIMENTS=(
  "imagenet64_linear_simple"
  "imagenet64_linear_hybrid"
  "imagenet64_linear_vlb"
  "imagenet64_cosine_simple"
  "imagenet64_cosine_hybrid"
  "imagenet64_cosine_vlb"
  "imagenet64_geometric_linear_simple"
  "imagenet64_geometric_linear_hybrid"
  "imagenet64_geometric_linear_vlb"
  "imagenet64_geometric_cosine_simple"
  "imagenet64_geometric_cosine_hybrid"
  "imagenet64_geometric_cosine_vlb"
)

echo "=========================================="
echo "Submitting 12 ImageNet-64 evaluation jobs"
echo "Results dir: $PARENT_EVAL_DIR"
echo "NLL samples: $NLL_NUM_SAMPLES"
echo "FID samples: $NUM_SAMPLES"
echo "Skip TV: $SKIP_TV"
echo "Force rerun: $FORCE"
echo "Dry run: $DRY_RUN"
echo "=========================================="

submitted=0
for exp_name in "${EXPERIMENTS[@]}"; do
  echo "--> Submitting job for: $exp_name"
  export_arg="ALL,EVAL_MODEL_NAME=${exp_name},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT},IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR},IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR},IMAGENET_TRAIN_STATS=${IMAGENET_TRAIN_STATS},NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES},NUM_SAMPLES=${NUM_SAMPLES},SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE},FORCE=${FORCE},SKIP_TV=${SKIP_TV}"
  if [ -n "${SKIP_NLL:-}" ]; then
    export_arg="${export_arg},SKIP_NLL=${SKIP_NLL}"
  fi
  if [ -n "${EVAL_TIMESTEP_RESPACING:-}" ]; then
    export_arg="${export_arg},EVAL_TIMESTEP_RESPACING=${EVAL_TIMESTEP_RESPACING}"
  fi

  if [ "$DRY_RUN" != "1" ]; then
    sbatch_output=$(
      env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
        sbatch \
        --export="$export_arg" \
        --job-name="eval_$exp_name" \
        --output="slurm_logs/eval_${exp_name}_%j.out" \
        --error="slurm_logs/eval_${exp_name}_%j.err" \
        evaluate_imagenet64_final.slurm
    )
    echo "$sbatch_output"
    job_id=$(awk '{print $4}' <<< "$sbatch_output")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date -Is)" "$job_id" "$exp_name" "$PARENT_EVAL_DIR/$exp_name" \
      "$NLL_NUM_SAMPLES" "$NUM_SAMPLES" "$SKIP_TV" >> "$SUBMISSION_TSV"
  fi

  submitted=$((submitted + 1))
  if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
    echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
    break
  fi
done

echo "=========================================="
echo "Submission complete"
echo "Submitted: $submitted"
echo "Results dir: $PARENT_EVAL_DIR"
echo "Submission manifest: $SUBMISSION_TSV"
echo "Aggregate after jobs finish:"
echo "  bash aggregate_imagenet64_evaluation_results.sh $PARENT_EVAL_DIR"
echo "=========================================="
