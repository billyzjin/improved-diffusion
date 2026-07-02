#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

RESULTS_DIR=${RESULTS_DIR:-results/richer_metrics}
MANIFEST=${MANIFEST:-$RESULTS_DIR/manifest.tsv}
EVALUATION_RESULTS=${EVALUATION_RESULTS:-evaluation_results_full.tsv}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-/project_gpfs/bata0/bjin0/richer_metrics_features}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SCHEDULES=${SCHEDULES:-linear,cosine,geometric_linear,geometric_cosine}
DATASETS=${DATASETS:-mnist,fashionmnist,cifar10,imagenet64}
OBJECTIVES=${OBJECTIVES:-simple,hybrid,vlb}
METRICS=${METRICS:-cmmd,kid,density_coverage}
CLIP_MODEL=${CLIP_MODEL:-ViT-B-32}
CLIP_PRETRAINED=${CLIP_PRETRAINED:-/project_gpfs/bata0/bjin0/model_cache/openclip/open_clip_pytorch_model.bin}
DRY_RUN=${DRY_RUN:-0}
MAX_ROWS=${MAX_ROWS:-0}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
RICHER_PARTITION=${RICHER_PARTITION:-}
RICHER_TIME=${RICHER_TIME:-}
SUBMISSION_TSV=${SUBMISSION_TSV:-$RESULTS_DIR/submission.tsv}
if [ -x /home/bjin0/improved-diffusion/.venv/bin/python ]; then
  PYTHON=${PYTHON:-/home/bjin0/improved-diffusion/.venv/bin/python}
else
  PYTHON=${PYTHON:-python3}
fi

mkdir -p "$RESULTS_DIR" "$SLURM_LOG_DIR"

"$PYTHON" scripts/prepare_richer_metrics_manifest.py \
  --evaluation_results "$EVALUATION_RESULTS" \
  --output "$MANIFEST" \
  --schedules "$SCHEDULES" \
  --datasets "$DATASETS" \
  --objectives "$OBJECTIVES"

row_count=$(($(wc -l < "$MANIFEST") - 1))
if [ "$row_count" -le 0 ]; then
  echo "ERROR: manifest has no rows: $MANIFEST"
  exit 1
fi
if [ "$MAX_ROWS" -gt 0 ] && [ "$MAX_ROWS" -lt "$row_count" ]; then
  row_count="$MAX_ROWS"
fi

bash -n richer_metrics.slurm
"$PYTHON" -m py_compile scripts/run_richer_metrics.py scripts/prepare_richer_metrics_manifest.py scripts/aggregate_richer_metrics.py

last_index=$((row_count - 1))
echo "=========================================="
echo "Submitting richer metrics"
echo "Evaluation results: $EVALUATION_RESULTS"
echo "Rows: $row_count"
echo "Manifest: $MANIFEST"
echo "Results dir: $RESULTS_DIR"
echo "Feature cache: $FEATURE_CACHE_ROOT"
echo "Metrics: $METRICS"
echo "Python: $PYTHON"
echo "CLIP model: $CLIP_MODEL"
echo "CLIP pretrained: $CLIP_PRETRAINED"
echo "Max concurrent array tasks: $MAX_CONCURRENT"
echo "Partition override: ${RICHER_PARTITION:-script default}"
echo "Time override: ${RICHER_TIME:-script default}"
echo "Dry run: $DRY_RUN"
echo "=========================================="

METRICS_EXPORT=${METRICS//,/;}
array_spec="0-${last_index}"
if [ "$MAX_CONCURRENT" -gt 0 ]; then
  array_spec="${array_spec}%${MAX_CONCURRENT}"
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN command:"
  echo "sbatch --array=${array_spec} richer_metrics.slurm"
  exit 0
fi

sbatch_overrides=()
if [ -n "$RICHER_PARTITION" ]; then
  sbatch_overrides+=(--partition="$RICHER_PARTITION")
fi
if [ -n "$RICHER_TIME" ]; then
  sbatch_overrides+=(--time="$RICHER_TIME")
fi

sbatch_output=$(
  env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
  sbatch \
    --array="$array_spec" \
    "${sbatch_overrides[@]}" \
    --export="ALL,MANIFEST=${MANIFEST},OUTPUT_DIR=${RESULTS_DIR},FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT},METRICS=${METRICS_EXPORT},CLIP_MODEL=${CLIP_MODEL},CLIP_PRETRAINED=${CLIP_PRETRAINED}" \
    --output="$SLURM_LOG_DIR/richer_metrics_%A_%a.out" \
    --error="$SLURM_LOG_DIR/richer_metrics_%A_%a.err" \
    richer_metrics.slurm
)
echo "$sbatch_output"
job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
if [ -n "$job_id" ]; then
  printf "submitted_at\tjob_id\tarray\tmanifest\tresults_dir\tevaluation_results\tmetrics\tfeature_cache_root\tclip_model\tclip_pretrained\tpartition\ttime_limit\tnotes\n" > "$SUBMISSION_TSV"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$job_id" \
    "$array_spec" \
    "$MANIFEST" \
    "$RESULTS_DIR" \
    "$EVALUATION_RESULTS" \
    "$METRICS" \
    "$FEATURE_CACHE_ROOT" \
    "$CLIP_MODEL" \
    "$CLIP_PRETRAINED" \
    "${RICHER_PARTITION:-script default}" \
    "${RICHER_TIME:-script default}" \
    "${NOTES:-}" >> "$SUBMISSION_TSV"
fi
echo "Aggregate after jobs finish:"
echo "  python3 scripts/aggregate_richer_metrics.py --manifest \"$MANIFEST\" --results_dir \"$RESULTS_DIR\" --output_tsv \"$RESULTS_DIR/richer_metrics_summary.tsv\""
