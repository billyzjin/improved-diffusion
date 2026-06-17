#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-/project_gpfs/bata0/bjin0/richer_metrics_features}
CLIP_MODEL=${CLIP_MODEL:-ViT-B-32}
CLIP_PRETRAINED=${CLIP_PRETRAINED:-/project_gpfs/bata0/bjin0/model_cache/openclip/open_clip_pytorch_model.bin}
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_FILE=${SUBMISSION_FILE:-results/clip_density_coverage_submissions_$(date +%Y%m%d_%H%M%S).tsv}
DRY_RUN=${DRY_RUN:-0}
SELECT_GROUPS=${SELECT_GROUPS:-main,linabar,svhn,cond_cifar10}
AGG_PARTITION=${AGG_PARTITION:-standard_hopper}

mkdir -p "$SLURM_LOG_DIR" "$(dirname "$SUBMISSION_FILE")"

printf "submitted_at\tgroup\tarray_job_id\taggregate_job_id\trows\tmanifest\tresults_dir\tmetrics\n" > "$SUBMISSION_FILE"

submit_group() {
  local group="$1"
  local manifest="$2"
  local outdir="$3"
  local rows
  rows=$(($(wc -l < "$manifest") - 1))
  if [ "$rows" -le 0 ]; then
    echo "ERROR: no rows in $manifest" >&2
    return 1
  fi

  local last=$((rows - 1))
  local array_spec="0-${last}"
  if [ "$MAX_CONCURRENT" -gt 0 ]; then
    array_spec="${array_spec}%${MAX_CONCURRENT}"
  fi

  echo "Submitting $group: rows=$rows array=$array_spec manifest=$manifest outdir=$outdir"
  if [ "$DRY_RUN" = "1" ]; then
    printf "%s\t%s\tDRY_RUN\tDRY_RUN\t%s\t%s\t%s\tclip_density_coverage\n" \
      "$(date -Iseconds)" "$group" "$rows" "$manifest" "$outdir" >> "$SUBMISSION_FILE"
    return 0
  fi

  local sbatch_out jid agg_out agg_jid
  sbatch_out=$(
    env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
    sbatch \
      --array="$array_spec" \
      --job-name="clipdc_${group}" \
      --export="ALL,MANIFEST=${manifest},OUTPUT_DIR=${outdir},FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT},METRICS=clip_density_coverage,CLIP_MODEL=${CLIP_MODEL},CLIP_PRETRAINED=${CLIP_PRETRAINED}" \
      --output="${SLURM_LOG_DIR}/clipdc_${group}_%A_%a.out" \
      --error="${SLURM_LOG_DIR}/clipdc_${group}_%A_%a.err" \
      richer_metrics.slurm
  )
  jid=$(printf "%s\n" "$sbatch_out" | awk '{print $4}')

  agg_out=$(
    env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
    sbatch \
      --account=bata0-external \
      --partition="$AGG_PARTITION" \
      --cpus-per-task=2 \
      --mem=8G \
      --time=00:30:00 \
      --dependency="afterok:${jid}" \
      --job-name="clipdc_${group}_agg" \
      --output="${SLURM_LOG_DIR}/clipdc_${group}_agg_%j.out" \
      --error="${SLURM_LOG_DIR}/clipdc_${group}_agg_%j.err" \
      --wrap="module load python/booth/3.12; cd /home/bjin0/improved-diffusion; python3 scripts/aggregate_richer_metrics.py --manifest ${manifest} --results_dir ${outdir} --output_tsv ${outdir}/richer_metrics_summary.tsv"
  )
  agg_jid=$(printf "%s\n" "$agg_out" | awk '{print $4}')

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\tclip_density_coverage\n" \
    "$(date -Iseconds)" "$group" "$jid" "$agg_jid" "$rows" "$manifest" "$outdir" >> "$SUBMISSION_FILE"
  echo "$group: $sbatch_out; aggregate: $agg_out"
}

should_submit() {
  case ",$SELECT_GROUPS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

if should_submit main; then
  submit_group main results/richer_metrics/manifest.tsv results/richer_metrics
fi
if should_submit linabar; then
  submit_group linabar results/linabar_richer_metrics_20260605_041727/manifest_completed_rows.tsv results/linabar_richer_metrics_20260605_041727
fi
if should_submit svhn; then
  submit_group svhn results/svhn_richer_metrics_20260615_080703/manifest.tsv results/svhn_richer_metrics_20260615_080703
fi
if should_submit cond_cifar10; then
  submit_group cond_cifar10 results/conditional_cifar10_richer_metrics_20260615_152836/manifest.tsv results/conditional_cifar10_richer_metrics_20260615_152836
fi

echo "Wrote submission record: $SUBMISSION_FILE"
cat "$SUBMISSION_FILE"
