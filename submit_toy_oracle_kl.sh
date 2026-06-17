#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

OUTPUT_DIR=${OUTPUT_DIR:-/project_gpfs/bata0/bjin0/toy_oracle_kl_$(date +%Y%m%d_%H%M%S)}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
TOY_DISTRIBUTION_LIST=${TOY_DISTRIBUTION_LIST:-gaussian_1d gmm_1d_symmetric_m1.5_sigma0.3 gmm_1d_symmetric_m3_sigma0.3 gmm_1d_symmetric_m5_sigma0.3 gmm_1d_skewed gmm_2d_grid_a2_sigma0.35 gmm_2d_grid_a4_sigma0.35}
TOY_TIMESTEPS_LIST=${TOY_TIMESTEPS_LIST:-50 100 250 1000}
SUBMISSION_TSV=${SUBMISSION_TSV:-$OUTPUT_DIR/submission.tsv}

mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
  printf "submitted_at\tstatus\tjob_id\tdistribution\ttimesteps\toutput_dir\n" > "$SUBMISSION_TSV"
fi

bash -n toy_oracle_kl.slurm

submitted=0
for distribution in $TOY_DISTRIBUTION_LIST; do
  for timesteps in $TOY_TIMESTEPS_LIST; do
    job_output_dir="$OUTPUT_DIR"
    echo "SUBMIT toy oracle distribution=$distribution T=$timesteps output=$job_output_dir"
    if [ "$DRY_RUN" = "1" ]; then
      printf "%s\tdry_run\tDRY_RUN\t%s\t%s\t%s\n" "$(date -Is)" "$distribution" "$timesteps" "$job_output_dir" >> "$SUBMISSION_TSV"
    else
      sbatch_output=$(
        env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
        sbatch \
          --export="ALL,OUTPUT_DIR=${job_output_dir},TOY_DISTRIBUTIONS=${distribution},TOY_TIMESTEPS=${timesteps}" \
          --job-name="toy_${distribution}_T${timesteps}" \
          --output="$SLURM_LOG_DIR/toy_${distribution}_T${timesteps}_%j.out" \
          --error="$SLURM_LOG_DIR/toy_${distribution}_T${timesteps}_%j.err" \
          toy_oracle_kl.slurm
      )
      echo "$sbatch_output"
      job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
      printf "%s\tsubmitted\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$job_id" "$distribution" "$timesteps" "$job_output_dir" >> "$SUBMISSION_TSV"
    fi
    submitted=$((submitted + 1))
    if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
      echo "Reached MAX_SUBMITS=$MAX_SUBMITS"
      exit 0
    fi
  done
done

echo "Submitted $submitted toy oracle jobs."
echo "Output dir: $OUTPUT_DIR"
echo "Manifest: $SUBMISSION_TSV"
echo "Aggregate after jobs finish:"
echo "  python3 scripts/aggregate_toy_oracle_kl.py --toy_dir \"$OUTPUT_DIR\""
