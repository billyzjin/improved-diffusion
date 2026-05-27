#!/bin/bash
set -euo pipefail

# Submit a FID-oriented geometric endpoint probe grid. Each job trains 50K CIFAR
# steps and then computes 10K-sample FID from the 50K EMA checkpoint.

cd /home/bjin0/improved-diffusion

PROBE_SLURM=${PROBE_SLURM:-train_probe_geometric_fid.slurm}
JOB_PREFIX=${JOB_PREFIX:-probe_fid}
BETA_VALUES=${BETA_VALUES:-"1e-4 3e-4 1e-3"}
ALPHA_VALUES=${ALPHA_VALUES:-"1e-3 3e-3 1e-2"}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/cifar10_geometric_fid_param_probe_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-256}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_TSV=${SUBMISSION_TSV:-$PARENT_EVAL_DIR/submission.tsv}

sanitize_tag() {
    local value="$1"
    value="${value//+/}"
    value="${value//./p}"
    value="${value//\//_}"
    value="${value//:/_}"
    value="${value//,/__}"
    printf '%s' "$value"
}

validate_pair() {
    local beta="$1"
    local alpha="$2"
    python3 - "$beta" "$alpha" <<'PY'
import sys

beta = float(sys.argv[1])
alpha = float(sys.argv[2])
if not (0.0 < beta < 1.0):
    raise SystemExit(f"invalid beta_1={beta}")
if not (0.0 < alpha < 1.0):
    raise SystemExit(f"invalid alpha_bar_T={alpha}")
if not (alpha < 1.0 - beta):
    raise SystemExit(
        f"invalid pair: need alpha_bar_T < 1 - beta_1, got alpha_bar_T={alpha}, "
        f"1-beta_1={1.0-beta}"
    )
PY
}

if [ ! -f "$PROBE_SLURM" ]; then
    echo "ERROR: probe slurm script not found: $PROBE_SLURM" >&2
    exit 1
fi

read -r -a beta_array <<< "$BETA_VALUES"
read -r -a alpha_array <<< "$ALPHA_VALUES"

mkdir -p "$PARENT_EVAL_DIR" "$SLURM_LOG_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tjob_id\tprobe_name\tbeta_1\talpha_bar_T\teval_dir\tnum_samples\n" > "$SUBMISSION_TSV"
fi

echo "=========================================="
echo "Submitting geometric FID parameter probe"
echo "Probe slurm: $PROBE_SLURM"
echo "Output dir: $PARENT_EVAL_DIR"
echo "Betas: ${beta_array[*]}"
echo "Alphas: ${alpha_array[*]}"
echo "Samples per probe: $NUM_SAMPLES"
echo "Dry run: $DRY_RUN"
echo "=========================================="

submitted=0
for beta in "${beta_array[@]}"; do
    for alpha in "${alpha_array[@]}"; do
        validate_pair "$beta" "$alpha"
        beta_tag=$(sanitize_tag "$beta")
        alpha_tag=$(sanitize_tag "$alpha")
        probe_name="${JOB_PREFIX}_b${beta_tag}_a${alpha_tag}"
        exp_dir="$PARENT_EVAL_DIR/cifar10_${probe_name}_step050000_n${NUM_SAMPLES}"

        export_arg="ALL,GEOMETRIC_BETA1=${beta},GEOMETRIC_ALPHA_BAR_T=${alpha},PROBE_NAME=${probe_name},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},NUM_SAMPLES=${NUM_SAMPLES},SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE}"
        echo "SUBMIT $probe_name beta=$beta alpha=$alpha"
        if [ "$DRY_RUN" != "1" ]; then
            sbatch_output=$(
                env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                    sbatch \
                    --account=bata0-external \
                    --partition=long_hopper \
                    --job-name="$probe_name" \
                    --output="$SLURM_LOG_DIR/${probe_name}_%j.out" \
                    --error="$SLURM_LOG_DIR/${probe_name}_%j.err" \
                    --export="$export_arg" \
                    "$PROBE_SLURM"
            )
            echo "$sbatch_output"
            job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
            if [ -z "$job_id" ]; then
                job_id="UNKNOWN"
            fi
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$(date -Is)" "$job_id" "$probe_name" "$beta" "$alpha" "$exp_dir" "$NUM_SAMPLES" >> "$SUBMISSION_TSV"
        fi

        submitted=$((submitted + 1))
        if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
            echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
            break 2
        fi
    done
done

echo "=========================================="
echo "Submission complete"
echo "Submitted: $submitted"
echo "Results dir: $PARENT_EVAL_DIR"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
