#!/bin/bash
set -euo pipefail

# Submit FID-only jobs for already-trained geometric probe checkpoints.

cd /home/bjin0/improved-diffusion

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/project_gpfs/bata0/bjin0/bjin0}
JOB_PREFIX=${JOB_PREFIX:-probe_hyb}
BETA_VALUES=${BETA_VALUES:-"3e-5"}
ALPHA_VALUES=${ALPHA_VALUES:-"3e-4 1e-3 3e-3"}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/cifar10_existing_geometric_probe_fid_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-256}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-050000}
KEEP_SAMPLES=${KEEP_SAMPLES:-0}
FORCE=${FORCE:-0}
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

find_checkpoint() {
    local probe_name="$1"
    find "$CHECKPOINT_ROOT" \
        -path "*/logs/cifar10_${probe_name}/ema_0.9999_${CHECKPOINT_STEP}.pt" \
        -type f \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

read -r -a beta_array <<< "$BETA_VALUES"
read -r -a alpha_array <<< "$ALPHA_VALUES"

mkdir -p "$PARENT_EVAL_DIR" "$SLURM_LOG_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tjob_id\tprobe_name\tbeta_1\talpha_bar_T\tcheckpoint_step\tmodel_path\teval_dir\tnum_samples\n" > "$SUBMISSION_TSV"
fi

echo "=========================================="
echo "Submitting existing geometric probe FID jobs"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Output dir: $PARENT_EVAL_DIR"
echo "Betas: ${beta_array[*]}"
echo "Alphas: ${alpha_array[*]}"
echo "Dry run: $DRY_RUN"
echo "=========================================="

submitted=0
missing=0
for beta in "${beta_array[@]}"; do
    for alpha in "${alpha_array[@]}"; do
        beta_tag=$(sanitize_tag "$beta")
        alpha_tag=$(sanitize_tag "$alpha")
        probe_name="${JOB_PREFIX}_b${beta_tag}_a${alpha_tag}"
        model_path=$(find_checkpoint "$probe_name")
        if [ -z "$model_path" ]; then
            echo "MISSING $probe_name step $CHECKPOINT_STEP"
            missing=$((missing + 1))
            continue
        fi

        exp_dir="$PARENT_EVAL_DIR/cifar10_${probe_name}_step${CHECKPOINT_STEP}_n${NUM_SAMPLES}"
        export_arg="ALL,PROBE_NAME=${probe_name},MODEL_PATH=${model_path},CHECKPOINT_STEP=${CHECKPOINT_STEP},GEOMETRIC_BETA1=${beta},GEOMETRIC_ALPHA_BAR_T=${alpha},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},NUM_SAMPLES=${NUM_SAMPLES},SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE},KEEP_SAMPLES=${KEEP_SAMPLES},FORCE=${FORCE}"

        echo "SUBMIT $probe_name beta=$beta alpha=$alpha -> $model_path"
        if [ "$DRY_RUN" != "1" ]; then
            sbatch_output=$(
                env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                    sbatch \
                    --export="$export_arg" \
                    --job-name="fid_${probe_name#${JOB_PREFIX}_}" \
                    --output="$SLURM_LOG_DIR/fid_${probe_name#${JOB_PREFIX}_}_%j.out" \
                    --error="$SLURM_LOG_DIR/fid_${probe_name#${JOB_PREFIX}_}_%j.err" \
                    evaluate_probe_geometric_fid.slurm
            )
            echo "$sbatch_output"
            job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
            if [ -z "$job_id" ]; then
                job_id="UNKNOWN"
            fi
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$(date -Is)" "$job_id" "$probe_name" "$beta" "$alpha" \
                "$CHECKPOINT_STEP" "$model_path" "$exp_dir" "$NUM_SAMPLES" >> "$SUBMISSION_TSV"
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
echo "Missing checkpoints: $missing"
echo "Results dir: $PARENT_EVAL_DIR"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
