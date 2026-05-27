#!/bin/bash
set -euo pipefail

# Submit a small FID screen for the best geometric endpoint probes.
#
# Defaults:
#   NUM_SAMPLES=10000
#   checkpoint step 050000, matching the 50K-step probe trainings.
#
# Useful overrides:
#   NUM_SAMPLES=50000
#   DRY_RUN=1
#   FORCE=1
#   MAX_SUBMITS=2

cd /home/bjin0/improved-diffusion

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/project_gpfs/bata0/bjin0/bjin0}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/cifar10_probe_fid_screen_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-256}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-050000}
KEEP_SAMPLES=${KEEP_SAMPLES:-0}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
SUBMISSION_TSV=${SUBMISSION_TSV:-$PARENT_EVAL_DIR/submission.tsv}

mkdir -p "$PARENT_EVAL_DIR" slurm_logs

# Format: PROBE_NAME|GEOMETRIC_BETA1|GEOMETRIC_ALPHA_BAR_T
probes=(
    "probe_hyb_b1e-5_a3e-3|1e-5|3e-3"
    "probe_hyb_b1e-5_a1e-2|1e-5|1e-2"
    "probe_hyb_b1e-5_a1e-3|1e-5|1e-3"
    "probe_hyb_b3e-5_a1e-2|3e-5|1e-2"
)

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

if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tjob_id\tprobe_name\tbeta_1\talpha_bar_T\tcheckpoint_step\tmodel_path\teval_dir\tnum_samples\n" > "$SUBMISSION_TSV"
fi

echo "=========================================="
echo "Submitting CIFAR-10 geometric probe FID screen"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Output dir: $PARENT_EVAL_DIR"
echo "Checkpoint step: $CHECKPOINT_STEP"
echo "Samples per probe: $NUM_SAMPLES"
echo "Keep sample npz files: $KEEP_SAMPLES"
echo "Dry run: $DRY_RUN"
echo "=========================================="

submitted=0
skipped_existing=0
missing=0

for entry in "${probes[@]}"; do
    IFS='|' read -r probe_name beta_1 alpha_bar_T <<< "$entry"
    model_path=$(find_checkpoint "$probe_name")
    if [ -z "$model_path" ]; then
        echo "MISSING $probe_name step $CHECKPOINT_STEP"
        missing=$((missing + 1))
        continue
    fi

    exp_dir="$PARENT_EVAL_DIR/cifar10_${probe_name}_step${CHECKPOINT_STEP}_n${NUM_SAMPLES}"
    if [ -s "$exp_dir/fid_results.txt" ] && [ "$FORCE" != "1" ]; then
        echo "SKIP existing $probe_name step $CHECKPOINT_STEP"
        skipped_existing=$((skipped_existing + 1))
        continue
    fi

    job_name="fid_${probe_name#probe_hyb_}"
    output_log="slurm_logs/${job_name}_%j.out"
    error_log="slurm_logs/${job_name}_%j.err"

    export_arg="ALL,PROBE_NAME=${probe_name},MODEL_PATH=${model_path},CHECKPOINT_STEP=${CHECKPOINT_STEP},GEOMETRIC_BETA1=${beta_1},GEOMETRIC_ALPHA_BAR_T=${alpha_bar_T},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},NUM_SAMPLES=${NUM_SAMPLES},SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE},KEEP_SAMPLES=${KEEP_SAMPLES},FORCE=${FORCE}"
    if [ -n "${EVAL_TIMESTEP_RESPACING:-}" ]; then
        export_arg="${export_arg},EVAL_TIMESTEP_RESPACING=${EVAL_TIMESTEP_RESPACING}"
    fi

    echo "SUBMIT $probe_name beta=$beta_1 alpha=$alpha_bar_T -> $model_path"
    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --export="$export_arg" \
                --job-name="$job_name" \
                --output="$output_log" \
                --error="$error_log" \
                evaluate_probe_geometric_fid.slurm
        )
        echo "$sbatch_output"
        job_id=$(awk '{print $4}' <<< "$sbatch_output")
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "$job_id" "$probe_name" "$beta_1" "$alpha_bar_T" \
            "$CHECKPOINT_STEP" "$model_path" "$exp_dir" "$NUM_SAMPLES" >> "$SUBMISSION_TSV"
    fi

    submitted=$((submitted + 1))
    if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
        echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
        break
    fi
done

echo "=========================================="
echo "FID screen submission complete"
echo "Submitted: $submitted"
echo "Skipped existing: $skipped_existing"
echo "Missing checkpoints: $missing"
echo "Output dir: $PARENT_EVAL_DIR"
echo "Submission manifest: $SUBMISSION_TSV"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo ""
echo "Aggregate after jobs finish:"
echo "  python3 scripts/aggregate_cifar10_checkpoint_fid.py $PARENT_EVAL_DIR"
echo "=========================================="
