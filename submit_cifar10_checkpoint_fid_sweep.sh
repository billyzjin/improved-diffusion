#!/bin/bash
set -euo pipefail

# Submit one FID-only evaluation job per CIFAR-10 EMA checkpoint.
#
# Defaults are intentionally set for a cheaper early-stopping probe:
#   NUM_SAMPLES=10000
# Use NUM_SAMPLES=50000 for paper-style FID once the promising steps are known.
#
# Useful overrides:
#   CHECKPOINT_STEPS="100000 150000 200000 250000 300000"
#   EXPERIMENTS="cifar10_linear_simple cifar10_cosine_hybrid"
#   NUM_SAMPLES=50000
#   DRY_RUN=1
#   MAX_SUBMITS=12
#   FORCE=1

cd /home/bjin0/improved-diffusion

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/project_gpfs/bata0/bjin0/bjin0}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/cifar10_fid_checkpoint_sweep_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-256}
KEEP_SAMPLES=${KEEP_SAMPLES:-0}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"050000 100000 150000 200000 250000 300000 350000 400000 450000 500000"}
EXPERIMENTS=${EXPERIMENTS:-"cifar10_linear_simple cifar10_linear_hybrid cifar10_linear_vlb cifar10_cosine_simple cifar10_cosine_hybrid cifar10_cosine_vlb cifar10_geometric_linear_simple cifar10_geometric_linear_hybrid cifar10_geometric_linear_vlb cifar10_geometric_cosine_simple cifar10_geometric_cosine_hybrid cifar10_geometric_cosine_vlb"}

mkdir -p "$PARENT_EVAL_DIR" slurm_logs

read -r -a steps <<< "$CHECKPOINT_STEPS"
read -r -a experiments <<< "$EXPERIMENTS"

echo "=========================================="
echo "Submitting CIFAR-10 checkpoint FID sweep"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Output dir: $PARENT_EVAL_DIR"
echo "Experiments: ${experiments[*]}"
echo "Steps: ${steps[*]}"
echo "Samples per checkpoint: $NUM_SAMPLES"
echo "Keep sample npz files: $KEEP_SAMPLES"
echo "Dry run: $DRY_RUN"
echo "=========================================="

find_checkpoint() {
    local exp_name="$1"
    local step="$2"
    find "$CHECKPOINT_ROOT" \
        -path "*/logs/${exp_name}/ema_0.9999_${step}.pt" \
        -type f \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

submitted=0
skipped_existing=0
missing=0

for exp_name in "${experiments[@]}"; do
    for step in "${steps[@]}"; do
        model_path=$(find_checkpoint "$exp_name" "$step")
        if [ -z "$model_path" ]; then
            echo "MISSING $exp_name step $step"
            missing=$((missing + 1))
            continue
        fi

        exp_dir="$PARENT_EVAL_DIR/${exp_name}_step${step}_n${NUM_SAMPLES}"
        if [ -s "$exp_dir/fid_results.txt" ] && [ "$FORCE" != "1" ]; then
            echo "SKIP existing $exp_name step $step"
            skipped_existing=$((skipped_existing + 1))
            continue
        fi

        job_name="fid_${exp_name#cifar10_}_${step}"
        output_log="slurm_logs/${job_name}_%j.out"
        error_log="slurm_logs/${job_name}_%j.err"

        export_arg="ALL,EVAL_MODEL_NAME=${exp_name},MODEL_PATH=${model_path},CHECKPOINT_STEP=${step},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},NUM_SAMPLES=${NUM_SAMPLES},SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE},KEEP_SAMPLES=${KEEP_SAMPLES},FORCE=${FORCE}"
        if [ -n "${EVAL_TIMESTEP_RESPACING:-}" ]; then
            export_arg="${export_arg},EVAL_TIMESTEP_RESPACING=${EVAL_TIMESTEP_RESPACING}"
        fi

        echo "SUBMIT $exp_name step $step -> $model_path"
        if [ "$DRY_RUN" != "1" ]; then
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --export="$export_arg" \
                --job-name="$job_name" \
                --output="$output_log" \
                --error="$error_log" \
                evaluate_cifar10_checkpoint_fid.slurm
        fi

        submitted=$((submitted + 1))
        if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
            echo "Reached MAX_SUBMITS=$MAX_SUBMITS; stopping."
            break 2
        fi
    done
done

echo "=========================================="
echo "Sweep submission complete"
echo "Submitted: $submitted"
echo "Skipped existing: $skipped_existing"
echo "Missing checkpoints: $missing"
echo "Output dir: $PARENT_EVAL_DIR"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo ""
echo "Aggregate after jobs finish:"
echo "  python3 scripts/aggregate_cifar10_checkpoint_fid.py $PARENT_EVAL_DIR"
echo "=========================================="
