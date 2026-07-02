#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SLURM=${TRAIN_SLURM:-train_cifar10_tuned_geometric.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/cifar10_tuned_geometric_training_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$SUBMISSION_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}
MAX_SUBMITS=${MAX_SUBMITS:-0}
HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT:-0.001}

mkdir -p "$SLURM_LOG_DIR" "$SUBMISSION_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tjob_id\trun_name\tobjective\tbeta_1\talpha_bar_T\treason\n" > "$SUBMISSION_TSV"
fi

# Format: run_name|objective|beta_1|alpha_bar_T|reason
runs=(
    "fid_b3e-3_a1e-2_hybrid|hybrid|3e-3|1e-2|best 50k probe FID"
    "fid_b3e-3_a1e-2_simple|simple|3e-3|1e-2|same FID-tuned schedule with simple objective"
    "nll_b1e-5_a3e-3_hybrid|hybrid|1e-5|3e-3|best 50k probe NLL"
    "balanced_b3e-5_a1e-3_hybrid|hybrid|3e-5|1e-3|best overlap point with both NLL and FID measured"
)

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: training slurm not found: $TRAIN_SLURM"
    exit 1
fi

echo "=========================================="
echo "Submitting tuned CIFAR-10 geometric training jobs"
echo "Submission dir: $SUBMISSION_DIR"
echo "Dry run: $DRY_RUN"
echo "Hybrid VB weight: $HYBRID_VB_WEIGHT"
echo "=========================================="

submitted=0
for entry in "${runs[@]}"; do
    IFS='|' read -r run_name objective beta alpha reason <<< "$entry"
    export_arg="ALL,RUN_NAME=${run_name},OBJECTIVE=${objective},HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT},GEOMETRIC_BETA1=${beta},GEOMETRIC_ALPHA_BAR_T=${alpha}"
    job_name="cifar_geo_${run_name}"
    echo "SUBMIT $run_name objective=$objective beta=$beta alpha=$alpha"
    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
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
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "$job_id" "$run_name" "$objective" "$beta" "$alpha" "$reason" >> "$SUBMISSION_TSV"
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
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
