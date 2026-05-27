#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

TRAIN_SLURM=${TRAIN_SLURM:-train_cifar10_tuned_geometric.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/cifar10_tuned_geometric_resume_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$SUBMISSION_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$SLURM_LOG_DIR" "$SUBMISSION_DIR"
printf "submitted_at\tjob_id\told_job_id\trun_name\tobjective\tbeta_1\talpha_bar_T\tresume_checkpoint\tresume_logdir\n" > "$SUBMISSION_TSV"

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: training slurm not found: $TRAIN_SLURM"
    exit 1
fi

# Format: old_job_id|run_name|objective|beta_1|alpha_bar_T|resume_logdir
runs=(
    "113346|fid_b3e-3_a1e-2_hybrid|hybrid|3e-3|1e-2|/project_gpfs/bata0/bjin0/bjin0/113346/logs/cifar10_tuned_geometric_fid_b3e-3_a1e-2_hybrid"
    "113347|fid_b3e-3_a1e-2_simple|simple|3e-3|1e-2|/project_gpfs/bata0/bjin0/bjin0/113347/logs/cifar10_tuned_geometric_fid_b3e-3_a1e-2_simple"
    "113348|nll_b1e-5_a3e-3_hybrid|hybrid|1e-5|3e-3|/project_gpfs/bata0/bjin0/bjin0/113348/logs/cifar10_tuned_geometric_nll_b1e-5_a3e-3_hybrid"
    "113349|balanced_b3e-5_a1e-3_hybrid|hybrid|3e-5|1e-3|/project_gpfs/bata0/bjin0/bjin0/113349/logs/cifar10_tuned_geometric_balanced_b3e-5_a1e-3_hybrid"
)

echo "=========================================="
echo "Submitting CIFAR-10 tuned geometric resume jobs"
echo "Submission dir: $SUBMISSION_DIR"
echo "Dry run: $DRY_RUN"
echo "=========================================="

for entry in "${runs[@]}"; do
    IFS='|' read -r old_job_id run_name objective beta alpha resume_logdir <<< "$entry"
    resume_checkpoint=$(ls -1 "$resume_logdir"/model*.pt 2>/dev/null | sort | tail -1)
    if [ -z "$resume_checkpoint" ]; then
        echo "ERROR: no model*.pt checkpoint found in $resume_logdir"
        exit 1
    fi

    export_arg="ALL,RUN_NAME=${run_name},OBJECTIVE=${objective},GEOMETRIC_BETA1=${beta},GEOMETRIC_ALPHA_BAR_T=${alpha},RESUME_LOGDIR=${resume_logdir},RESUME_CHECKPOINT=${resume_checkpoint}"
    job_name="cifar_geo_resume_${run_name}"
    echo "SUBMIT old=$old_job_id run=$run_name checkpoint=$resume_checkpoint"

    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
                sbatch \
                --time=1-00:00:00 \
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
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "$job_id" "$old_job_id" "$run_name" "$objective" "$beta" "$alpha" "$resume_checkpoint" "$resume_logdir" >> "$SUBMISSION_TSV"
    fi
done

echo "=========================================="
echo "Resume submission complete"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
