#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

EVAL_SLURM=${EVAL_SLURM:-evaluate_models_final.slurm}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-slurm_logs}
PARENT_EVAL_DIR=${PARENT_EVAL_DIR:-/project_gpfs/bata0/bjin0/cifar10_conditional_evaluation_$(date +%Y%m%d_%H%M%S)}
SUBMISSION_TSV=${SUBMISSION_TSV:-$PARENT_EVAL_DIR/submission.tsv}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
SKIP_NLL=${SKIP_NLL:-0}
SKIP_FID=${SKIP_FID:-0}
SKIP_TV=${SKIP_TV:-1}
TRAIN_RUN_GLOB=${TRAIN_RUN_GLOB:-/project_gpfs/bata0/bjin0/bjin0/12255[3-8]/logs}

experiments=(
    cifar10_cond_cosine_simple
    cifar10_cond_cosine_hybrid
    cifar10_cond_cosine_vlb
    cifar10_cond_geometric_cosine_simple
    cifar10_cond_geometric_cosine_hybrid
    cifar10_cond_geometric_cosine_vlb
)

mkdir -p "$SLURM_LOG_DIR" "$PARENT_EVAL_DIR"
if [ ! -f "$SUBMISSION_TSV" ]; then
    printf "submitted_at\tstatus\tjob_id\texperiment\tmodel_path\teval_dir\tclass_cond\tnum_classes\tskip_nll\tskip_fid\tskip_tv\n" > "$SUBMISSION_TSV"
fi

if [ ! -f "$EVAL_SLURM" ]; then
    echo "ERROR: evaluation slurm not found: $EVAL_SLURM"
    exit 1
fi

echo "=========================================="
echo "Submitting class-conditional CIFAR-10 evaluations"
echo "Evaluation dir: $PARENT_EVAL_DIR"
echo "Dry run: $DRY_RUN"
echo "Skip NLL: $SKIP_NLL"
echo "Skip FID: $SKIP_FID"
echo "Skip TV:  $SKIP_TV"
echo "=========================================="

for exp_name in "${experiments[@]}"; do
    model_path=$(find $TRAIN_RUN_GLOB -maxdepth 2 -name "ema_0.9999_500000.pt" -path "*/$exp_name/*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
        echo "ERROR: missing final EMA checkpoint for $exp_name"
        exit 1
    fi

    echo "SUBMIT $exp_name model=$model_path"
    if [ "$DRY_RUN" != "1" ]; then
        sbatch_output=$(
            env -u SBATCH_PARTITION -u SBATCH_ACCOUNT -u SBATCH_QOS -u SBATCH_GRES -u SBATCH_CONSTRAINT \
            sbatch \
                --account=bata0-external \
                --partition=long_hopper \
                --gres=gpu:h100:1 \
                --export="ALL,EVAL_MODEL_NAME=${exp_name},PARENT_EVAL_DIR=${PARENT_EVAL_DIR},CLASS_COND=True,NUM_CLASSES=10,SKIP_NLL=${SKIP_NLL},SKIP_FID=${SKIP_FID},SKIP_TV=${SKIP_TV},FORCE=${FORCE}" \
                --job-name="eval_${exp_name}" \
                --output="$SLURM_LOG_DIR/eval_${exp_name}_%j.out" \
                --error="$SLURM_LOG_DIR/eval_${exp_name}_%j.err" \
                "$EVAL_SLURM"
        )
        echo "$sbatch_output"
        job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
        if [ -z "$job_id" ]; then
            job_id="UNKNOWN"
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "submitted" "$job_id" "$exp_name" "$model_path" "$PARENT_EVAL_DIR/$exp_name" "True" "10" "$SKIP_NLL" "$SKIP_FID" "$SKIP_TV" >> "$SUBMISSION_TSV"
    else
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Is)" "dry_run" "DRY_RUN" "$exp_name" "$model_path" "$PARENT_EVAL_DIR/$exp_name" "True" "10" "$SKIP_NLL" "$SKIP_FID" "$SKIP_TV" >> "$SUBMISSION_TSV"
    fi
done

echo "=========================================="
echo "Submission complete"
echo "Manifest: $SUBMISSION_TSV"
echo "=========================================="
