#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=cifar10_tuned_geometric

set -euo pipefail

required_vars=(
    RUN_NAME
    OBJECTIVE
    GEOMETRIC_BETA1
    GEOMETRIC_ALPHA_BAR_T
)
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: required environment variable $var is not set"
        exit 1
    fi
done

module load python/booth/3.12

cd /home/bjin0/improved-diffusion
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "cifar_train" ]; then
    echo "ERROR: CIFAR-10 train directory missing: $(pwd)/cifar_train"
    exit 1
fi

TRAIN_COUNT=$(find cifar_train -name "*.png" | wc -l)
if [ "$TRAIN_COUNT" -lt 50000 ]; then
    echo "ERROR: expected 50000 CIFAR train images, found $TRAIN_COUNT"
    exit 1
fi

DEFAULT_LOGDIR="/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_tuned_geometric_${RUN_NAME}"
RESUME_LOGDIR=${RESUME_LOGDIR:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

if [ -n "$RESUME_LOGDIR" ]; then
    LOGDIR="$RESUME_LOGDIR"
else
    LOGDIR="$DEFAULT_LOGDIR"
fi
export OPENAI_LOGDIR="$LOGDIR"
export OPENAI_LOG_FORMAT="stdout,log"
mkdir -p "$LOGDIR"

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="$MODEL_FLAGS --num_heads 4 --attention_resolutions 16,8 --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --ema_rate 0.9999 --log_interval 1000 --save_interval 50000"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule geometric --geometric_beta1 ${GEOMETRIC_BETA1} --geometric_alpha_bar_T ${GEOMETRIC_ALPHA_BAR_T}"

if [ -n "$RESUME_LOGDIR" ] && [ -z "$RESUME_CHECKPOINT" ]; then
    RESUME_CHECKPOINT=$(ls -1 "$RESUME_LOGDIR"/model*.pt 2>/dev/null | sort | tail -1)
    if [ -z "$RESUME_CHECKPOINT" ]; then
        echo "ERROR: RESUME_LOGDIR provided but no model*.pt checkpoints found in $RESUME_LOGDIR"
        exit 1
    fi
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ ! -f "$RESUME_CHECKPOINT" ]; then
        echo "ERROR: RESUME_CHECKPOINT does not exist: $RESUME_CHECKPOINT"
        exit 1
    fi
    TRAIN_FLAGS="$TRAIN_FLAGS --resume_checkpoint $RESUME_CHECKPOINT"
    export OPENAI_LOG_FORMAT="stdout"
fi

dropout="0.3"
HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT:-0.001}
objective_flags=()
case "$OBJECTIVE" in
    simple)
        objective_flags=(--dropout "$dropout" --learn_sigma False --use_kl False)
        ;;
    hybrid)
        objective_flags=(--dropout "$dropout" --learn_sigma True --use_kl False --rescale_learned_sigmas True --hybrid_vb_weight "$HYBRID_VB_WEIGHT")
        ;;
    vlb)
        objective_flags=(--dropout "$dropout" --learn_sigma True --use_kl True --schedule_sampler loss-second-moment)
        ;;
    *)
        echo "ERROR: unknown OBJECTIVE=$OBJECTIVE; expected simple, hybrid, or vlb"
        exit 1
        ;;
esac

if [ -n "$RESUME_CHECKPOINT" ]; then
    METADATA_FILE="$LOGDIR/metadata_resume_${SLURM_JOB_ID:-manual}.tsv"
    STDOUT_LOG_FILE="$LOGDIR/resume_${SLURM_JOB_ID:-manual}.stdout.log"
else
    METADATA_FILE="$LOGDIR/metadata.tsv"
    STDOUT_LOG_FILE="$LOGDIR/log.txt"
fi

cat > "$METADATA_FILE" <<EOF
run_name	$RUN_NAME
objective	$OBJECTIVE
hybrid_vb_weight	$([ "$OBJECTIVE" = "hybrid" ] && echo "$HYBRID_VB_WEIGHT" || echo "none")
geometric_beta1	$GEOMETRIC_BETA1
geometric_alpha_bar_T	$GEOMETRIC_ALPHA_BAR_T
train_steps	500000
data_dir	$(pwd)/cifar_train
resume_checkpoint	${RESUME_CHECKPOINT:-none}
start_time	$(date -Is)
EOF

echo "=========================================="
echo "CIFAR-10 TUNED GEOMETRIC TRAINING"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Run: $RUN_NAME"
echo "Objective: $OBJECTIVE"
echo "Hybrid VB weight: $([ "$OBJECTIVE" = "hybrid" ] && echo "$HYBRID_VB_WEIGHT" || echo "none")"
echo "beta_1: $GEOMETRIC_BETA1"
echo "alpha_bar_T: $GEOMETRIC_ALPHA_BAR_T"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-none}"
echo "Logdir: $LOGDIR"
echo "=========================================="

python3 scripts/image_train_no_mpi.py \
    --data_dir ./cifar_train \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    "${objective_flags[@]}" \
    $TRAIN_FLAGS \
    --lr_anneal_steps 500000 \
    2>&1 | tee "$STDOUT_LOG_FILE"
train_exit=${PIPESTATUS[0]}

{
    echo -e "end_time\t$(date -Is)"
    echo -e "exit_code\t$train_exit"
} >> "$METADATA_FILE"

if [ "$train_exit" -ne 0 ]; then
    echo "ERROR: training failed with exit code $train_exit"
    exit "$train_exit"
fi

echo "Training completed successfully: $LOGDIR"
dcgmi stats --verbose --job "${SLURM_JOB_ID:-}" || true
