#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=tuned_geometric

set -euo pipefail

required_vars=(
    DATASET
    RUN_NAME
    SCHEDULE_NAME
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
export OPENAI_LOG_FORMAT="stdout,log"

case "$DATASET" in
    mnist)
        DATA_DIR="./mnist_train"
        EXPECTED_TRAIN_COUNT=60000
        IMAGE_SIZE=32
        NUM_CHANNELS=128
        NUM_RES_BLOCKS=3
        NUM_HEADS=4
        ATTENTION_RESOLUTIONS="16,8"
        BATCH_SIZE=${BATCH_SIZE:-128}
        LOG_INTERVAL=${LOG_INTERVAL:-1000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
        LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-500000}
        USE_FP16=${USE_FP16:-False}
        MICRO_BATCH=${MICRO_BATCH:-}
        ;;
    fashionmnist)
        DATA_DIR="./fashion_train"
        EXPECTED_TRAIN_COUNT=60000
        IMAGE_SIZE=32
        NUM_CHANNELS=128
        NUM_RES_BLOCKS=3
        NUM_HEADS=4
        ATTENTION_RESOLUTIONS="16,8"
        BATCH_SIZE=${BATCH_SIZE:-128}
        LOG_INTERVAL=${LOG_INTERVAL:-1000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
        LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-500000}
        USE_FP16=${USE_FP16:-False}
        MICRO_BATCH=${MICRO_BATCH:-}
        ;;
    cifar10)
        DATA_DIR="./cifar_train"
        EXPECTED_TRAIN_COUNT=50000
        IMAGE_SIZE=32
        NUM_CHANNELS=128
        NUM_RES_BLOCKS=3
        NUM_HEADS=4
        ATTENTION_RESOLUTIONS="16,8"
        BATCH_SIZE=${BATCH_SIZE:-128}
        LOG_INTERVAL=${LOG_INTERVAL:-1000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
        LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-500000}
        USE_FP16=${USE_FP16:-False}
        MICRO_BATCH=${MICRO_BATCH:-}
        ;;
    imagenet64)
        IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
        DATA_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
        EXPECTED_TRAIN_COUNT=0
        IMAGE_SIZE=64
        NUM_CHANNELS=128
        NUM_RES_BLOCKS=3
        NUM_HEADS=4
        ATTENTION_RESOLUTIONS="16,8"
        BATCH_SIZE=${BATCH_SIZE:-128}
        LOG_INTERVAL=${LOG_INTERVAL:-500}
        SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
        LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-200000}
        USE_FP16=${USE_FP16:-True}
        MICRO_BATCH=${MICRO_BATCH:-16}
        ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET; expected mnist, fashionmnist, cifar10, or imagenet64"
        exit 1
        ;;
esac

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: dataset directory missing for $DATASET: $DATA_DIR"
    exit 1
fi

if [ "$EXPECTED_TRAIN_COUNT" -gt 0 ]; then
    TRAIN_COUNT=$(find "$DATA_DIR" -name "*.png" | wc -l)
    if [ "$TRAIN_COUNT" -lt "$EXPECTED_TRAIN_COUNT" ]; then
        echo "ERROR: expected at least $EXPECTED_TRAIN_COUNT $DATASET training images, found $TRAIN_COUNT"
        exit 1
    fi
else
    TRAIN_COUNT="not_counted"
fi

DEFAULT_LOGDIR="/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/${DATASET}_tuned_${RUN_NAME}"
RESUME_LOGDIR=${RESUME_LOGDIR:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

if [ -n "$RESUME_LOGDIR" ]; then
    LOGDIR="$RESUME_LOGDIR"
else
    LOGDIR="$DEFAULT_LOGDIR"
fi
export OPENAI_LOGDIR="$LOGDIR"
mkdir -p "$LOGDIR"

MODEL_FLAGS="--image_size ${IMAGE_SIZE} --num_channels ${NUM_CHANNELS} --num_res_blocks ${NUM_RES_BLOCKS}"
MODEL_FLAGS="$MODEL_FLAGS --num_heads ${NUM_HEADS} --attention_resolutions ${ATTENTION_RESOLUTIONS} --use_scale_shift_norm True"

DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule geometric"
DIFFUSION_FLAGS="$DIFFUSION_FLAGS --geometric_beta1 ${GEOMETRIC_BETA1} --geometric_alpha_bar_T ${GEOMETRIC_ALPHA_BAR_T}"

TRAIN_FLAGS="--lr 1e-4 --batch_size ${BATCH_SIZE} --ema_rate 0.9999"
TRAIN_FLAGS="$TRAIN_FLAGS --log_interval ${LOG_INTERVAL} --save_interval ${SAVE_INTERVAL} --lr_anneal_steps ${LR_ANNEAL_STEPS}"
HYBRID_VB_WEIGHT=${HYBRID_VB_WEIGHT:-0.001}
if [ "$USE_FP16" = "True" ]; then
    TRAIN_FLAGS="$TRAIN_FLAGS --use_fp16 True"
fi
if [ -n "$MICRO_BATCH" ]; then
    TRAIN_FLAGS="$TRAIN_FLAGS --microbatch ${MICRO_BATCH}"
fi

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

dropout=${DROPOUT:-0.3}
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

{
    echo -e "dataset\t$DATASET"
    echo -e "run_name\t$RUN_NAME"
    echo -e "schedule_name\t$SCHEDULE_NAME"
    echo -e "objective\t$OBJECTIVE"
    echo -e "hybrid_vb_weight\t$([ "$OBJECTIVE" = "hybrid" ] && echo "$HYBRID_VB_WEIGHT" || echo "none")"
    echo -e "geometric_beta1\t$GEOMETRIC_BETA1"
    echo -e "geometric_alpha_bar_T\t$GEOMETRIC_ALPHA_BAR_T"
    echo -e "dropout\t$dropout"
    echo -e "image_size\t$IMAGE_SIZE"
    echo -e "batch_size\t$BATCH_SIZE"
    echo -e "microbatch\t${MICRO_BATCH:-none}"
    echo -e "use_fp16\t$USE_FP16"
    echo -e "train_steps\t$LR_ANNEAL_STEPS"
    echo -e "data_dir\t$DATA_DIR"
    echo -e "train_count\t$TRAIN_COUNT"
    echo -e "resume_checkpoint\t${RESUME_CHECKPOINT:-none}"
    echo -e "start_time\t$(date -Is)"
} > "$METADATA_FILE"

echo "=========================================="
echo "TUNED GEOMETRIC TRAINING"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Dataset: $DATASET"
echo "Run: $RUN_NAME"
echo "Schedule: $SCHEDULE_NAME"
echo "Objective: $OBJECTIVE"
echo "Hybrid VB weight: $([ "$OBJECTIVE" = "hybrid" ] && echo "$HYBRID_VB_WEIGHT" || echo "none")"
echo "beta_1: $GEOMETRIC_BETA1"
echo "alpha_bar_T: $GEOMETRIC_ALPHA_BAR_T"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-none}"
echo "Data dir: $DATA_DIR"
echo "Logdir: $LOGDIR"
echo "=========================================="

python3 scripts/image_train_no_mpi.py \
    --data_dir "$DATA_DIR" \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    "${objective_flags[@]}" \
    --class_cond False \
    $TRAIN_FLAGS \
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
