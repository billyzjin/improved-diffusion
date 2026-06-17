#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=linabar_smoke

set -euo pipefail

SCHEDULE=${SCHEDULE:-linabar_cosine}
OBJECTIVE=${OBJECTIVE:-hybrid}
SMOKE_STEPS=${SMOKE_STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-64}
SAMPLE_COUNT=${SAMPLE_COUNT:-16}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-16}
DATA_DIR=${DATA_DIR:-./cifar_train}
RUN_NAME=${RUN_NAME:-cifar10_${SCHEDULE}_${OBJECTIVE}_smoke_${SMOKE_STEPS}}

module load python/booth/3.12
if module load cuda/12.8 2>/dev/null; then
    echo "Loaded cuda/12.8"
elif module load cuda/12.4 2>/dev/null; then
    echo "Loaded cuda/12.4"
elif module load cuda/12.2 2>/dev/null; then
    echo "Loaded cuda/12.2"
elif module load cuda/11.8 2>/dev/null; then
    echo "Loaded cuda/11.8"
else
    echo "ERROR: No CUDA module found"
    exit 1
fi

cd /home/bjin0/improved-diffusion
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OPENAI_LOG_FORMAT="stdout,log,csv"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: CIFAR-10 training directory not found: $DATA_DIR"
    exit 1
fi

python3 - <<'PY'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this smoke test")
PY

python3 scripts/check_linabar_schedules.py --timesteps 4000

LOGDIR="/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/${RUN_NAME}"
mkdir -p "$LOGDIR"
export OPENAI_LOGDIR="$LOGDIR"

objective_flags=()
case "$OBJECTIVE" in
    simple)
        objective_flags=(--dropout 0.3 --learn_sigma False --use_kl False)
        ;;
    hybrid)
        objective_flags=(--dropout 0.3 --learn_sigma True --use_kl False --rescale_learned_sigmas True)
        ;;
    vlb)
        objective_flags=(--dropout 0.3 --learn_sigma True --use_kl True --schedule_sampler loss-second-moment)
        ;;
    *)
        echo "ERROR: unknown OBJECTIVE=$OBJECTIVE; expected simple, hybrid, or vlb"
        exit 1
        ;;
esac

python3 - <<PY > "$LOGDIR/schedule_metadata.tsv"
import numpy as np
from improved_diffusion.gaussian_diffusion import get_named_beta_schedule

schedule = "${SCHEDULE}"
betas = get_named_beta_schedule(schedule, 4000)
alpha_bar = np.cumprod(1.0 - betas)
print(f"schedule\t{schedule}")
print("diffusion_steps\t4000")
print(f"beta1\t{betas[0]:.17g}")
print(f"max_beta\t{betas.max():.17g}")
print(f"last_beta\t{betas[-1]:.17g}")
print(f"alpha_bar_T\t{alpha_bar[-1]:.17g}")
print(f"alpha_bar_T_log10\t{np.log10(alpha_bar[-1]):.17g}")
PY

{
    echo -e "run_name\t$RUN_NAME"
    echo -e "schedule\t$SCHEDULE"
    echo -e "objective\t$OBJECTIVE"
    echo -e "smoke_steps\t$SMOKE_STEPS"
    echo -e "batch_size\t$BATCH_SIZE"
    echo -e "sample_count\t$SAMPLE_COUNT"
    echo -e "data_dir\t$DATA_DIR"
    echo -e "logdir\t$LOGDIR"
    echo -e "start_time\t$(date -Is)"
} > "$LOGDIR/smoke_metadata.tsv"

echo "=========================================="
echo "LINEAR-ALPHABAR SMOKE TRAINING"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Run: $RUN_NAME"
echo "Schedule: $SCHEDULE"
echo "Objective: $OBJECTIVE"
echo "Steps: $SMOKE_STEPS"
echo "Logdir: $LOGDIR"
echo "=========================================="
cat "$LOGDIR/schedule_metadata.tsv"

python3 scripts/image_train_no_mpi.py \
    --data_dir "$DATA_DIR" \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --num_heads 4 \
    --attention_resolutions 16,8 \
    --use_scale_shift_norm True \
    "${objective_flags[@]}" \
    --diffusion_steps 4000 \
    --noise_schedule "$SCHEDULE" \
    --class_cond False \
    --use_checkpoint False \
    --rescale_timesteps True \
    --predict_xstart False \
    --lr 1e-4 \
    --batch_size "$BATCH_SIZE" \
    --ema_rate 0.9999 \
    --log_interval 50 \
    --save_interval "$SMOKE_STEPS" \
    --lr_anneal_steps "$SMOKE_STEPS" \
    2>&1 | tee "$LOGDIR/train_stdout.log"

MODEL_PATH=$(ls -1t "$LOGDIR"/ema_0.9999_*.pt 2>/dev/null | head -1)
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: no EMA checkpoint found in $LOGDIR"
    exit 1
fi
echo "Using smoke checkpoint: $MODEL_PATH"

export OPENAI_LOGDIR="$LOGDIR/sample_logger"
export OPENAI_LOG_FORMAT="stdout,log"
mkdir -p "$OPENAI_LOGDIR"

python3 scripts/image_sample_no_mpi.py \
    --model_path "$MODEL_PATH" \
    --num_samples "$SAMPLE_COUNT" \
    --batch_size "$SAMPLE_BATCH_SIZE" \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --num_heads 4 \
    --attention_resolutions 16,8 \
    --use_scale_shift_norm True \
    "${objective_flags[@]}" \
    --diffusion_steps 4000 \
    --noise_schedule "$SCHEDULE" \
    --class_cond False \
    --use_checkpoint False \
    --rescale_timesteps True \
    --predict_xstart False \
    --clip_denoised True \
    --use_ddim False \
    2>&1 | tee "$LOGDIR/sample_stdout.log"

{
    echo -e "end_time\t$(date -Is)"
    echo -e "checkpoint\t$MODEL_PATH"
} >> "$LOGDIR/smoke_metadata.tsv"

echo "Smoke test completed successfully: $LOGDIR"
