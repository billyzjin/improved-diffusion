#!/bin/bash
#SBATCH --partition=long_hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --account=bata0-external

# Probe run for geometric schedule endpoint tuning.
# Train for 50K steps on CIFAR-10 (45K train split), then evaluate NLL on 5K val split.
#
# Required environment variables:
#   GEOMETRIC_BETA1       - first-step noise parameter
#   GEOMETRIC_ALPHA_BAR_T - terminal signal level
#   PROBE_NAME            - experiment label (e.g., "probe_b1e-6")
#
# Usage:
#   sbatch --export=GEOMETRIC_BETA1=1e-6,GEOMETRIC_ALPHA_BAR_T=1.52e-10,PROBE_NAME=probe_b1e-6 \
#          --job-name=probe_b1e-6 train_probe_geometric.slurm

set -euo pipefail

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

echo "Testing PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || { echo "ERROR: PyTorch test failed"; exit 1; }

# Make the in-repo package importable without requiring a separate install step.
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"

echo "=========================================="
echo "GEOMETRIC ENDPOINT PROBE: ${PROBE_NAME}"
echo "  beta_1      = ${GEOMETRIC_BETA1}"
echo "  alpha_bar_T = ${GEOMETRIC_ALPHA_BAR_T}"
echo "=========================================="

LOGDIR="/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_${PROBE_NAME}"
export OPENAI_LOGDIR="$LOGDIR"
export OPENAI_LOG_FORMAT="stdout,log,csv"
mkdir -p "$LOGDIR"

cd /home/bjin0/improved-diffusion

# ---- Phase 1: Train for 50K steps ----
echo "Phase 1: Training (50K steps)..."
python3 scripts/image_train_no_mpi.py \
    --data_dir ./cifar_train_45k \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --num_heads 4 \
    --attention_resolutions 16,8 \
    --learn_sigma True \
    --dropout 0.3 \
    --diffusion_steps 4000 \
    --noise_schedule geometric \
    --geometric_beta1 ${GEOMETRIC_BETA1} \
    --geometric_alpha_bar_T ${GEOMETRIC_ALPHA_BAR_T} \
    --lr 1e-4 \
    --batch_size 128 \
    --class_cond False \
    --use_checkpoint False \
    --rescale_timesteps True \
    --rescale_learned_sigmas True \
    --use_scale_shift_norm True \
    --lr_anneal_steps 50000 \
    --ema_rate 0.9999 \
    --log_interval 500 \
    --save_interval 50000

echo "Training complete."

# ---- Phase 2: Evaluate NLL on validation split ----
echo "Phase 2: Evaluating NLL on validation split..."

# Find the EMA checkpoint
MODEL_PATH=$(ls -t "$LOGDIR"/ema_0.9999_*.pt 2>/dev/null | head -1)
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No EMA checkpoint found in $LOGDIR"
    exit 1
fi
echo "Using model: $MODEL_PATH"

EVAL_LOGDIR="${LOGDIR}/eval_val"
export OPENAI_LOGDIR="$EVAL_LOGDIR"
mkdir -p "$EVAL_LOGDIR"

python3 scripts/image_nll_no_mpi.py \
    --model_path "$MODEL_PATH" \
    --data_dir ./cifar_val \
    --batch_size 128 \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --num_heads 4 \
    --attention_resolutions 16,8 \
    --learn_sigma True \
    --dropout 0.3 \
    --diffusion_steps 4000 \
    --noise_schedule geometric \
    --geometric_beta1 ${GEOMETRIC_BETA1} \
    --geometric_alpha_bar_T ${GEOMETRIC_ALPHA_BAR_T} \
    --class_cond False \
    --use_checkpoint False \
    --rescale_learned_sigmas True \
    --use_scale_shift_norm True \
    --rescale_timesteps True \
    --num_samples 5000 \
    --clip_denoised True \
    2>&1 | tee "$EVAL_LOGDIR/nll_results.txt"

echo "=========================================="
echo "PROBE COMPLETE: ${PROBE_NAME}"
echo "  Results in: $EVAL_LOGDIR/nll_results.txt"
echo "=========================================="
