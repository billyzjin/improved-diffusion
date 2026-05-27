#!/bin/bash
#SBATCH --partition=long_hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --account=bata0-external

# Train a short CIFAR-10 geometric endpoint probe, then compute 10k-sample FID.

set -euo pipefail

required_vars=(
    GEOMETRIC_BETA1
    GEOMETRIC_ALPHA_BAR_T
    PROBE_NAME
    PARENT_EVAL_DIR
)
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: required environment variable $var is not set"
        exit 1
    fi
done

NUM_SAMPLES=${NUM_SAMPLES:-10000}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-256}
KEEP_SAMPLES=${KEEP_SAMPLES:-0}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-050000}
CIFAR_STATS_FILE=${CIFAR_STATS_FILE:-/project_gpfs/bata0/bjin0/cifar10_train_stats.npz}

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

python3 - <<'PY'
import importlib.util
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"pytorch_fid available: {importlib.util.find_spec('pytorch_fid') is not None}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this job")
if importlib.util.find_spec("pytorch_fid") is None:
    raise SystemExit("pytorch_fid is required for this job")
PY

echo "=========================================="
echo "GEOMETRIC FID PROBE: ${PROBE_NAME}"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "beta_1: ${GEOMETRIC_BETA1}"
echo "alpha_bar_T: ${GEOMETRIC_ALPHA_BAR_T}"
echo "FID samples: ${NUM_SAMPLES}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "=========================================="

TRAIN_LOGDIR="/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_${PROBE_NAME}"
mkdir -p "$TRAIN_LOGDIR"
export OPENAI_LOGDIR="$TRAIN_LOGDIR"
export OPENAI_LOG_FORMAT="stdout,log,csv"

echo "Phase 1: Training 50K steps..."
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
    --geometric_beta1 "${GEOMETRIC_BETA1}" \
    --geometric_alpha_bar_T "${GEOMETRIC_ALPHA_BAR_T}" \
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

MODEL_PATH="$TRAIN_LOGDIR/ema_0.9999_${CHECKPOINT_STEP}.pt"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH=$(ls -t "$TRAIN_LOGDIR"/ema_0.9999_*.pt 2>/dev/null | head -1)
fi
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: no EMA checkpoint found in $TRAIN_LOGDIR"
    exit 1
fi
echo "Using checkpoint: $MODEL_PATH"

echo "Phase 2: Sampling and FID..."
eval_model_name="cifar10_${PROBE_NAME}"
exp_dir="$PARENT_EVAL_DIR/${eval_model_name}_step${CHECKPOINT_STEP}_n${NUM_SAMPLES}"
mkdir -p "$exp_dir"

cat > "$exp_dir/metadata.tsv" <<EOF
experiment	$eval_model_name
probe_name	$PROBE_NAME
checkpoint_step	$CHECKPOINT_STEP
model_path	$MODEL_PATH
num_samples	$NUM_SAMPLES
sample_batch_size	$SAMPLE_BATCH_SIZE
noise_schedule	geometric
geometric_beta1	$GEOMETRIC_BETA1
geometric_alpha_bar_T	$GEOMETRIC_ALPHA_BAR_T
dropout	0.3
learn_sigma	True
use_kl	False
rescale_learned_sigmas	True
training_logdir	$TRAIN_LOGDIR
start_time	$(date -Is)
EOF

export OPENAI_LOGDIR="$exp_dir/logger"
export OPENAI_LOG_FORMAT="stdout,log"
mkdir -p "$OPENAI_LOGDIR"
rm -f "$OPENAI_LOGDIR"/samples_*.npz

python3 scripts/image_sample_no_mpi.py \
    --model_path "$MODEL_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$SAMPLE_BATCH_SIZE" \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --num_heads 4 \
    --attention_resolutions 16,8 \
    --use_scale_shift_norm True \
    --dropout 0.3 \
    --learn_sigma True \
    --use_kl False \
    --rescale_learned_sigmas True \
    --diffusion_steps 4000 \
    --noise_schedule geometric \
    --geometric_beta1 "$GEOMETRIC_BETA1" \
    --geometric_alpha_bar_T "$GEOMETRIC_ALPHA_BAR_T" \
    --class_cond False \
    --use_checkpoint False \
    --rescale_timesteps True \
    --predict_xstart False \
    --clip_denoised True \
    --use_ddim False \
    2>&1 | tee "$exp_dir/sample_results.txt"

sample_file=$(ls -1t "$OPENAI_LOGDIR"/samples_*.npz 2>/dev/null | head -1)
if [ -z "$sample_file" ] || [ ! -f "$sample_file" ]; then
    echo "ERROR: sample npz not found in $OPENAI_LOGDIR"
    exit 1
fi

if [ ! -f "$CIFAR_STATS_FILE" ]; then
    echo "Precomputing CIFAR-10 train stats: $CIFAR_STATS_FILE"
    python3 -m pytorch_fid ./cifar_train "$CIFAR_STATS_FILE" --save-stats --device cuda \
        2>&1 | tee "$exp_dir/fid_stats_build.txt"
fi

python3 scripts/compute_fid_from_npz.py \
    --samples_npz "$sample_file" \
    --stats_npz "$CIFAR_STATS_FILE" \
    --device cuda \
    --tmp_root "/project_gpfs/bata0/bjin0" \
    --output_txt "$exp_dir/fid_results.txt" \
    2>&1 | tee "$exp_dir/fid_compute_log.txt"

if [ "$KEEP_SAMPLES" = "1" ]; then
    stable_sample_file="$exp_dir/samples_${NUM_SAMPLES}x32x32x3.npz"
    mv "$sample_file" "$stable_sample_file"
    echo "Kept samples: $stable_sample_file"
else
    rm -f "$sample_file"
fi

{
    echo -e "end_time\t$(date -Is)"
    echo -e "fid\t$(cat "$exp_dir/fid_results.txt")"
} >> "$exp_dir/metadata.tsv"

echo "Done. FID: $(cat "$exp_dir/fid_results.txt")"
echo "Results: $exp_dir"
