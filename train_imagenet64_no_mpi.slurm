#!/bin/bash

#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100:1

#---------------------------------------------------------------------------------
# Job specific name
#SBATCH --job-name=imagenet64_train
#SBATCH --output=imagenet64_train_%j.out
#SBATCH --error=imagenet64_train_%j.err

set -o pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"
echo "Node: $SLURM_NODELIST"

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

export CUDA_VISIBLE_DEVICES=0
export OPENAI_LOG_FORMAT="stdout,log"

cd /home/bjin0/improved-diffusion || exit 1

echo "Using clean in-repo no-MPI entrypoints (no runtime patching)."

echo "Checking if improved-diffusion package is installed..."
if python3 -c "import improved_diffusion; print('Package already installed')" 2>/dev/null; then
  echo "Package already installed, skipping installation."
else
  echo "Installing improved-diffusion package..."
  pip3 install -e . 2>&1 || { echo "ERROR: pip install failed"; exit 1; }
fi

#---------------------------------------------------------------------------------
# Dataset locations (must be provided by user/environment)
#---------------------------------------------------------------------------------
IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
IMAGENET_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
IMAGENET_VAL_DIR=${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}

if [ ! -d "$IMAGENET_TRAIN_DIR" ]; then
  echo "ERROR: IMAGENET_TRAIN_DIR does not exist: $IMAGENET_TRAIN_DIR"
  exit 1
fi
if [ ! -d "$IMAGENET_VAL_DIR" ]; then
  echo "ERROR: IMAGENET_VAL_DIR does not exist: $IMAGENET_VAL_DIR"
  exit 1
fi

echo "ImageNet-64 train dir: $IMAGENET_TRAIN_DIR"
echo "ImageNet-64  val dir: $IMAGENET_VAL_DIR"

#---------------------------------------------------------------------------------
# Experiment config (mirror CIFAR/MNIST/Fashion structure)
#---------------------------------------------------------------------------------
EXPERIMENT=${EXPERIMENT:-"linear_simple"}
echo "Running experiment: $EXPERIMENT"

# Model architecture flags (unconditional ImageNet-64)
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="$MODEL_FLAGS --num_heads 4 --attention_resolutions 16,8 --use_scale_shift_norm True"

# Diffusion steps
DIFFUSION_STEPS="--diffusion_steps 4000"

# 200K iterations by default, disk-friendly logging/checkpointing.
# Override IMAGENET_LR_ANNEAL_STEPS for smoke tests.
IMAGENET_BATCH_SIZE=${IMAGENET_BATCH_SIZE:-128}
IMAGENET_MICROBATCH=${IMAGENET_MICROBATCH:-16}
IMAGENET_USE_FP16=${IMAGENET_USE_FP16:-True}
IMAGENET_LOG_INTERVAL=${IMAGENET_LOG_INTERVAL:-500}
IMAGENET_SAVE_INTERVAL=${IMAGENET_SAVE_INTERVAL:-20000}
IMAGENET_LR_ANNEAL_STEPS=${IMAGENET_LR_ANNEAL_STEPS:-200000}
TRAIN_FLAGS="--lr 1e-4 --batch_size ${IMAGENET_BATCH_SIZE} --microbatch ${IMAGENET_MICROBATCH} --use_fp16 ${IMAGENET_USE_FP16} --ema_rate 0.9999 --log_interval ${IMAGENET_LOG_INTERVAL} --save_interval ${IMAGENET_SAVE_INTERVAL} --lr_anneal_steps ${IMAGENET_LR_ANNEAL_STEPS}"

echo "ImageNet-64 batch size:  $IMAGENET_BATCH_SIZE"
echo "ImageNet-64 microbatch: $IMAGENET_MICROBATCH"
echo "ImageNet-64 use_fp16:   $IMAGENET_USE_FP16"
echo "ImageNet-64 steps:      $IMAGENET_LR_ANNEAL_STEPS"

RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}
if [ -n "$RESUME_CHECKPOINT" ]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --resume_checkpoint $RESUME_CHECKPOINT"
  echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

RESUME_LOGDIR=${RESUME_LOGDIR:-""}
if [ -n "$RESUME_LOGDIR" ]; then
  export OPENAI_LOGDIR="$RESUME_LOGDIR"
  mkdir -p "$OPENAI_LOGDIR"
  echo "Using RESUME_LOGDIR as OPENAI_LOGDIR: $OPENAI_LOGDIR"
  if [ -z "$RESUME_CHECKPOINT" ]; then
    AUTO_CKPT=$(ls -1 "$OPENAI_LOGDIR"/model*.pt 2>/dev/null | sort | tail -1)
    if [ -z "$AUTO_CKPT" ]; then
      echo "ERROR: RESUME_LOGDIR provided but no model*.pt checkpoints found in $OPENAI_LOGDIR"
      exit 1
    fi
    TRAIN_FLAGS="$TRAIN_FLAGS --resume_checkpoint $AUTO_CKPT"
    echo "Auto-resuming from latest checkpoint: $AUTO_CKPT"
  fi
fi

case $EXPERIMENT in
  linear_simple)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_linear_simple
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule linear \
      --dropout 0.1 \
      --learn_sigma False \
      --use_kl False \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  linear_hybrid)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_linear_hybrid
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule linear \
      --dropout 0.1 \
      --learn_sigma True \
      --use_kl False \
      --rescale_learned_sigmas True \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  linear_vlb)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_linear_vlb
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule linear \
      --dropout 0.1 \
      --learn_sigma True \
      --use_kl True \
      --schedule_sampler loss-second-moment \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  cosine_simple)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_cosine_simple
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule cosine \
      --dropout 0.3 \
      --learn_sigma False \
      --use_kl False \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  cosine_hybrid)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_cosine_hybrid
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule cosine \
      --dropout 0.3 \
      --learn_sigma True \
      --use_kl False \
      --rescale_learned_sigmas True \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  cosine_vlb)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_cosine_vlb
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule cosine \
      --dropout 0.3 \
      --learn_sigma True \
      --use_kl True \
      --schedule_sampler loss-second-moment \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  ours_simple)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_ours_simple
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule ours \
      --dropout 0.3 \
      --learn_sigma False \
      --use_kl False \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  ours_hybrid)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_ours_hybrid
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule ours \
      --dropout 0.3 \
      --learn_sigma True \
      --use_kl False \
      --rescale_learned_sigmas True \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  ours_vlb)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_ours_vlb
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule ours \
      --dropout 0.3 \
      --learn_sigma True \
      --use_kl True \
      --schedule_sampler loss-second-moment \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  ours_v2_simple)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_ours_v2_simple
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule ours_v2 \
      --dropout 0.3 \
      --learn_sigma False \
      --use_kl False \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  ours_v2_hybrid)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_ours_v2_hybrid
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule ours_v2 \
      --dropout 0.3 \
      --learn_sigma True \
      --use_kl False \
      --rescale_learned_sigmas True \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  ours_v2_vlb)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_ours_v2_vlb
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule ours_v2 \
      --dropout 0.3 \
      --learn_sigma True \
      --use_kl True \
      --schedule_sampler loss-second-moment \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  geometric_linear_simple)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_geometric_linear_simple
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule geometric_linear \
      --dropout 0.3 \
      --learn_sigma False --use_kl False \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  geometric_linear_hybrid)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_geometric_linear_hybrid
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule geometric_linear \
      --dropout 0.3 \
      --learn_sigma True --use_kl False --rescale_learned_sigmas True \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  geometric_linear_vlb)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_geometric_linear_vlb
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule geometric_linear \
      --dropout 0.3 \
      --learn_sigma True --use_kl True --schedule_sampler loss-second-moment \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  geometric_cosine_simple)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_geometric_cosine_simple
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule geometric_cosine \
      --dropout 0.3 \
      --learn_sigma False --use_kl False \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  geometric_cosine_hybrid)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_geometric_cosine_hybrid
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule geometric_cosine \
      --dropout 0.3 \
      --learn_sigma True --use_kl False --rescale_learned_sigmas True \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  geometric_cosine_vlb)
    if [ -z "$RESUME_LOGDIR" ]; then
      export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/imagenet64_geometric_cosine_vlb
      mkdir -p "$OPENAI_LOGDIR"
    fi
    python3 scripts/image_train_no_mpi.py \
      --data_dir "$IMAGENET_TRAIN_DIR" \
      $MODEL_FLAGS $DIFFUSION_STEPS \
      --noise_schedule geometric_cosine \
      --dropout 0.3 \
      --learn_sigma True --use_kl True --schedule_sampler loss-second-moment \
      --class_cond False \
      $TRAIN_FLAGS \
      2>&1 | tee "$OPENAI_LOGDIR/log.txt"
    ;;
  *)
    echo "ERROR: Unknown experiment: $EXPERIMENT"
    echo "Valid options: linear_simple, linear_hybrid, linear_vlb, cosine_simple, cosine_hybrid, cosine_vlb, ours_simple, ours_hybrid, ours_vlb, ours_v2_simple, ours_v2_hybrid, ours_v2_vlb, geometric_linear_simple, geometric_linear_hybrid, geometric_linear_vlb, geometric_cosine_simple, geometric_cosine_hybrid, geometric_cosine_vlb"
    exit 1
    ;;
esac

EXIT_CODE=${PIPESTATUS[0]:-$?}
if [ $EXIT_CODE -ne 0 ]; then
  echo "Training failed with exit code: $EXIT_CODE"
  exit $EXIT_CODE
fi

dcgmi stats --verbose --job ${SLURM_JOB_ID} || true
echo "Training completed successfully!"
