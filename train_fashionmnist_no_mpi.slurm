#!/bin/bash
set -o pipefail

#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:h100:1

#---------------------------------------------------------------------------------
# Job specific name
#SBATCH --job-name=fashionmnist_paper_repro
#SBATCH --output=fashionmnist_paper_%j.out
#SBATCH --error=fashionmnist_paper_%j.err

#---------------------------------------------------------------------------------
# Print some useful variables
echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job
module load python/booth/3.12

#---------------------------------------------------------------------------------
# Set environment variables
# Default: write outputs to a job-specific directory. For resuming, you can override with RESUME_LOGDIR.
export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_paper_repro
export CUDA_VISIBLE_DEVICES=0
export OPENAI_LOG_FORMAT="stdout,log"

mkdir -p "$OPENAI_LOGDIR"

# NOTE: Aggressive cleanup of old .pt files was removed to avoid accidentally
# deleting checkpoints from other experiments (e.g. ImageNet jobs).
# Scratch space auto-cleans after 35 days anyway.

cd /home/bjin0/improved-diffusion

echo "Using clean in-repo no-MPI entrypoints (no runtime patching)."

# Ensure package is installed
echo "Checking if improved-diffusion package is installed..."
if python3 -c "import improved_diffusion; print('Package already installed')" 2>/dev/null; then
    echo "Package already installed, skipping installation."
else
    echo "Installing improved-diffusion package..."
    pip3 install -e . 2>&1 || { echo "ERROR: pip install failed"; exit 1; }
fi

#---------------------------------------------------------------------------------
# Prepare dataset
#---------------------------------------------------------------------------------
echo "Preparing Fashion-MNIST dataset..."
if [ ! -d "fashion_train" ]; then
    echo "ERROR: Fashion-MNIST dataset not found in current directory!"
    echo "Please download it on the login node first:"
    echo "  python3 datasets/fashionmnist.py"
    exit 1
fi

TRAIN_COUNT=$(find fashion_train -name "*.png" | wc -l)
TEST_COUNT=$(find fashion_test -name "*.png" | wc -l)
echo "Found $TRAIN_COUNT training images and $TEST_COUNT test images"

if [ $TRAIN_COUNT -lt 60000 ]; then
    echo "ERROR: Incomplete Fashion-MNIST dataset! Expected 60,000 training images, found $TRAIN_COUNT"
    exit 1
fi
if [ $TEST_COUNT -lt 10000 ]; then
    echo "ERROR: Incomplete Fashion-MNIST dataset! Expected 10,000 test images, found $TEST_COUNT"
    exit 1
fi

echo "Fashion-MNIST dataset verified successfully"

#---------------------------------------------------------------------------------
# Train models (mirror CIFAR-10 experiment matrix)
#---------------------------------------------------------------------------------
echo "=========================================="
echo "RUNNING EXPERIMENTS ON FASHION-MNIST"
echo "=========================================="

EXPERIMENT=${EXPERIMENT:-"linear_simple"}
echo "Running experiment: $EXPERIMENT"

# Architecture flags (mirror CIFAR-10; inputs are RGB 32x32 from datasets/fashionmnist.py)
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="$MODEL_FLAGS --num_heads 4 --attention_resolutions 16,8 --use_scale_shift_norm True"

# Training flags (disk-optimized like CIFAR)
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --ema_rate 0.9999 --log_interval 1000 --save_interval 50000"

RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_FLAGS="$TRAIN_FLAGS --resume_checkpoint $RESUME_CHECKPOINT"
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

# Optional: resume in-place from an existing experiment directory.
# Usage example:
#   sbatch --export=EXPERIMENT=cosine_simple,RESUME_LOGDIR=/project_gpfs/bata0/bjin0/$USER/<old_jobid>/logs/fashionmnist_cosine_simple train_fashionmnist_no_mpi.slurm
RESUME_LOGDIR=${RESUME_LOGDIR:-""}
if [ -n "$RESUME_LOGDIR" ]; then
    export OPENAI_LOGDIR="$RESUME_LOGDIR"
    mkdir -p "$OPENAI_LOGDIR"
    echo "Using RESUME_LOGDIR as OPENAI_LOGDIR: $OPENAI_LOGDIR"

    # If no explicit RESUME_CHECKPOINT was provided, auto-pick the newest model*.pt in the logdir.
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

DIFFUSION_STEPS="--diffusion_steps 4000"

case $EXPERIMENT in
    linear_simple)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_linear_simple
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma False \
            --use_kl False \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    linear_hybrid)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_linear_hybrid
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    linear_vlb)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_linear_vlb
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    cosine_simple)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_cosine_simple
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma False \
            --use_kl False \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    cosine_hybrid)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_cosine_hybrid
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    cosine_vlb)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_cosine_vlb
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    ours_simple)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_ours_simple
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours \
            --dropout 0.3 \
            --learn_sigma False \
            --use_kl False \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    ours_hybrid)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_ours_hybrid
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    ours_vlb)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_ours_vlb
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    ours_v2_simple)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_ours_v2_simple
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours_v2 \
            --dropout 0.3 \
            --learn_sigma False \
            --use_kl False \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    ours_v2_hybrid)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_ours_v2_hybrid
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours_v2 \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    ours_v2_vlb)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_ours_v2_vlb
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours_v2 \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            --class_cond False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    geometric_linear_simple)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_geometric_linear_simple
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_linear \
            --dropout 0.3 \
            --learn_sigma False --use_kl False \
            --class_cond False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    geometric_linear_hybrid)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_geometric_linear_hybrid
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_linear \
            --dropout 0.3 \
            --learn_sigma True --use_kl False --rescale_learned_sigmas True \
            --class_cond False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    geometric_linear_vlb)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_geometric_linear_vlb
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_linear \
            --dropout 0.3 \
            --learn_sigma True --use_kl True --schedule_sampler loss-second-moment \
            --class_cond False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    geometric_cosine_simple)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_geometric_cosine_simple
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_cosine \
            --dropout 0.3 \
            --learn_sigma False --use_kl False \
            --class_cond False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    geometric_cosine_hybrid)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_geometric_cosine_hybrid
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_cosine \
            --dropout 0.3 \
            --learn_sigma True --use_kl False --rescale_learned_sigmas True \
            --class_cond False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    geometric_cosine_vlb)
        if [ -z "$RESUME_LOGDIR" ]; then
            export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/fashionmnist_geometric_cosine_vlb
            mkdir -p "$OPENAI_LOGDIR"
        fi
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./fashion_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_cosine \
            --dropout 0.3 \
            --learn_sigma True --use_kl True --schedule_sampler loss-second-moment \
            --class_cond False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    *)
        echo "ERROR: Unknown experiment: $EXPERIMENT"
        echo "Valid options: linear_simple, linear_hybrid, linear_vlb, cosine_simple, cosine_hybrid, cosine_vlb, ours_simple, ours_hybrid, ours_vlb, ours_v2_simple, ours_v2_hybrid, ours_v2_vlb, geometric_linear_simple, geometric_linear_hybrid, geometric_linear_vlb, geometric_cosine_simple, geometric_cosine_hybrid, geometric_cosine_vlb"
        exit 1
        ;;
esac

TRAINING_EXIT_CODE=${PIPESTATUS[0]:-$?}
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "EXPERIMENT FAILED! Exit code: $TRAINING_EXIT_CODE"
    exit 1
fi

dcgmi stats --verbose --job ${SLURM_JOB_ID} || true
echo "Training completed successfully!"


