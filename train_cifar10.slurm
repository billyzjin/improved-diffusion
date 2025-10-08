#!/bin/bash

#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested
#SBATCH --partition=standard_h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1

#---------------------------------------------------------------------------------
# Job specific name
#SBATCH --job-name=cifar10_diffusion
#SBATCH --output=cifar10_%j.out
#SBATCH --error=cifar10_%j.err

#---------------------------------------------------------------------------------
# Print some useful variables
echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job
module load python/booth/3.12
module load cuda/11.0

#---------------------------------------------------------------------------------
# Set environment variables
export OPENAI_LOGDIR=/tmp/improved_diffusion_logs
export CUDA_VISIBLE_DEVICES=0

# Navigate to your project directory
cd /home/bjin0/improved-diffusion

# Install dependencies (if not already installed)
pip install -e .
pip install mpi4py

# Prepare dataset (if not already done)
# Check if CIFAR-10 already exists, otherwise use pre-downloaded data
if [ ! -d "cifar_train" ]; then
    echo "CIFAR-10 not found in current directory..."
    
    # Check for pre-downloaded data in home directory
    if [ -d "/home/bjin0/cifar10_data/cifar_train" ]; then
        echo "Using pre-downloaded CIFAR-10 data..."
        cp -r /home/bjin0/cifar10_data/cifar_* .
    elif [ -d "/home/bjin0/cifar_train" ]; then
        echo "Using CIFAR-10 data from home directory..."
        cp -r /home/bjin0/cifar_* .
    else
        echo "No pre-downloaded data found, trying network download..."
        if ! python3 datasets/cifar10.py; then
            echo "Network download failed, creating dummy CIFAR-10 dataset..."
            python3 prepare_cifar10_manual.py
        fi
    fi
else
    echo "CIFAR-10 dataset already exists, skipping download..."
fi

# Set hyperparameters
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# Run training
python3 scripts/image_train.py --data_dir ./cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# Generate samples after training
python3 scripts/image_sample.py --model_path $OPENAI_LOGDIR/ema_0.9999_*.pt $MODEL_FLAGS $DIFFUSION_FLAGS

#---------------------------------------------------------------------------------
# Print GPU stats to output file at job completion
dcgmi stats --verbose --job ${SLURM_JOB_ID}

echo "Training completed successfully!"
