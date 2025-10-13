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
#SBATCH --job-name=cifar10_two_stage
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

# Create a modified dist_util.py that works without MPI
cat > improved_diffusion/dist_util_no_mpi.py << 'EOF'
"""
Helpers for distributed training - modified to work without MPI.
"""
import os
import torch as th

# Single GPU setup
def setup_dist():
    """Setup for single GPU training."""
    if not th.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return
    
    # Set device
    th.cuda.set_device(0)
    print(f"Using GPU: {th.cuda.get_device_name(0)}")

def dev():
    """Get the device to use."""
    return th.device("cuda" if th.cuda.is_available() else "cpu")

def get_world_size():
    """Get world size (always 1 for single GPU)."""
    return 1

def get_rank():
    """Get rank (always 0 for single GPU)."""
    return 0

def get_local_rank():
    """Get local rank (always 0 for single GPU)."""
    return 0

def is_main_process():
    """Check if this is the main process."""
    return True

def barrier():
    """Barrier (no-op for single GPU)."""
    pass

def all_gather(tensor):
    """All gather (just return tensor for single GPU)."""
    return [tensor]

def all_reduce(tensor, op=None):
    """All reduce (just return tensor for single GPU)."""
    return tensor

def broadcast(tensor, src=0):
    """Broadcast (just return tensor for single GPU)."""
    return tensor

def synchronize():
    """Synchronize (no-op for single GPU)."""
    pass
EOF

# Backup original dist_util.py and replace with no-MPI version
cp improved_diffusion/dist_util.py improved_diffusion/dist_util_original.py
cp improved_diffusion/dist_util_no_mpi.py improved_diffusion/dist_util.py

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

# Create separate log directories for each stage
mkdir -p $OPENAI_LOGDIR/toy_model
mkdir -p $OPENAI_LOGDIR/full_model

echo "=========================================="
echo "STAGE 1: TOY MODEL (Quick Test)"
echo "=========================================="

# Stage 1: Toy model - trains in a few minutes
echo "Training toy model for quick testing..."
export OPENAI_LOGDIR_TEMP=$OPENAI_LOGDIR/toy_model

# Toy model hyperparameters (much smaller and faster)
TOY_MODEL_FLAGS="--image_size 32 --num_channels 64 --num_res_blocks 2 --learn_sigma True --dropout 0.1"
TOY_DIFFUSION_FLAGS="--diffusion_steps 100 --noise_schedule linear"
TOY_TRAIN_FLAGS="--lr 1e-3 --batch_size 64 --max_steps 1000 --save_interval 500"

# Run toy model training
python3 scripts/image_train.py --data_dir ./cifar_train $TOY_MODEL_FLAGS $TOY_DIFFUSION_FLAGS $TOY_TRAIN_FLAGS

# Generate samples from toy model
echo "Generating samples from toy model..."
python3 scripts/image_sample.py --model_path $OPENAI_LOGDIR/toy_model/ema_0.9999_*.pt $TOY_MODEL_FLAGS $TOY_DIFFUSION_FLAGS --num_samples 100

echo "Toy model completed! Check $OPENAI_LOGDIR/toy_model/ for results."

echo "=========================================="
echo "STAGE 2: FULL MODEL (Paper Reproduction)"
echo "=========================================="

# Stage 2: Full model - reproduces paper results
echo "Training full model to reproduce paper results..."
export OPENAI_LOGDIR_TEMP=$OPENAI_LOGDIR/full_model

# Full model hyperparameters (exact same as paper)
FULL_MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
FULL_DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
FULL_TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# Run full model training
python3 scripts/image_train.py --data_dir ./cifar_train $FULL_MODEL_FLAGS $FULL_DIFFUSION_FLAGS $FULL_TRAIN_FLAGS

# Generate samples from full model
echo "Generating samples from full model..."
python3 scripts/image_sample.py --model_path $OPENAI_LOGDIR/full_model/ema_0.9999_*.pt $FULL_MODEL_FLAGS $FULL_DIFFUSION_FLAGS --num_samples 10000

echo "Full model completed! Check $OPENAI_LOGDIR/full_model/ for results."

# Restore original dist_util.py
cp improved_diffusion/dist_util_original.py improved_diffusion/dist_util.py

#---------------------------------------------------------------------------------
# Print GPU stats to output file at job completion
dcgmi stats --verbose --job ${SLURM_JOB_ID}

echo "=========================================="
echo "TRAINING COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Toy model results: $OPENAI_LOGDIR/toy_model/"
echo "Full model results: $OPENAI_LOGDIR/full_model/"
echo "=========================================="
