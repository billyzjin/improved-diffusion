#!/bin/bash

#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested
#SBATCH --partition=long_h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-120:00:00
#SBATCH --gres=gpu:4

#---------------------------------------------------------------------------------
# Job specific name
#SBATCH --job-name=lsun_diffusion
#SBATCH --output=lsun_%j.out
#SBATCH --error=lsun_%j.err

#---------------------------------------------------------------------------------
# Print some useful variables
echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job
module load python/booth/3.12
module load cuda/11.0
module load openmpi

#---------------------------------------------------------------------------------
# Set environment variables
export OPENAI_LOGDIR=/tmp/improved_diffusion_logs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Navigate to your project directory
cd /home/bjin0/improved-diffusion

# Install dependencies (if not already installed)
pip install -e .
pip install mpi4py

# Set hyperparameters for LSUN bedroom
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# Run distributed training
mpiexec -n 4 python3 scripts/image_train.py --data_dir /path/to/lsun_bedroom $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# Generate samples after training
python3 scripts/image_sample.py --model_path $OPENAI_LOGDIR/ema_0.9999_*.pt $MODEL_FLAGS $DIFFUSION_FLAGS

#---------------------------------------------------------------------------------
# Print GPU stats to output file at job completion
dcgmi stats --verbose --job ${SLURM_JOB_ID}

echo "LSUN bedroom training completed successfully!"
