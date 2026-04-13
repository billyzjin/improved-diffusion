#!/bin/bash

#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:1

#---------------------------------------------------------------------------------
# Job specific name
#SBATCH --job-name=cifar10_paper_repro
#SBATCH --output=cifar10_paper_%j.out
#SBATCH --error=cifar10_paper_%j.err

#---------------------------------------------------------------------------------
# Print some useful variables
echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job
module load python/booth/3.12
# Skip CUDA module - it's already included with Python

#---------------------------------------------------------------------------------
# Set environment variables
# Use a persistent directory instead of /tmp to preserve results across runs
# Use scratch space for logs (50TB available vs 14GB home directory)
export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_paper_repro
export CUDA_VISIBLE_DEVICES=0

# OPTIMIZE LOGGING FOR DISK SPACE
# Only log essential formats: stdout (terminal) and log (text file)
# Removed csv format which creates large CSV files with every metric
export OPENAI_LOG_FORMAT="stdout,log"

# Create log directory
mkdir -p $OPENAI_LOGDIR

# NOTE: Aggressive cleanup of old .pt files was removed to avoid accidentally
# deleting checkpoints from other experiments (e.g. ImageNet jobs).
# Scratch space auto-cleans after 35 days anyway.

# Navigate to your project directory
cd /home/bjin0/improved-diffusion

echo "Using clean in-repo no-MPI entrypoints (no runtime patching)."

# Check if package is already installed, install if needed
echo "Checking if improved-diffusion package is installed..."
if python3 -c "import improved_diffusion; print('Package already installed')" 2>/dev/null; then
    echo "Package already installed, skipping installation."
else
    echo "Installing improved-diffusion package..."
    if ! pip3 install -e . 2>&1; then
        echo "Editable install failed, trying regular install..."
        if ! pip3 install . 2>&1; then
            echo "ERROR: Both editable and regular install failed!"
            echo "Last pip error output:"
            pip3 install . 2>&1 | tail -20
    exit 1
fi
    fi
fi

# Verify installation
echo "Verifying package installation..."
if ! python3 -c "import improved_diffusion; print('Package installed successfully')" 2>&1; then
    echo "ERROR: Package installation verification failed!"
    echo "Trying to import improved_diffusion:"
    python3 -c "import improved_diffusion" 2>&1
    exit 1
fi

#---------------------------------------------------------------------------------
# Prepare dataset
#---------------------------------------------------------------------------------

echo "Preparing CIFAR-10 dataset..."
if [ ! -d "cifar_train" ]; then
    echo "ERROR: CIFAR-10 dataset not found in current directory!"
    echo ""
    echo "The script requires the real CIFAR-10 dataset to reproduce paper results."
    echo "Please download it on the login node first:"
    echo "  python3 datasets/cifar10.py"
    echo ""
    echo "This will create cifar_train/ and cifar_test/ directories with 50,000 and 10,000 images respectively."
    exit 1
fi

# Verify dataset completeness
TRAIN_COUNT=$(find cifar_train -name "*.png" | wc -l)
TEST_COUNT=$(find cifar_test -name "*.png" | wc -l)

echo "Found $TRAIN_COUNT training images and $TEST_COUNT test images"

if [ $TRAIN_COUNT -lt 50000 ]; then
    echo "ERROR: Incomplete CIFAR-10 dataset! Expected 50,000 training images, found $TRAIN_COUNT"
    echo "Please re-download the dataset:"
    echo "  rm -rf cifar_train cifar_test"
    echo "  python3 datasets/cifar10.py"
    exit 1
fi

if [ $TEST_COUNT -lt 10000 ]; then
    echo "ERROR: Incomplete CIFAR-10 dataset! Expected 10,000 test images, found $TEST_COUNT"
    echo "Please re-download the dataset:"
    echo "  rm -rf cifar_train cifar_test"
    echo "  python3 datasets/cifar10.py"
    exit 1
fi

echo "CIFAR-10 dataset verified successfully"

echo "CIFAR-10 dataset ready: $(find cifar_train -name "*.png" | wc -l) images found"

#---------------------------------------------------------------------------------
# Train models to reproduce paper results
#---------------------------------------------------------------------------------

echo "=========================================="
echo "REPRODUCING PAPER RESULTS FOR CIFAR-10"
echo "=========================================="
echo ""
echo "This script trains ONE experiment at a time."
echo "To reproduce full paper results, run all 5 scripts:"
echo "  1. train_cifar10_linear_simple.slurm"
echo "  2. train_cifar10_linear_hybrid.slurm"
echo "  2b. train_cifar10_linear_vlb.slurm (extra, not in paper Table 2)"
echo "  3. train_cifar10_cosine_simple.slurm"
echo "  4. train_cifar10_cosine_hybrid.slurm"
echo "  5. train_cifar10_cosine_vlb.slurm"
echo ""
echo "Each model trains for 500K iterations (~12-24 hours on H100)"
echo "=========================================="
echo ""

# Determine which experiment to run based on environment variable
# Default to linear_simple if not specified
EXPERIMENT=${EXPERIMENT:-"linear_simple"}

echo "Running experiment: $EXPERIMENT"
echo ""

# Paper hyperparameters for CIFAR-10:
# - Architecture: 3 resblocks per stage, channels [128, 256, 256, 256]
# - Batch size: 128
# - Learning rate: 1e-4
# - EMA rate: 0.9999
# - Image size: 32x32
# - Training iterations: 500K
# - Diffusion timesteps: 4000

# Model architecture flags (from paper)
# Note: channel_mult is hardcoded to (1,2,2,2) for image_size=32 in script_util.py
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="$MODEL_FLAGS --num_heads 4 --attention_resolutions 16,8 --use_scale_shift_norm True"

# Training flags (from paper) - OPTIMIZED FOR DISK SPACE
# Reduced log_interval from 100 to 1000 to cut logging by 10x (500,000 steps = 500 logs instead of 5,000)
# Increased save_interval from 10,000 to 50,000 to reduce checkpoints by 5x (500,000 steps = 10 checkpoints instead of 50)
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --ema_rate 0.9999 --log_interval 1000 --save_interval 50000"

# Resume checkpoint (set via environment variable)
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_FLAGS="$TRAIN_FLAGS --resume_checkpoint $RESUME_CHECKPOINT"
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

# Diffusion steps (from paper)
DIFFUSION_STEPS="--diffusion_steps 4000"

# Run the appropriate experiment based on EXPERIMENT variable
case $EXPERIMENT in
    linear_simple)
        echo "=========================================="
        echo "EXPERIMENT: Linear schedule, L_simple"
        echo "Target: FID = 2.90 (best FID in paper)"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_linear_simple
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma False \
            --use_kl False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    linear_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Linear schedule, L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_linear_hybrid
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;

    linear_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Linear schedule, L_vlb (extra)"
        echo "NOTE: Not reported in Table 2 of the paper"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_linear_vlb
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    cosine_simple)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_simple"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_cosine_simple
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma False \
            --use_kl False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    cosine_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_cosine_hybrid
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    cosine_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_vlb"
        echo "Target: NLL = 2.94 bits/dim (best NLL in paper)"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_cosine_vlb
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
    
    ours_simple)
        echo "=========================================="
        echo "EXPERIMENT: Our Custom schedule, L_simple"
        echo "Identical to cosine_simple except noise schedule"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_ours_simple
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours \
            --dropout 0.3 \
            --learn_sigma False \
            --use_kl False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;
    
    ours_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Our Custom schedule, L_hybrid"
        echo "Identical to cosine_hybrid except noise schedule"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_ours_hybrid
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;
    
    ours_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Our Custom schedule, L_vlb"
        echo "Identical to cosine_vlb except noise schedule"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_ours_vlb
        mkdir -p $OPENAI_LOGDIR

        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    ours_v2_simple)
        echo "=========================================="
        echo "EXPERIMENT: Optimal schedule (v2), L_simple"
        echo "beta_t = 1/(T-t+1), theoretically optimal"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_ours_v2_simple
        mkdir -p $OPENAI_LOGDIR

        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours_v2 \
            --dropout 0.3 \
            --learn_sigma False \
            --use_kl False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    ours_v2_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Optimal schedule (v2), L_hybrid"
        echo "beta_t = 1/(T-t+1), theoretically optimal"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_ours_v2_hybrid
        mkdir -p $OPENAI_LOGDIR

        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours_v2 \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl False \
            --rescale_learned_sigmas True \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    ours_v2_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Optimal schedule (v2), L_vlb"
        echo "beta_t = 1/(T-t+1), theoretically optimal"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_ours_v2_vlb
        mkdir -p $OPENAI_LOGDIR

        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule ours_v2 \
            --dropout 0.3 \
            --learn_sigma True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    geometric_linear_simple)
        echo "=========================================="
        echo "EXPERIMENT: Geometric schedule (linear endpoints), L_simple"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_geometric_linear_simple
        mkdir -p $OPENAI_LOGDIR
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_linear \
            --dropout 0.3 \
            --learn_sigma False --use_kl False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    geometric_linear_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Geometric schedule (linear endpoints), L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_geometric_linear_hybrid
        mkdir -p $OPENAI_LOGDIR
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_linear \
            --dropout 0.3 \
            --learn_sigma True --use_kl False --rescale_learned_sigmas True \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    geometric_linear_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Geometric schedule (linear endpoints), L_vlb"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_geometric_linear_vlb
        mkdir -p $OPENAI_LOGDIR
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_linear \
            --dropout 0.3 \
            --learn_sigma True --use_kl True --schedule_sampler loss-second-moment \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    geometric_cosine_simple)
        echo "=========================================="
        echo "EXPERIMENT: Geometric schedule (cosine endpoints), L_simple"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_geometric_cosine_simple
        mkdir -p $OPENAI_LOGDIR
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_cosine \
            --dropout 0.3 \
            --learn_sigma False --use_kl False \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    geometric_cosine_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Geometric schedule (cosine endpoints), L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_geometric_cosine_hybrid
        mkdir -p $OPENAI_LOGDIR
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_cosine \
            --dropout 0.3 \
            --learn_sigma True --use_kl False --rescale_learned_sigmas True \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    geometric_cosine_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Geometric schedule (cosine endpoints), L_vlb"
        echo "=========================================="
        export OPENAI_LOGDIR=/project_gpfs/bata0/bjin0/${SLURM_JOB_USER}/${SLURM_JOB_ID}/logs/cifar10_geometric_cosine_vlb
        mkdir -p $OPENAI_LOGDIR
        python3 scripts/image_train_no_mpi.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS $DIFFUSION_STEPS \
            --noise_schedule geometric_cosine \
            --dropout 0.3 \
            --learn_sigma True --use_kl True --schedule_sampler loss-second-moment \
            $TRAIN_FLAGS --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        TRAINING_EXIT_CODE=$?
        ;;

    *)
        echo "ERROR: Unknown experiment: $EXPERIMENT"
        echo "Valid options: linear_simple, linear_hybrid, linear_vlb, cosine_simple, cosine_hybrid, cosine_vlb, ours_simple, ours_hybrid, ours_vlb, ours_v2_simple, ours_v2_hybrid, ours_v2_vlb, geometric_linear_simple, geometric_linear_hybrid, geometric_linear_vlb, geometric_cosine_simple, geometric_cosine_hybrid, geometric_cosine_vlb"
        exit 1
        ;;
esac

# Check if training completed successfully
TRAINING_EXIT_CODE=$?
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Results saved in: $OPENAI_LOGDIR"
    echo ""
    echo "NOTE: Logs are stored in scratch space (/project_gpfs/) which has 50TB available"
    echo "vs your 14GB home directory. Files will auto-delete after 35 days."
    echo ""
    echo "To evaluate FID and NLL:"
echo "  python3 scripts/image_sample_no_mpi.py --model_path $OPENAI_LOGDIR/ema_0.9999_500000.pt ..."
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "EXPERIMENT FAILED!"
    echo "=========================================="
    echo "Exit code: $TRAINING_EXIT_CODE"
    echo ""
    echo "Last 50 lines of training output:"
    echo "----------------------------------------"
    tail -50 "$OPENAI_LOGDIR/log.txt" 2>/dev/null || echo "No log file found"
    echo "----------------------------------------"
    echo ""
    echo "Check the full log file for details: $OPENAI_LOGDIR/log.txt"
    echo "=========================================="
    exit 1
fi

#---------------------------------------------------------------------------------
# Print GPU stats
#---------------------------------------------------------------------------------
dcgmi stats --verbose --job ${SLURM_JOB_ID} || true

echo "Training completed successfully!"
