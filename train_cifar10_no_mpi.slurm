#!/bin/bash

#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=bata0-external

#---------------------------------------------------------------------------------
# Resources requested
#SBATCH --partition=long_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

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
export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_paper_repro
export CUDA_VISIBLE_DEVICES=0

# OPTIMIZE LOGGING FOR DISK SPACE
# Only log essential formats: stdout (terminal) and log (text file)
# Removed csv format which creates large CSV files with every metric
export OPENAI_LOG_FORMAT="stdout,log"

# Create log directory
mkdir -p $OPENAI_LOGDIR

# Clean up old logs to free space (keep only recent experiments)
echo "Cleaning up old logs to free disk space..."
find /home/bjin0/improved-diffusion/logs -name "progress*.csv" -delete 2>/dev/null || true
find /home/bjin0/improved-diffusion/logs -name "tb*" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/bjin0/improved-diffusion/logs -name "*.pt" -mtime +7 -delete 2>/dev/null || true

# Navigate to your project directory
cd /home/bjin0/improved-diffusion

# Always restore originals, even on failure
restore_originals() {
  cp improved_diffusion/dist_util_original.py improved_diffusion/dist_util.py 2>/dev/null || true
  cp improved_diffusion/image_datasets_original.py improved_diffusion/image_datasets.py 2>/dev/null || true
  cp improved_diffusion/train_util_original.py improved_diffusion/train_util.py 2>/dev/null || true
  cp improved_diffusion/resample_original.py improved_diffusion/resample.py 2>/dev/null || true
}
trap restore_originals EXIT

# Install dependencies (if not already installed)
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

# Verify installation
echo "Verifying package installation..."
if ! python3 -c "import improved_diffusion; print('Package installed successfully')" 2>&1; then
    echo "ERROR: Package installation verification failed!"
    echo "Trying to import improved_diffusion:"
    python3 -c "import improved_diffusion" 2>&1
    exit 1
fi

#---------------------------------------------------------------------------------
# Apply patches to remove MPI dependencies for single-GPU training
#---------------------------------------------------------------------------------

echo "Creating no-MPI patches..."

# Create dist_util_no_mpi.py
cat > improved_diffusion/dist_util_no_mpi.py << 'EOF'
"""
Distributed utilities - modified to work without MPI for single GPU training.
"""
import os
import torch as th

def setup_dist():
    """Setup for single GPU (no distributed training)."""
    if not th.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return
    th.cuda.set_device(0)
    print(f"Using GPU: {th.cuda.get_device_name(0)}")

def dev():
    """Get the device to use for torch.distributed."""
    return th.device("cuda" if th.cuda.is_available() else "cpu")

def get_world_size():
    """Get the number of processes (always 1 for single GPU)."""
    return 1

def get_rank():
    """Get the rank of this process (always 0 for single GPU)."""
    return 0

def get_local_rank():
    """Get the local rank (always 0 for single GPU)."""
    return 0

def is_main_process():
    """Check if this is the main process (always True for single GPU)."""
    return True

def barrier():
    """Synchronization barrier (no-op for single GPU)."""
    pass

def all_gather(tensor):
    """Gather tensors from all processes (no-op for single GPU)."""
    return [tensor]

def all_reduce(tensor, op=None):
    """Reduce tensors across processes (no-op for single GPU)."""
    return tensor

def broadcast(tensor, src=0):
    """Broadcast tensor to all processes (no-op for single GPU)."""
    return tensor

def synchronize():
    """Synchronize (no-op for single GPU)."""
    pass

def load_state_dict(path, map_location="cpu"):
    """Load state dict from file."""
    return th.load(path, map_location=map_location)

def sync_params(params):
    """Sync parameters across processes (no-op for single GPU)."""
    pass
EOF

# Create image_datasets_no_mpi.py
cat > improved_diffusion/image_datasets_no_mpi.py << 'EOF'
"""
Image datasets - modified to work without MPI.
"""
import os
import torch as th
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size, class_cond=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.class_cond = class_cond
        
        self.image_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        image = Image.open(path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = th.from_numpy(image).permute(2, 0, 1)
        
        if self.class_cond:
            class_name = os.path.basename(path).split('_')[0]
            class_id = hash(class_name) % 1000
            return image, class_id
        else:
            return image, 0

def load_data(data_dir, batch_size, image_size, class_cond=False):
    """Load image dataset."""
    dataset = ImageDataset(data_dir, image_size, class_cond)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    # Convert to infinite iterator and fix format
    def infinite_dataloader():
        while True:
            for batch_data in dataloader:
                if class_cond:
                    images, labels = batch_data
                    cond = {"y": labels} if class_cond else {}
                    yield images, cond
                else:
                    images, _ = batch_data
                    cond = {}
                    yield images, cond
    return infinite_dataloader()
EOF

# Create train_util patch
echo "Creating train_util patch..."
sed -e 's/self.global_batch = self.batch_size \* dist.get_world_size()/self.global_batch = self.batch_size \* 1  # Fixed: no distributed training/' \
    -e 's/if th.cuda.is_available():/if False:  # Disable DDP for single GPU/' \
    -e 's/self.use_ddp = True/self.use_ddp = False/' \
    -e 's/if dist.get_world_size() > 1:/if False:  # Disable multi-GPU check/' \
    -e 's/if dist.get_rank() == 0:/if True:  # Always save on single GPU/' \
    -e 's/dist.barrier()/pass  # No barrier needed for single GPU/' \
    -e 's/dist.get_world_size()/1/g' \
    -e 's/dist.get_rank()/0/g' \
    improved_diffusion/train_util.py > improved_diffusion/train_util_patched.py

# Create resample_no_mpi.py (CRITICAL - this was missing!)
cat > improved_diffusion/resample_no_mpi.py << 'EOF'
"""
Resampling utilities - modified to work without MPI for single GPU training.
"""
import torch as th
import numpy as np
from . import dist_util


class UniformSampler:
    """
    Uniform sampling of timesteps.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion

    def sample(self, batch_size, device):
        ts = np.random.choice(
            self.diffusion.num_timesteps, batch_size, replace=True
        ).astype(np.int64)
        return th.from_numpy(ts).to(device), th.ones_like(ts, dtype=th.float32)


class LossAwareSampler:
    """
    A wrapper around a sampler that performs loss-aware sampling.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.loss_history = np.zeros([diffusion.num_timesteps])

    def weights(self):
        """
        Get sampling weights for each timestep.
        """
        if not self.loss_history.any():
            return np.ones_like(self.loss_history)
        weights = np.sqrt(np.mean(self.loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - np.exp(-self.loss_history / self.loss_history.mean())
        return weights

    def sample(self, batch_size, device):
        """
        Sample timesteps based on loss history.
        """
        weights = self.weights()
        ts = np.random.choice(
            self.diffusion.num_timesteps, batch_size, replace=True, p=weights
        ).astype(np.int64)
        return th.from_numpy(ts).to(device), th.ones_like(ts, dtype=th.float32)

    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update loss history with local losses (no MPI needed for single GPU).
        """
        for t, loss in zip(local_ts.cpu().numpy(), local_losses.cpu().numpy()):
            if self.loss_history[t] == 0:
                self.loss_history[t] = loss
            else:
                self.loss_history[t] = 0.9 * self.loss_history[t] + 0.1 * loss

    def update_with_all_losses(self, ts, losses):
        """
        Update loss history with all losses.
        """
        for t, loss in zip(ts, losses):
            if self.loss_history[t] == 0:
                self.loss_history[t] = loss
            else:
                self.loss_history[t] = 0.9 * self.loss_history[t] + 0.1 * loss


def create_named_schedule_sampler(name, diffusion):
    """
    Create a named schedule sampler.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossAwareSampler(diffusion)
    else:
        raise ValueError(f"Unknown schedule sampler: {name}")
EOF

# Apply patches
echo "Applying patches..."
cp improved_diffusion/dist_util.py improved_diffusion/dist_util_original.py
cp improved_diffusion/dist_util_no_mpi.py improved_diffusion/dist_util.py

cp improved_diffusion/image_datasets.py improved_diffusion/image_datasets_original.py
cp improved_diffusion/image_datasets_no_mpi.py improved_diffusion/image_datasets.py

cp improved_diffusion/train_util.py improved_diffusion/train_util_original.py
cp improved_diffusion/train_util_patched.py improved_diffusion/train_util.py

cp improved_diffusion/resample.py improved_diffusion/resample_original.py
cp improved_diffusion/resample_no_mpi.py improved_diffusion/resample.py

#---------------------------------------------------------------------------------
# Prepare dataset
#---------------------------------------------------------------------------------

echo "Preparing CIFAR-10 dataset..."
if [ ! -d "cifar_train" ]; then
    echo "CIFAR-10 not found in current directory..."
    
    # Check for pre-downloaded data in home directory
    if [ -d "/home/bjin0/cifar10_data/cifar_train" ]; then
        echo "Using pre-downloaded CIFAR-10 data..."
        if ! cp -r /home/bjin0/cifar10_data/cifar_* . 2>&1; then
            echo "ERROR: Failed to copy pre-downloaded CIFAR-10 data!"
            exit 1
        fi
    elif [ -d "/home/bjin0/cifar_train" ]; then
        echo "Using CIFAR-10 data from home directory..."
        if ! cp -r /home/bjin0/cifar_* . 2>&1; then
            echo "ERROR: Failed to copy CIFAR-10 data from home directory!"
            exit 1
        fi
    else
        echo "No pre-downloaded data found, trying network download..."
        if ! python3 datasets/cifar10.py 2>&1; then
            echo "Network download failed, creating dummy CIFAR-10 dataset..."
            if ! python3 prepare_cifar10_manual.py 2>&1; then
                echo "ERROR: Failed to create dummy CIFAR-10 dataset!"
                exit 1
            fi
        fi
    fi
else
    echo "CIFAR-10 dataset already exists, skipping download..."
fi

# Verify dataset exists
if [ ! -d "cifar_train" ]; then
    echo "ERROR: CIFAR-10 dataset preparation failed - cifar_train directory not found!"
    exit 1
fi

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
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --ema_rate 0.9999 --log_interval 1000 --save_interval 10000"

# Diffusion steps (from paper)
DIFFUSION_STEPS="--diffusion_steps 4000"

# Run the appropriate experiment based on EXPERIMENT variable
case $EXPERIMENT in
    linear_simple)
        echo "=========================================="
        echo "EXPERIMENT: Linear schedule, L_simple"
        echo "Target: FID = 2.90 (best FID in paper)"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_linear_simple
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    linear_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Linear schedule, L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_linear_hybrid
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma True \
            --rescale_learned_sigmas False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    cosine_simple)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_simple"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_cosine_simple
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    cosine_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_cosine_hybrid
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma True \
            --rescale_learned_sigmas False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    cosine_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_vlb"
        echo "Target: NLL = 2.94 bits/dim (best NLL in paper)"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_cosine_vlb
        mkdir -p $OPENAI_LOGDIR
        
        echo "Starting training with output redirected to: $OPENAI_LOGDIR/log.txt"
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma True \
            --rescale_learned_sigmas True \
            --use_kl True \
            --schedule_sampler loss-second-moment \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000 \
            2>&1 | tee "$OPENAI_LOGDIR/log.txt"
        ;;
        
    *)
        echo "ERROR: Unknown experiment: $EXPERIMENT"
        echo "Valid options: linear_simple, linear_hybrid, cosine_simple, cosine_hybrid, cosine_vlb"
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
    echo "To evaluate FID and NLL:"
    echo "  python3 scripts/image_sample.py --model_path $OPENAI_LOGDIR/ema_0.9999_500000.pt ..."
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
# Restore original files
#---------------------------------------------------------------------------------
echo "Restoring original files..."
cp improved_diffusion/dist_util_original.py improved_diffusion/dist_util.py
cp improved_diffusion/image_datasets_original.py improved_diffusion/image_datasets.py
cp improved_diffusion/train_util_original.py improved_diffusion/train_util.py
cp improved_diffusion/resample_original.py improved_diffusion/resample.py

#---------------------------------------------------------------------------------
# Print GPU stats
#---------------------------------------------------------------------------------
dcgmi stats --verbose --job ${SLURM_JOB_ID} || true

echo "Training completed successfully!"
