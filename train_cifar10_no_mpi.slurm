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

# Create log directory
mkdir -p $OPENAI_LOGDIR

# Navigate to your project directory
cd /home/bjin0/improved-diffusion

# Always restore originals, even on failure
restore_originals() {
  cp improved_diffusion/dist_util_original.py improved_diffusion/dist_util.py 2>/dev/null || true
  cp improved_diffusion/image_datasets_original.py improved_diffusion/image_datasets.py 2>/dev/null || true
  cp improved_diffusion/train_util_original.py improved_diffusion/train_util.py 2>/dev/null || true
}
trap restore_originals EXIT

# Install dependencies (if not already installed)
echo "Installing improved-diffusion package..."
if ! pip3 install -e .; then
    echo "Editable install failed, trying regular install..."
    pip3 install .
fi

# Verify installation
python3 -c "import improved_diffusion; print('Package installed successfully')" || {
    echo "ERROR: Package installation failed!"
    exit 1
}

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
    improved_diffusion/train_util.py > improved_diffusion/train_util_patched.py

# Apply patches
echo "Applying patches..."
cp improved_diffusion/dist_util.py improved_diffusion/dist_util_original.py
cp improved_diffusion/dist_util_no_mpi.py improved_diffusion/dist_util.py

cp improved_diffusion/image_datasets.py improved_diffusion/image_datasets_original.py
cp improved_diffusion/image_datasets_no_mpi.py improved_diffusion/image_datasets.py

cp improved_diffusion/train_util.py improved_diffusion/train_util_original.py
cp improved_diffusion/train_util_patched.py improved_diffusion/train_util.py

#---------------------------------------------------------------------------------
# Prepare dataset
#---------------------------------------------------------------------------------

echo "Preparing CIFAR-10 dataset..."
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

# Training flags (from paper)
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --ema_rate 0.9999 --log_interval 100 --save_interval 10000"

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
        
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000
        ;;
        
    linear_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Linear schedule, L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_linear_hybrid
        mkdir -p $OPENAI_LOGDIR
        
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule linear \
            --dropout 0.1 \
            --learn_sigma True \
            --rescale_learned_sigmas False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000
        ;;
        
    cosine_simple)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_simple"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_cosine_simple
        mkdir -p $OPENAI_LOGDIR
        
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000
        ;;
        
    cosine_hybrid)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_hybrid"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_cosine_hybrid
        mkdir -p $OPENAI_LOGDIR
        
        python3 scripts/image_train.py \
            --data_dir ./cifar_train \
            $MODEL_FLAGS \
            $DIFFUSION_STEPS \
            --noise_schedule cosine \
            --dropout 0.3 \
            --learn_sigma True \
            --rescale_learned_sigmas False \
            $TRAIN_FLAGS \
            --lr_anneal_steps 500000
        ;;
        
    cosine_vlb)
        echo "=========================================="
        echo "EXPERIMENT: Cosine schedule, L_vlb"
        echo "Target: NLL = 2.94 bits/dim (best NLL in paper)"
        echo "=========================================="
        export OPENAI_LOGDIR=/home/bjin0/improved-diffusion/logs/cifar10_cosine_vlb
        mkdir -p $OPENAI_LOGDIR
        
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
            --lr_anneal_steps 500000
        ;;
        
    *)
        echo "ERROR: Unknown experiment: $EXPERIMENT"
        echo "Valid options: linear_simple, linear_hybrid, cosine_simple, cosine_hybrid, cosine_vlb"
        exit 1
        ;;
esac

# Check if training completed successfully
if [ $? -eq 0 ]; then
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
    echo "Check the error messages above for details."
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

#---------------------------------------------------------------------------------
# Print GPU stats
#---------------------------------------------------------------------------------
dcgmi stats --verbose --job ${SLURM_JOB_ID} || true

echo "Training completed successfully!"
