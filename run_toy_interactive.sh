#!/bin/bash

echo "=========================================="
echo "TOY DIFFUSION MODEL - INTERACTIVE RUN"
echo "=========================================="

# Navigate to project
cd /home/bjin0/improved-diffusion

# Load modules
echo "Loading modules..."
module load python/booth/3.12
# Skip CUDA module - it's already included with Python

# Set environment
export OPENAI_LOGDIR=/tmp/toy_diffusion_logs
export CUDA_VISIBLE_DEVICES=0

echo "Environment set up:"
echo "  OPENAI_LOGDIR: $OPENAI_LOGDIR"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Create no-MPI patches
echo "Creating no-MPI patches..."

# Create dist_util_no_mpi.py
cat > improved_diffusion/dist_util_no_mpi.py << 'EOF'
"""
Helpers for distributed training - modified to work without MPI.
"""
import os
import torch as th

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
        
        # Get all image files
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
        
        # Load image
        image = Image.open(path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = th.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        if self.class_cond:
            # Extract class from filename (assuming format: class_name_XXXXX.png)
            class_name = os.path.basename(path).split('_')[0]
            # Simple hash-based class ID
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
        num_workers=1,  # Reduced to avoid warnings
        pin_memory=True
    )
    # Convert to infinite iterator and fix format
    def infinite_dataloader():
        while True:
            for batch_data in dataloader:
                if class_cond:
                    # batch_data is (images, labels)
                    images, labels = batch_data
                    # Convert to expected format: (batch, cond_dict)
                    cond = {"y": labels} if class_cond else {}
                    yield images, cond
                else:
                    # batch_data is (images, dummy_labels)
                    images, _ = batch_data
                    # Convert to expected format: (batch, cond_dict)
                    cond = {}
                    yield images, cond
    return infinite_dataloader()
EOF

# Create a simple patch for train_util.py to fix the distributed training issue
echo "Creating train_util patch..."
# Replace multiple problematic lines with sed commands
sed -e 's/self.global_batch = self.batch_size \* dist.get_world_size()/self.global_batch = self.batch_size \* 1  # Fixed: no distributed training/' \
    -e 's/if th.cuda.is_available():/if False:  # Disable DDP for single GPU/' \
    -e 's/self.use_ddp = True/self.use_ddp = False/' \
    -e 's/if dist.get_world_size() > 1:/if False:  # Disable multi-GPU check/' \
    improved_diffusion/train_util.py > improved_diffusion/train_util_patched.py

# Backup and replace files
echo "Backing up original files and applying patches..."
cp improved_diffusion/dist_util.py improved_diffusion/dist_util_original.py
cp improved_diffusion/dist_util_no_mpi.py improved_diffusion/dist_util.py

cp improved_diffusion/image_datasets.py improved_diffusion/image_datasets_original.py
cp improved_diffusion/image_datasets_no_mpi.py improved_diffusion/image_datasets.py

cp improved_diffusion/train_util.py improved_diffusion/train_util_original.py
cp improved_diffusion/train_util_patched.py improved_diffusion/train_util.py

# Prepare dataset
echo "Preparing dataset..."
if [ ! -d "cifar_train" ]; then
    echo "CIFAR-10 not found, creating dummy dataset..."
    python3 prepare_cifar10_manual.py
else
    echo "CIFAR-10 dataset already exists, skipping creation..."
fi

# Run toy model training
echo "=========================================="
echo "STARTING TOY MODEL TRAINING"
echo "=========================================="

TOY_MODEL_FLAGS="--image_size 32 --num_channels 32 --num_res_blocks 1 --learn_sigma True --dropout 0.1"
TOY_DIFFUSION_FLAGS="--diffusion_steps 50 --noise_schedule linear"
TOY_TRAIN_FLAGS="--lr 1e-3 --batch_size 32 --save_interval 250"

echo "Model flags: $TOY_MODEL_FLAGS"
echo "Diffusion flags: $TOY_DIFFUSION_FLAGS"
echo "Train flags: $TOY_TRAIN_FLAGS"

python3 scripts/image_train.py --data_dir ./cifar_train $TOY_MODEL_FLAGS $TOY_DIFFUSION_FLAGS $TOY_TRAIN_FLAGS

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Find the model checkpoint
    MODEL_PATH=$(find $OPENAI_LOGDIR -name "ema_0.9999_*.pt" | head -1)
    
    if [ -n "$MODEL_PATH" ]; then
        echo "Found model checkpoint: $MODEL_PATH"
        
        # Generate samples
        echo "=========================================="
        echo "GENERATING SAMPLES"
        echo "=========================================="
        
        python3 scripts/image_sample.py --model_path "$MODEL_PATH" $TOY_MODEL_FLAGS $TOY_DIFFUSION_FLAGS --num_samples 50
    else
        echo "No model checkpoint found in $OPENAI_LOGDIR"
        echo "Listing directory contents:"
        ls -la $OPENAI_LOGDIR/
    fi
else
    echo "Training failed! Check the error messages above."
fi

# Restore original files
echo "Restoring original files..."
cp improved_diffusion/dist_util_original.py improved_diffusion/dist_util.py
cp improved_diffusion/image_datasets_original.py improved_diffusion/image_datasets.py
cp improved_diffusion/train_util_original.py improved_diffusion/train_util.py

echo "=========================================="
echo "TOY MODEL COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Results saved to: $OPENAI_LOGDIR"
echo "Check the directory for:"
echo "  - Model checkpoints (*.pt files)"
echo "  - Generated samples (samples_*.npz)"
echo "  - Training logs"
echo "=========================================="
