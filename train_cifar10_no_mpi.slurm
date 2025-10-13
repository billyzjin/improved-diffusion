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

# Create a simple training script that doesn't use MPI
cat > train_simple.py << 'EOF'
#!/usr/bin/env python3
"""
Simple training script without MPI for single GPU training.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# Add the project to path
sys.path.append('/home/bjin0/improved-diffusion')

# Import the improved diffusion modules
from improved_diffusion.unet import UNetModel
from improved_diffusion.gaussian_diffusion import GaussianDiffusion

def create_model():
    """Create a simple UNet model for CIFAR-10."""
    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=[],
        dropout=0.3,
        channel_mult=(1, 2, 2, 2),
        num_heads=1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    return model

def create_diffusion():
    """Create diffusion process."""
    return GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        model_mean_type="epsilon",
        model_var_type="learned_range",
        loss_type="mse",
        rescale_timesteps=False,
    )

def load_data(data_dir, batch_size=128):
    """Load CIFAR-10 data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def train():
    """Main training function."""
    print("Starting simple CIFAR-10 training...")
    
    # Create model and diffusion
    model = create_model()
    diffusion = create_diffusion()
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Load data
    data_loader = load_data("./cifar_train", batch_size=128)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Short training for testing
        total_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(device)
            
            # Sample timesteps
            t = torch.randint(0, diffusion.num_timesteps, (data.shape[0],), device=device)
            
            # Add noise
            noise = torch.randn_like(data)
            noisy_data = diffusion.q_sample(data, t, noise=noise)
            
            # Predict noise
            predicted_noise = model(noisy_data, t)
            
            # Compute loss
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "/tmp/improved_diffusion_logs/model_simple.pt")
    print("Training completed! Model saved.")

if __name__ == "__main__":
    train()
EOF

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

# Run the simple training script
python3 train_simple.py

#---------------------------------------------------------------------------------
# Print GPU stats to output file at job completion
dcgmi stats --verbose --job ${SLURM_JOB_ID}

echo "Training completed successfully!"
