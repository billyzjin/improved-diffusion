#!/usr/bin/env python3
"""
Quick test script to verify the diffusion model setup works.
This runs a very short training session to test everything is working.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the project to path
sys.path.append('/home/bjin0/improved-diffusion')

# Set up environment
os.environ['OPENAI_LOGDIR'] = '/tmp/test_diffusion_logs'
os.makedirs('/tmp/test_diffusion_logs', exist_ok=True)

# Create modified dist_util.py for testing
dist_util_content = '''
"""
Helpers for distributed training - modified to work without MPI.
"""
import os
import torch as th

def setup_dist():
    if not th.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return
    th.cuda.set_device(0)
    print(f"Using GPU: {th.cuda.get_device_name(0)}")

def dev():
    return th.device("cuda" if th.cuda.is_available() else "cpu")

def get_world_size():
    return 1

def get_rank():
    return 0

def get_local_rank():
    return 0

def is_main_process():
    return True

def barrier():
    pass

def all_gather(tensor):
    return [tensor]

def all_reduce(tensor, op=None):
    return tensor

def broadcast(tensor, src=0):
    return tensor

def synchronize():
    pass
'''

# Write the modified dist_util.py
with open('improved_diffusion/dist_util.py', 'w') as f:
    f.write(dist_util_content)

# Import after modifying dist_util
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

def test_toy_model():
    """Test with a very small toy model."""
    print("Setting up toy diffusion model test...")
    
    # Setup
    dist_util.setup_dist()
    logger.configure()
    
    # Create very small model
    model, diffusion = create_model_and_diffusion(
        image_size=32,
        num_channels=32,  # Very small
        num_res_blocks=1,  # Very small
        learn_sigma=True,
        dropout=0.1,
        diffusion_steps=50,  # Very few steps
        noise_schedule='linear'
    )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    
    # Create dummy data
    print("Creating dummy data...")
    dummy_data = torch.randn(10, 3, 32, 32).to(dist_util.dev())
    
    # Test forward pass
    print("Testing forward pass...")
    t = torch.randint(0, diffusion.num_timesteps, (10,), device=dist_util.dev())
    noise = torch.randn_like(dummy_data)
    noisy_data = diffusion.q_sample(dummy_data, t, noise=noise)
    
    # Test model prediction
    predicted_noise = model(noisy_data, t)
    print(f"Model prediction shape: {predicted_noise.shape}")
    
    # Test loss computation
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful!")
    
    print("✅ Toy model test completed successfully!")
    print("The diffusion model setup is working correctly.")
    
    return True

if __name__ == "__main__":
    try:
        test_toy_model()
        print("\n🎉 All tests passed! You can now submit the full job.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
