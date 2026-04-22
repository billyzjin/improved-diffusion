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
