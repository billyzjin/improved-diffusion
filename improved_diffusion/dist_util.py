"""
Helpers for distributed training.
"""

import os
import torch

def setup_dist():
    pass

def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_state_dict(path, map_location=None):
    return torch.load(path, map_location=map_location)

def sync_params(params):
    pass

def get_world_size():
    return 1

def get_rank():
    return 0

def is_main_process():
    return True

def barrier():
    pass

def broadcast(tensor, src=0):
    return tensor

def all_gather(tensor):
    return [tensor]

def all_reduce(tensor, op=None):
    return tensor
