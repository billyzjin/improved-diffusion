"""
Helpers for single-process (no MPI / no torch.distributed) training and eval.

This module exists so the original paper codepath can remain untouched, while
SLURM jobs can use an explicit no-MPI entrypoint.
"""

from __future__ import annotations

import torch


def setup_dist() -> None:
    """
    No-op "distributed" setup for single-process jobs.

    We still set the CUDA device to 0 if available so behavior is consistent
    across clusters that expose multiple GPUs to a job.
    """

    if torch.cuda.is_available():
        torch.cuda.set_device(0)


def dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def sync_params(_params) -> None:
    # No-op for single process.
    return None


def get_world_size() -> int:
    return 1


def get_rank() -> int:
    return 0


def is_main_process() -> bool:
    return True


def barrier() -> None:
    return None


def broadcast(tensor, src: int = 0):
    _ = src
    return tensor


def all_gather(tensor):
    return [tensor]


def all_reduce(tensor, op=None):
    _ = op
    return tensor


