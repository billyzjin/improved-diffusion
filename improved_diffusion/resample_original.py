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
        ts_tensor = th.from_numpy(ts).to(device)
        return ts_tensor, th.ones_like(ts_tensor, dtype=th.float32)


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
            # Return uniform weights if no loss history
            weights = np.ones_like(self.loss_history)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            return weights

        # Calculate weights based on loss history
        weights = np.sqrt(np.mean(self.loss_history**2, axis=-1))

        # Apply exponential scaling
        weights *= 1 - np.exp(-self.loss_history / self.loss_history.mean())

        # Ensure minimum weight to avoid zero probabilities
        weights = np.maximum(weights, 1e-8)

        # Normalize to ensure probabilities sum to 1
        weights = weights / np.sum(weights)

        return weights

    def sample(self, batch_size, device):
        """
        Sample timesteps based on loss history.
        """
        weights = self.weights()
        ts = np.random.choice(
            self.diffusion.num_timesteps, batch_size, replace=True, p=weights
        ).astype(np.int64)
        ts_tensor = th.from_numpy(ts).to(device)
        return ts_tensor, th.ones_like(ts_tensor, dtype=th.float32)

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
