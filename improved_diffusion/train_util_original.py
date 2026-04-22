import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
# import torch.distributed as dist  # Not needed for single GPU
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP  # Not needed for single GPU
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * 1  # Fixed: no distributed training
        self.num_accumulation_rounds = 1
        self.sync_cuda = th.cuda.is_available()

        self.use_ddp = False  # Disabled for single GPU
        self.ddp_model = self.model

        self.scaler = th.cuda.amp.GradScaler() if self.use_fp16 else None

        assert not (
            not th.cuda.is_available() and self.use_fp16
        ), "Cannot use fp16 without CUDA."

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed as well
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = False  # Disabled for single GPU
            self.ddp_model = self.model  # No DDP wrapper needed for single GPU

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if bf.exists(ema_checkpoint):
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return params

    def _master_params_to_state_dict(self, master_params):
        state_dict = {}
        for (name, _), value in zip(self.model.named_parameters(), master_params):
            state_dict[name] = value.clone()
        return state_dict

    def _master_params_to_model_params(self, master_params):
        for param, master_param in zip(self.model_params, master_params):
            param.detach().copy_(master_param)

    def _model_params_to_master_params(self, model_params, master_params):
        for param, master_param in zip(model_params, master_params):
            master_param.detach().copy_(param.detach())

    def _unscale_and_clip_grads(self, master_params):
        # Unscale the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.opt)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        th.nn.utils.clip_grad_norm_(master_params, 1.0)

    def _log_grad_norm(self, master_params):
        sqsum = 0.0
        for p in master_params:
            sqsum += (p.grad.data**2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _save(self, rate, params):
        state_dict = self._master_params_to_state_dict(params)
        if True:  # Always save on single GPU
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

    def _save_optimizer_state(self):
        if True:  # Always save on single GPU
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

    def _save_checkpoint(self):
        def is_checkpointing():
            return True  # Always checkpoint on single GPU

        if is_checkpointing():
            self._save(0, self.master_params)
            for rate, params in zip(self.ema_rate, self.ema_params):
                self._save(rate, params)
            self._save_optimizer_state()

    def _load_checkpoint(self):
        checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        logger.log(f"loading model from checkpoint: {checkpoint}...")
        self.resume_step = parse_resume_step_from_filename(checkpoint)

        if True:  # Always load on single GPU
            state_dict = dist_util.load_state_dict(checkpoint, map_location=dist_util.dev())
            self._state_dict_to_master_params(state_dict)

    def run_loop(self):
        if self.resume_step:
            self._load_checkpoint()

        while self.step + self.resume_step < self.lr_anneal_steps or self.lr_anneal_steps == 0:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self._save_checkpoint()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self._save_checkpoint()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self._unscale_and_clip_grads(self.master_params)
            self._log_grad_norm(self.master_params)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self._log_grad_norm(self.master_params)
            self.opt.step()
        self._update_ema()
        self._anneal_lr()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            if self.use_fp16:
                loss_scale = 2**self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate directory, such as "/tmp/experiments"
    # to avoid polluting the repository.
    return os.environ.get("OPENAI_LOGDIR", "logs")


def find_resume_checkpoint():
    # On your system, you might want to change this to always return the
    # latest checkpoint on your system.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (ie [0.1, 0.5, 0.9]) for each timestep.
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
