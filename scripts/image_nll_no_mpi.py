"""
Approximate the bits/dimension for an image model (no MPI / single process).

This mirrors `scripts/image_nll.py` but avoids importing/using torch.distributed.
"""

import argparse
import os

import numpy as np
import torch as th

from improved_diffusion import logger
from improved_diffusion.dist_util_no_mpi import setup_dist, dev, load_state_dict
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(load_state_dict(args.model_path, map_location="cpu"))
    model.to(dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    """
    Compute the bits per dimension of a model.
    """
    model.eval()
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    all_counts = []  # number of samples per batch
    num_complete = 0

    with th.no_grad():
        for batch, model_kwargs in data:
            if num_complete >= num_samples:
                break

            # Trim the last batch so we evaluate exactly num_samples
            remaining = num_samples - num_complete
            if batch.shape[0] > remaining:
                batch = batch[:remaining]
                model_kwargs = {k: v[:remaining] for k, v in model_kwargs.items()}

            batch = batch.to(dev())
            model_kwargs = {k: v.to(dev()) for k, v in model_kwargs.items()}

            minibatch_metrics = diffusion.calc_bpd_loop(
                model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
            )

            n = batch.shape[0]
            for key, term_list in all_metrics.items():
                terms = minibatch_metrics[key].sum(dim=0)  # sum over batch, not mean
                term_list.append(terms.detach().cpu().numpy())
            all_counts.append(n)

            total_bpd = minibatch_metrics["total_bpd"]
            # Fail fast on NaNs/Infs so downstream metrics (FID/TV) don't look "mysteriously bad".
            if not th.isfinite(total_bpd).all().item():
                raise RuntimeError(
                    "Non-finite total_bpd encountered during NLL evaluation. "
                    "This usually indicates a diverged checkpoint (NaNs/Infs in model outputs)."
                )
            all_bpd.extend(total_bpd.cpu().numpy())
            num_complete += n

            logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd):.6f}")

    # Save per-step means across all evaluated samples.
    total_count = sum(all_counts)
    for name, terms in all_metrics.items():
        out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
        logger.log(f"saving {name} terms to {out_path}")
        per_step_mean = np.sum(np.stack(terms, axis=0), axis=0) / total_count
        np.savez(out_path, per_step_mean)

    logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd):.6f}")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=1000,
        batch_size=1,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

