"""
Generate image samples from a model and save them as an .npz (no MPI / single process).

This mirrors `scripts/image_sample.py` but avoids importing/using torch.distributed.
"""

import argparse
import os

import numpy as np
import torch as th

from improved_diffusion import logger
from improved_diffusion.dist_util_no_mpi import setup_dist, dev, load_state_dict
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

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            if not 0 < args.num_classes:
                raise ValueError(f"num_classes must be positive, got {args.num_classes}")
            if args.class_label >= args.num_classes:
                raise ValueError(
                    f"class_label={args.class_label} is outside num_classes={args.num_classes}"
                )
            if args.class_label >= 0:
                classes = th.full(
                    (args.batch_size,), args.class_label, dtype=th.long, device=dev()
                )
            else:
                classes = th.randint(
                    low=0, high=args.num_classes, size=(args.batch_size,), device=dev()
                )
            model_kwargs["y"] = classes
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # Fail fast if sampling becomes non-finite (prevents silently saving garbage images).
        if not th.isfinite(sample).all().item():
            raise RuntimeError(
                "Non-finite values encountered during sampling. "
                "This typically indicates a diverged checkpoint (NaNs/Infs in model outputs)."
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        all_images.append(sample.cpu().numpy())
        if args.class_cond:
            all_labels.append(classes.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)[: args.num_samples]

    shape_str = "x".join(map(str, arr.shape))
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        class_label=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

