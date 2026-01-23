#!/usr/bin/env python3
"""
Prepare an ImageNet-64 dataset directory compatible with this repo's loader.

How the loader works (see `improved_diffusion/image_datasets.py`):
  - Reads all *.png/*.jpg/*.jpeg recursively from a directory.
  - If you later enable class-conditional training, it infers labels from the
    filename prefix before the first underscore (e.g. `n01440764_123.JPEG`).

This script does NOT provide ImageNet credentials or bypass licensing.
Instead, it helps you convert the *official* ImageNet downsampled releases
from the ImageNet website into a directory-of-images layout.

As of recent ImageNet site updates, "Download downsampled image data (64x64)"
is provided as **.npz** shards (train part1/part2, and val).
This script supports:
  - extracting legacy archives (.zip/.tar/.tar.gz/.tgz) into directories, and/or
  - converting ImageNet 64x64 **.npz** files into:
      <out_root>/train/
      <out_root>/val/

Usage examples:

  # From already-downloaded archives:
  python3 datasets/imagenet64.py \
    --train_archive /path/to/imagenet64_train.zip \
    --val_archive   /path/to/imagenet64_val.zip \
    --out_root imagenet64

  # If you have direct URLs (only works if your environment allows it):
  python3 datasets/imagenet64.py \
    --train_url https://.../Train_64x64.zip \
    --val_url   https://.../Val_64x64.zip \
    --out_root imagenet64

  # From the current official ImageNet downsampled 64x64 .npz downloads:
  python3 datasets/imagenet64.py \
    --train_npz_part1 /path/to/train_64x64_part1.npz \
    --train_npz_part2 /path/to/train_64x64_part2.npz \
    --val_npz         /path/to/val_64x64.npz \
    --out_root imagenet64

  # If you only have a single train archive (no official val archive),
  # create a val split directory from the extracted train images (using symlinks):
  python3 datasets/imagenet64.py \
    --train_archive /path/to/imagenet64_train.zip \
    --out_root imagenet64 \
    --make_val_from_train 0.01

Afterwards, point your SLURM scripts at:
  IMAGENET_TRAIN_DIR=<out_root>/train
  IMAGENET_VAL_DIR=<out_root>/val
"""

import argparse
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    print(f"Downloading {url} -> {out_path}")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def _extract(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(archive_path.suffixes).lower()

    print(f"Extracting {archive_path} -> {out_dir}")

    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out_dir)
        return

    if suffix.endswith(".tar") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
        mode = "r"
        if suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
            mode = "r:gz"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(out_dir)
        return

    raise ValueError(f"Unsupported archive type: {archive_path}")


def _maybe_flatten_single_topdir(out_dir: Path) -> None:
    """
    If extraction produced exactly one top-level directory, move its contents up.
    """
    entries = [p for p in out_dir.iterdir() if p.name not in [".DS_Store"]]
    if len(entries) == 1 and entries[0].is_dir():
        top = entries[0]
        print(f"Flattening single top-level directory: {top.name}")
        for child in top.iterdir():
            shutil.move(str(child), str(out_dir / child.name))
        top.rmdir()


def _count_images(root: Path, limit: int = 20000) -> Tuple[int, bool]:
    """
    Count up to `limit` images quickly, to sanity-check extraction.
    Returns (count, hit_limit).
    """
    n = 0
    hit = False
    for p in root.rglob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            n += 1
            if n >= limit:
                hit = True
                break
    return n, hit


def _npz_pick_arrays(npz: Dict[str, Any]):
    """
    Try to infer (images, labels) arrays from a variety of common ImageNet-downsampled npz formats.
    Returns (imgs, labels) where labels may be None.
    """
    import numpy as np

    # Common key candidates.
    img_keys = ["data", "images", "x", "arr_0"]
    lbl_keys = ["labels", "y", "targets", "arr_1"]

    imgs = None
    labels = None

    for k in img_keys:
        if k in npz:
            imgs = npz[k]
            break
    if imgs is None:
        # Fallback: pick the largest array-like value.
        arrays = [(k, v) for k, v in npz.items() if hasattr(v, "shape")]
        if arrays:
            arrays.sort(key=lambda kv: int(np.prod(kv[1].shape)), reverse=True)
            imgs = arrays[0][1]

    for k in lbl_keys:
        if k in npz:
            labels = npz[k]
            break

    if imgs is None:
        raise ValueError("Could not infer image array from npz file (no known keys like 'data').")

    return imgs, labels


def _npz_iter_images(imgs: Any, labels: Optional[Any], image_size: int = 64) -> Iterator[Tuple[int, Any, Optional[int]]]:
    """
    Yield (idx, img_hwc_uint8, label_int_or_None).
    Handles common shapes:
      - (N, 3, H, W)
      - (N, H, W, 3)
      - (N, H*W*3)
    """
    import numpy as np

    imgs = np.asarray(imgs)
    n = imgs.shape[0]

    for i in range(n):
        x = imgs[i]
        if x.ndim == 3:
            # (3,H,W) or (H,W,3)
            if x.shape[0] == 3 and x.shape[1] == image_size and x.shape[2] == image_size:
                x = np.transpose(x, (1, 2, 0))
            elif x.shape[2] == 3 and x.shape[0] == image_size and x.shape[1] == image_size:
                pass
            else:
                raise ValueError(f"Unexpected 3D image shape: {x.shape}")
        elif x.ndim == 1:
            # Flattened
            if x.shape[0] == image_size * image_size * 3:
                x = x.reshape(image_size, image_size, 3)
            else:
                raise ValueError(f"Unexpected 1D image length: {x.shape[0]}")
        else:
            raise ValueError(f"Unexpected image array ndim: {x.ndim}")

        if x.dtype != np.uint8:
            # Some formats store uint8 already; if not, try to convert safely.
            x = np.clip(x, 0, 255).astype(np.uint8)

        y = None  # type: Optional[int]
        if labels is not None:
            yv = labels[i]
            # labels may be scalar array type.
            try:
                y = int(yv)
            except Exception:
                y = None

        yield i, x, y


def _write_npz_as_images(
    npz_path: Path,
    out_dir: Path,
    split: str,
    start_index: int = 0,
    image_size: int = 64,
    shard_by_class: bool = True,
    max_images: int = 0,
) -> int:
    """
    Convert an ImageNet-64 npz shard to PNG files.
    Returns the next global index after writing.
    """
    import numpy as np
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading npz: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        imgs, labels = _npz_pick_arrays(dict(z))

    written = 0
    next_index = start_index
    for local_i, x_hwc, y in _npz_iter_images(imgs, labels, image_size=image_size):
        if max_images and written >= max_images:
            break
        if y is None:
            class_name = "class0000"
        else:
            class_name = f"class{y:04d}"

        if shard_by_class:
            cls_dir = out_dir / class_name
        else:
            cls_dir = out_dir
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Filename prefix enables future class_cond via `image_datasets.py`'s convention.
        fname = f"{class_name}_{next_index:08d}.png"
        img = Image.fromarray(x_hwc, mode="RGB")
        img.save(cls_dir / fname, format="PNG")

        written += 1
        next_index += 1

        if written % 50000 == 0:
            print(f"[{split}] wrote {written} images so far...")

    print(f"[{split}] wrote {written} images from {npz_path}")
    return next_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="imagenet64", help="Output root directory")

    ap.add_argument("--train_archive", default="", help="Local path to ImageNet-64 train archive (.zip/.tar/.tar.gz/.tgz)")
    ap.add_argument("--val_archive", default="", help="Local path to ImageNet-64 val archive (.zip/.tar/.tar.gz/.tgz)")
    ap.add_argument("--train_url", default="", help="Optional URL to download train archive")
    ap.add_argument("--val_url", default="", help="Optional URL to download val archive")
    ap.add_argument("--download_dir", default=".imagenet64_downloads", help="Where to save downloaded archives")

    # Current official downsampled ImageNet downloads use npz shards.
    ap.add_argument("--train_npz_part1", default="", help="Local path to ImageNet-64 train npz part1")
    ap.add_argument("--train_npz_part2", default="", help="Local path to ImageNet-64 train npz part2")
    ap.add_argument("--val_npz", default="", help="Local path to ImageNet-64 val npz")
    ap.add_argument(
        "--train_npz_dir_part1",
        default="",
        help="Directory containing ImageNet64 train shards for part1 (e.g. train_data_batch_1..5.npz).",
    )
    ap.add_argument(
        "--train_npz_dir_part2",
        default="",
        help="Directory containing ImageNet64 train shards for part2 (e.g. train_data_batch_6..10.npz).",
    )
    ap.add_argument(
        "--make_val_from_train",
        type=float,
        default=0.0,
        help="If >0 and no val archive is provided, create a val split from train with this fraction (e.g. 0.01).",
    )
    ap.add_argument(
        "--val_split_seed",
        type=int,
        default=0,
        help="Seed used when creating a val split from train.",
    )
    ap.add_argument(
        "--val_split_link",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize val split files (default: symlink).",
    )
    ap.add_argument(
        "--max_images_per_split",
        type=int,
        default=0,
        help="If >0, only write this many images per split (for quick smoke tests).",
    )
    ap.add_argument(
        "--no_shard_by_class",
        action="store_true",
        help="If set, write all images into a single directory instead of class subdirs.",
    )

    args = ap.parse_args()

    out_root = Path(args.out_root)
    train_out = out_root / "train"
    val_out = out_root / "val"

    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve archives (download if URL provided).
    train_archive = Path(args.train_archive) if args.train_archive else None
    val_archive = Path(args.val_archive) if args.val_archive else None

    if args.train_url:
        dl_dir = Path(args.download_dir)
        dl_dir.mkdir(parents=True, exist_ok=True)
        fname = os.path.basename(args.train_url.split("?")[0]) or "imagenet64_train"
        train_archive = dl_dir / fname
        _download(args.train_url, train_archive)

    if args.val_url:
        dl_dir = Path(args.download_dir)
        dl_dir.mkdir(parents=True, exist_ok=True)
        fname = os.path.basename(args.val_url.split("?")[0]) or "imagenet64_val"
        val_archive = dl_dir / fname
        _download(args.val_url, val_archive)

    if not train_archive:
        # If no archive, maybe we're using npz inputs.
        if not args.train_npz_part1 and not args.train_npz_part2:
            raise SystemExit(
                "You must provide a train source.\n"
                "Use --train_archive/--train_url OR --train_npz_part1/--train_npz_part2."
            )

    if train_archive and (not train_archive.exists()):
        raise SystemExit(f"Train archive not found: {train_archive}")
    if val_archive and (not val_archive.exists()):
        raise SystemExit(f"Val archive not found: {val_archive}")

    # Prefer NPZ conversion if provided; otherwise fall back to extracting archives.
    shard_by_class = not args.no_shard_by_class

    if args.train_npz_dir_part1 or args.train_npz_dir_part2 or args.train_npz_part1 or args.train_npz_part2 or args.val_npz:
        # Validate NPZ inputs.
        if not args.val_npz:
            raise SystemExit("For ImageNet-64 npz conversion, provide --val_npz.")

        # Determine train shard lists.
        train_files_p1 = []
        train_files_p2 = []

        if args.train_npz_dir_part1 or args.train_npz_dir_part2:
            if not args.train_npz_dir_part1 or not args.train_npz_dir_part2:
                raise SystemExit("Provide both --train_npz_dir_part1 and --train_npz_dir_part2.")
            d1 = Path(args.train_npz_dir_part1)
            d2 = Path(args.train_npz_dir_part2)
            if not d1.is_dir():
                raise SystemExit(f"train_npz_dir_part1 is not a directory: {d1}")
            if not d2.is_dir():
                raise SystemExit(f"train_npz_dir_part2 is not a directory: {d2}")
            train_files_p1 = sorted(d1.glob("train_data_batch_*.npz"))
            train_files_p2 = sorted(d2.glob("train_data_batch_*.npz"))
            if not train_files_p1:
                raise SystemExit(f"No train_data_batch_*.npz found in {d1}")
            if not train_files_p2:
                raise SystemExit(f"No train_data_batch_*.npz found in {d2}")
        else:
            if not args.train_npz_part1 or not args.train_npz_part2:
                raise SystemExit("Provide both --train_npz_part1 and --train_npz_part2, or use --train_npz_dir_part1/--train_npz_dir_part2.")
            train_files_p1 = [Path(args.train_npz_part1)]
            train_files_p2 = [Path(args.train_npz_part2)]

        val_p = Path(args.val_npz)
        for p in train_files_p1 + train_files_p2 + [val_p]:
            if not p.exists():
                raise SystemExit(f"NPZ file not found: {p}")

        if train_out.exists() and any(train_out.iterdir()):
            print(f"Skipping train npz conversion; directory not empty: {train_out}")
        else:
            idx0 = 0
            for p in train_files_p1:
                idx0 = _write_npz_as_images(
                    p,
                    train_out,
                    split="train",
                    start_index=idx0,
                    shard_by_class=shard_by_class,
                    max_images=args.max_images_per_split,
                )
            for p in train_files_p2:
                idx0 = _write_npz_as_images(
                    p,
                    train_out,
                    split="train",
                    start_index=idx0,
                    shard_by_class=shard_by_class,
                    max_images=args.max_images_per_split,
                )

        if val_out.exists() and any(val_out.iterdir()):
            print(f"Skipping val npz conversion; directory not empty: {val_out}")
        else:
            _write_npz_as_images(
                val_p,
                val_out,
                split="val",
                start_index=0,
                shard_by_class=shard_by_class,
                max_images=args.max_images_per_split,
            )
    else:
        # Extract if output dirs are empty/nonexistent.
        if train_out.exists() and any(train_out.iterdir()):
            print(f"Skipping train extraction; directory not empty: {train_out}")
        else:
            train_out.mkdir(parents=True, exist_ok=True)
            _extract(train_archive, train_out)
            _maybe_flatten_single_topdir(train_out)

    if val_archive:
        if val_out.exists() and any(val_out.iterdir()):
            print(f"Skipping val extraction; directory not empty: {val_out}")
        else:
            val_out.mkdir(parents=True, exist_ok=True)
            _extract(val_archive, val_out)
            _maybe_flatten_single_topdir(val_out)
    else:
        # Optional: make a val split directory from the extracted train files.
        if args.make_val_from_train <= 0:
            raise SystemExit(
                "No val archive provided.\n"
                "Either provide --val_archive/--val_url or set --make_val_from_train (e.g. 0.01)."
            )
        if val_out.exists() and any(val_out.iterdir()):
            print(f"Skipping val split creation; directory not empty: {val_out}")
        else:
            import random

            all_imgs = [p for p in train_out.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            if not all_imgs:
                raise SystemExit(f"No images found under extracted train dir: {train_out}")

            rnd = random.Random(args.val_split_seed)
            rnd.shuffle(all_imgs)
            k = max(1, int(len(all_imgs) * args.make_val_from_train))
            val_imgs = all_imgs[:k]

            val_out.mkdir(parents=True, exist_ok=True)
            print(
                f"Creating val split from train: {k}/{len(all_imgs)} images "
                f"({args.make_val_from_train:.4f}) via {args.val_split_link}"
            )

            for src in val_imgs:
                # Mirror relative path under train_out so classes/subdirs (if any) are preserved.
                rel = src.relative_to(train_out)
                dst = val_out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    continue
                if args.val_split_link == "symlink":
                    dst.symlink_to(src)
                elif args.val_split_link == "hardlink":
                    os.link(src, dst)
                else:
                    shutil.copy2(src, dst)

    train_n, train_hit = _count_images(train_out)
    val_n, val_hit = _count_images(val_out)

    print("==========================================")
    print("ImageNet-64 prep complete.")
    print(f"Train dir: {train_out} (found at least {train_n}{'+' if train_hit else ''} images)")
    print(f"Val dir:   {val_out} (found at least {val_n}{'+' if val_hit else ''} images)")
    if not val_archive:
        print("")
        print("WARNING: val was created as a split from train (not an official validation set).")
    print("")
    print("Use with SLURM scripts:")
    print(f"  export IMAGENET_TRAIN_DIR={train_out}")
    print(f"  export IMAGENET_VAL_DIR={val_out}")
    print("==========================================")


if __name__ == "__main__":
    main()

