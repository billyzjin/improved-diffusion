#!/usr/bin/env python3
"""
Audit and rebuild the official downsampled ImageNet-64 dataset.

Nichol & Dhariwal's unconditional ImageNet-64 runs use the official
downsampled ImageNet-64 release, not center-cropped ILSVRC2012 images. This
script treats the official NPZ shards as the source of truth and verifies a
converted directory-of-PNGs against those shards.

Supported workflows:

  # Audit source shards only.
  python3 datasets/verify_imagenet64_official.py \
    --mode audit-sources \
    --train-npz-dir-part1 /path/to/Imagenet64_train_part1_npz \
    --train-npz-dir-part2 /path/to/Imagenet64_train_part2_npz \
    --val-npz /path/to/Imagenet64_val_npz/val_data.npz \
    --manifest-path /tmp/imagenet64_source_manifest.json

  # Rebuild a fresh verified tree, then audit and spot-check it.
  python3 datasets/verify_imagenet64_official.py \
    --mode all \
    --train-npz-dir-part1 /path/to/Imagenet64_train_part1_npz \
    --train-npz-dir-part2 /path/to/Imagenet64_train_part2_npz \
    --val-npz /path/to/Imagenet64_val_npz/val_data.npz \
    --out-root /project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505

The official flattened NPZ layout is interpreted as channel planes:

  flat.reshape(3, 64, 64).transpose(1, 2, 0)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_SIZE = 64
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
EXPECTED_TRAIN_COUNT = 1_281_167
EXPECTED_VAL_COUNT = 50_000
EXPECTED_TRAIN_SHARDS = tuple(range(1, 11))
VALID_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SourceShards:
    train: List[Path]
    val: Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and rebuild official downsampled ImageNet-64 NPZ shards."
    )
    parser.add_argument(
        "--mode",
        choices=["audit-sources", "convert", "audit-tree", "verify", "all"],
        default="all",
        help=(
            "audit-sources: inspect NPZ shards; convert: write PNG tree; "
            "audit-tree: inspect an image tree; verify: audit source/tree plus "
            "source-to-PNG spot checks; all: audit source, convert, then verify."
        ),
    )

    source = parser.add_argument_group("source shards")
    source.add_argument(
        "--source-root",
        default="",
        help="Directory to search recursively for train_data_batch_*.npz and val_data.npz.",
    )
    source.add_argument(
        "--train-npz-dir",
        default="",
        help="Directory containing all train_data_batch_*.npz shards.",
    )
    source.add_argument(
        "--train-npz-dir-part1",
        default="",
        help="Directory containing official train_data_batch_1..5.npz shards.",
    )
    source.add_argument(
        "--train-npz-dir-part2",
        default="",
        help="Directory containing official train_data_batch_6..10.npz shards.",
    )
    source.add_argument(
        "--train-npz-files",
        nargs="*",
        default=[],
        help="Explicit train_data_batch_*.npz paths.",
    )
    source.add_argument(
        "--val-npz",
        default="",
        help="Path to official val_data.npz.",
    )
    source.add_argument(
        "--skip-source-sha256",
        action="store_true",
        help="Skip full SHA256 checksums of NPZ files for faster smoke runs.",
    )

    output = parser.add_argument_group("converted tree")
    output.add_argument(
        "--out-root",
        default="",
        help="Fresh output root for conversion. Creates train/ and val/ under this directory.",
    )
    output.add_argument(
        "--tree-root",
        default="",
        help="Existing converted tree root to audit. Defaults to --out-root.",
    )
    output.add_argument(
        "--manifest-path",
        default="",
        help="Where to write the JSON manifest. Defaults to <tree-or-out-root>/verification/manifest.json.",
    )
    output.add_argument(
        "--sample-dir",
        default="",
        help="Where to write sample grids. Defaults to <tree-or-out-root>/verification/samples.",
    )
    output.add_argument(
        "--max-images-per-split",
        type=positive_int,
        default=0,
        help="If >0, convert/audit this many images per split for a smoke test.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files during conversion.",
    )
    output.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-existing PNG files during conversion.",
    )
    output.add_argument(
        "--png-compress-level",
        type=int,
        default=0,
        choices=range(0, 10),
        metavar="[0-9]",
        help="PNG compression level for conversion. 0 is fastest and avoids partial-write ambiguity.",
    )

    verify = parser.add_argument_group("verification")
    verify.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker processes for converted-tree image audit.",
    )
    verify.add_argument(
        "--chunksize",
        type=int,
        default=256,
        help="Multiprocessing chunksize for image audit.",
    )
    verify.add_argument(
        "--seed",
        type=int,
        default=20260505,
        help="Seed for deterministic samples and spot checks.",
    )
    verify.add_argument(
        "--sample-count",
        type=int,
        default=64,
        help="Number of tree images to hash and place in each sample grid.",
    )
    verify.add_argument(
        "--spot-check-count",
        type=int,
        default=128,
        help="Number of deterministic source-to-PNG exact spot checks per split.",
    )
    verify.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail converted-tree count checks when running a partial smoke test.",
    )
    verify.add_argument(
        "--progress-interval",
        type=int,
        default=50_000,
        help="Print image-audit progress every N files.",
    )

    return parser.parse_args()


def batch_number(path: Path) -> int:
    match = re.search(r"train_data_batch_(\d+)$", path.stem)
    if not match:
        return 10**9
    return int(match.group(1))


def sorted_train_shards(paths: Iterable[Path]) -> List[Path]:
    return sorted((p for p in paths if p.name.endswith(".npz")), key=batch_number)


def resolve_source_shards(args: argparse.Namespace) -> SourceShards:
    train_files: List[Path] = []

    if args.train_npz_files:
        train_files = [Path(p).expanduser() for p in args.train_npz_files]
    elif args.train_npz_dir_part1 or args.train_npz_dir_part2:
        if not args.train_npz_dir_part1 or not args.train_npz_dir_part2:
            raise SystemExit("Provide both --train-npz-dir-part1 and --train-npz-dir-part2.")
        d1 = Path(args.train_npz_dir_part1).expanduser()
        d2 = Path(args.train_npz_dir_part2).expanduser()
        train_files = sorted_train_shards(d1.glob("train_data_batch_*.npz"))
        train_files += sorted_train_shards(d2.glob("train_data_batch_*.npz"))
    elif args.train_npz_dir:
        d = Path(args.train_npz_dir).expanduser()
        train_files = sorted_train_shards(d.glob("train_data_batch_*.npz"))
    elif args.source_root:
        root = Path(args.source_root).expanduser()
        train_files = sorted_train_shards(root.rglob("train_data_batch_*.npz"))

    if args.val_npz:
        val_file = Path(args.val_npz).expanduser()
    elif args.source_root:
        val_candidates = sorted(Path(args.source_root).expanduser().rglob("val_data.npz"))
        if not val_candidates:
            raise SystemExit(f"No val_data.npz found under --source-root={args.source_root}")
        val_file = val_candidates[0]
    else:
        raise SystemExit("Provide --val-npz or --source-root.")

    if not train_files:
        raise SystemExit(
            "No train_data_batch_*.npz shards found. Provide --train-npz-dir*, "
            "--train-npz-files, or --source-root."
        )

    missing = [p for p in train_files + [val_file] if not p.is_file()]
    if missing:
        raise SystemExit("Missing source shard(s):\n" + "\n".join(f"  {p}" for p in missing))

    return SourceShards(train=train_files, val=val_file)


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_npz_arrays(arrays: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray, Optional[str], Optional[np.ndarray]]:
    image_keys = ("data", "images", "x", "arr_0")
    label_keys = ("labels", "y", "targets", "arr_1")

    image_key = ""
    for key in image_keys:
        if key in arrays:
            image_key = key
            break
    if not image_key:
        candidates = [(key, value) for key, value in arrays.items() if hasattr(value, "shape")]
        if not candidates:
            raise ValueError("NPZ contains no array-like entries.")
        candidates.sort(key=lambda kv: int(np.prod(kv[1].shape)), reverse=True)
        image_key = candidates[0][0]

    label_key: Optional[str] = None
    for key in label_keys:
        if key in arrays and key != image_key:
            label_key = key
            break

    labels = arrays[label_key] if label_key is not None else None
    return image_key, arrays[image_key], label_key, labels


def image_layout(images: np.ndarray) -> str:
    if images.ndim == 2 and images.shape[1] == IMAGE_SIZE * IMAGE_SIZE * 3:
        return "flat_chw_channel_planes"
    if images.ndim == 4 and images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE):
        return "nchw"
    if images.ndim == 4 and images.shape[1:] == IMAGE_SHAPE:
        return "nhwc"
    return "unknown"


def to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 1:
        if x.shape[0] != IMAGE_SIZE * IMAGE_SIZE * 3:
            raise ValueError(f"Unexpected flat image length: {x.shape[0]}")
        x = x.reshape(3, IMAGE_SIZE, IMAGE_SIZE).transpose(1, 2, 0)
    elif x.ndim == 3:
        if x.shape == (3, IMAGE_SIZE, IMAGE_SIZE):
            x = x.transpose(1, 2, 0)
        elif x.shape == IMAGE_SHAPE:
            pass
        else:
            raise ValueError(f"Unexpected 3D image shape: {x.shape}")
    else:
        raise ValueError(f"Unexpected image ndim: {x.ndim}")

    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)


def label_to_int(labels: Optional[np.ndarray], index: int) -> Optional[int]:
    if labels is None:
        return None
    try:
        return int(labels[index])
    except Exception:
        return None


def class_name(label: Optional[int]) -> str:
    if label is None:
        return "class0000"
    return f"class{label:04d}"


def expected_png_path(split_root: Path, label: Optional[int], global_index: int) -> Path:
    cls = class_name(label)
    return split_root / cls / f"{cls}_{global_index:08d}.png"


def npz_array_manifest(path: Path, compute_sha256: bool) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
    }
    if compute_sha256:
        info["sha256"] = sha256_file(path)

    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    info["array_keys"] = list(arrays.keys())
    info["arrays"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }

    image_key, images, label_key, labels = pick_npz_arrays(arrays)
    layout = image_layout(images)
    count = int(images.shape[0])
    info.update(
        {
            "image_key": image_key,
            "image_count": count,
            "image_dtype": str(images.dtype),
            "image_shape": list(images.shape),
            "image_layout": layout,
            "label_key": label_key,
        }
    )

    if labels is not None:
        flat_labels = np.asarray(labels).reshape(-1)
        info["label_shape"] = list(labels.shape)
        info["label_dtype"] = str(labels.dtype)
        info["label_count"] = int(flat_labels.shape[0])
        info["label_min"] = int(flat_labels.min()) if flat_labels.size else None
        info["label_max"] = int(flat_labels.max()) if flat_labels.size else None
        unique, counts = np.unique(flat_labels.astype(np.int64), return_counts=True)
        info["label_unique_count"] = int(unique.shape[0])
        info["label_counts"] = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    return info


def official_identity_checks(source: SourceShards, source_manifest: Dict[str, Any]) -> Dict[str, Any]:
    train_batches = [batch_number(path) for path in source.train]
    train_count = int(source_manifest["train"]["total_images"])
    val_count = int(source_manifest["val"]["total_images"])
    train_layouts = sorted(
        set(shard.get("image_layout", "unknown") for shard in source_manifest["train"]["shards"])
    )
    val_layouts = sorted(
        set(shard.get("image_layout", "unknown") for shard in source_manifest["val"]["shards"])
    )

    return {
        "source_format_inferred": (
            "official_downsampled_imagenet64_npz"
            if train_batches == list(EXPECTED_TRAIN_SHARDS)
            and source.val.name == "val_data.npz"
            and train_count == EXPECTED_TRAIN_COUNT
            and val_count == EXPECTED_VAL_COUNT
            and "flat_chw_channel_planes" in train_layouts
            else "unconfirmed"
        ),
        "train_batch_numbers": train_batches,
        "expected_train_batch_numbers": list(EXPECTED_TRAIN_SHARDS),
        "train_shard_names_ok": train_batches == list(EXPECTED_TRAIN_SHARDS),
        "val_name_ok": source.val.name == "val_data.npz",
        "train_count_ok": train_count == EXPECTED_TRAIN_COUNT,
        "val_count_ok": val_count == EXPECTED_VAL_COUNT,
        "train_layouts": train_layouts,
        "val_layouts": val_layouts,
        "official_flat_chw_layout_seen": "flat_chw_channel_planes" in train_layouts,
    }


def audit_sources(source: SourceShards, compute_sha256: bool) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    train_shards: List[Dict[str, Any]] = []
    train_label_counts: collections.Counter[int] = collections.Counter()
    train_total = 0

    print(f"Auditing {len(source.train)} train NPZ shard(s)...", flush=True)
    for path in source.train:
        print(f"  source train shard: {path}", flush=True)
        shard = npz_array_manifest(path, compute_sha256=compute_sha256)
        train_shards.append(shard)
        train_total += int(shard["image_count"])
        if shard["image_dtype"] != "uint8":
            errors.append(f"{path}: image dtype is {shard['image_dtype']}, expected uint8")
        if shard["image_layout"] == "unknown":
            errors.append(f"{path}: unrecognized image shape {shard['image_shape']}")
        if shard.get("label_count") is not None and shard["label_count"] != shard["image_count"]:
            errors.append(f"{path}: label count does not match image count")
        if "label_counts" in shard:
            for key, value in shard["label_counts"].items():
                train_label_counts[int(key)] += int(value)

    print(f"Auditing val NPZ shard: {source.val}", flush=True)
    val_shard = npz_array_manifest(source.val, compute_sha256=compute_sha256)
    val_total = int(val_shard["image_count"])
    if val_shard["image_dtype"] != "uint8":
        errors.append(f"{source.val}: image dtype is {val_shard['image_dtype']}, expected uint8")
    if val_shard["image_layout"] == "unknown":
        errors.append(f"{source.val}: unrecognized image shape {val_shard['image_shape']}")
    if val_shard.get("label_count") is not None and val_shard["label_count"] != val_shard["image_count"]:
        errors.append(f"{source.val}: label count does not match image count")

    manifest = {
        "train": {
            "expected_images": EXPECTED_TRAIN_COUNT,
            "total_images": train_total,
            "shards": train_shards,
            "class_label_counts": {
                str(key): int(train_label_counts[key]) for key in sorted(train_label_counts)
            },
        },
        "val": {
            "expected_images": EXPECTED_VAL_COUNT,
            "total_images": val_total,
            "shards": [val_shard],
        },
    }
    manifest["identity"] = official_identity_checks(source, manifest)

    if train_total != EXPECTED_TRAIN_COUNT:
        errors.append(f"source train count {train_total} != expected {EXPECTED_TRAIN_COUNT}")
    if val_total != EXPECTED_VAL_COUNT:
        errors.append(f"source val count {val_total} != expected {EXPECTED_VAL_COUNT}")
    if not manifest["identity"]["train_shard_names_ok"]:
        errors.append(
            "train NPZ shard names are not exactly train_data_batch_1.npz through train_data_batch_10.npz"
        )
    if not manifest["identity"]["val_name_ok"]:
        errors.append("val NPZ shard is not named val_data.npz")
    if manifest["identity"]["source_format_inferred"] != "official_downsampled_imagenet64_npz":
        errors.append("source identity could not be confirmed as official downsampled ImageNet-64 NPZ")

    return manifest, errors


def iter_npz_images(
    path: Path,
    start_global_index: int,
    max_images: int = 0,
) -> Iterator[Tuple[int, np.ndarray, Optional[int]]]:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    _, images, _, labels = pick_npz_arrays(arrays)
    count = int(images.shape[0])
    limit = count if max_images <= 0 else min(count, max_images)
    for local_index in range(limit):
        global_index = start_global_index + local_index
        yield global_index, to_hwc_uint8(images[local_index]), label_to_int(labels, local_index)


def split_source_counts(source: SourceShards) -> Dict[str, List[int]]:
    counts = {"train": [], "val": []}
    for path in source.train:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        _, images, _, _ = pick_npz_arrays(arrays)
        counts["train"].append(int(images.shape[0]))
    with np.load(source.val, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    _, images, _, _ = pick_npz_arrays(arrays)
    counts["val"].append(int(images.shape[0]))
    return counts


def ensure_conversion_target(split_root: Path, overwrite: bool, resume: bool) -> None:
    if not split_root.exists():
        split_root.mkdir(parents=True, exist_ok=True)
        return
    has_entries = any(split_root.iterdir())
    if has_entries and not overwrite and not resume:
        raise SystemExit(
            f"Refusing to write into non-empty split directory without --overwrite or --resume: {split_root}"
        )
    split_root.mkdir(parents=True, exist_ok=True)


def save_png_atomic(array: np.ndarray, destination: Path, compress_level: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(array, mode="RGB").save(
            tmp,
            format="PNG",
            compress_level=compress_level,
        )
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def convert_source_to_tree(
    source: SourceShards,
    out_root: Path,
    max_images_per_split: int,
    overwrite: bool,
    resume: bool,
    compress_level: int,
) -> Dict[str, Any]:
    train_root = out_root / "train"
    val_root = out_root / "val"
    ensure_conversion_target(train_root, overwrite=overwrite, resume=resume)
    ensure_conversion_target(val_root, overwrite=overwrite, resume=resume)

    summary: Dict[str, Any] = {
        "out_root": str(out_root),
        "png_compress_level": compress_level,
        "max_images_per_split": max_images_per_split,
        "splits": {},
    }

    for split, shard_paths, split_root in (
        ("train", source.train, train_root),
        ("val", [source.val], val_root),
    ):
        split_start = time.time()
        print(f"Converting {split} -> {split_root}", flush=True)
        written = 0
        skipped = 0
        global_index = 0
        stop_after = max_images_per_split if max_images_per_split > 0 else None

        for shard_path in shard_paths:
            remaining = 0 if stop_after is None else stop_after - written - skipped
            if stop_after is not None and remaining <= 0:
                break
            with np.load(shard_path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            _, images, _, labels = pick_npz_arrays(arrays)
            shard_count = int(images.shape[0])
            shard_limit = shard_count if stop_after is None else min(shard_count, remaining)

            for local_index in range(shard_limit):
                idx = global_index + local_index
                array = to_hwc_uint8(images[local_index])
                label = label_to_int(labels, local_index)
                dst = expected_png_path(split_root, label, idx)
                if dst.exists() and resume and not overwrite:
                    skipped += 1
                else:
                    save_png_atomic(array, dst, compress_level=compress_level)
                    written += 1
                total = written + skipped
                if total % 50_000 == 0:
                    print(f"  [{split}] materialized {total} images", flush=True)

            global_index += shard_count

        summary["splits"][split] = {
            "root": str(split_root),
            "written": written,
            "skipped_existing": skipped,
            "elapsed_seconds": round(time.time() - split_start, 3),
        }
        print(
            f"Finished {split}: wrote={written} skipped_existing={skipped} "
            f"elapsed={summary['splits'][split]['elapsed_seconds']}s",
            flush=True,
        )

    return summary


def is_three_by_three_tiled(array: np.ndarray) -> bool:
    # Detect exact or near-exact 3x3 tiling over the largest common 63x63 area.
    crop = array[:63, :63, :]
    base = crop[:21, :21, :].astype(np.int16)
    for row in range(3):
        for col in range(3):
            tile = crop[row * 21 : (row + 1) * 21, col * 21 : (col + 1) * 21, :].astype(np.int16)
            if np.mean(np.abs(base - tile)) > 1.0:
                return False
    return True


def audit_image_file(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    result: Dict[str, Any] = {
        "path": path_str,
        "count": 1,
        "zero_byte": 0,
        "unreadable": 0,
        "wrong_shape": 0,
        "wrong_mode": 0,
        "grayscale_equal_channels": 0,
        "tiled_3x3_like": 0,
        "mode": None,
        "size": None,
        "class_label": None,
        "sum": [0, 0, 0],
        "sumsq": [0, 0, 0],
        "pixels": 0,
        "error": "",
    }

    match = re.search(r"class(\d{4})", str(path.parent))
    if match:
        result["class_label"] = int(match.group(1))

    try:
        if path.stat().st_size == 0:
            result["zero_byte"] = 1
            result["error"] = "zero-byte file"
            return result
    except OSError as exc:
        result["unreadable"] = 1
        result["error"] = f"stat failed: {exc}"
        return result

    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            result["mode"] = mode
            result["size"] = f"{size[0]}x{size[1]}"
            if mode != "RGB":
                result["wrong_mode"] = 1
            if size != (IMAGE_SIZE, IMAGE_SIZE):
                result["wrong_shape"] = 1
            arr = np.asarray(image.convert("RGB"))
    except Exception as exc:
        result["unreadable"] = 1
        result["error"] = f"open/read failed: {exc}"
        return result

    if arr.shape != IMAGE_SHAPE:
        result["wrong_shape"] = 1
        result["error"] = f"decoded shape {arr.shape}, expected {IMAGE_SHAPE}"
        return result

    result["pixels"] = IMAGE_SIZE * IMAGE_SIZE
    sums = arr.sum(axis=(0, 1), dtype=np.uint64)
    sumsq = np.square(arr.astype(np.uint64)).sum(axis=(0, 1), dtype=np.uint64)
    result["sum"] = [int(x) for x in sums]
    result["sumsq"] = [int(x) for x in sumsq]
    if np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2]):
        result["grayscale_equal_channels"] = 1
    if is_three_by_three_tiled(arr):
        result["tiled_3x3_like"] = 1
    return result


def iter_image_paths(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def reservoir_update(sample: List[Path], item: Path, seen: int, limit: int, rng: random.Random) -> None:
    if limit <= 0:
        return
    if len(sample) < limit:
        sample.append(item)
    else:
        j = rng.randint(0, seen - 1)
        if j < limit:
            sample[j] = item


def merge_counter(counter: collections.Counter, key: Any, amount: int = 1) -> None:
    if key is not None:
        counter[key] += amount


def audit_tree_split(
    split_root: Path,
    expected_count: int,
    workers: int,
    chunksize: int,
    sample_count: int,
    seed: int,
    progress_interval: int,
    allow_partial: bool,
) -> Tuple[Dict[str, Any], List[str], List[Path]]:
    if not split_root.is_dir():
        return {"root": str(split_root), "exists": False}, [f"missing split directory: {split_root}"], []

    rng = random.Random(seed)
    sample_paths: List[Path] = []
    files_seen = 0

    counters = collections.Counter()
    modes = collections.Counter()
    sizes = collections.Counter()
    class_counts = collections.Counter()
    sum_channels = np.zeros(3, dtype=np.float64)
    sumsq_channels = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    bad_examples: List[Dict[str, Any]] = []

    def path_strings() -> Iterator[str]:
        nonlocal files_seen
        for path in iter_image_paths(split_root):
            files_seen += 1
            reservoir_update(sample_paths, path, files_seen, sample_count, rng)
            yield str(path)

    iterator: Iterable[Dict[str, Any]]
    if workers <= 1:
        iterator = map(audit_image_file, path_strings())
    else:
        pool = mp.Pool(processes=workers)
        iterator = pool.imap_unordered(audit_image_file, path_strings(), chunksize=chunksize)

    start = time.time()
    processed = 0
    try:
        for result in iterator:
            processed += 1
            counters["files"] += 1
            counters["zero_byte"] += int(result["zero_byte"])
            counters["unreadable"] += int(result["unreadable"])
            counters["wrong_shape"] += int(result["wrong_shape"])
            counters["wrong_mode"] += int(result["wrong_mode"])
            counters["grayscale_equal_channels"] += int(result["grayscale_equal_channels"])
            counters["tiled_3x3_like"] += int(result["tiled_3x3_like"])
            merge_counter(modes, result["mode"])
            merge_counter(sizes, result["size"])
            merge_counter(class_counts, result["class_label"])
            if result["pixels"]:
                total_pixels += int(result["pixels"])
                sum_channels += np.asarray(result["sum"], dtype=np.float64)
                sumsq_channels += np.asarray(result["sumsq"], dtype=np.float64)
            if result["error"] and len(bad_examples) < 50:
                bad_examples.append({"path": result["path"], "error": result["error"]})
            if progress_interval and processed % progress_interval == 0:
                print(
                    f"  audited {processed} images under {split_root} "
                    f"({time.time() - start:.1f}s)",
                    flush=True,
                )
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    errors: List[str] = []
    if counters["files"] != expected_count and not allow_partial:
        errors.append(f"{split_root}: found {counters['files']} images, expected {expected_count}")
    if counters["zero_byte"]:
        errors.append(f"{split_root}: found {counters['zero_byte']} zero-byte files")
    if counters["unreadable"]:
        errors.append(f"{split_root}: found {counters['unreadable']} unreadable files")
    if counters["wrong_shape"]:
        errors.append(f"{split_root}: found {counters['wrong_shape']} images not {IMAGE_SIZE}x{IMAGE_SIZE}")
    if counters["wrong_mode"]:
        errors.append(f"{split_root}: found {counters['wrong_mode']} files whose original mode is not RGB")
    if counters["tiled_3x3_like"]:
        errors.append(f"{split_root}: found {counters['tiled_3x3_like']} exact/near 3x3 tiled images")

    if total_pixels:
        means = sum_channels / total_pixels
        variances = np.maximum(sumsq_channels / total_pixels - means**2, 0.0)
        stds = np.sqrt(variances)
    else:
        means = np.zeros(3, dtype=np.float64)
        stds = np.zeros(3, dtype=np.float64)

    summary = {
        "root": str(split_root),
        "exists": True,
        "expected_images": expected_count,
        "files": int(counters["files"]),
        "zero_byte": int(counters["zero_byte"]),
        "unreadable": int(counters["unreadable"]),
        "wrong_shape": int(counters["wrong_shape"]),
        "wrong_mode": int(counters["wrong_mode"]),
        "grayscale_equal_channels": int(counters["grayscale_equal_channels"]),
        "tiled_3x3_like": int(counters["tiled_3x3_like"]),
        "modes": {str(k): int(v) for k, v in sorted(modes.items(), key=lambda kv: str(kv[0]))},
        "sizes": {str(k): int(v) for k, v in sorted(sizes.items(), key=lambda kv: str(kv[0]))},
        "class_counts": {str(k): int(class_counts[k]) for k in sorted(class_counts)},
        "channel_mean_rgb": [round(float(x), 6) for x in means],
        "channel_std_rgb": [round(float(x), 6) for x in stds],
        "bad_examples": bad_examples,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    return summary, errors, sample_paths


def hash_sample_files(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(paths):
        row: Dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        try:
            row["sha256"] = sha256_file(path)
        except Exception as exc:
            row["sha256_error"] = str(exc)
        rows.append(row)
    return rows


def make_grid(arrays: Sequence[np.ndarray], destination: Path, columns: int = 8) -> Optional[str]:
    if not arrays:
        return None
    columns = max(1, min(columns, len(arrays)))
    rows = int(math.ceil(len(arrays) / columns))
    canvas = Image.new("RGB", (columns * IMAGE_SIZE, rows * IMAGE_SIZE), (0, 0, 0))
    for index, array in enumerate(arrays):
        img = Image.fromarray(to_hwc_uint8(array), mode="RGB")
        x = (index % columns) * IMAGE_SIZE
        y = (index // columns) * IMAGE_SIZE
        canvas.paste(img, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
    return str(destination)


def load_tree_sample_arrays(paths: Sequence[Path], limit: int) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for path in sorted(paths)[:limit]:
        try:
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert("RGB")))
        except Exception:
            continue
    return arrays


def audit_tree(
    tree_root: Path,
    workers: int,
    chunksize: int,
    sample_count: int,
    seed: int,
    progress_interval: int,
    allow_partial: bool,
    max_images_per_split: int,
    sample_dir: Optional[Path],
) -> Tuple[Dict[str, Any], List[str]]:
    expected_train = min(max_images_per_split, EXPECTED_TRAIN_COUNT) if max_images_per_split else EXPECTED_TRAIN_COUNT
    expected_val = min(max_images_per_split, EXPECTED_VAL_COUNT) if max_images_per_split else EXPECTED_VAL_COUNT

    manifest = {"root": str(tree_root), "splits": {}}
    all_errors: List[str] = []
    for split, expected, split_seed in (
        ("train", expected_train, seed),
        ("val", expected_val, seed + 1),
    ):
        print(f"Auditing converted {split} tree...", flush=True)
        split_manifest, split_errors, sample_paths = audit_tree_split(
            tree_root / split,
            expected_count=expected,
            workers=workers,
            chunksize=chunksize,
            sample_count=sample_count,
            seed=split_seed,
            progress_interval=progress_interval,
            allow_partial=allow_partial,
        )
        split_manifest["sample_hashes"] = hash_sample_files(sample_paths)
        if sample_dir is not None:
            arrays = load_tree_sample_arrays(sample_paths, sample_count)
            grid_path = sample_dir / f"{split}_tree_sample_grid.png"
            split_manifest["sample_grid"] = make_grid(arrays, grid_path)
        manifest["splits"][split] = split_manifest
        all_errors.extend(split_errors)
    return manifest, all_errors


def choose_spot_indices(total: int, count: int, seed: int) -> List[int]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    n = min(count, total)
    return sorted(rng.sample(range(total), n))


def load_source_indices(
    shard_paths: Sequence[Path],
    target_indices: Sequence[int],
) -> Dict[int, Tuple[np.ndarray, Optional[int]]]:
    targets = sorted(set(target_indices))
    if not targets:
        return {}

    out: Dict[int, Tuple[np.ndarray, Optional[int]]] = {}
    target_pos = 0
    global_offset = 0
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        _, images, _, labels = pick_npz_arrays(arrays)
        shard_count = int(images.shape[0])
        shard_start = global_offset
        shard_end = global_offset + shard_count

        while target_pos < len(targets) and targets[target_pos] < shard_start:
            target_pos += 1
        while target_pos < len(targets) and shard_start <= targets[target_pos] < shard_end:
            global_index = targets[target_pos]
            local_index = global_index - shard_start
            out[global_index] = (
                to_hwc_uint8(images[local_index]),
                label_to_int(labels, local_index),
            )
            target_pos += 1
        global_offset = shard_end
        if target_pos >= len(targets):
            break
    return out


def spot_check_split(
    split: str,
    shard_paths: Sequence[Path],
    split_root: Path,
    expected_count: int,
    count: int,
    seed: int,
    sample_dir: Optional[Path],
) -> Tuple[Dict[str, Any], List[str]]:
    indices = choose_spot_indices(expected_count, count, seed)
    source_items = load_source_indices(shard_paths, indices)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    source_arrays: List[np.ndarray] = []
    tree_arrays: List[np.ndarray] = []

    for global_index in indices:
        if global_index not in source_items:
            errors.append(f"{split}: source index {global_index} could not be loaded")
            continue
        source_array, label = source_items[global_index]
        path = expected_png_path(split_root, label, global_index)
        row: Dict[str, Any] = {
            "index": global_index,
            "label": label,
            "path": str(path),
            "source_pixel_sha256": hashlib.sha256(source_array.tobytes()).hexdigest(),
            "exists": path.is_file(),
            "exact_match": False,
        }
        source_arrays.append(source_array)
        if not path.is_file():
            errors.append(f"{split}: missing expected PNG for source index {global_index}: {path}")
            rows.append(row)
            continue
        try:
            with Image.open(path) as image:
                tree_array = np.asarray(image.convert("RGB"))
            row["png_pixel_sha256"] = hashlib.sha256(tree_array.tobytes()).hexdigest()
            row["exact_match"] = bool(np.array_equal(source_array, tree_array))
            tree_arrays.append(tree_array)
        except Exception as exc:
            row["read_error"] = str(exc)
            errors.append(f"{split}: failed to read expected PNG {path}: {exc}")
        if not row["exact_match"]:
            errors.append(f"{split}: PNG does not exactly match source index {global_index}: {path}")
        rows.append(row)

    manifest: Dict[str, Any] = {
        "split": split,
        "requested": count,
        "checked": len(rows),
        "exact_matches": sum(1 for row in rows if row.get("exact_match")),
        "rows": rows,
    }
    if sample_dir is not None:
        manifest["source_grid"] = make_grid(
            source_arrays[: min(len(source_arrays), 64)],
            sample_dir / f"{split}_source_spot_grid.png",
        )
        manifest["tree_grid"] = make_grid(
            tree_arrays[: min(len(tree_arrays), 64)],
            sample_dir / f"{split}_tree_spot_grid.png",
        )
    return manifest, errors


def spot_check_tree_against_sources(
    source: SourceShards,
    tree_root: Path,
    count: int,
    seed: int,
    max_images_per_split: int,
    sample_dir: Optional[Path],
) -> Tuple[Dict[str, Any], List[str]]:
    train_expected = min(max_images_per_split, EXPECTED_TRAIN_COUNT) if max_images_per_split else EXPECTED_TRAIN_COUNT
    val_expected = min(max_images_per_split, EXPECTED_VAL_COUNT) if max_images_per_split else EXPECTED_VAL_COUNT
    manifest = {"splits": {}}
    errors: List[str] = []

    for split, paths, expected, split_seed in (
        ("train", source.train, train_expected, seed + 100),
        ("val", [source.val], val_expected, seed + 200),
    ):
        print(f"Spot-checking {split} PNGs against source arrays...", flush=True)
        split_manifest, split_errors = spot_check_split(
            split=split,
            shard_paths=paths,
            split_root=tree_root / split,
            expected_count=expected,
            count=count,
            seed=split_seed,
            sample_dir=sample_dir,
        )
        manifest["splits"][split] = split_manifest
        errors.extend(split_errors)
    return manifest, errors


def default_manifest_path(args: argparse.Namespace, tree_root: Optional[Path]) -> Path:
    if args.manifest_path:
        return Path(args.manifest_path).expanduser()
    if tree_root is not None:
        return tree_root / "verification" / "manifest.json"
    return Path("imagenet64_official_manifest.json")


def default_sample_dir(args: argparse.Namespace, tree_root: Optional[Path]) -> Optional[Path]:
    if args.sample_dir:
        return Path(args.sample_dir).expanduser()
    if tree_root is not None:
        return tree_root / "verification" / "samples"
    return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    started = time.time()
    source_required = args.mode in {"audit-sources", "convert", "verify", "all"}
    tree_required = args.mode in {"audit-tree", "verify"}
    source: Optional[SourceShards] = resolve_source_shards(args) if source_required else None

    out_root = Path(args.out_root).expanduser() if args.out_root else None
    tree_root = Path(args.tree_root).expanduser() if args.tree_root else out_root
    if args.mode in {"convert", "all"} and out_root is None:
        raise SystemExit("--out-root is required for --mode convert/all")
    if tree_required and tree_root is None:
        raise SystemExit("--tree-root or --out-root is required for --mode audit-tree/verify")

    allow_partial = bool(args.allow_partial or args.max_images_per_split > 0)
    sample_dir = default_sample_dir(args, tree_root)
    manifest_path = default_manifest_path(args, tree_root)
    manifest: Dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "mode": args.mode,
        "created_unix_time": time.time(),
        "configuration": {
            "image_size": IMAGE_SIZE,
            "expected_train_images": EXPECTED_TRAIN_COUNT,
            "expected_val_images": EXPECTED_VAL_COUNT,
            "max_images_per_split": args.max_images_per_split,
            "allow_partial": allow_partial,
            "workers": args.workers,
            "seed": args.seed,
            "sample_count": args.sample_count,
            "spot_check_count": args.spot_check_count,
            "source_sha256": not args.skip_source_sha256,
        },
        "errors": [],
    }

    if source is not None:
        manifest["source_paths"] = {
            "train": [str(path) for path in source.train],
            "val": str(source.val),
        }

    errors: List[str] = []

    if args.mode in {"audit-sources", "verify", "all"}:
        assert source is not None
        source_manifest, source_errors = audit_sources(
            source, compute_sha256=not args.skip_source_sha256
        )
        manifest["sources"] = source_manifest
        errors.extend(source_errors)

    if args.mode in {"convert", "all"}:
        assert source is not None and out_root is not None
        conversion_manifest = convert_source_to_tree(
            source=source,
            out_root=out_root,
            max_images_per_split=args.max_images_per_split,
            overwrite=args.overwrite,
            resume=args.resume,
            compress_level=args.png_compress_level,
        )
        manifest["conversion"] = conversion_manifest
        tree_root = out_root

    if args.mode in {"audit-tree", "verify", "all"}:
        assert tree_root is not None
        tree_manifest, tree_errors = audit_tree(
            tree_root=tree_root,
            workers=max(1, args.workers),
            chunksize=max(1, args.chunksize),
            sample_count=max(0, args.sample_count),
            seed=args.seed,
            progress_interval=args.progress_interval,
            allow_partial=allow_partial,
            max_images_per_split=args.max_images_per_split,
            sample_dir=sample_dir,
        )
        manifest["tree"] = tree_manifest
        errors.extend(tree_errors)

    if args.mode in {"verify", "all"}:
        assert source is not None and tree_root is not None
        spot_manifest, spot_errors = spot_check_tree_against_sources(
            source=source,
            tree_root=tree_root,
            count=max(0, args.spot_check_count),
            seed=args.seed,
            max_images_per_split=args.max_images_per_split,
            sample_dir=sample_dir,
        )
        manifest["spot_checks"] = spot_manifest
        errors.extend(spot_errors)

    manifest["errors"] = errors
    manifest["ok"] = not errors
    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    write_json(manifest_path, manifest)

    print("==========================================")
    print(f"Manifest: {manifest_path}")
    print(f"OK: {manifest['ok']}")
    if errors:
        print("Errors:")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more")
    print("==========================================")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
