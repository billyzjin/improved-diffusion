"""Convert LSUN LMDB splits into center-cropped image folders."""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

from PIL import Image
import lmdb
import numpy as np


def progress(iterable, total=None, desc=None):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        return iterable


def open_lmdb(path: Path):
    if not path.is_dir():
        raise SystemExit(f"LMDB directory not found: {path}")
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=32,
    )


def source_count(lmdb_path: Path) -> int:
    env = open_lmdb(lmdb_path)
    try:
        return int(env.stat()["entries"])
    finally:
        env.close()


def iter_lmdb_images(lmdb_path: Path):
    env = open_lmdb(lmdb_path)
    try:
        with env.begin(write=False) as transaction:
            cursor = transaction.cursor()
            for _, image_data in cursor:
                yield image_data
    finally:
        env.close()


def center_crop_resize(image_data: bytes, image_size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resample = getattr(Image, "Resampling", Image).BOX
    image = image.resize((int(round(scale * width)), int(round(scale * height))), resample=resample)
    arr = np.array(image)
    h, w, _ = arr.shape
    h_off = (h - image_size) // 2
    w_off = (w - image_size) // 2
    return arr[h_off : h_off + image_size, w_off : w_off + image_size]


def completed_manifest(path: Path, expected_count: int, image_size: int) -> bool:
    manifest = path / "_manifest.tsv"
    if not manifest.is_file():
        return False
    values: dict[str, str] = {}
    with manifest.open() as f:
        for line in f:
            key, _, value = line.rstrip("\n").partition("\t")
            values[key] = value
    return (
        values.get("status") == "complete"
        and int(values.get("output_count", "-1")) == expected_count
        and int(values.get("image_size", "-1")) == image_size
    )


def write_manifest(path: Path, split: str, lmdb_path: Path, source_entries: int, output_count: int, image_size: int):
    with (path / "_manifest.tsv").open("w") as f:
        f.write("status\tcomplete\n")
        f.write(f"split\t{split}\n")
        f.write(f"source_lmdb\t{lmdb_path}\n")
        f.write(f"source_entries\t{source_entries}\n")
        f.write(f"output_count\t{output_count}\n")
        f.write(f"image_size\t{image_size}\n")


def write_png_atomic(arr: np.ndarray, out_path: Path) -> None:
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        Image.fromarray(arr).save(tmp_path)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def prepare_split(split: str, root: Path, out_root: Path, category: str, image_size: int, overwrite: bool, max_images):
    lmdb_path = root / f"{category}_{split}_lmdb"
    out_dir = out_root / ("val" if split == "val" else split)
    entries = source_count(lmdb_path)
    expected = min(entries, max_images) if max_images else entries

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if completed_manifest(out_dir, expected, image_size):
        print(f"skipping split {split}; completed manifest found in {out_dir}")
        return

    print(f"converting {expected} / {entries} LSUN {category} {split} images to {out_dir}")
    prefix = f"{category}_{split}"
    processed = 0
    saved = 0
    skipped = 0
    for i, image_data in enumerate(progress(iter_lmdb_images(lmdb_path), total=expected, desc=split)):
        if max_images and i >= max_images:
            break
        out_path = out_dir / f"{prefix}_{i:07d}.png"
        if out_path.exists() and not overwrite:
            if out_path.stat().st_size == 0:
                out_path.unlink()
            else:
                skipped += 1
                processed += 1
                continue
        if out_path.exists() and overwrite:
            out_path.unlink()
        if not out_path.exists():
            arr = center_crop_resize(image_data, image_size)
            write_png_atomic(arr, out_path)
            saved += 1
            processed += 1
            continue
        if out_path.exists() and not overwrite:
            skipped += 1
            processed += 1
            continue

    if processed != expected:
        raise SystemExit(f"split {split}: processed {processed} images, expected {expected}")
    write_manifest(out_dir, split, lmdb_path, entries, expected, image_size)
    print(f"split {split}: complete output_count={expected} saved={saved} skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Convert LSUN Bedroom LMDB splits to image folders.")
    parser.add_argument("--root", default="/project_gpfs/bata0/bjin0/lsun_bedroom_64x64/source")
    parser.add_argument("--out_root", default="/project_gpfs/bata0/bjin0/lsun_bedroom_64x64")
    parser.add_argument("--category", default="bedroom")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        max_images = args.max_train_images if split == "train" else args.max_val_images
        prepare_split(split, root, out_root, args.category, args.size, args.overwrite, max_images or None)


if __name__ == "__main__":
    main()
