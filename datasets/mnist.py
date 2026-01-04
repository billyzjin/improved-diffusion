import gzip
import os
import struct
import urllib.request
from pathlib import Path

from PIL import Image

CLASSES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)

# Mirrors the Fashion-MNIST dataset prep style: RGB 32x32 PNGs, class-prefixed names.

BASE_URL = "http://yann.lecun.com/exdb/mnist/"

FILES = {
    "train": {
        "images": "train-images-idx3-ubyte.gz",
        "labels": "train-labels-idx1-ubyte.gz",
    },
    "test": {
        "images": "t10k-images-idx3-ubyte.gz",
        "labels": "t10k-labels-idx1-ubyte.gz",
    },
}


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    print(f"Downloading {url} -> {out_path}")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())


def _read_idx_images_gz(path: Path):
    """
    Read IDX image file (gzipped) without numpy.
    Returns (n, rows, cols, raw_bytes) where raw_bytes has length n*rows*cols.
    """
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, n, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Bad magic for images file {path}: {magic}")
        data = f.read()
    expected = n * rows * cols
    if len(data) != expected:
        raise ValueError(f"Unexpected image payload size in {path}: got {len(data)}, expected {expected}")
    return n, rows, cols, data


def _read_idx_labels_gz(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, n = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Bad magic for labels file {path}: {magic}")
        data = f.read()
    if len(data) != n:
        raise ValueError(f"Label count mismatch in {path}: expected {n}, got {len(data)}")
    return data


def _dump_split(split: str, cache_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)

    img_gz = cache_dir / FILES[split]["images"]
    lab_gz = cache_dir / FILES[split]["labels"]
    _download(BASE_URL + FILES[split]["images"], img_gz)
    _download(BASE_URL + FILES[split]["labels"], lab_gz)

    n, rows, cols, images = _read_idx_images_gz(img_gz)  # raw bytes
    labels = _read_idx_labels_gz(lab_gz)  # raw bytes
    if n != len(labels):
        raise ValueError(f"Image/label count mismatch: {n} vs {len(labels)}")

    print(f"Dumping {n} {split} images to {out_dir} ...")
    stride = rows * cols
    for i in range(n):
        lab = labels[i]
        cls = CLASSES[lab]
        offset = i * stride
        img_bytes = images[offset : offset + stride]
        img = Image.frombytes("L", (cols, rows), img_bytes).convert("RGB")
        img = img.resize((32, 32), resample=Image.BICUBIC)
        img.save(out_dir / f"{cls}_{i:05d}.png")
        if (i + 1) % 5000 == 0:
            print(f"  wrote {i+1}/{n}")


def main() -> None:
    cache_dir = Path(".mnist_cache")
    for split in ["train", "test"]:
        out_dir = Path(f"mnist_{split}")
        if out_dir.exists():
            print(f"skipping split {split} since {out_dir} already exists.")
            continue
        _dump_split(split=split, cache_dir=cache_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()


