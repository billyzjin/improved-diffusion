import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# Use filesystem-safe class names (Fashion-MNIST label order 0..9).
CLASSES = (
    "tshirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankleboot",
)

BASE_URL = "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

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


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, n, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Bad magic for images file {path}: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    arr = arr.reshape(n, rows, cols)
    return arr


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, n = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Bad magic for labels file {path}: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.shape[0] != n:
        raise ValueError(f"Label count mismatch in {path}: expected {n}, got {arr.shape[0]}")
    return arr


def _dump_split(split: str, cache_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)

    img_gz = cache_dir / FILES[split]["images"]
    lab_gz = cache_dir / FILES[split]["labels"]
    _download(BASE_URL + FILES[split]["images"], img_gz)
    _download(BASE_URL + FILES[split]["labels"], lab_gz)

    images = _read_idx_images_gz(img_gz)  # [N,28,28], uint8
    labels = _read_idx_labels_gz(lab_gz)  # [N], uint8
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}")

    n = images.shape[0]
    print(f"Dumping {n} {split} images to {out_dir} ...")
    for i in range(n):
        lab = int(labels[i])
        cls = CLASSES[lab]
        img = Image.fromarray(images[i], mode="L").convert("RGB")
        img = img.resize((32, 32), resample=Image.BICUBIC)
        (out_dir / f"{cls}_{i:05d}.png").write_bytes(_pil_to_png_bytes(img))
        if (i + 1) % 5000 == 0:
            print(f"  wrote {i+1}/{n}")


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    # Avoid relying on tqdm/torchvision; keep dependencies minimal.
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    # Prefer torchvision if available (faster + simpler), but fall back to raw download.
    try:
        import tempfile

        import torchvision

        print("torchvision detected; using torchvision.datasets.FashionMNIST")
        for split in ["train", "test"]:
            out_dir = Path(f"fashion_{split}")
            if out_dir.exists():
                print(f"skipping split {split} since {out_dir} already exists.")
                continue

            print("downloading via torchvision...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                dataset = torchvision.datasets.FashionMNIST(
                    root=tmp_dir, train=split == "train", download=True
                )

            out_dir.mkdir()
            print(f"dumping {len(dataset)} images to {out_dir} ...")
            for i in range(len(dataset)):
                image, label = dataset[i]  # PIL image in mode "L"
                image = image.convert("RGB").resize((32, 32), resample=Image.BICUBIC)
                filename = out_dir / f"{CLASSES[label]}_{i:05d}.png"
                image.save(filename)
                if (i + 1) % 5000 == 0:
                    print(f"  wrote {i+1}/{len(dataset)}")
        return
    except ModuleNotFoundError as e:
        if e.name != "torchvision":
            raise
        print("torchvision not found; falling back to raw Fashion-MNIST download.")

    cache_dir = Path(".fashionmnist_cache")
    for split in ["train", "test"]:
        out_dir = Path(f"fashion_{split}")
        if out_dir.exists():
            print(f"skipping split {split} since {out_dir} already exists.")
            continue
        _dump_split(split=split, cache_dir=cache_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()


