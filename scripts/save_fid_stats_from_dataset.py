#!/usr/bin/env python3
"""Save FID real-data statistics from an image directory or LSUN-style LMDB."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def is_lmdb_dir(path: Path) -> bool:
    return path.is_dir() and (path / "data.mdb").is_file()


def center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resample = getattr(Image, "Resampling", Image).BOX
    image = image.resize((int(round(scale * width)), int(round(scale * height))), resample=resample)
    left = (image.width - image_size) // 2
    top = (image.height - image_size) // 2
    return image.crop((left, top, left + image_size, top + image_size))


class ImageDirTensorDataset(Dataset):
    def __init__(self, root: Path, image_size: int, center_crop: bool):
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.image_size = image_size
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.center_crop:
            image = center_crop_resize(image, self.image_size)
        else:
            image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


class LmdbTensorDataset(Dataset):
    def __init__(self, root: Path, image_size: int):
        self.root = root
        self.image_size = image_size
        self._env = None
        env = self._open_env()
        with env.begin(write=False) as transaction:
            self.keys = list(transaction.cursor().iternext(keys=True, values=False))
        env.close()
        self._env = None

    def _open_env(self) -> Any:
        if self._env is None:
            import lmdb

            self._env = lmdb.open(
                str(self.root),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=32,
            )
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        env = self._open_env()
        with env.begin(write=False) as transaction:
            image_data = transaction.get(self.keys[idx])
        if image_data is None:
            raise KeyError(f"LMDB key disappeared: {self.keys[idx]!r}")
        image = center_crop_resize(Image.open(io.BytesIO(image_data)), self.image_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def make_dataset(data_path: Path, image_size: int, center_crop_dirs: bool) -> Dataset:
    if is_lmdb_dir(data_path):
        return LmdbTensorDataset(data_path, image_size)
    if data_path.is_dir():
        return ImageDirTensorDataset(data_path, image_size, center_crop_dirs)
    raise FileNotFoundError(data_path)


def build_model(device: str, dims: int) -> Any:
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    return InceptionV3([block_idx]).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--center_crop_dirs",
        action="store_true",
        help="Use LSUN-style center-crop preprocessing for image directories too.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(data_path, args.image_size, args.center_crop_dirs)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))

    model = build_model(args.device, args.dims)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    n = 0
    feat_sum = np.zeros(args.dims, dtype=np.float64)
    feat_outer = np.zeros((args.dims, args.dims), dtype=np.float64)
    with torch.no_grad():
        try:
            for batch in loader:
                batch = batch.to(args.device, non_blocking=True)
                pred = model(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                feats = pred.squeeze(3).squeeze(2).detach().cpu().numpy().astype(np.float64, copy=False)
                n += feats.shape[0]
                feat_sum += feats.sum(axis=0)
                feat_outer += feats.T @ feats
                if n % (args.batch_size * 100) == 0:
                    print(f"processed {n} images", flush=True)
        finally:
            close = getattr(dataset, "close", None)
            if close is not None:
                close()

    if n < 2:
        raise RuntimeError(f"Need at least 2 images for covariance, got {n}")
    mu = feat_sum / n
    sigma = (feat_outer - n * np.outer(mu, mu)) / (n - 1)

    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("wb") as f:
        np.savez(f, mu=mu, sigma=sigma)
    os.replace(tmp_path, out_path)
    print(f"saved FID stats for {n} images to {out_path}")


if __name__ == "__main__":
    main()
