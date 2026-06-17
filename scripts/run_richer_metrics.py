#!/usr/bin/env python3
"""Compute CMMD, KID, and density/coverage for generated sample NPZ files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_metrics(value: str) -> set[str]:
    aliases = {
        "dc": "density_coverage",
        "density": "density_coverage",
        "coverage": "density_coverage",
        "clip_dc": "clip_density_coverage",
        "clip_density": "clip_density_coverage",
        "clip_coverage": "clip_density_coverage",
        "density_coverage_clip": "clip_density_coverage",
    }
    out = set()
    for item in re.split(r"[,;:+\s]+", value):
        item = item.strip().lower()
        if not item:
            continue
        out.add(aliases.get(item, item))
    valid = {"cmmd", "kid", "density_coverage", "clip_density_coverage"}
    bad = out - valid
    if bad:
        raise ValueError(f"Unknown metrics {sorted(bad)}; valid metrics are {sorted(valid)}")
    return out


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_image_paths(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return sorted(paths)


def select_indices(n_total: int, n_requested: int | None, seed: int) -> np.ndarray:
    if n_requested is None or n_requested <= 0 or n_requested >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_requested, replace=False))


def load_npz_images(path: Path, n_requested: int | None) -> np.ndarray:
    data = np.load(path)
    if "arr_0" not in data:
        raise KeyError(f"{path} does not contain arr_0")
    arr = data["arr_0"]
    data.close()
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D samples in {path}, got {arr.shape}")
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB NHWC samples in {path}, got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 samples in {path}, got {arr.dtype}")
    if n_requested and n_requested > 0:
        arr = arr[: min(n_requested, arr.shape[0])]
    return arr


class NpzImageDataset:
    def __init__(self, images: np.ndarray, transform: Any):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> Any:
        img = Image.fromarray(self.images[idx], "RGB")
        return self.transform(img) if self.transform is not None else img


class PathImageDataset:
    def __init__(self, paths: list[Path], transform: Any):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img) if self.transform is not None else img


def dataloader_features(dataset: Any, model: Any, device: str, batch_size: int, num_workers: int, kind: str) -> np.ndarray:
    import torch
    from torch.nn.functional import adaptive_avg_pool2d

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    feats: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if kind == "inception":
                pred = model(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze(3).squeeze(2)
            elif kind == "clip":
                pred = model.encode_image(batch)
                pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                raise ValueError(kind)
            feats.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(feats, axis=0)


def build_inception_model(device: str, dims: int) -> Any:
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    return InceptionV3([block_idx]).to(device)


def build_clip_model(device: str, model_name: str, pretrained: str) -> tuple[Any, Any, str]:
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device)
        return model, preprocess, f"open_clip:{model_name}:{pretrained}"
    except ImportError:
        pass
    try:
        import clip

        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess, f"clip:{model_name}"
    except ImportError as exc:
        raise RuntimeError(
            "CMMD requires either open_clip_torch or OpenAI clip in the active Python environment. "
            "Install/load one of them, or run without --metrics cmmd."
        ) from exc


def feature_paths(cache_root: Path, row: dict[str, str], split: str, kind: str, n_requested: int, seed: int) -> tuple[Path, Path]:
    if split == "real":
        base = cache_root / row["dataset"] / "real"
    else:
        cache_key = row.get("cache_key", "")
        if not cache_key:
            cache_key_source = row.get("samples_npz") or row.get("source_eval_dir") or json.dumps(row, sort_keys=True)
            cache_key = hashlib.sha256(cache_key_source.encode("utf-8")).hexdigest()[:16]
        base = cache_root / row["dataset"] / row["objective"] / row["schedule"] / cache_key / "generated"
    stem = f"{kind}_n{n_requested}_seed{seed}"
    return base / f"{stem}.npy", base / f"{stem}.metadata.json"


def acquire_feature_lock(feature_path: Path, meta_path: Path, force: bool, wait_seconds: int = 43_200) -> Path | None:
    """Return a lock dir to release, or None if another worker completed the cache."""
    if feature_path.is_file() and meta_path.is_file() and not force:
        return None

    lock_dir = feature_path.with_suffix(feature_path.suffix + ".lock")
    start = time.time()
    while True:
        if feature_path.is_file() and meta_path.is_file() and not force:
            return None
        try:
            lock_dir.mkdir()
            return lock_dir
        except FileExistsError:
            if time.time() - start > wait_seconds:
                raise TimeoutError(f"Timed out waiting for feature lock: {lock_dir}")
            print(f"Waiting for feature cache lock: {lock_dir}", flush=True)
            time.sleep(30)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = tempfile.NamedTemporaryFile(prefix=path.name, suffix=".tmp", dir=str(path.parent), delete=False)
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            np.save(tmp, array)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def extract_or_load_features(
    *,
    row: dict[str, str],
    split: str,
    kind: str,
    cache_root: Path,
    n_requested: int,
    seed: int,
    device: str,
    batch_size: int,
    num_workers: int,
    inception_dims: int,
    clip_model_name: str,
    clip_pretrained: str,
    force: bool,
) -> np.ndarray:
    feature_path, meta_path = feature_paths(cache_root, row, split, kind, n_requested, seed)
    if feature_path.is_file() and meta_path.is_file() and not force:
        return np.load(feature_path)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = acquire_feature_lock(feature_path, meta_path, force)
    if lock_dir is None:
        return np.load(feature_path)

    try:
        if split == "generated":
            source_path = Path(row["samples_npz"])
            images = load_npz_images(source_path, n_requested)
            selected_count = int(images.shape[0])
            source_kind = "npz"
            source_digest = file_sha256(source_path)
        else:
            source_path = Path(row["real_dir"])
            all_paths = collect_image_paths(source_path)
            indices = select_indices(len(all_paths), n_requested, seed)
            all_paths = [all_paths[int(i)] for i in indices]
            selected_count = len(all_paths)
            source_kind = "image_dir"
            source_digest = ""

        if kind == "inception_pool3":
            import torchvision.transforms as transforms

            transform = transforms.ToTensor()
            model = build_inception_model(device, inception_dims)
            dataset = NpzImageDataset(images, transform) if split == "generated" else PathImageDataset(all_paths, transform)
            features = dataloader_features(dataset, model, device, batch_size, num_workers, "inception")
            model_label = f"pytorch_fid_inception_v3_pool3_dims{inception_dims}"
            preprocessing = "torchvision.transforms.ToTensor"
        elif kind == "clip":
            model, transform, model_label = build_clip_model(device, clip_model_name, clip_pretrained)
            dataset = NpzImageDataset(images, transform) if split == "generated" else PathImageDataset(all_paths, transform)
            features = dataloader_features(dataset, model, device, batch_size, num_workers, "clip")
            preprocessing = "clip_preprocess_l2_normalized"
        else:
            raise ValueError(kind)

        metadata = {
            "dataset": row["dataset"],
            "objective": row["objective"],
            "schedule": row["schedule"],
            "split": split,
            "kind": kind,
            "source_kind": source_kind,
            "source_path": str(source_path),
            "source_sha256": source_digest,
            "n_requested": n_requested,
            "n_features": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "seed": seed,
            "model": model_label,
            "preprocessing": preprocessing,
            "image_range": "uint8 [0,255] before preprocessing",
        }
        if selected_count != features.shape[0]:
            raise RuntimeError(f"Feature count mismatch for {feature_path}: selected={selected_count} features={features.shape[0]}")
        atomic_save_npy(feature_path, features.astype(np.float32, copy=False))
        atomic_write_json(meta_path, metadata)
        print(f"Saved {kind} {split} features: {feature_path} shape={features.shape}", flush=True)
        return features
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def subset_features(features: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0 or n >= features.shape[0]:
        return features
    idx = select_indices(features.shape[0], n, seed)
    return features[idx]


def torch_pairwise_sq(x: Any, y: Any) -> Any:
    import torch

    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).T
    return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0.0)


def median_bandwidth_sq(features_a: np.ndarray, features_b: np.ndarray, seed: int, max_points: int) -> float:
    import torch

    combined = np.concatenate([features_a, features_b], axis=0)
    idx = select_indices(combined.shape[0], min(max_points, combined.shape[0]), seed)
    z = torch.from_numpy(combined[idx].astype(np.float32, copy=False))
    d = torch_pairwise_sq(z, z).cpu().numpy()
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def rbf_sum(features_a: np.ndarray, features_b: np.ndarray, sigma_sq: float, device: str, chunk_size: int) -> float:
    import torch

    xb = torch.from_numpy(features_b.astype(np.float32, copy=False)).to(device)
    total = 0.0
    for start in range(0, features_a.shape[0], chunk_size):
        xa = torch.from_numpy(features_a[start : start + chunk_size].astype(np.float32, copy=False)).to(device)
        d = torch_pairwise_sq(xa, xb)
        total += torch.exp(-d / (2.0 * sigma_sq)).sum().item()
    return float(total)


def compute_cmmd(real: np.ndarray, gen: np.ndarray, device: str, seed: int, n: int, bandwidth_sample: int, chunk_size: int) -> dict[str, Any]:
    real = subset_features(real, n, seed)
    gen = subset_features(gen, n, seed + 1)
    n_real = real.shape[0]
    n_gen = gen.shape[0]
    if n_real < 2 or n_gen < 2:
        raise ValueError("CMMD requires at least two real and generated features")
    sigma_sq = median_bandwidth_sq(real, gen, seed, bandwidth_sample)
    kxx = rbf_sum(real, real, sigma_sq, device, chunk_size)
    kyy = rbf_sum(gen, gen, sigma_sq, device, chunk_size)
    kxy = rbf_sum(real, gen, sigma_sq, device, chunk_size)
    mmd2 = (kxx - n_real) / (n_real * (n_real - 1)) + (kyy - n_gen) / (n_gen * (n_gen - 1)) - 2.0 * kxy / (n_real * n_gen)
    return {
        "metric": "CMMD",
        "value": float(mmd2),
        "estimator": "unbiased_mmd2",
        "kernel": "rbf",
        "bandwidth": "median_heuristic_squared_distance",
        "bandwidth_sq": float(sigma_sq),
        "feature_normalization": "l2",
        "num_real": int(n_real),
        "num_generated": int(n_gen),
    }


def polynomial_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    d = x.shape[1]
    kxx = (x @ x.T / d + 1.0) ** 3
    kyy = (y @ y.T / d + 1.0) ** 3
    kxy = (x @ y.T / d + 1.0) ** 3
    m = x.shape[0]
    n = y.shape[0]
    return float(
        (np.sum(kxx) - np.trace(kxx)) / (m * (m - 1))
        + (np.sum(kyy) - np.trace(kyy)) / (n * (n - 1))
        - 2.0 * np.mean(kxy)
    )


def compute_kid(real: np.ndarray, gen: np.ndarray, seed: int, subsets: int, subset_size: int) -> dict[str, Any]:
    m = min(subset_size, real.shape[0], gen.shape[0])
    if m < 2:
        raise ValueError("KID requires at least two real and generated features")
    rng = np.random.default_rng(seed)
    values = []
    real64 = real.astype(np.float64, copy=False)
    gen64 = gen.astype(np.float64, copy=False)
    for _ in range(subsets):
        ridx = rng.choice(real64.shape[0], size=m, replace=False)
        gidx = rng.choice(gen64.shape[0], size=m, replace=False)
        values.append(polynomial_mmd2(real64[ridx], gen64[gidx]))
    arr = np.array(values, dtype=np.float64)
    return {
        "metric": "KID",
        "value": float(np.mean(arr)),
        "standard_error": float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "estimator": "unbiased_mmd2",
        "subsets": int(subsets),
        "subset_size": int(m),
        "num_real": int(real.shape[0]),
        "num_generated": int(gen.shape[0]),
        "kernel": "polynomial ((x.y/d)+1)^3",
    }


def kth_real_radii(real: np.ndarray, k: int, device: str, chunk_size: int) -> np.ndarray:
    import torch

    real_t = torch.from_numpy(real.astype(np.float32, copy=False)).to(device)
    radii = np.empty(real.shape[0], dtype=np.float32)
    for start in range(0, real.shape[0], chunk_size):
        stop = min(start + chunk_size, real.shape[0])
        d = torch_pairwise_sq(real_t[start:stop], real_t)
        rows = torch.arange(stop - start, device=device)
        cols = torch.arange(start, stop, device=device)
        d[rows, cols] = float("inf")
        radii[start:stop] = torch.kthvalue(d, k=k, dim=1).values.detach().cpu().numpy()
    return radii


def compute_density_coverage(
    real: np.ndarray,
    gen: np.ndarray,
    seed: int,
    n: int,
    k: int,
    device: str,
    chunk_size: int,
    feature_extractor: str,
) -> dict[str, Any]:
    import torch

    real = subset_features(real, n, seed)
    gen = subset_features(gen, n, seed + 1)
    if real.shape[0] <= k:
        raise ValueError(f"Density/coverage requires more than k real features, got {real.shape[0]} <= {k}")

    radii = kth_real_radii(real, k, device, chunk_size)
    radii_t = torch.from_numpy(radii).to(device)
    real_t = torch.from_numpy(real.astype(np.float32, copy=False)).to(device)
    min_real_to_gen = torch.full((real.shape[0],), float("inf"), device=device)
    density_sum = 0.0

    for start in range(0, gen.shape[0], chunk_size):
        gen_t = torch.from_numpy(gen[start : start + chunk_size].astype(np.float32, copy=False)).to(device)
        d = torch_pairwise_sq(gen_t, real_t)
        density_sum += ((d <= radii_t[None, :]).sum(dim=1).float() / float(k)).sum().item()
        min_real_to_gen = torch.minimum(min_real_to_gen, d.min(dim=0).values)

    coverage = (min_real_to_gen <= radii_t).float().mean().item()
    density = density_sum / gen.shape[0]
    return {
        "metric": "density_coverage",
        "density": float(density),
        "coverage": float(coverage),
        "k": int(k),
        "num_real": int(real.shape[0]),
        "num_generated": int(gen.shape[0]),
        "feature_extractor": feature_extractor,
    }


def row_output_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / "rows" / f"{row['dataset']}_{row['objective']}_{row['schedule']}.json"


def run_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    metrics = parse_metrics(args.metrics)
    n_features = int(row["n_samples"] or row["fid_samples"] or 0)
    if n_features <= 0:
        raise ValueError(f"Could not infer n_features from manifest row: {row}")

    result_path = row_output_path(Path(args.output_dir), row)
    if result_path.is_file() and not args.force_metrics:
        result = json.loads(result_path.read_text())
        existing_metrics = set(result.get("metrics", {}))
        missing_metrics = metrics - existing_metrics
        if not missing_metrics:
            print(f"Metric JSON already complete; skipping {result_path}")
            return result
        print(f"Metric JSON exists but is missing {sorted(missing_metrics)}; updating {result_path}")
        metrics = missing_metrics
    elif result_path.is_file() and args.force_metrics:
        result = json.loads(result_path.read_text())
        result.setdefault("metrics", {})
    else:
        result = {
            "dataset": row["dataset"],
            "objective": row["objective"],
            "schedule": row["schedule"],
            "samples_npz": row["samples_npz"],
            "real_dir": row["real_dir"],
            "nll_bpd": row.get("nll_bpd", ""),
            "fid": row.get("fid", ""),
            "n_features_requested": n_features,
            "metrics": {},
        }

    cache_root = Path(args.feature_cache_root)
    result.update(
        {
            "dataset": row["dataset"],
            "objective": row["objective"],
            "schedule": row["schedule"],
            "samples_npz": row["samples_npz"],
            "real_dir": row["real_dir"],
            "nll_bpd": row.get("nll_bpd", result.get("nll_bpd", "")),
            "fid": row.get("fid", result.get("fid", "")),
            "n_features_requested": n_features,
        }
    )
    result.setdefault("metrics", {})

    need_inception = bool(metrics & {"kid", "density_coverage"})
    need_clip = bool(metrics & {"cmmd", "clip_density_coverage"})
    if need_inception:
        real_inc = extract_or_load_features(
            row=row,
            split="real",
            kind="inception_pool3",
            cache_root=cache_root,
            n_requested=n_features,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            inception_dims=args.inception_dims,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            force=args.force_features,
        )
        gen_inc = extract_or_load_features(
            row=row,
            split="generated",
            kind="inception_pool3",
            cache_root=cache_root,
            n_requested=n_features,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            inception_dims=args.inception_dims,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            force=args.force_features,
        )
        if "kid" in metrics:
            result["metrics"]["kid"] = compute_kid(real_inc, gen_inc, args.seed, args.kid_subsets, args.kid_subset_size)
        if "density_coverage" in metrics:
            result["metrics"]["density_coverage"] = compute_density_coverage(
                real_inc,
                gen_inc,
                args.seed,
                args.dc_n,
                args.dc_k,
                args.device,
                args.distance_chunk_size,
                "inception_v3_pool3",
            )

    if need_clip:
        real_clip = extract_or_load_features(
            row=row,
            split="real",
            kind="clip",
            cache_root=cache_root,
            n_requested=n_features,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            inception_dims=args.inception_dims,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            force=args.force_features,
        )
        gen_clip = extract_or_load_features(
            row=row,
            split="generated",
            kind="clip",
            cache_root=cache_root,
            n_requested=n_features,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            inception_dims=args.inception_dims,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            force=args.force_features,
        )
        if "cmmd" in metrics:
            result["metrics"]["cmmd"] = compute_cmmd(
                real_clip,
                gen_clip,
                args.device,
                args.seed,
                args.cmmd_n,
                args.cmmd_bandwidth_sample,
                args.distance_chunk_size,
            )
            result["metrics"]["cmmd"]["clip_model"] = args.clip_model
        if "clip_density_coverage" in metrics:
            result["metrics"]["clip_density_coverage"] = compute_density_coverage(
                real_clip,
                gen_clip,
                args.seed,
                args.dc_n,
                args.dc_k,
                args.device,
                args.distance_chunk_size,
                f"clip:{args.clip_model}",
            )
            result["metrics"]["clip_density_coverage"]["feature_normalization"] = "l2"

    result_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_path, result)
    print(f"Wrote metrics: {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row_index", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--schedule", default=None)
    parser.add_argument("--metrics", default="cmmd,kid,density_coverage")
    parser.add_argument("--output_dir", default="results/richer_metrics")
    parser.add_argument("--feature_cache_root", default="/project_gpfs/bata0/bjin0/richer_metrics_features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--inception_dims", type=int, default=2048)
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--cmmd_n", type=int, default=10000)
    parser.add_argument("--cmmd_bandwidth_sample", type=int, default=2048)
    parser.add_argument("--dc_n", type=int, default=10000)
    parser.add_argument("--dc_k", type=int, default=5)
    parser.add_argument("--distance_chunk_size", type=int, default=1024)
    parser.add_argument("--force_features", action="store_true")
    parser.add_argument("--force_metrics", action="store_true")
    args = parser.parse_args()

    rows = load_manifest_rows(Path(args.manifest))
    if args.row_index is not None:
        rows = [rows[args.row_index]]
    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]
    if args.objective:
        rows = [r for r in rows if r["objective"] == args.objective]
    if args.schedule:
        rows = [r for r in rows if r["schedule"] == args.schedule]
    if not rows:
        raise RuntimeError("No manifest rows selected")

    for row in rows:
        print(f"Running richer metrics for {row['dataset']} {row['objective']} {row['schedule']}", flush=True)
        run_row(row, args)


if __name__ == "__main__":
    main()
