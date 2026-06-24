#!/usr/bin/env python3
"""Build a manifest for richer metric evaluation from saved NLL/FID results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


DEFAULT_SCHEDULES = ("linear", "cosine", "geometric_linear", "geometric_cosine")
DEFAULT_DATASETS = ("mnist", "fashionmnist", "cifar10", "imagenet64")
DEFAULT_OBJECTIVES = ("simple", "hybrid", "vlb")
DATASET_IMAGE_SIZES = {
    "imagenet64": 64,
    "celeba64": 64,
    "lsun_bedroom64": 64,
}

REAL_DIR_DEFAULTS = {
    "mnist": "mnist_train",
    "fashionmnist": "fashion_train",
    "cifar10": "cifar_train",
    "cifar100": "/project_gpfs/bata0/bjin0/cifar100_32x32/train",
    "imagenet64": "/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505/train",
    "svhn": "/project_gpfs/bata0/bjin0/svhn_32x32/train",
    "celeba64": "/project_gpfs/bata0/bjin0/celeba_64x64/train",
    "lsun_bedroom64": "/project_gpfs/bata0/bjin0/lsun_bedroom_64x64/source/bedroom_train_lmdb",
}


def parse_csv_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def sample_shape_from_name(path: Path) -> tuple[int | None, int | None]:
    match = re.search(r"samples_(\d+)x(\d+)x\2x3\.npz$", path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def find_samples(source: Path) -> Path | None:
    if not source.exists():
        return None
    candidates = sorted(source.glob("samples_*x*x*x3.npz"))
    candidates.extend(sorted(source.glob("*/samples_*x*x*x3.npz")))
    if not candidates:
        candidates = sorted(source.rglob("samples_*x*x*x3.npz"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return candidates[-1]


def real_dir_for_dataset(dataset: str, args: argparse.Namespace) -> str:
    override = getattr(args, f"{dataset}_real_dir", None)
    return override or REAL_DIR_DEFAULTS[dataset]


def cache_key_for_sample(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_results", default="evaluation_results_full.tsv")
    parser.add_argument("--output", default="results/richer_metrics/manifest.tsv")
    parser.add_argument("--schedules", default=",".join(DEFAULT_SCHEDULES))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--objectives", default=",".join(DEFAULT_OBJECTIVES))
    parser.add_argument("--mnist_real_dir", default=None)
    parser.add_argument("--fashionmnist_real_dir", default=None)
    parser.add_argument("--cifar10_real_dir", default=None)
    parser.add_argument("--imagenet64_real_dir", default=None)
    parser.add_argument("--svhn_real_dir", default=None)
    parser.add_argument("--cifar100_real_dir", default=None)
    parser.add_argument("--celeba64_real_dir", default=None)
    parser.add_argument("--lsun_bedroom64_real_dir", default=None)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    schedules = set(parse_csv_list(args.schedules, DEFAULT_SCHEDULES))
    datasets = set(parse_csv_list(args.datasets, DEFAULT_DATASETS))
    objectives = set(parse_csv_list(args.objectives, DEFAULT_OBJECTIVES))

    eval_path = Path(args.evaluation_results)
    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)

    rows = []
    missing = []
    with eval_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["dataset"] not in datasets:
                continue
            if row["schedule"] not in schedules:
                continue
            if row["objective"] not in objectives:
                continue

            source = Path(row["source"])
            samples = find_samples(source)
            real_dir = Path(real_dir_for_dataset(row["dataset"], args))
            if samples is None or not samples.is_file():
                missing.append((row["dataset"], row["schedule"], row["objective"], "samples", str(source)))
                if not args.allow_missing:
                    continue
            if not real_dir.is_dir():
                missing.append((row["dataset"], row["schedule"], row["objective"], "real_dir", str(real_dir)))
                if not args.allow_missing:
                    continue

            n_from_name, size_from_name = sample_shape_from_name(samples) if samples else (None, None)
            rows.append(
                {
                    "dataset": row["dataset"],
                    "objective": row["objective"],
                    "schedule": row["schedule"],
                    "samples_npz": str(samples) if samples else "",
                    "real_dir": str(real_dir),
                    "n_samples": str(n_from_name or row.get("fid_samples", "")),
                    "image_size": str(size_from_name or DATASET_IMAGE_SIZES.get(row["dataset"], 32)),
                    "nll_bpd": row.get("nll_bpd", ""),
                    "fid": row.get("fid", ""),
                    "nll_samples": row.get("nll_samples", ""),
                    "fid_samples": row.get("fid_samples", ""),
                    "sampling_steps": row.get("sampling_steps", ""),
                    "source_eval_dir": row["source"],
                    "cache_key": cache_key_for_sample(samples) if samples else "",
                }
            )

    rows.sort(key=lambda r: (r["dataset"], r["objective"], r["schedule"]))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "objective",
        "schedule",
        "samples_npz",
        "real_dir",
        "n_samples",
        "image_size",
        "nll_bpd",
        "fid",
        "nll_samples",
        "fid_samples",
        "sampling_steps",
        "source_eval_dir",
        "cache_key",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} richer-metrics manifest rows to {out_path}")
    if missing:
        print("Missing inputs:")
        for item in missing:
            print("\t".join(item))
        if not args.allow_missing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
