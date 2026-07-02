#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def read_metadata(path: Path) -> dict[str, str]:
    out = {}
    if not path.is_file():
        return out
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            out[key] = value
    return out


def parse_nll(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    matches = re.findall(r"done\s+\d+\s+samples:\s+bpd=([0-9.]+)", text)
    return matches[-1] if matches else ""


def parse_fid(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    return matches[-1] if matches else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate generic image-folder NLL/FID outputs.")
    parser.add_argument("eval_root", help="Evaluation root containing one directory per evaluated run.")
    parser.add_argument("--output", default=None, help="Output TSV path.")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.is_dir():
        raise FileNotFoundError(eval_root)

    rows = []
    for exp_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        metadata = read_metadata(exp_dir / "metadata.tsv")
        if not metadata:
            continue
        rows.append(
            {
                "dataset": metadata.get("dataset", ""),
                "schedule": metadata.get("schedule_name", ""),
                "objective": metadata.get("objective", ""),
                "hybrid_vb_weight": metadata.get(
                    "hybrid_vb_weight",
                    "0.001" if metadata.get("objective", "") == "hybrid" else "",
                ),
                "beta_1": "",
                "alpha_bar_T": "",
                "nll_bpd": parse_nll(exp_dir / "nll_results.txt"),
                "fid": parse_fid(exp_dir / "fid_results.txt"),
                "nll_samples": metadata.get("nll_num_samples", ""),
                "fid_samples": metadata.get("fid_num_samples", ""),
                "sampling_steps": "4000",
                "source": str(exp_dir),
            }
        )

    rows.sort(key=lambda r: (r["dataset"], r["schedule"], r["objective"], r["hybrid_vb_weight"]))
    output = Path(args.output) if args.output else eval_root / "results_summary.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "schedule",
        "objective",
        "hybrid_vb_weight",
        "beta_1",
        "alpha_bar_T",
        "nll_bpd",
        "fid",
        "nll_samples",
        "fid_samples",
        "sampling_steps",
        "source",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
