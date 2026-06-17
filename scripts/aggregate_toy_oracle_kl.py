#!/usr/bin/env python3
"""Aggregate toy oracle per-run summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy_dir", required=True)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_tsv", default=None)
    args = parser.parse_args()

    toy_dir = Path(args.toy_dir)
    summaries = []
    for path in sorted(toy_dir.glob("*/*/*/summary.json")):
        summaries.append(json.loads(path.read_text()))
    summaries.sort(key=lambda r: (r["distribution"], r["T"], r["schedule"]))

    jsonl_path = Path(args.output_jsonl) if args.output_jsonl else toy_dir / "summary.jsonl"
    tsv_path = Path(args.output_tsv) if args.output_tsv else toy_dir / "summary.tsv"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w") as f:
        for row in summaries:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    fields = [
        "distribution",
        "T",
        "schedule",
        "endpoint_family",
        "beta1",
        "alpha_bar_T",
        "sum_K_bulk",
        "sum_K_bulk_se",
        "sum_Psi_bulk",
        "max_K_bulk",
        "max_Psi_bulk",
        "first_step_K",
        "first_step_K_se",
        "n_x",
        "m_post",
        "seed",
        "warning_count",
    ]
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} toy oracle rows to {jsonl_path} and {tsv_path}")


if __name__ == "__main__":
    main()
