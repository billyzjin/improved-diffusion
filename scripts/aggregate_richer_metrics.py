#!/usr/bin/env python3
"""Aggregate richer metric JSON rows into TSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_manifest(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {(r["dataset"], r["objective"], r["schedule"]): r for r in rows}


def metric_value(metrics: dict, name: str, field: str = "value") -> str:
    if name not in metrics:
        return ""
    value = metrics[name].get(field, "")
    if value == "":
        return ""
    return f"{float(value):.9g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/richer_metrics/manifest.tsv")
    parser.add_argument("--results_dir", default="results/richer_metrics")
    parser.add_argument("--output_tsv", default="results/richer_metrics/richer_metrics_summary.tsv")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    rows = []
    for path in sorted((Path(args.results_dir) / "rows").glob("*.json")):
        data = json.loads(path.read_text())
        key = (data["dataset"], data["objective"], data["schedule"])
        base = manifest.get(key, {})
        metrics = data.get("metrics", {})
        dc = metrics.get("density_coverage", {})
        kid = metrics.get("kid", {})
        rows.append(
            {
                "dataset": data["dataset"],
                "objective": data["objective"],
                "schedule": data["schedule"],
                "nll_bpd": base.get("nll_bpd", data.get("nll_bpd", "")),
                "fid": base.get("fid", data.get("fid", "")),
                "cmmd": metric_value(metrics, "cmmd"),
                "kid": metric_value(metrics, "kid"),
                "kid_se": f"{float(kid.get('standard_error', 'nan')):.9g}" if "standard_error" in kid else "",
                "density": f"{float(dc.get('density', 'nan')):.9g}" if "density" in dc else "",
                "coverage": f"{float(dc.get('coverage', 'nan')):.9g}" if "coverage" in dc else "",
                "n_features_requested": str(data.get("n_features_requested", "")),
                "samples_npz": data.get("samples_npz", ""),
                "real_dir": data.get("real_dir", ""),
            }
        )

    rows.sort(key=lambda r: (r["dataset"], r["objective"], r["schedule"]))
    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "objective",
        "schedule",
        "nll_bpd",
        "fid",
        "cmmd",
        "kid",
        "kid_se",
        "density",
        "coverage",
        "n_features_requested",
        "samples_npz",
        "real_dir",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
