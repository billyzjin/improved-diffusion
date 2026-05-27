#!/usr/bin/env python3
"""
Collect and rank geometric probe runs by validation NLL.

Expected run layout from train_probe_geometric.slurm:
  /project_gpfs/bata0/bjin0/$USER/<jobid>/logs/cifar10_<PROBE_NAME>/
    ema_0.9999_50000.pt
    eval_val/nll_results.txt

Example:
  python3 scripts/collect_probe_geometric_results.py

  python3 scripts/collect_probe_geometric_results.py \
      --search-root /project_gpfs/bata0/bjin0/$USER \
      --job-prefix probe_hyb \
      --show-paths \
      --output-csv /tmp/probe_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


NLL_RE = re.compile(r"bpd=([0-9]+(?:\.[0-9]+)?)")
NAME_RE = re.compile(r"^cifar10_(?P<probe_name>.+)_b(?P<beta_tag>[^_]+)_a(?P<alpha_tag>.+)$")
EMA_RE = re.compile(r"^ema_0\.9999_(?P<step>[0-9]+)\.pt$")


@dataclass
class ProbeResult:
    probe_name: str
    beta_tag: str
    alpha_tag: str
    beta_text: str
    alpha_text: str
    beta_value: Optional[float]
    alpha_value: Optional[float]
    nll: Optional[float]
    status: str
    job_id: Optional[int]
    run_dir: Path
    nll_file: Optional[Path]
    model_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "")
    default_root = f"/project_gpfs/bata0/bjin0/{user}" if user else "/project_gpfs/bata0/bjin0"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--search-root",
        default=default_root,
        help="Root under which job directories live (default: /project_gpfs/bata0/bjin0/$USER).",
    )
    ap.add_argument(
        "--job-prefix",
        default="probe_hyb",
        help="Probe name prefix used by submit_probe_geometric_grid.sh (default: probe_hyb). "
        "May include shell-style wildcards.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show only the top N rows with parsed NLL (default: show all).",
    )
    ap.add_argument(
        "--show-paths",
        action="store_true",
        help="Print run directory and model path under each row.",
    )
    ap.add_argument(
        "--show-all-runs",
        action="store_true",
        help="Do not deduplicate repeated probe names; show every discovered run.",
    )
    ap.add_argument(
        "--output-csv",
        default="",
        help="Optional tall CSV output path.",
    )
    ap.add_argument(
        "--output-matrix-csv",
        default="",
        help="Optional beta-by-alpha matrix CSV output path.",
    )
    ap.add_argument(
        "--output-matrix-md",
        default="",
        help="Optional beta-by-alpha matrix Markdown output path.",
    )
    return ap.parse_args()


def decode_tag(tag: str) -> tuple[str, Optional[float]]:
    text = tag.replace("p", ".")
    try:
        value = float(text)
    except ValueError:
        value = None
    return text, value


def parse_nll(nll_file: Path) -> tuple[Optional[float], str]:
    try:
        text = nll_file.read_text(errors="replace")
    except OSError:
        return None, "read_error"
    matches = NLL_RE.findall(text)
    if not matches:
        return None, "parse_error"
    return float(matches[-1]), "ok"


def infer_job_id(run_dir: Path) -> Optional[int]:
    try:
        return int(run_dir.parents[1].name)
    except (IndexError, ValueError):
        return None


def newest_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def ema_step(path: Path) -> int:
    m = EMA_RE.match(path.name)
    if not m:
        return -1
    return int(m.group("step"))


def collect_results(search_root: Path, job_prefix: str) -> list[ProbeResult]:
    pattern = f"*/logs/cifar10_{job_prefix}_b*_a*"
    results: list[ProbeResult] = []
    for run_dir in sorted(search_root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        m = NAME_RE.match(run_dir.name)
        if not m:
            continue

        probe_name = m.group("probe_name")
        beta_tag = m.group("beta_tag")
        alpha_tag = m.group("alpha_tag")
        beta_text, beta_value = decode_tag(beta_tag)
        alpha_text, alpha_value = decode_tag(alpha_tag)

        nll_file = run_dir / "eval_val" / "nll_results.txt"
        if nll_file.is_file():
            nll, status = parse_nll(nll_file)
            nll_path: Optional[Path] = nll_file
        else:
            nll, status = None, "missing_nll"
            nll_path = None

        model_candidates = list(run_dir.glob("ema_0.9999_*.pt"))
        model_path = max(model_candidates, key=ema_step) if model_candidates else None

        results.append(
            ProbeResult(
                probe_name=probe_name,
                beta_tag=beta_tag,
                alpha_tag=alpha_tag,
                beta_text=beta_text,
                alpha_text=alpha_text,
                beta_value=beta_value,
                alpha_value=alpha_value,
                nll=nll,
                status=status,
                job_id=infer_job_id(run_dir),
                run_dir=run_dir,
                nll_file=nll_path,
                model_path=model_path,
            )
        )
    return results


def prefer_result(lhs: ProbeResult, rhs: ProbeResult) -> ProbeResult:
    lhs_ok = lhs.nll is not None
    rhs_ok = rhs.nll is not None
    if lhs_ok != rhs_ok:
        return lhs if lhs_ok else rhs

    lhs_job = lhs.job_id if lhs.job_id is not None else -1
    rhs_job = rhs.job_id if rhs.job_id is not None else -1
    if lhs_job != rhs_job:
        return lhs if lhs_job > rhs_job else rhs

    lhs_time = newest_mtime(lhs.run_dir)
    rhs_time = newest_mtime(rhs.run_dir)
    return lhs if lhs_time >= rhs_time else rhs


def dedupe_results(results: list[ProbeResult]) -> list[ProbeResult]:
    deduped: dict[tuple[str, str, str], ProbeResult] = {}
    for result in results:
        key = (result.probe_name, result.beta_tag, result.alpha_tag)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = result
        else:
            deduped[key] = prefer_result(existing, result)
    return list(deduped.values())


def sort_key(result: ProbeResult):
    nll_missing = result.nll is None
    nll_value = result.nll if result.nll is not None else float("inf")
    beta_value = result.beta_value if result.beta_value is not None else float("inf")
    alpha_value = result.alpha_value if result.alpha_value is not None else float("inf")
    job_id = result.job_id if result.job_id is not None else -1
    return (nll_missing, nll_value, beta_value, alpha_value, -job_id, result.probe_name)


def write_csv(results: list[ProbeResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "probe_name",
                "beta_1",
                "alpha_bar_T",
                "val_nll",
                "status",
                "job_id",
                "run_dir",
                "nll_file",
                "model_path",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "probe_name": r.probe_name,
                    "beta_1": r.beta_text,
                    "alpha_bar_T": r.alpha_text,
                    "val_nll": "" if r.nll is None else f"{r.nll:.6f}",
                    "status": r.status,
                    "job_id": "" if r.job_id is None else r.job_id,
                    "run_dir": str(r.run_dir),
                    "nll_file": "" if r.nll_file is None else str(r.nll_file),
                    "model_path": "" if r.model_path is None else str(r.model_path),
                }
            )


def matrix_sort_key(text: str) -> tuple[bool, float, str]:
    try:
        return (False, float(text), text)
    except ValueError:
        return (True, float("inf"), text)


def prefer_matrix_result(lhs: ProbeResult, rhs: ProbeResult) -> ProbeResult:
    lhs_ok = lhs.nll is not None
    rhs_ok = rhs.nll is not None
    if lhs_ok != rhs_ok:
        return lhs if lhs_ok else rhs
    if lhs.nll is not None and rhs.nll is not None and lhs.nll != rhs.nll:
        return lhs if lhs.nll < rhs.nll else rhs

    lhs_job = lhs.job_id if lhs.job_id is not None else -1
    rhs_job = rhs.job_id if rhs.job_id is not None else -1
    return lhs if lhs_job >= rhs_job else rhs


def make_matrix(results: list[ProbeResult]) -> tuple[list[str], list[str], dict[tuple[str, str], ProbeResult]]:
    beta_values = sorted({r.beta_text for r in results}, key=matrix_sort_key)
    alpha_values = sorted({r.alpha_text for r in results}, key=matrix_sort_key)
    cells: dict[tuple[str, str], ProbeResult] = {}
    for result in results:
        key = (result.beta_text, result.alpha_text)
        existing = cells.get(key)
        if existing is None:
            cells[key] = result
        else:
            cells[key] = prefer_matrix_result(existing, result)
    return beta_values, alpha_values, cells


def write_matrix_csv(results: list[ProbeResult], output_csv: Path) -> None:
    beta_values, alpha_values, cells = make_matrix(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta_1 \\ alpha_bar_T", *alpha_values])
        for beta in beta_values:
            row = [beta]
            for alpha in alpha_values:
                result = cells.get((beta, alpha))
                row.append("" if result is None or result.nll is None else f"{result.nll:.6f}")
            writer.writerow(row)


def write_matrix_markdown(results: list[ProbeResult], output_md: Path) -> None:
    beta_values, alpha_values, cells = make_matrix(results)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| `beta_1` \\ `alpha_bar_T` | " + " | ".join(f"`{a}`" for a in alpha_values) + " |")
    lines.append("|---:|" + "|".join(["---:"] * len(alpha_values)) + "|")
    for beta in beta_values:
        row = [f"`{beta}`"]
        for alpha in alpha_values:
            result = cells.get((beta, alpha))
            row.append("" if result is None or result.nll is None else f"`{result.nll:.6f}`")
        lines.append("| " + " | ".join(row) + " |")
    output_md.write_text("\n".join(lines) + "\n")


def print_table(results: list[ProbeResult], show_paths: bool) -> None:
    headers = ("Rank", "Probe Name", "beta_1", "alpha_bar_T", "val_nll", "Job ID", "Status")
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append(
            (
                str(i),
                r.probe_name,
                r.beta_text,
                r.alpha_text,
                "" if r.nll is None else f"{r.nll:.6f}",
                "" if r.job_id is None else str(r.job_id),
                r.status,
            )
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row, result in zip(rows, results):
        print(fmt.format(*row))
        if show_paths:
            print(f"      run_dir: {result.run_dir}")
            if result.model_path is not None:
                print(f"      model:   {result.model_path}")
            if result.nll_file is not None:
                print(f"      nll:     {result.nll_file}")


def main() -> int:
    args = parse_args()
    search_root = Path(args.search_root)
    if not search_root.is_dir():
        print(f"ERROR: search root not found: {search_root}", file=sys.stderr)
        return 1

    results = collect_results(search_root, args.job_prefix)
    if not results:
        print(
            f"No probe runs found under {search_root} matching job prefix '{args.job_prefix}'.",
            file=sys.stderr,
        )
        return 1

    if not args.show_all_runs:
        results = dedupe_results(results)
    results.sort(key=sort_key)

    ok_count = sum(r.nll is not None for r in results)
    missing_count = len(results) - ok_count

    visible = results
    if args.limit > 0:
        ok_results = [r for r in results if r.nll is not None][: args.limit]
        bad_results = [r for r in results if r.nll is None]
        visible = ok_results + bad_results

    print(f"Search root:  {search_root}")
    print(f"Job prefix:   {args.job_prefix}")
    print(f"Runs found:   {len(results)}")
    print(f"With NLL:     {ok_count}")
    print(f"Missing NLL:  {missing_count}")
    print("")
    print_table(visible, args.show_paths)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        write_csv(results, output_csv)
        print("")
        print(f"Wrote CSV: {output_csv}")
    if args.output_matrix_csv:
        output_matrix_csv = Path(args.output_matrix_csv)
        write_matrix_csv(results, output_matrix_csv)
        print(f"Wrote matrix CSV: {output_matrix_csv}")
    if args.output_matrix_md:
        output_matrix_md = Path(args.output_matrix_md)
        write_matrix_markdown(results, output_matrix_md)
        print(f"Wrote matrix Markdown: {output_matrix_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
