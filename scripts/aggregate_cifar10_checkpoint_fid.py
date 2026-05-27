#!/usr/bin/env python3
"""Aggregate CIFAR-10 FID-by-checkpoint sweep results."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


RESULT_RE = re.compile(
    r"^(?P<experiment>cifar10_.+)_step(?P<step>[0-9]{6})_n(?P<num_samples>[0-9]+)$"
)

SCHEDULE_ORDER = {
    "linear": 0,
    "cosine": 1,
    "geometric_linear": 2,
    "geometric_cosine": 3,
}
OBJECTIVE_ORDER = {"simple": 0, "hybrid": 1, "vlb": 2}


@dataclass(frozen=True)
class Result:
    experiment: str
    schedule: str
    objective: str
    step: int
    step_label: str
    num_samples: int
    fid: float
    result_dir: Path
    model_path: str


def parse_experiment(name: str) -> tuple[str, str]:
    body = name.removeprefix("cifar10_")
    for objective in ("simple", "hybrid", "vlb"):
        suffix = f"_{objective}"
        if body.endswith(suffix):
            return body[: -len(suffix)], objective
    return body, "unknown"


def read_metadata(path: Path) -> dict[str, str]:
    metadata_path = path / "metadata.tsv"
    metadata: dict[str, str] = {}
    if not metadata_path.is_file():
        return metadata
    for line in metadata_path.read_text().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            metadata[parts[0]] = parts[1]
    return metadata


def read_fid(path: Path) -> float | None:
    fid_path = path / "fid_results.txt"
    if not fid_path.is_file():
        return None
    text = fid_path.read_text().strip()
    if not text:
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        return None


def collect(parent: Path) -> list[Result]:
    results: list[Result] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        match = RESULT_RE.match(child.name)
        if not match:
            continue
        fid = read_fid(child)
        if fid is None:
            continue
        experiment = match.group("experiment")
        schedule, objective = parse_experiment(experiment)
        metadata = read_metadata(child)
        step_label = match.group("step")
        results.append(
            Result(
                experiment=experiment,
                schedule=schedule,
                objective=objective,
                step=int(step_label),
                step_label=step_label,
                num_samples=int(match.group("num_samples")),
                fid=fid,
                result_dir=child,
                model_path=metadata.get("model_path", ""),
            )
        )
    return sorted(
        results,
        key=lambda r: (
            SCHEDULE_ORDER.get(r.schedule, 99),
            OBJECTIVE_ORDER.get(r.objective, 99),
            r.experiment,
            r.num_samples,
            r.step,
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_fid(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def best_rows(results: list[Result]) -> list[dict[str, object]]:
    by_experiment: dict[tuple[str, int], list[Result]] = {}
    for result in results:
        by_experiment.setdefault((result.experiment, result.num_samples), []).append(result)

    rows: list[dict[str, object]] = []
    for (experiment, num_samples), group in sorted(
        by_experiment.items(),
        key=lambda item: (
            SCHEDULE_ORDER.get(parse_experiment(item[0][0])[0], 99),
            OBJECTIVE_ORDER.get(parse_experiment(item[0][0])[1], 99),
            item[0][0],
            item[0][1],
        ),
    ):
        schedule, objective = parse_experiment(experiment)
        best = min(group, key=lambda r: r.fid)
        final = max(group, key=lambda r: r.step)
        rows.append(
            {
                "experiment": experiment,
                "schedule": schedule,
                "objective": objective,
                "num_samples": num_samples,
                "best_step": best.step_label,
                "best_fid": f"{best.fid:.6f}",
                "final_step": final.step_label,
                "final_fid": f"{final.fid:.6f}",
                "final_minus_best": f"{final.fid - best.fid:.6f}",
                "best_result_dir": str(best.result_dir),
                "best_model_path": best.model_path,
            }
        )
    return rows


def write_markdown(parent: Path, results: list[Result], best: list[dict[str, object]]) -> Path:
    md_path = parent / "fid_by_checkpoint_summary.md"
    steps = sorted({r.step_label for r in results})
    by_experiment: dict[tuple[str, int], dict[str, Result]] = {}
    for result in results:
        by_experiment.setdefault((result.experiment, result.num_samples), {})[
            result.step_label
        ] = result

    lines: list[str] = []
    lines.append("# CIFAR-10 FID by Checkpoint")
    lines.append("")
    lines.append("Lower FID is better. `final_minus_best > 0` means an earlier checkpoint improved over the latest evaluated checkpoint.")
    lines.append("")
    lines.append("## Best Checkpoint")
    lines.append("")
    lines.append("| Experiment | Samples | Best Step | Best FID | Final Step | Final FID | Final - Best |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in best:
        lines.append(
            "| {experiment} | {num_samples} | {best_step} | {best_fid} | {final_step} | {final_fid} | {final_minus_best} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Trajectory")
    lines.append("")
    header = "| Experiment | Samples | " + " | ".join(steps) + " |"
    divider = "|---|---:|" + "|".join(["---:"] * len(steps)) + "|"
    lines.append(header)
    lines.append(divider)
    for (experiment, num_samples), step_map in sorted(
        by_experiment.items(),
        key=lambda item: (
            SCHEDULE_ORDER.get(parse_experiment(item[0][0])[0], 99),
            OBJECTIVE_ORDER.get(parse_experiment(item[0][0])[1], 99),
            item[0][0],
            item[0][1],
        ),
    ):
        cells = [fmt_fid(step_map[step].fid) if step in step_map else "" for step in steps]
        lines.append(f"| {experiment} | {num_samples} | " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_eval_dir", type=Path)
    args = parser.parse_args()

    parent = args.parent_eval_dir
    if not parent.is_dir():
        raise SystemExit(f"not a directory: {parent}")

    results = collect(parent)
    if not results:
        raise SystemExit(f"no completed FID results found under {parent}")

    trajectory_rows = [
        {
            "experiment": r.experiment,
            "schedule": r.schedule,
            "objective": r.objective,
            "step": r.step_label,
            "num_samples": r.num_samples,
            "fid": f"{r.fid:.6f}",
            "result_dir": str(r.result_dir),
            "model_path": r.model_path,
        }
        for r in results
    ]
    trajectory_fields = [
        "experiment",
        "schedule",
        "objective",
        "step",
        "num_samples",
        "fid",
        "result_dir",
        "model_path",
    ]
    write_csv(parent / "fid_by_checkpoint.csv", trajectory_rows, trajectory_fields)

    best = best_rows(results)
    best_fields = [
        "experiment",
        "schedule",
        "objective",
        "num_samples",
        "best_step",
        "best_fid",
        "final_step",
        "final_fid",
        "final_minus_best",
        "best_result_dir",
        "best_model_path",
    ]
    write_csv(parent / "best_fid_by_experiment.csv", best, best_fields)
    md_path = write_markdown(parent, results, best)

    print(f"Wrote {parent / 'fid_by_checkpoint.csv'}")
    print(f"Wrote {parent / 'best_fid_by_experiment.csv'}")
    print(f"Wrote {md_path}")
    print("")
    print("Best checkpoint by experiment:")
    for row in best:
        print(
            "{experiment:38s} n={num_samples:<6} best_step={best_step} "
            "best_fid={best_fid} final_fid={final_fid} final-best={final_minus_best}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
