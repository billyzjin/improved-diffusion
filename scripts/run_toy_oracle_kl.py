#!/usr/bin/env python3
"""Oracle Gaussianization KL for small Gaussian-mixture toy problems.

This estimates, for each diffusion step, the KL between the exact reverse
posterior L(S_t | X_t) and its moment-matched Gaussian approximation. The
implementation follows the formulas in geometric_odds_experiment_implementation_plan.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import logsumexp

try:
    from improved_diffusion.gaussian_diffusion import get_named_beta_schedule
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    def _fallback_betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas, dtype=np.float64)

    def _fallback_linear_alphabar_betas(num_diffusion_timesteps, beta1, alpha_bar_T):
        t = np.arange(1, num_diffusion_timesteps + 1, dtype=np.float64)
        alpha_bar_1 = 1.0 - beta1
        alpha_bar = alpha_bar_1 + (alpha_bar_T - alpha_bar_1) * (t - 1.0) / (num_diffusion_timesteps - 1.0)
        betas = np.empty(num_diffusion_timesteps, dtype=np.float64)
        betas[0] = beta1
        betas[1:] = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas

    def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, geometric_beta1=0.0, geometric_alpha_bar_T=0.0):
        """No-torch fallback matching improved_diffusion.gaussian_diffusion schedules."""
        t_steps = num_diffusion_timesteps
        if schedule_name == "linear":
            scale = 1000 / t_steps
            return np.linspace(scale * 0.0001, scale * 0.02, t_steps, dtype=np.float64)
        if schedule_name == "cosine":
            return _fallback_betas_for_alpha_bar(
                t_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        if schedule_name in ("geometric_linear", "geometric_cosine"):
            if schedule_name == "geometric_linear":
                baseline = get_named_beta_schedule("linear", t_steps)
            else:
                baseline = get_named_beta_schedule("cosine", t_steps)
            beta_1 = float(baseline[0])
            alpha_bar_T = float(np.prod(1.0 - baseline))
        elif schedule_name == "geometric":
            beta_1 = geometric_beta1
            alpha_bar_T = geometric_alpha_bar_T
        else:
            beta_1 = None
            alpha_bar_T = None

        if schedule_name.startswith("geometric"):
            z_1 = beta_1 / (1.0 - beta_1)
            z_T = (1.0 - alpha_bar_T) / alpha_bar_T
            q = (z_T / z_1) ** (1.0 / (t_steps - 1))
            betas = np.zeros(t_steps, dtype=np.float64)
            betas[0] = beta_1
            z_prev = z_1
            for i in range(1, t_steps):
                betas[i] = (q - 1) * z_prev / (1.0 + q * z_prev)
                z_prev *= q
            return np.minimum(betas, 0.999)

        if schedule_name in ("linabar_linear", "linabar_cosine"):
            baseline_name = "linear" if schedule_name == "linabar_linear" else "cosine"
            baseline = get_named_beta_schedule(baseline_name, t_steps)
            return _fallback_linear_alphabar_betas(t_steps, float(baseline[0]), float(np.prod(1.0 - baseline)))

        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


DEFAULT_SCHEDULES = (
    "linear",
    "cosine",
    "geometric_linear",
    "geometric_cosine",
    "linabar_linear",
    "linabar_cosine",
)

DEFAULT_TIMESTEPS = (50, 100, 250, 1000)


@dataclass(frozen=True)
class GMM:
    name: str
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])


def as_covs(scales: Iterable[float], dim: int) -> np.ndarray:
    return np.stack([(float(s) ** 2) * np.eye(dim, dtype=np.float64) for s in scales])


def build_distributions() -> dict[str, GMM]:
    out: dict[str, GMM] = {}
    out["gaussian_1d"] = GMM(
        name="gaussian_1d",
        weights=np.array([1.0], dtype=np.float64),
        means=np.array([[0.0]], dtype=np.float64),
        covs=as_covs([1.0], 1),
    )
    for m in (1.5, 3.0, 5.0):
        key = f"gmm_1d_symmetric_m{m:g}_sigma0.3"
        out[key] = GMM(
            name=key,
            weights=np.array([0.5, 0.5], dtype=np.float64),
            means=np.array([[-m], [m]], dtype=np.float64),
            covs=as_covs([0.3, 0.3], 1),
        )
    out["gmm_1d_skewed"] = GMM(
        name="gmm_1d_skewed",
        weights=np.array([0.8, 0.2], dtype=np.float64),
        means=np.array([[-1.0], [3.0]], dtype=np.float64),
        covs=as_covs([0.4, 0.7], 1),
    )
    for a in (2.0, 4.0):
        key = f"gmm_2d_grid_a{a:g}_sigma0.35"
        means = np.array(
            [[-a, -a], [-a, a], [a, -a], [a, a]],
            dtype=np.float64,
        )
        out[key] = GMM(
            name=key,
            weights=np.full(4, 0.25, dtype=np.float64),
            means=means,
            covs=as_covs([0.35] * 4, 2),
        )
    return out


def endpoint_family(schedule: str) -> str:
    if schedule.endswith("_linear") or schedule == "linear":
        return "linear"
    if schedule.endswith("_cosine") or schedule == "cosine":
        return "cosine"
    return "custom"


def bulk_ratios(betas: np.ndarray) -> np.ndarray:
    alpha_bar = np.cumprod(1.0 - betas)
    return betas[1:] / ((1.0 - betas[1:]) * (1.0 - alpha_bar[:-1]))


def psi(r: np.ndarray) -> np.ndarray:
    return 0.5 * r * r - r + np.log1p(r)


def cholesky_with_jitter(cov: np.ndarray, warnings: list[str], label: str) -> np.ndarray:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.eye(cov.shape[-1], dtype=np.float64)
        warnings.append(f"added 1e-10 diagonal jitter for {label}")
        return np.linalg.cholesky(cov + jitter)


def log_gaussian(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, warnings: list[str], label: str) -> np.ndarray:
    """Log N(samples; mean, cov).

    samples can be (..., d), mean can be broadcast to samples, and cov is (d, d).
    """
    d = cov.shape[0]
    chol = cholesky_with_jitter(cov, warnings, label)
    centered = samples - mean
    flat = centered.reshape(-1, d).T
    solved = np.linalg.solve(chol, flat)
    maha = np.sum(solved * solved, axis=0).reshape(centered.shape[:-1])
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + maha)


def weighted_choice_from_uniform(u: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, u, side="right")


def prepare_common_randoms(gmm: GMM, n_x: int, m_post: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "u_x": rng.random(n_x, dtype=np.float64),
        "eps_x": rng.standard_normal((n_x, gmm.dim), dtype=np.float64),
        "u_post": rng.random((n_x, m_post), dtype=np.float64),
        "eps_post": rng.standard_normal((n_x, m_post, gmm.dim), dtype=np.float64),
    }


def posterior_step_kl(
    gmm: GMM,
    beta_t: float,
    alpha_bar_t: float,
    alpha_bar_prev: float,
    randoms: dict[str, np.ndarray],
    x_chunk_size: int,
) -> tuple[float, float, list[str]]:
    """Estimate E_x KL(q(S_t|X_t=x) || Gaussian moment match)."""
    warnings: list[str] = []
    d = gmm.dim
    k = gmm.n_components
    n_x = randoms["u_x"].shape[0]
    m_post = randoms["u_post"].shape[1]

    eta_t = (1.0 - beta_t) * (1.0 - alpha_bar_prev)
    comp_means = math.sqrt(alpha_bar_t) * gmm.means
    comp_covs = alpha_bar_t * gmm.covs + eta_t * np.eye(d, dtype=np.float64)[None, :, :]
    obs_covs = comp_covs + beta_t * np.eye(d, dtype=np.float64)[None, :, :]

    # Draw X_t from its marginal mixture using common uniforms/normals.
    comp_x = weighted_choice_from_uniform(randoms["u_x"], gmm.weights)
    x = np.empty((n_x, d), dtype=np.float64)
    for c in range(k):
        mask = comp_x == c
        if not np.any(mask):
            continue
        chol = cholesky_with_jitter(obs_covs[c], warnings, f"obs_cov component {c}")
        x[mask] = comp_means[c] + randoms["eps_x"][mask] @ chol.T

    kl_by_x = np.empty(n_x, dtype=np.float64)
    log_weights_prior = np.log(gmm.weights)

    for start in range(0, n_x, x_chunk_size):
        stop = min(start + x_chunk_size, n_x)
        xb = x[start:stop]
        bsz = xb.shape[0]

        log_joint = np.empty((bsz, k), dtype=np.float64)
        post_means = np.empty((bsz, k, d), dtype=np.float64)
        post_covs = np.empty((k, d, d), dtype=np.float64)

        for c in range(k):
            log_joint[:, c] = log_weights_prior[c] + log_gaussian(
                xb,
                comp_means[c],
                obs_covs[c],
                warnings,
                f"obs logpdf component {c}",
            )
            gain = comp_covs[c] @ np.linalg.inv(obs_covs[c])
            post_means[:, c, :] = comp_means[c] + (xb - comp_means[c]) @ gain.T
            post_covs[c] = comp_covs[c] - gain @ comp_covs[c]
            post_covs[c] = 0.5 * (post_covs[c] + post_covs[c].T)

        log_norm = logsumexp(log_joint, axis=1)
        weights = np.exp(log_joint - log_norm[:, None])

        # Moment-matched Gaussian for every x in the chunk.
        mm_mean = np.sum(weights[:, :, None] * post_means, axis=1)
        mm_cov = np.empty((bsz, d, d), dtype=np.float64)
        for i in range(bsz):
            cov = np.zeros((d, d), dtype=np.float64)
            for c in range(k):
                diff = post_means[i, c] - mm_mean[i]
                cov += weights[i, c] * (post_covs[c] + np.outer(diff, diff))
            mm_cov[i] = 0.5 * (cov + cov.T)

        # Draw posterior samples using common uniforms/normals.
        u_post = randoms["u_post"][start:stop]
        eps_post = randoms["eps_post"][start:stop]
        cdf = np.cumsum(weights, axis=1)
        cdf[:, -1] = 1.0
        comp_post = np.sum(u_post[:, :, None] > cdf[:, None, :], axis=2)

        samples = np.empty((bsz, m_post, d), dtype=np.float64)
        for c in range(k):
            mask = comp_post == c
            if not np.any(mask):
                continue
            chol = cholesky_with_jitter(post_covs[c], warnings, f"posterior cov component {c}")
            base = post_means[:, c, :][:, None, :]
            samples[mask] = (base + eps_post @ chol.T)[mask]

        log_q_comp = np.empty((bsz, m_post, k), dtype=np.float64)
        for c in range(k):
            lp = log_gaussian(
                samples,
                post_means[:, c, :][:, None, :],
                post_covs[c],
                warnings,
                f"posterior sample logpdf component {c}",
            )
            log_q_comp[:, :, c] = np.log(weights[:, c])[:, None] + lp
        log_q = logsumexp(log_q_comp, axis=2)

        log_g = np.empty((bsz, m_post), dtype=np.float64)
        for i in range(bsz):
            log_g[i] = log_gaussian(
                samples[i],
                mm_mean[i],
                mm_cov[i],
                warnings,
                f"moment matched cov chunk_index {start + i}",
            )
        kl_by_x[start:stop] = np.mean(log_q - log_g, axis=1)

    return float(np.mean(kl_by_x)), float(np.std(kl_by_x, ddof=1) / math.sqrt(n_x)), warnings


def save_plots(base_dir: Path, all_summaries: list[dict], all_curves: dict[tuple[str, int, str], dict[str, np.ndarray]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on cluster environment
        (base_dir / "plot_warnings.txt").write_text(f"matplotlib unavailable; skipped plots: {exc}\n")
        return

    by_dist_t: dict[tuple[str, int], list[str]] = {}
    for dist, t_steps, sched in all_curves:
        by_dist_t.setdefault((dist, t_steps), []).append(sched)

    for (dist, t_steps), schedules in by_dist_t.items():
        out_dir = base_dir / dist / str(t_steps)
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        for sched in sorted(schedules):
            curve = all_curves[(dist, t_steps, sched)]
            x = np.arange(2, t_steps + 1)
            y = np.maximum(curve["kt_actual"][1:], 1e-300)
            plt.plot(x, y, label=sched)
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("K_t")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "kt_bulk.png", dpi=160)
        plt.close()

        plt.figure()
        for sched in sorted(schedules):
            curve = all_curves[(dist, t_steps, sched)]
            x = np.arange(2, t_steps + 1)
            y = np.maximum(curve["psi_bulk"], 1e-300)
            plt.plot(x, y, label=sched)
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("Psi(r_t)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "psi_bulk.png", dpi=160)
        plt.close()

        plt.figure()
        for sched in sorted(schedules):
            curve = all_curves[(dist, t_steps, sched)]
            plt.plot(np.arange(1, t_steps + 1), np.cumsum(curve["kt_actual"]), label=sched)
        plt.xlabel("t")
        plt.ylabel("cumulative K")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "cumulative_kt.png", dpi=160)
        plt.close()

        plt.figure()
        for sched in sorted(schedules):
            curve = all_curves[(dist, t_steps, sched)]
            plt.scatter(curve["psi_bulk"], curve["kt_actual"][1:], s=4, label=sched)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Psi(r_t)")
        plt.ylabel("K_t")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "kt_vs_psi.png", dpi=160)
        plt.close()

    by_dist_sched: dict[tuple[str, str], list[dict]] = {}
    for row in all_summaries:
        by_dist_sched.setdefault((row["distribution"], row["schedule"]), []).append(row)
    for dist in sorted({row["distribution"] for row in all_summaries}):
        out_dir = base_dir / dist
        plt.figure()
        for sched in sorted({s for d, s in by_dist_sched if d == dist}):
            rows = sorted(by_dist_sched[(dist, sched)], key=lambda r: r["T"])
            plt.plot([r["T"] for r in rows], [r["sum_K_bulk"] for r in rows], marker="o", label=sched)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("T")
        plt.ylabel("sum bulk K")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "sum_bulk_kt_by_T.png", dpi=160)
        plt.close()


def run_one(
    gmm: GMM,
    schedule: str,
    timesteps: int,
    n_x: int,
    m_post: int,
    seed: int,
    x_chunk_size: int,
    out_root: Path,
) -> tuple[dict, dict[str, np.ndarray]]:
    betas = get_named_beta_schedule(schedule, timesteps).astype(np.float64)
    alpha_bar = np.cumprod(1.0 - betas)
    alpha_bar_prev = np.concatenate([np.array([1.0], dtype=np.float64), alpha_bar[:-1]])
    r_bulk = bulk_ratios(betas)
    psi_bulk = psi(r_bulk)
    randoms = prepare_common_randoms(gmm, n_x, m_post, seed)

    kt_actual = np.empty(timesteps, dtype=np.float64)
    kt_actual_se = np.empty(timesteps, dtype=np.float64)
    all_warnings: list[str] = []

    for i in range(timesteps):
        kt, se, warnings = posterior_step_kl(
            gmm,
            beta_t=float(betas[i]),
            alpha_bar_t=float(alpha_bar[i]),
            alpha_bar_prev=float(alpha_bar_prev[i]),
            randoms=randoms,
            x_chunk_size=x_chunk_size,
        )
        kt_actual[i] = max(kt, 0.0)
        kt_actual_se[i] = se
        all_warnings.extend(warnings)
        if (i + 1) % max(1, timesteps // 10) == 0:
            print(f"{gmm.name} T={timesteps} {schedule}: completed {i + 1}/{timesteps}", flush=True)

    out_dir = out_root / gmm.name / str(timesteps) / schedule
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "betas.npy", betas)
    np.save(out_dir / "alpha_bar.npy", alpha_bar)
    np.save(out_dir / "r_bulk.npy", r_bulk)
    np.save(out_dir / "psi_bulk.npy", psi_bulk)
    np.save(out_dir / "kt_actual.npy", kt_actual)
    np.save(out_dir / "kt_actual_se.npy", kt_actual_se)

    bulk = kt_actual[1:]
    bulk_se = kt_actual_se[1:]
    summary = {
        "distribution": gmm.name,
        "dim": gmm.dim,
        "T": timesteps,
        "schedule": schedule,
        "endpoint_family": endpoint_family(schedule),
        "beta1": float(betas[0]),
        "alpha_bar_T": float(alpha_bar[-1]),
        "n_x": n_x,
        "m_post": m_post,
        "seed": seed,
        "sum_K_bulk": float(np.sum(bulk)),
        "sum_K_bulk_se": float(math.sqrt(np.sum(bulk_se * bulk_se))),
        "sum_Psi_bulk": float(np.sum(psi_bulk)),
        "max_K_bulk": float(np.max(bulk)) if bulk.size else float("nan"),
        "max_Psi_bulk": float(np.max(psi_bulk)) if psi_bulk.size else float("nan"),
        "first_step_K": float(kt_actual[0]),
        "first_step_K_se": float(kt_actual_se[0]),
        "warning_count": len(all_warnings),
        "warnings": sorted(set(all_warnings))[:25],
        "notes": "K_1 excluded from bulk comparisons",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary, {
        "betas": betas,
        "alpha_bar": alpha_bar,
        "r_bulk": r_bulk,
        "psi_bulk": psi_bulk,
        "kt_actual": kt_actual,
        "kt_actual_se": kt_actual_se,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="toy_results")
    parser.add_argument("--distributions", nargs="+", default=None)
    parser.add_argument("--schedules", nargs="+", default=list(DEFAULT_SCHEDULES))
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--n_x_1d", type=int, default=20_000)
    parser.add_argument("--m_post_1d", type=int, default=64)
    parser.add_argument("--n_x_2d", type=int, default=10_000)
    parser.add_argument("--m_post_2d", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--x_chunk_size", type=int, default=1024)
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    distributions = build_distributions()
    selected = args.distributions or list(distributions)
    unknown = sorted(set(selected) - set(distributions))
    if unknown:
        raise ValueError(f"Unknown distributions: {unknown}. Available: {sorted(distributions)}")

    summaries: list[dict] = []
    curves: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for dist_name in selected:
        gmm = distributions[dist_name]
        n_x = args.n_x_1d if gmm.dim == 1 else args.n_x_2d
        m_post = args.m_post_1d if gmm.dim == 1 else args.m_post_2d
        for timesteps in args.timesteps:
            for schedule in args.schedules:
                summary, curve = run_one(
                    gmm=gmm,
                    schedule=schedule,
                    timesteps=timesteps,
                    n_x=n_x,
                    m_post=m_post,
                    seed=args.seed,
                    x_chunk_size=args.x_chunk_size,
                    out_root=out_root,
                )
                summaries.append(summary)
                curves[(dist_name, timesteps, schedule)] = curve

    summary_dir = out_root / "summary_shards"
    summary_dir.mkdir(parents=True, exist_ok=True)
    dist_part = "_".join(selected)
    time_part = "_".join(str(t) for t in args.timesteps)
    sched_part = "_".join(args.schedules)
    shard_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"summary_{dist_part}_T{time_part}_{sched_part}.jsonl")
    summary_path = summary_dir / shard_name
    with summary_path.open("w") as f:
        for row in summaries:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if not args.skip_plots:
        save_plots(out_root, summaries, curves)

    print(f"Wrote {len(summaries)} toy oracle summaries to {summary_path}")


if __name__ == "__main__":
    main()
