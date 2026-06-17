#!/usr/bin/env python3
"""Sanity checks for endpoint-matched linear-in-alpha_bar schedules."""

import argparse

import numpy as np

from improved_diffusion.gaussian_diffusion import get_named_beta_schedule


def alpha_bar(betas):
    return np.cumprod(1.0 - betas)


def relerr(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def check_pair(timesteps, baseline_name, linabar_name):
    baseline = get_named_beta_schedule(baseline_name, timesteps)
    linabar = get_named_beta_schedule(linabar_name, timesteps)
    baseline_ab = alpha_bar(baseline)
    linabar_ab = alpha_bar(linabar)

    beta1_err = abs(linabar[0] - baseline[0])
    alpha_t_err = relerr(linabar_ab[-1], baseline_ab[-1])
    diffs = np.diff(linabar_ab)
    linearity = np.std(diffs) / max(abs(np.mean(diffs)), 1e-300)

    assert linabar.shape == (timesteps,)
    assert np.all(np.isfinite(linabar))
    assert np.all((linabar > 0) & (linabar < 1))
    assert beta1_err < 1e-14
    assert alpha_t_err < 1e-6
    assert linearity < 1e-8

    return {
        "timesteps": timesteps,
        "baseline": baseline_name,
        "schedule": linabar_name,
        "beta1": linabar[0],
        "target_alpha_bar_T": baseline_ab[-1],
        "actual_alpha_bar_T": linabar_ab[-1],
        "alpha_bar_T_relerr": alpha_t_err,
        "max_beta": linabar.max(),
        "last_beta": linabar[-1],
        "alpha_bar_linearity_cv": linearity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, nargs="+", default=[1000, 4000])
    args = parser.parse_args()

    rows = []
    for timesteps in args.timesteps:
        rows.append(check_pair(timesteps, "linear", "linabar_linear"))
        rows.append(check_pair(timesteps, "cosine", "linabar_cosine"))

    header = (
        "T baseline schedule beta1 target_alpha_bar_T actual_alpha_bar_T "
        "relerr max_beta last_beta alpha_bar_linearity_cv"
    )
    print(header)
    for row in rows:
        print(
            "{timesteps} {baseline} {schedule} "
            "{beta1:.10g} {target_alpha_bar_T:.10g} {actual_alpha_bar_T:.10g} "
            "{alpha_bar_T_relerr:.3e} {max_beta:.10g} {last_beta:.10g} "
            "{alpha_bar_linearity_cv:.3e}".format(**row)
        )


if __name__ == "__main__":
    main()
