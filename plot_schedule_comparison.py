"""
Visualize and compare noise schedules.

Plots beta_t, alpha_bar_t, z_t (odds), and r_t (noise-to-smoothing ratio)
for all schedules. Reproduces and extends Figure 1 from the paper.

Usage:
    python3 plot_schedule_comparison.py
    python3 plot_schedule_comparison.py --T 1000
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from improved_diffusion.gaussian_diffusion import get_named_beta_schedule


SCHEDULE_COLORS = {
    "linear": "#1f77b4",
    "cosine": "#ff7f0e",
    "ours": "#2ca02c",
    "geometric_linear": "#9467bd",
    "geometric_cosine": "#e377c2",
}

SCHEDULE_ORDER = ["linear", "cosine", "ours", "geometric_linear", "geometric_cosine"]


def compute_schedule_quantities(betas):
    """Compute derived quantities from a beta schedule."""
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    z = (1.0 - alpha_bar) / alpha_bar  # odds

    # r_t = beta_t / eta_t for t >= 2, where eta_t = (1-beta_t)(1-alpha_bar_{t-1})
    eta = (1.0 - betas[1:]) * (1.0 - alpha_bar[:-1])
    r = betas[1:] / eta

    return alpha_bar, z, r


def main():
    parser = argparse.ArgumentParser(description="Visualize noise schedules")
    parser.add_argument("--T", type=int, default=4000, help="Number of timesteps")
    parser.add_argument("--output_dir", default="plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    T = args.T

    # Compute all schedules
    schedules = {}
    for name in SCHEDULE_ORDER:
        betas = get_named_beta_schedule(name, T)
        alpha_bar, z, r = compute_schedule_quantities(betas)
        schedules[name] = {
            "betas": betas,
            "alpha_bar": alpha_bar,
            "z": z,
            "r": r,
        }

    timesteps = np.arange(T)
    timesteps_r = np.arange(1, T)  # r_t starts at t=2

    # --- 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): alpha_bar_t
    ax = axes[0, 0]
    for name in SCHEDULE_ORDER:
        ax.plot(timesteps, schedules[name]["alpha_bar"],
                label=name, color=SCHEDULE_COLORS[name], linewidth=1.2)
    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("$\\bar{\\alpha}_t$", fontsize=12)
    ax.set_title("(a) Signal retention $\\bar{\\alpha}_t$", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (b): z_t (odds) on log scale
    ax = axes[0, 1]
    for name in SCHEDULE_ORDER:
        ax.plot(timesteps, schedules[name]["z"],
                label=name, color=SCHEDULE_COLORS[name], linewidth=1.2)
    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("$z_t = (1-\\bar{\\alpha}_t)/\\bar{\\alpha}_t$", fontsize=12)
    ax.set_title("(b) Odds variable $z_t$ (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (c): r_t (noise-to-smoothing ratio)
    ax = axes[1, 0]
    for name in SCHEDULE_ORDER:
        ax.plot(timesteps_r, schedules[name]["r"],
                label=name, color=SCHEDULE_COLORS[name], linewidth=1.2)
    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("$r_t = \\beta_t / \\eta_t$", fontsize=12)
    ax.set_title("(c) Noise-to-smoothing ratio $r_t$", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (d): per-step cost Psi(r_t)
    ax = axes[1, 1]
    for name in SCHEDULE_ORDER:
        r = schedules[name]["r"]
        psi = 0.5 * r**2 - r + np.log(1.0 + r)
        ax.plot(timesteps_r, psi,
                label=name, color=SCHEDULE_COLORS[name], linewidth=1.2)
    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("$\\Psi(r_t)$", fontsize=12)
    ax.set_title("(d) Per-step cost $\\Psi(r_t)$", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Noise schedule comparison ($T={T}$)", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, f"schedule_comparison_T{T}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- Also plot beta_t separately (it's informative but clutters the 4-panel) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in SCHEDULE_ORDER:
        ax.plot(timesteps, schedules[name]["betas"],
                label=name, color=SCHEDULE_COLORS[name], linewidth=1.2)
    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("$\\beta_t$", fontsize=12)
    ax.set_title(f"Noise parameters $\\beta_t$ ($T={T}$)", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, f"beta_schedules_T{T}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Print summary statistics
    print(f"\nSchedule summary (T={T}):")
    print(f"{'schedule':<20} {'beta_1':>12} {'alpha_bar_T':>14} {'r_t range':>25} {'sum Psi(r_t)':>14}")
    for name in SCHEDULE_ORDER:
        s = schedules[name]
        r = s["r"]
        psi = 0.5 * r**2 - r + np.log(1.0 + r)
        print(f"{name:<20} {s['betas'][0]:>12.6e} {s['alpha_bar'][-1]:>14.6e} [{r.min():.6e}, {r.max():.6e}] {psi.sum():>14.6e}")


if __name__ == "__main__":
    main()
