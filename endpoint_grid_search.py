"""
Two-parameter grid search for optimal geometric schedule endpoints.

Estimates the total NLL as:
  NLL ~ L_0(beta_1) + bulk(beta_1, alpha_bar_T) + L_T(alpha_bar_T)

L_0 is computed via 1D Gauss-Hermite quadrature (no Monte Carlo, no training).
Key insight: for the naive decoder (mean = x_1/sqrt(1-beta_1)), the per-pixel
residual is N(0, beta_1/(1-beta_1)), so the expected discretized NLL is a 1D
integral over the residual, identical for all interior pixels.

Usage:
    python3 endpoint_grid_search.py
    python3 endpoint_grid_search.py --T 4000
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr  # standard normal CDF


# ---- Discretized Gaussian NLL via quadrature ----

def L0_per_pixel_interior(residual_std, decoder_sigma):
    """
    Expected discretized Gaussian NLL for an interior pixel.

    The decoder predicts mean = x_0 + delta, where delta ~ N(0, residual_std^2).
    The decoder outputs a Gaussian with std = decoder_sigma.
    The pixel bin is [x_0 - 1/255, x_0 + 1/255] (half-width = 1/255).

    Returns NLL in nats (per pixel).
    """
    half_bin = 1.0 / 255.0

    # Gauss-Hermite quadrature: integrate over delta ~ N(0, residual_std^2)
    # Change of variables: delta = residual_std * sqrt(2) * t
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    # nodes are for exp(-t^2), so delta = residual_std * sqrt(2) * t
    delta = residual_std * np.sqrt(2.0) * nodes  # [64]

    # For each delta, compute -log P(x_0 in bin | mean = x_0 + delta, sigma)
    upper = (half_bin - delta) / decoder_sigma
    lower = (-half_bin - delta) / decoder_sigma
    cdf_diff = np.clip(ndtr(upper) - ndtr(lower), 1e-12, None)
    nll = -np.log(cdf_diff)  # nats per pixel

    # Weighted average (Gauss-Hermite weights include 1/sqrt(pi) normalization)
    return np.sum(weights * nll) / np.sqrt(np.pi)


def L0_per_pixel_edge(residual_std, decoder_sigma, side="left"):
    """
    Expected NLL for an edge pixel (x_0 = -1 or x_0 = +1).

    For x_0 = -1: bin is (-inf, -1 + 1/255], so P = Phi((1/255 - delta)/sigma)
    For x_0 = +1: bin is [1 - 1/255, inf), so P = 1 - Phi((-1/255 - delta)/sigma)
    """
    half_bin = 1.0 / 255.0
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    delta = residual_std * np.sqrt(2.0) * nodes

    if side == "left":
        cdf_val = np.clip(ndtr((half_bin - delta) / decoder_sigma), 1e-12, None)
    else:
        cdf_val = np.clip(1.0 - ndtr((-half_bin - delta) / decoder_sigma), 1e-12, None)

    nll = -np.log(cdf_val)
    return np.sum(weights * nll) / np.sqrt(np.pi)


def compute_L0(beta1_values, sigma_candidates, frac_left=0.0, frac_right=0.0):
    """
    Compute L_0 in bits/dim for a range of beta_1 values (fully vectorized).

    For the naive decoder: residual_std = sqrt(beta_1 / (1 - beta_1)).
    Optimizes decoder_sigma over candidates for each beta_1.
    Also computes oracle L_0 (residual_std = 0).
    """
    frac_interior = 1.0 - frac_left - frac_right
    half_bin = 1.0 / 255.0

    # Gauss-Hermite nodes/weights
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    # weights sum to sqrt(pi); normalize: w_i / sqrt(pi) are the integration weights
    w = weights / np.sqrt(np.pi)  # [64]

    beta1 = np.asarray(beta1_values)  # [n_beta]
    sigma = np.asarray(sigma_candidates)  # [n_sigma]
    res_std = np.sqrt(beta1 / (1.0 - beta1))  # [n_beta]

    # delta values for each (beta1, node): [n_beta, 64]
    delta = res_std[:, None] * np.sqrt(2.0) * nodes[None, :]

    # For each (beta1, sigma, node), compute NLL
    # Shapes: delta[n_beta, 1, 64], sigma[1, n_sigma, 1]
    d3 = delta[:, None, :]      # [n_beta, 1, 64]
    s3 = sigma[None, :, None]   # [1, n_sigma, 64]

    # Interior pixels
    upper = (half_bin - d3) / s3   # [n_beta, n_sigma, 64]
    lower = (-half_bin - d3) / s3
    cdf_diff = np.clip(ndtr(upper) - ndtr(lower), 1e-12, None)
    nll_int = -np.log(cdf_diff)    # [n_beta, n_sigma, 64]
    # Weighted average over quadrature nodes → [n_beta, n_sigma]
    E_nll_int = np.einsum("bsn,n->bs", nll_int, w)

    # Edge pixels (left: x=-1)
    cdf_left = np.clip(ndtr((half_bin - d3) / s3), 1e-12, None)
    E_nll_left = np.einsum("bsn,n->bs", -np.log(cdf_left), w)

    # Edge pixels (right: x=+1)
    cdf_right = np.clip(1.0 - ndtr((-half_bin - d3) / s3), 1e-12, None)
    E_nll_right = np.einsum("bsn,n->bs", -np.log(cdf_right), w)

    # Combined NLL per pixel → [n_beta, n_sigma]
    nll_total = frac_interior * E_nll_int + frac_left * E_nll_left + frac_right * E_nll_right

    # Optimize over sigma for each beta1
    best_idx = np.argmin(nll_total, axis=1)  # [n_beta]
    L0_naive = nll_total[np.arange(len(beta1)), best_idx] / np.log(2.0)
    best_sigmas = sigma[best_idx]

    # Oracle: residual_std = 0, so delta = 0 for all nodes
    # Interior: nll = -log(Phi(half_bin/sigma) - Phi(-half_bin/sigma))
    cdf_oracle = np.clip(ndtr(half_bin / sigma) - ndtr(-half_bin / sigma), 1e-12, None)
    nll_oracle_int = -np.log(cdf_oracle)
    nll_oracle_left = -np.log(np.clip(ndtr(half_bin / sigma), 1e-12, None))
    nll_oracle_right = -np.log(np.clip(1.0 - ndtr(-half_bin / sigma), 1e-12, None))
    nll_oracle = frac_interior * nll_oracle_int + frac_left * nll_oracle_left + frac_right * nll_oracle_right
    best_oracle_idx = np.argmin(nll_oracle)
    L0_oracle = np.full(len(beta1), nll_oracle[best_oracle_idx] / np.log(2.0))

    return L0_naive, L0_oracle, best_sigmas


# ---- Analytical schedule quantities ----

def psi(r):
    """Psi(r) = r^2/2 - r + log(1+r)."""
    return 0.5 * r**2 - r + np.log1p(r)


def geometric_r(beta1, alpha_bar_T, T):
    """Constant r_t for geometric schedule."""
    z1 = beta1 / (1.0 - beta1)
    zT = (1.0 - alpha_bar_T) / alpha_bar_T
    q = (zT / z1) ** (1.0 / (T - 1))
    return q - 1.0


def bulk_sum_psi(beta1, alpha_bar_T, T):
    """(T-1) * Psi(r) for the geometric schedule."""
    r = geometric_r(beta1, alpha_bar_T, T)
    return (T - 1) * psi(r)


def prior_kl_bpd(alpha_bar_T, E_norm_sq, d):
    """ELBO prior term in bits/dim."""
    aT = alpha_bar_T
    kl_nats = 0.5 * (aT * E_norm_sq + d * (1.0 - aT) - d - d * np.log(1.0 - aT))
    return kl_nats / (d * np.log(2.0))


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4000)
    parser.add_argument("--output_dir", default="plots")
    args = parser.parse_args()

    T = args.T
    d = 3 * 32 * 32
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data statistics from CIFAR-10 ----
    print("Loading CIFAR-10 for data statistics...")
    from torchvision import datasets, transforms
    import torch

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1),
    ])
    cifar_test = datasets.CIFAR10(
        root="./cifar10_data", train=False, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(cifar_test, batch_size=10000, shuffle=False)
    x0_all, _ = next(iter(loader))
    x0_flat = x0_all.view(x0_all.shape[0], -1).numpy()

    E_norm_sq = (x0_flat ** 2).sum(axis=1).mean()

    # Fraction of edge pixels (x = -1 or x = +1)
    frac_left = (x0_flat < -0.999).mean()
    frac_right = (x0_flat > 0.999).mean()
    print(f"  d={d}, E[||x||^2]={E_norm_sq:.2f}")
    print(f"  Edge pixel fractions: left={frac_left:.4f}, right={frac_right:.4f}")

    # KL(P_0 || N(0,I)) estimate
    pixel_mean = x0_flat.mean(axis=0)
    pixel_var = x0_flat.var(axis=0)
    KL_gauss = 0.5 * ((pixel_var.sum() + (pixel_mean**2).sum()) - d - np.log(pixel_var).sum())
    print(f"  KL(N(mu,Sigma)||N(0,I)) = {KL_gauss:.1f} nats")

    # ---- L_0 computation ----
    print("\nComputing L_0 via quadrature...")
    beta1_fine = np.logspace(-7, -2.5, 200)
    sigma_candidates = np.logspace(-5, -0.5, 200)

    L0_naive, L0_oracle, best_sigmas = compute_L0(
        beta1_fine, sigma_candidates, frac_left, frac_right
    )
    print(f"  Done. L0 range: [{L0_naive.min():.4f}, {L0_naive.max():.4f}] bits/dim")

    # Empirical data points from trained models (hybrid objective)
    emp = {
        "geo_linear":  {"b1": 2.5e-5,       "L0": 1.010, "total": 3.012},
        "geo_cosine":  {"b1": 9.865819e-6,   "L0": 0.488, "total": 2.921},
        "linear":      {"b1": 2.5e-5,        "L0": 1.090},
        "cosine":      {"b1": 9.865819e-6,   "L0": 0.488},  # Note: the actual cosine beta_1 is the same
    }

    # ---- Bulk calibration from two geometric experiments ----
    aT_gl, aT_gc = 4.246652e-5, 1.517980e-10
    psi_gl = bulk_sum_psi(2.5e-5, aT_gl, T)
    psi_gc = bulk_sum_psi(9.865819e-6, aT_gc, T)
    bulk_gl = 3.012 - 1.010  # total - L0 (L_T negligible)
    bulk_gc = 2.921 - 0.488

    C1 = (bulk_gc - bulk_gl) / (psi_gc - psi_gl)
    C0 = bulk_gl - C1 * psi_gl
    print(f"\n  Bulk calibration: bulk ~ {C0:.3f} + {C1:.0f} * sum_Psi")

    # ---- 2D grid ----
    print("\nRunning 2D grid...")
    beta1_2d = np.logspace(-7, -2.5, 300)
    log_aT_2d = np.linspace(-12, -2, 300)
    aT_2d = 10.0 ** log_aT_2d

    B1, AT = np.meshgrid(beta1_2d, aT_2d, indexing="ij")
    Z1 = B1 / (1.0 - B1)
    ZT = (1.0 - AT) / AT
    Q = (ZT / Z1) ** (1.0 / (T - 1))
    R = Q - 1.0
    sum_psi_grid = (T - 1) * psi(R)
    bulk_grid = C0 + C1 * sum_psi_grid
    prior_grid = prior_kl_bpd(AT, E_norm_sq, d)

    L0_2d = np.interp(np.log10(beta1_2d), np.log10(beta1_fine), L0_naive)
    total_grid = L0_2d[:, None] + bulk_grid + prior_grid

    opt_idx = np.unravel_index(np.argmin(total_grid), total_grid.shape)
    opt_b1 = beta1_2d[opt_idx[0]]
    opt_aT = aT_2d[opt_idx[1]]
    opt_total = total_grid[opt_idx]
    opt_L0 = L0_2d[opt_idx[0]]
    opt_bulk = bulk_grid[opt_idx]
    opt_prior = prior_grid[opt_idx]
    opt_r = geometric_r(opt_b1, opt_aT, T)

    print(f"\n{'='*70}")
    print(f"OPTIMAL ENDPOINTS (T={T})")
    print(f"{'='*70}")
    print(f"  beta_1*      = {opt_b1:.6e}")
    print(f"  alpha_bar_T* = {opt_aT:.6e}")
    print(f"  r*           = {opt_r:.6e}")
    print(f"  L_0*         = {opt_L0:.4f} bits/dim")
    print(f"  bulk*        = {opt_bulk:.4f} bits/dim")
    print(f"  L_T*         = {opt_prior:.2e} bits/dim")
    print(f"  total*       = {opt_total:.4f} bits/dim")

    print(f"\nComparison with trained models (hybrid):")
    print(f"  {'schedule':<20} {'beta_1':>12} {'aT':>14} {'est_total':>10} {'actual':>8}")
    for name, b1, aT, actual in [
        ("geometric_linear", 2.5e-5, aT_gl, 3.012),
        ("geometric_cosine", 9.865819e-6, aT_gc, 2.921),
    ]:
        l0 = np.interp(np.log10(b1), np.log10(beta1_fine), L0_naive)
        bk = C0 + C1 * bulk_sum_psi(b1, aT, T)
        pr = prior_kl_bpd(aT, E_norm_sq, d)
        print(f"  {name:<20} {b1:>12.2e} {aT:>14.2e} {l0+bk+pr:>10.4f} {actual:>8.3f}")

    # ---- Plots ----
    print("\nGenerating plots...")

    # Plot 1: L_0 vs beta_1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(beta1_fine, L0_naive, "b-", linewidth=2, label="Naive decoder (no training)")
    ax.plot(beta1_fine, L0_oracle, "g--", linewidth=2, label="Oracle decoder (lower bound)")
    for name, info, marker, color in [
        ("geometric_linear", emp["geo_linear"], "o", "red"),
        ("geometric_cosine", emp["geo_cosine"], "o", "red"),
        ("linear baseline", emp["linear"], "^", "orange"),
        ("cosine baseline", emp["cosine"], "^", "orange"),
    ]:
        ax.scatter([info["b1"]], [info["L0"]], c=color, s=100, zorder=5,
                   marker=marker, label=f"Trained: {name}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta_1$", fontsize=13)
    ax.set_ylabel(r"$L_0$ (bits/dim)", fontsize=13)
    ax.set_title(r"Decoder NLL $L_0$ vs $\beta_1$ (CIFAR-10)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "L0_vs_beta1.png"), dpi=150)
    plt.close(fig)
    print("  Saved L0_vs_beta1.png")

    # Plot 2: 1D tradeoff — components vs beta_1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    aT_fixed = 1e-10
    bulk_1d = C0 + C1 * np.array([bulk_sum_psi(b, aT_fixed, T) for b in beta1_fine])
    prior_1d = prior_kl_bpd(aT_fixed, E_norm_sq, d)
    total_1d = L0_naive + bulk_1d + prior_1d
    opt_1d = np.argmin(total_1d)

    ax = axes[0]
    ax.plot(beta1_fine, L0_naive, "b-", linewidth=1.5, label=r"$L_0$ (decoder NLL)")
    ax.plot(beta1_fine, bulk_1d, "r-", linewidth=1.5, label=r"Bulk (calibrated)")
    ax.plot(beta1_fine, total_1d, "k-", linewidth=2.5, label="Total")
    ax.axvline(beta1_fine[opt_1d], color="purple", linestyle="--", alpha=0.6)
    ax.scatter([beta1_fine[opt_1d]], [total_1d[opt_1d]], c="purple", s=120, zorder=5,
               marker="*", label=rf"Opt: $\beta_1$={beta1_fine[opt_1d]:.2e}")
    for b1, name, c in [(2.5e-5, "geo_linear", "green"), (9.865819e-6, "geo_cosine", "orange")]:
        ax.axvline(b1, color=c, linestyle="--", alpha=0.4, label=rf"{name} $\beta_1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta_1$", fontsize=12)
    ax.set_ylabel("bits/dim", fontsize=12)
    ax.set_title(rf"NLL components ($\bar{{\alpha}}_T = 10^{{-10}}$, $T={T}$)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for log_aT, color in [(-3, "#1f77b4"), (-5, "#ff7f0e"), (-7, "#2ca02c"),
                            (-9, "#d62728"), (-11, "#9467bd")]:
        aT_val = 10.0 ** log_aT
        bk = C0 + C1 * np.array([bulk_sum_psi(b, aT_val, T) for b in beta1_fine])
        pr = prior_kl_bpd(aT_val, E_norm_sq, d)
        tot = L0_naive + bk + pr
        j = np.argmin(tot)
        ax.plot(beta1_fine, tot, color=color, linewidth=1.5,
                label=rf"$\bar{{\alpha}}_T=10^{{{log_aT}}}$, opt $\beta_1$={beta1_fine[j]:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta_1$", fontsize=12)
    ax.set_ylabel("Estimated total NLL (bits/dim)", fontsize=12)
    ax.set_title(f"Total NLL for different $\\bar{{\\alpha}}_T$ ($T={T}$)", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "endpoint_1d_search.png"), dpi=150)
    plt.close(fig)
    print("  Saved endpoint_1d_search.png")

    # Plot 3: 2D heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    lb1 = np.log10(beta1_2d)
    ext = [lb1[0], lb1[-1], log_aT_2d[0], log_aT_2d[-1]]

    ax = axes[0]
    L0_img = np.broadcast_to(L0_2d[:, None], total_grid.shape)
    im = ax.imshow(L0_img.T, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=ax, label="bits/dim")
    ax.set_xlabel(r"$\log_{10}(\beta_1)$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\bar{\alpha}_T)$", fontsize=12)
    ax.set_title(r"$L_0(\beta_1)$", fontsize=13)

    ax = axes[1]
    im = ax.imshow(bulk_grid.T, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=ax, label="bits/dim")
    ax.set_xlabel(r"$\log_{10}(\beta_1)$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\bar{\alpha}_T)$", fontsize=12)
    ax.set_title("Bulk (calibrated)", fontsize=13)

    ax = axes[2]
    vmin = opt_total - 0.3
    vmax = min(opt_total + 1.5, np.percentile(total_grid, 90))
    im = ax.imshow(total_grid.T, origin="lower", aspect="auto", extent=ext,
                   cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="bits/dim")
    ax.plot(np.log10(opt_b1), np.log10(opt_aT), "r*", markersize=15,
            label=f"Opt: ({opt_b1:.1e}, {opt_aT:.1e})")
    ax.plot(np.log10(2.5e-5), np.log10(aT_gl), "wo", markersize=8,
            markeredgecolor="red", label="geo_linear")
    ax.plot(np.log10(9.865819e-6), np.log10(aT_gc), "w^", markersize=8,
            markeredgecolor="red", label="geo_cosine")
    ax.set_xlabel(r"$\log_{10}(\beta_1)$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\bar{\alpha}_T)$", fontsize=12)
    ax.set_title("Estimated total NLL", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Endpoint optimization for geometric schedule ($T={T}$)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "endpoint_2d_search.png"), dpi=150)
    plt.close(fig)
    print("  Saved endpoint_2d_search.png")

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"Optimal beta_1 for various alpha_bar_T (T={T})")
    print(f"{'='*70}")
    print(f"{'log10(aT)':>10} {'opt_beta1':>14} {'opt_r':>12} "
          f"{'L0':>8} {'bulk':>8} {'L_T':>10} {'total':>8}")
    for log_aT in [-3, -4, -5, -6, -7, -8, -9, -10, -11, -12]:
        aT_val = 10.0 ** log_aT
        bk = C0 + C1 * np.array([bulk_sum_psi(b, aT_val, T) for b in beta1_fine])
        pr = prior_kl_bpd(aT_val, E_norm_sq, d)
        tot = L0_naive + bk + pr
        j = np.argmin(tot)
        b = beta1_fine[j]
        r = geometric_r(b, aT_val, T)
        print(f"{log_aT:>10} {b:>14.4e} {r:>12.4e} "
              f"{L0_naive[j]:>8.4f} {bk[j]:>8.4f} {pr:>10.2e} {tot[j]:>8.4f}")


if __name__ == "__main__":
    main()
