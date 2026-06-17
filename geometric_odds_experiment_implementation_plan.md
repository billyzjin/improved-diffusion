# Implementation Plan: Geometric-Odds Diffusion Schedule Experiments

## Purpose

This document is an implementation brief for an agent who will run the next round of experiments for the geometric-odds diffusion schedule paper.

The experiments should support the following core claim:

> For finite-step DDPM-style diffusion, the geometric-odds schedule is the natural endpoint-matched schedule for reducing bulk Gaussianization error. This mechanism should be validated directly on oracle toy problems, then tested in trained image models through likelihood, sample-quality metrics, reduced-NFE sampling, and one additional natural-image dataset.

The selected experiment set is:

1. **Oracle Gaussianization-KL toy experiments**.
2. **Additional metrics**: CMMD, KID, and fidelity/diversity diagnostics such as precision/recall or density/coverage.
3. **NFE / respacing sweep**.
4. **One additional dataset**: start with **FFHQ-64**.
5. **Linear-in-$\bar\alpha$ (linear-decay) schedule** on the existing setups (four datasets x three objectives), as the §6.3 cumulant-optimal counterpart to the geometric-odds schedule. See **Experiment 5**.

The matched-endpoint image comparison (linear/cosine versus their geometric-odds counterparts, on the four datasets x three objectives) is **already complete**: those NLL/FID/TV results are in `evaluation_results_full.tsv` and Table 1 of the paper, so it is **not** re-run here. It only needs curation for the write-up (main table = `L_hybrid`/`L_vlb`, `L_simple` to an appendix), and the new metrics in item 2 can be layered onto its existing samples (see Phase 3).

Do **not** implement the stitched geometric/linear-decay **hybrid** schedule for now. Note the distinction: the standalone linear-in-$\bar\alpha$ schedule (item 5) **is** in scope this round; the hybrid that combines it with geometric odds is **not**. Also do **not** prioritize COCO, endpoint-grid tuning, class-conditional CIFAR-10, or multiple training seeds in this round.

---

## High-level experimental story

The final experiments section should make the following points, in this order:

1. **Mechanism validation:** On synthetic distributions where the true reverse posterior is computable, geometric odds reduces and equalizes the actual finite-step Gaussianization KL better than linear/cosine baselines with matched endpoints.
2. **Image-model validation:** On trained DDPM-style models, geometric odds gives robust NLL / bits-dim improvements under the well-specified objectives `L_hybrid` and `L_vlb`.
3. **Metric robustness:** Because FID is mixed on larger datasets, evaluate CMMD, KID, and precision/recall or density/coverage to determine whether geometric odds changes fidelity, diversity, or semantic distribution matching.
4. **Practical sampling:** Test whether the schedule remains competitive under reduced numbers of function evaluations, even though the main theorem concerns the full finite reverse chain.
5. **Dataset generality:** Add FFHQ-64 as the first additional natural-image dataset because it is unconditional and avoids the confounds of class/text conditioning.

---

## Important theory/schedule definitions

### DDPM notation

For a discrete schedule with `T` steps,

X_t = \sqrt{1-\beta_t}X_{t-1}+\sqrt{\beta_t}Z_t,
\qquad
\bar\alpha_t=\prod_{i=1}^t(1-\beta_i).

The noise-to-signal odds are

z_t=\frac{1-\bar\alpha_t}{\bar\alpha_t}.

For bulk steps `t >= 2`, define



\eta_t=(1-\beta_t)(1-\bar\alpha_{t-1}),
\qquad
r_t=\frac{\beta_t}{\eta_t}

\frac{\beta_t}{(1-\beta_t)(1-\bar\alpha_{t-1})}.

The universal per-step proxy is

\Psi(r)=\frac{r^2}{2}-r+\log(1+r).

Use `log1p(r)` in code for numerical stability.

### Geometric-odds schedule

Given:

- number of steps `T`,
- first-step noise `beta_1`,
- terminal signal level `alpha_bar_T`,

compute:

z_1=\frac{\beta_1}{1-\beta_1},
\qquad
z_T=\frac{1-\bar\alpha_T}{\bar\alpha_T},

q=\left(\frac{z_T}{z_1}\right)^{1/(T-1)},
\qquad
z_t=z_1q^{t-1}, \quad t=1,\ldots,T.

Then

\bar\alpha_t=\frac{1}{1+z_t},

and the correct beta formula is

\beta_1 \text{ fixed},
\qquad
\beta_t=\frac{(q-1)z_{t-1}}{1+qz_{t-1}},
\quad t=2,\ldots,T.

Under this schedule, all bulk ratios satisfy

r_2=r_3=\cdots=r_T=q-1.

### Implementation status (already in the repo)

The geometric-odds schedule is **already implemented** in `improved_diffusion/gaussian_diffusion.py` and wired through the training/eval CLI -- do not reimplement it:

- `get_named_beta_schedule(num_steps, "geometric", geometric_beta1=b1, geometric_alpha_bar_T=aT)` -- generic geometric-odds with user-specified endpoints.
- `"geometric_linear"` / `"geometric_cosine"` -- the same schedule with endpoints auto-matched to the DDPM linear / cosine baselines (no endpoints to pass).
- CLI flags `--noise_schedule`, `--geometric_beta1`, `--geometric_alpha_bar_T` thread through `script_util.py`.

Matched-endpoint runs therefore need no new schedule code: use `--noise_schedule geometric_linear` / `geometric_cosine`, or `--noise_schedule geometric` with explicit endpoints.

For toy / diagnostic analysis only (Experiment 1 and the Phase 0 plots) you need the per-step ratio `r_t` and proxy `Psi(r_t)`. `psi(r)` already exists in `endpoint_grid_search.py`; `r_t` is a one-liner on a `betas` array:

```python
def bulk_ratios(betas):  # r_t for t = 2..T
    ab = np.cumprod(1.0 - betas)
    return betas[1:] / ((1.0 - betas[1:]) * (1.0 - ab[:-1]))
```

Sanity-check the existing schedule (not a reimplementation) before running:

```python
from improved_diffusion.gaussian_diffusion import get_named_beta_schedule
betas = get_named_beta_schedule("geometric_linear", 4000)
r = bulk_ratios(betas)
assert np.all((betas > 0) & (betas < 1))
assert np.std(r) / np.mean(r) < 1e-6   # geometric odds => constant r_t by construction
```

### Linear-in-$\bar\alpha$ (linear-decay) schedule

**Naming warning.** This is the schedule whose cumulative product $\bar\alpha_t$ is linear in $t$. It is **not** the DDPM `linear` schedule, which is linear in $\beta_t$. To avoid confusion, name it `linabar` in filenames and "linear-$\bar\alpha$" (or "linear-decay", following the paper) in text. Never reuse the bare name `linear` for it.

**Motivation.** This is the schedule the paper's §6.3 (`sec:t_dependent_prefactor`, eq. `linear_decay_optimal`) derives as the minimizer of the *refined*, cumulant-aware bulk objective $\sum_t r_t^3/z_t^3$, whereas the geometric-odds schedule minimizes the *universal* objective $\sum_t \Psi(r_t)$. Both bounds are proven; they disagree on the optimal shape. Testing linear-in-$\bar\alpha$ directly probes which proven bound better predicts trained-model behavior.

**Paper form (one parameter, $\bar\alpha_T$ only).**

\bar\alpha_t = 1 - (1-\bar\alpha_T)\frac{t}{T},
\qquad
\beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}.

This forces $\beta_1 = (1-\bar\alpha_T)/T$, so it matches $\bar\alpha_T$ but **not** $\beta_1$. This pure form is **already in the repo** as `ours_v2` (`gaussian_diffusion.py`: `beta_t = 1/(T-t+1)`, with `beta` clipped at `0.999`) -- reuse it rather than reimplementing.

**Matched-endpoint form (recommended for comparison).** To compare apples-to-apples against the baselines and the geometric schedule (which match *both* endpoints), interpolate $\bar\alpha_t$ linearly from $\bar\alpha_1 = 1-\beta_1$ (at $t=1$) to $\bar\alpha_T$ (at $t=T$). This matches both endpoints, so only the schedule shape differs. Produce two matched variants, `linabar-linear` and `linabar-cosine`, inheriting endpoints from the linear and cosine baselines respectively, exactly as `geometric-linear` / `geometric-cosine` do. Implement these as two new branches in `get_named_beta_schedule` -- `linabar_linear` and `linabar_cosine` -- right next to the existing `geometric_linear` / `geometric_cosine` cases; the reference NumPy below is the branch body (the only genuinely new schedule code this round).

**Late-step behavior / clipping.** Because $\bar\alpha_t$ decreases linearly, this schedule front-loads signal retention and dumps most of the noise into the final steps. The final $\beta_t$ is governed by $\bar\alpha_T$: for the linear endpoints ($\bar\alpha_T = 4\times10^{-5}$) the last $\beta_t \approx 0.86$ (large but fine); for the cosine endpoints ($\bar\alpha_T \approx 2\times10^{-9}$) the last $\beta_t \approx 0.99999$, an almost-total-noise step. For the primary matched-endpoint experiment, **do not clip** `linabar-cosine`: clipping at `0.999` changes the terminal $\bar\alpha_T$ by orders of magnitude and invalidates the fixed-endpoint comparison. Instead, match the cosine schedule's actual $\bar\alpha_T$ directly, record the resulting `max_beta`, and run a short smoke test before launching the full slate. A clipped `linabar-cosine-clipped` variant may be run as a diagnostic only, but it must be labeled separately and excluded from the main matched-endpoint tables.

```python
def linear_alphabar_betas(T: int, beta1: float, alpha_bar_T: float) -> np.ndarray:
    """Matched-endpoint linear-in-alpha_bar schedule.

    alpha_bar interpolates linearly from alpha_bar_1 = 1 - beta1 (at t=1)
    to alpha_bar_T (at t=T), matching BOTH beta1 and alpha_bar_T so it is
    directly comparable to the geometric schedule under the same endpoints.

    NOTE: 'linear in alpha_bar', NOT the DDPM 'linear' (which is linear in beta).
    """
    if T < 2:
        raise ValueError("T must be at least 2")
    if not (0.0 < beta1 < 1.0):
        raise ValueError("beta1 must be in (0, 1)")
    if not (0.0 < alpha_bar_T < 1.0):
        raise ValueError("alpha_bar_T must be in (0, 1)")

    a1 = 1.0 - beta1
    if not (a1 > alpha_bar_T):
        raise ValueError("Need alpha_bar_1 > alpha_bar_T so the chain adds noise")

    t = np.arange(1, T + 1, dtype=np.float64)                    # t = 1..T
    alpha_bar = a1 + (alpha_bar_T - a1) * (t - 1.0) / (T - 1.0)   # a1 at t=1, alpha_bar_T at t=T

    betas = np.empty(T, dtype=np.float64)
    betas[0] = beta1
    betas[1:] = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas
```

(The pure one-parameter form is `ours_v2` in the repo; see above. Only `linear_alphabar_betas` -- the endpoint-matched variant -- is new.)

Tests:

```python
betas = linear_alphabar_betas(4000, 1e-4, 4e-5)
ab = np.cumprod(1.0 - betas)
assert betas.shape == (4000,)
assert np.all(betas > 0) and np.all(betas < 1)
assert abs(betas[0] - 1e-4) < 1e-12
assert abs(ab[-1] - 4e-5) / 4e-5 < 1e-8
# alpha_bar must be linear in t => constant decrements:
d = np.diff(ab)
assert np.std(d) / abs(np.mean(d)) < 1e-6
```

---

## Repository / artifact structure

Use or adapt this structure.

```text
experiments/
  # Schedules already live in improved_diffusion/gaussian_diffusion.py
  # (get_named_beta_schedule); do NOT build a parallel schedules module.
  # Add the new linabar_linear / linabar_cosine branches there.
  schedules/
    plot_schedules.py        # diagnostics only (alpha_bar_t, z_t, r_t, Psi)
  toy_oracle_kl/
    gmm_posteriors.py
    estimate_gaussianization_kl.py
    run_toy_suite.py
    configs/
      gaussian_1d.yaml
      gmm_1d_symmetric.yaml
      gmm_1d_skewed.yaml
      gmm_2d_grid.yaml
      boundary_proxy_1d.yaml
  image_runs/
    configs/
      cifar10_hybrid_linear.yaml
      cifar10_hybrid_glinear.yaml
      cifar10_hybrid_cosine.yaml
      cifar10_hybrid_gcosine.yaml
      ...
    train_or_reuse.py
    sample_full_chain.py
    evaluate_nll.py
  metrics/
    compute_fid.py
    compute_cmmd.py
    compute_kid.py
    compute_precision_recall.py
    compute_density_coverage.py
    aggregate_metrics.py
  nfe_sweep/
    make_respaced_diffusion.py
    sample_nfe_sweep.py
    evaluate_nfe_sweep.py
  ffhq64/
    prepare_ffhq64.py
    configs/
      ffhq64_hybrid_linear.yaml
      ffhq64_hybrid_glinear.yaml
      ffhq64_hybrid_cosine.yaml
      ffhq64_hybrid_gcosine.yaml
  reports/
    tables/
    figures/
    metrics_json/
```

Every run should produce:

```text
outputs/<dataset>/<objective>/<schedule_name>/
  config.yaml
  betas.npy
  alpha_bar.npy
  schedule_diagnostics.json
  checkpoint.pt or checkpoint_path.txt
  samples/
    samples_50k.npz
    sample_grid.png
  metrics/
    nll.json
    fid.json
    cmmd.json
    kid.json
    prdc_or_pr.json
```

---

# Experiment 1: Oracle Gaussianization-KL toy experiments

## Goal

Directly estimate the object studied by the theory:

K_t=\mathbb E_{X_t}
D_{\mathrm{KL}}
\left(
\mathcal L(X_{t-1}\mid X_t)

\text{best Gaussian approximation}
\right),

or equivalently after the affine rescaling used in the proof,

K_t=\mathbb E_{X_t}
D_{\mathrm{KL}}
\left(
\mathcal L(S_t\mid X_t)

\text{moment-matched Gaussian}
\right),
\qquad
S_t=\sqrt{1-\beta_t}X_{t-1}.

This avoids neural-network training and directly tests whether the geometric-odds schedule reduces the actual finite-step Gaussianization error.

## Distributions to implement

Implement these first:

### 1D Gaussian sanity check

```text
X0 ~ N(0, 1)
```

Expected result: actual Gaussianization KL is numerically zero for every step and every schedule.

### 1D symmetric Gaussian mixture

```text
X0 ~ 0.5 N(-m, sigma_x^2) + 0.5 N(m, sigma_x^2)
```

Use:

```text
m in {1.5, 3.0, 5.0}
sigma_x = 0.3
```

Purpose: multimodality without skewness.

### 1D skewed Gaussian mixture

```text
X0 ~ 0.8 N(-1, 0.4^2) + 0.2 N(3, 0.7^2)
```

Purpose: nonzero skewness / third cumulant.

### 2D four-mode grid mixture

```text
X0 ~ uniform mixture over means {(-a,-a), (-a,a), (a,-a), (a,a)}
component covariance = sigma_x^2 I
```

Use:

```text
a in {2, 4}
sigma_x = 0.35
```

Purpose: simple multimodal 2D problem with visualizable posterior geometry.

### Optional boundary-layer proxy

Use either:

```text
X0 ~ Uniform[-1, 1]
```

with 1D quadrature, or approximate compact support by a fine grid mixture of narrow Gaussians.

Purpose: show why the first step is delicate. Keep this optional and do not let it delay the GMM suite.

## Schedules to compare

For each toy distribution and each `T`, compare:

```text
linear
geometric-linear
cosine
geometric-cosine
```

Use matched endpoints:

- `geometric-linear` inherits `beta_1` and `alpha_bar_T` from the linear schedule.
- `geometric-cosine` inherits `beta_1` and `alpha_bar_T` from the cosine schedule.

Run at:

```text
T in {50, 100, 250, 1000}
```

Use `T=4000` only if the estimator is fast enough.

## How to compute the actual KL for GMM data

Assume

X_0\mid C=c\sim N(\mu_c,\Sigma_c),
\qquad
P(C=c)=\pi_c.

For diffusion step `t`, define

S_t=\sqrt{\bar\alpha_t}X_0+\sqrt{\eta_t}W,
\qquad
X_t=S_t+\sqrt{\beta_t}Z,

where

\eta_t=(1-\beta_t)(1-\bar\alpha_{t-1}).

Conditional on component `c`,

S_t\mid C=c\sim N(m_c, P_c),

with

m_c=\sqrt{\bar\alpha_t}\mu_c,
\qquad
P_c=\bar\alpha_t\Sigma_c+\eta_t I.

The observation is

X_t\mid C=c\sim N(m_c, P_c+\beta_t I).

Given `X_t = x`, posterior component weights are

w_c(x)\propto \pi_cN(x;m_c,P_c+\beta_t I).

The component posterior is Gaussian:

S_t\mid X_t=x,C=c\sim N(\tilde m_c(x),\tilde P_c),

where

K_c=P_c(P_c+\beta_t I)^{-1},

\tilde m_c(x)=m_c+K_c(x-m_c),

\tilde P_c=P_c-P_c(P_c+\beta_t I)^{-1}P_c.

Thus the exact posterior is a mixture:

q_t(s\mid x)=\sum_c w_c(x)N(s;\tilde m_c(x),\tilde P_c).

The KL-best Gaussian is the moment-matched Gaussian:

g_t^\star(s\mid x)=N(m(x),C(x)),

with

m(x)=\sum_c w_c(x)\tilde m_c(x),

C(x)=\sum_c w_c(x)\left[\tilde P_c+
(\tilde m_c(x)-m(x))(\tilde m_c(x)-m(x))^\top\right].

Then estimate



D_{\mathrm{KL}}(q_t(\cdot\mid x)g_t^\star(\cdot\mid x))

\mathbb E_{s\sim q_t(\cdot\mid x)}
\left[\log q_t(s\mid x)-\log g_t^\star(s\mid x)\right].

Use Monte Carlo over both `x` and posterior samples `s`.

## Recommended estimator settings

Start with:

```text
N_y = 20_000 marginal samples for 1D
M_post = 64 posterior samples per y for 1D

N_y = 10_000 marginal samples for 2D
M_post = 32 posterior samples per y for 2D
```

Use common random numbers across schedules whenever practical:

- same component draws,
- same standard-normal draws,
- same posterior-sampling seeds.

This reduces estimator noise when comparing schedules.

## Numerical stability requirements

Use:

- `float64` for all toy oracle computations;
- Cholesky decompositions for Gaussian log densities;
- `scipy.special.logsumexp` for mixture log densities;
- diagonal jitter `1e-10 I` only if Cholesky fails, and log a warning if used.

## Outputs

For each distribution / schedule / `T`, save:

```text
toy_results/<distribution>/<T>/<schedule>/
  betas.npy
  alpha_bar.npy
  r_bulk.npy
  psi_bulk.npy
  kt_actual.npy
  kt_actual_se.npy
  summary.json
```

`summary.json` should contain:

```json
{
  "distribution": "gmm_1d_symmetric_m3_sigma0.3",
  "T": 1000,
  "schedule": "geometric_linear",
  "endpoint_family": "linear",
  "beta1": 0.0001,
  "alpha_bar_T": 0.00004,
  "sum_K_bulk": 0.0,
  "sum_K_bulk_se": 0.0,
  "sum_Psi_bulk": 0.0,
  "max_K_bulk": 0.0,
  "max_Psi_bulk": 0.0,
  "first_step_K": 0.0,
  "notes": "K_1 excluded from bulk comparisons"
}
```

## Plots

Generate these plots:

1. `K_t` versus `t` for all four schedules, log y-axis, bulk steps only.
2. `Psi(r_t)` versus `t` for all four schedules, log y-axis, bulk steps only.
3. Cumulative `sum_{s<=t} K_s` versus `t`.
4. Total bulk `sum K_t` versus `T`, log-log if possible.
5. Scatter plot: actual `K_t` versus proxy `Psi(r_t)`.
6. For 2D GMM, optional posterior contour examples at representative timesteps.

## Success criteria

The experiment is successful if:

- Gaussian sanity check gives `K_t` approximately zero.
- Geometric schedules make `r_t` and `Psi(r_t)` flat by construction.
- Actual `K_t` is more evenly distributed under geometric schedules than under matched baselines.
- Total bulk `sum K_t` is lower or comparable for geometric schedules, especially at large `T`.
- `K_1` is reported separately and not included in the bulk theorem validation.

---

# Experiment 2: Add CMMD, KID, and fidelity/diversity diagnostics

## Goal

The current FID pattern is mixed on larger datasets. Add metrics that help distinguish:

- perceptual fidelity,
- semantic distribution matching,
- diversity / coverage,
- memorization or mode dropping.

## Metrics to implement

### FID

Keep existing FID pipeline for continuity.

Requirements:

- Use the same preprocessing for real and generated images.
- Use training-set real statistics if that is the existing convention.
- Clearly label sample count, e.g. `FID-50K` or `FID-10K`.

### CMMD

Compute MMD in CLIP feature space.

Recommended implementation path:

- Use an existing reliable implementation if available.
- Otherwise implement:
  1. load a fixed CLIP image encoder,
  2. extract features for real and generated images,
  3. normalize features according to the chosen CMMD convention,
  4. compute squared MMD using an RBF kernel,
  5. use fixed kernel bandwidth or median heuristic, but record which one was used.

Record:

```json
{
  "metric": "CMMD",
  "clip_model": "<model name>",
  "feature_normalization": "<none/l2/etc>",
  "kernel": "rbf",
  "bandwidth": "<value or median heuristic>",
  "num_real": 50000,
  "num_generated": 50000,
  "value": 0.0
}
```

Use CMMD mainly for CIFAR-10, ImageNet-64, and FFHQ-64. It is less important for MNIST/Fashion-MNIST.

### KID

Compute KID in Inception feature space.

Recommended settings:

```text
feature extractor: InceptionV3 pool3
kernel: polynomial kernel, standard KID convention
subsets: 100
subset size: 1000 if enough samples are available
report: mean KID and standard error over subsets
```

KID is cheap once Inception features are cached.

### Precision/Recall or Density/Coverage

Implement one of these families; density/coverage is preferred if a reliable implementation is available.

Feature space:

```text
InceptionV3 pool3 features
```

Default nearest-neighbor parameter:

```text
k = 5
```

Record:

```json
{
  "metric": "density_coverage",
  "feature_extractor": "inception_v3_pool3",
  "k": 5,
  "density": 0.0,
  "coverage": 0.0
}
```

or

```json
{
  "metric": "precision_recall",
  "feature_extractor": "inception_v3_pool3",
  "k": 5,
  "precision": 0.0,
  "recall": 0.0
}
```

## Feature caching

Cache features to avoid recomputing.

```text
features/<dataset>/real/inception_pool3.npy
features/<dataset>/real/clip.npy
features/<dataset>/<objective>/<schedule>/generated/inception_pool3.npy
features/<dataset>/<objective>/<schedule>/generated/clip.npy
```

Store metadata:

```text
features/<...>/metadata.json
```

Metadata must include:

- dataset name,
- split,
- number of images,
- preprocessing transform,
- feature model,
- model version / checkpoint identifier,
- image range convention before preprocessing.

## Aggregated metric table

For each dataset and objective, generate a table:

```text
schedule | NLL | FID | CMMD | KID | precision | recall | density | coverage
```

If both precision/recall and density/coverage are implemented, include both. Otherwise include one.

## Success criteria

This experiment is successful if it clarifies whether geometric odds:

- improves NLL but hurts perceptual fidelity,
- improves distributional matching beyond FID,
- changes diversity/coverage,
- behaves differently under linear endpoints versus cosine endpoints.

---

# Experiment 3: NFE / respacing sweep

## Goal

Test whether the schedule remains useful under practical reduced-step sampling.

This is not the main theorem validation, because the theorem concerns the full finite reverse chain. Treat this as a practical diagnostic.

## First dataset / objective

Run first on:

```text
Dataset: CIFAR-10
Objective: L_hybrid
```

If results are promising and compute allows, repeat for:

```text
Objective: L_vlb
Dataset: FFHQ-64 after the FFHQ checkpoints exist
```

## Schedules

Use:

```text
linear
geometric-linear
cosine
geometric-cosine
```

Use the already-trained full-chain models. Do not retrain for each NFE.

## NFE values

Use:

```text
NFE in {25, 50, 100, 250, 1000, 4000}
```

`4000` is the full-chain reference.

## Respacing rule

Primary rule: uniform respacing in original timestep index.

For a selected sequence

```text
0 = s_0 < s_1 < ... < s_n = T
```

construct the respaced diffusion with cumulative alpha values inherited from the original schedule:

\bar\alpha'*k = \bar\alpha*{s_k},
\qquad
\beta'*k=1-\frac{\bar\alpha'*{k}}{\bar\alpha'_{k-1}}.

Use the trained model at the corresponding original timestep embedding `s_k`.

If using OpenAI Improved-Diffusion, this is essentially the `SpacedDiffusion` construction. Make sure each schedule uses the same respacing rule.

Optional sensitivity check, only if cheap:

```text
uniform in log-SNR / log-odds
```

Do not make the optional rule part of the main claims unless it is fully run for all schedules.

## Metrics

For each NFE, evaluate generated samples with:

```text
FID
CMMD
KID
precision/recall or density/coverage
```

Do not report NLL for respaced sampling unless the likelihood calculation is explicitly valid for the respaced chain. Main NLL remains full-chain only.

## Sample counts

For CIFAR-10:

```text
50k generated samples per schedule per NFE if feasible.
```

If compute-constrained, start with 10k for all runs to debug, then run 50k for the final table.

## Plots

Generate:

1. FID versus NFE.
2. CMMD versus NFE.
3. KID versus NFE.
4. Precision and recall versus NFE, or density and coverage versus NFE.
5. One sample grid per schedule at `NFE=50` and `NFE=250`.

Use log scale for NFE on the x-axis.

## Success criteria

This experiment is successful if it answers:

- Does geometric odds help only in the full-chain setting, or also under reduced NFE?
- Are failures under reduced NFE concentrated in FID/perceptual metrics or also visible in CMMD/KID?
- Does geometric-linear behave differently from geometric-cosine?

---

# Experiment 4: First additional dataset — FFHQ-64

## Recommendation

Run **FFHQ-64** as the first new dataset.

Reason:

- It is unconditional.
- It is a natural-image dataset.
- It is semantically cleaner than ImageNet and much less confounded than COCO.
- It tests whether the CIFAR/ImageNet FID behavior is due to complex multi-class semantics or more general natural-image structure.

Do **not** start with COCO. COCO introduces captions, text encoders, text-image alignment metrics, classifier-free guidance, and many confounds unrelated to the schedule question.

## Data preparation

Prepare FFHQ at 64x64.

Use a deterministic train/validation split if no existing split is already standard in the codebase:

```text
train: all but 5k images
validation: 5k images
split seed: 1234
```

Use the validation split for NLL / bits-dim. Use training-set real images for FID/CMMD/KID if that is the convention used for the other datasets.

Record:

```json
{
  "dataset": "FFHQ-64",
  "resolution": 64,
  "split_seed": 1234,
  "num_train": "<fill in>",
  "num_val": "<fill in>",
  "preprocessing": "center crop / resize / normalization details"
}
```

## Schedules

Use:

```text
linear
geometric-linear
cosine
geometric-cosine
```

No hybrid.

## Objective

Run first:

```text
L_hybrid
```

Add `L_vlb` only after `L_hybrid` produces a clear signal or if compute is abundant.

## Architecture

Use the 64x64 architecture already used for ImageNet-64 unless there is a strong implementation reason to change it:

```text
U-Net, 128 base channels
3 residual blocks per resolution
attention at 16 and 8
channel multipliers = (1, 2, 3, 4)
T = 4000
learning rate = 1e-4
batch size = 128, or the largest stable batch size with gradient accumulation
EMA = 0.9999
```

Training length:

```text
Start with 500k iterations for final runs.
Use 50k-100k iteration smoke runs to debug configs and samples.
```

## Metrics

For FFHQ-64, report:

```text
NLL / bits-dim on validation split
FID-50K if feasible
CMMD-50K if feasible
KID
density/coverage or precision/recall
```

If only 10k generated samples are feasible initially, label all metrics accordingly and later repeat with 50k for final results.

## Deliverables

```text
ffhq64_results/
  schedules.png
  sample_grids/
    linear.png
    geometric_linear.png
    cosine.png
    geometric_cosine.png
  metrics_table.csv
  metrics_table.tex
  full_metrics.json
```

## Success criteria

FFHQ-64 is successful if it gives a clean unconditional natural-image test of the schedule. The result does not need to show universal FID improvement. It should clarify whether the NLL advantage persists and whether perceptual/diversity metrics behave more like CIFAR-10 or more like MNIST/Fashion-MNIST.

---

# Experiment 5: Linear-in-$\bar\alpha$ (linear-decay) schedule on the existing setups

## Goal

Test the linear-in-$\bar\alpha$ schedule (defined in "Linear-in-$\bar\alpha$ (linear-decay) schedule" above) on trained image models. This is the schedule §6.3 of the paper derives as the optimizer of the *refined*, cumulant-aware bound, in contrast to the geometric-odds schedule, which optimizes the *universal* bound. The two proven bounds disagree on the optimal shape; this experiment tests which one better predicts downstream NLL and sample quality.

This complements **Experiment 1**: the oracle measures the actual finite-step Gaussianization KL directly, while this experiment measures the downstream effect on trained models. If linear-in-$\bar\alpha$ is also added to Experiment 1, the oracle and trained-model rankings of the same four-to-six schedules can be compared head to head.

## Scope (this round)

Run **first** on the existing setups, reusing the existing trained image checkpoints and the CMMD/KID/PRDC metrics pipeline:

```text
Datasets:    MNIST32, FashionMNIST32, CIFAR-10, ImageNet-64 (if reusable)
Objectives:  L_simple, L_hybrid, L_vlb   (all three)
```

All three objectives are requested, including `L_simple`. Treat `L_simple` as a diagnostic: its loss weighting is schedule-dependent and is expected to misbehave under a schedule whose $\beta_t$ profile differs sharply from DDPM-linear, which linear-in-$\bar\alpha$ does (large late-step $\beta_t$). `L_hybrid` and `L_vlb` remain the well-specified comparisons.

Do **not** extend to FFHQ-64 or the NFE sweep in this round; those are later options once the existing-setup results are in. The stitched geometric/linear-decay hybrid remains deferred.

## Schedules

Add the matched-endpoint linear-in-$\bar\alpha$ variants to the existing comparison so each endpoint family has three contenders at fixed endpoints:

```text
linear endpoints:  linear  vs  geometric-linear  vs  linabar-linear
cosine endpoints:  cosine  vs  geometric-cosine  vs  linabar-cosine
```

Compute the `linabar-*` endpoints from the actual baseline `betas` arrays (never hard-code), exactly as for the geometric variants:

```python
beta1_match = baseline_betas[0]
alpha_bar_T_match = np.cumprod(1.0 - baseline_betas)[-1]
linabar_betas = linear_alphabar_betas(T, beta1_match, alpha_bar_T_match)
```

Do **not** clip the primary `linabar-*` variants: the point of this experiment is to match both baseline endpoints exactly. In particular, clipping `linabar-cosine` at `0.999` would substantially increase its terminal $\bar\alpha_T$ and turn it into a different, non-matched schedule. Record `max_beta` and the exact endpoint errors for both variants. Optionally also run a clipped `linabar-cosine-clipped` diagnostic and/or the pure linear-decay form (already in the repo as `ours_v2`, matching $\bar\alpha_T$ only) on CIFAR-10 to check sensitivity; label these clearly and keep them out of the matched-endpoint tables.

## Training / sampling / metrics

These are **new training runs** (`linabar-linear` and `linabar-cosine` checkpoints do not exist yet): up to 2 schedules x 4 datasets x 3 objectives. Use the same training configuration as the existing 84-model study: T=4000, lr 1e-4, batch 128, EMA 0.9999; U-Net with 128 base channels, 3 residual blocks, attention at 16 and 8, channel multipliers (1,2,2,2) for the 32x32 datasets and (1,2,3,4) for ImageNet-64; 500k iterations for MNIST/Fashion-MNIST/CIFAR-10 and 200k for ImageNet-64; one fixed, recorded training seed. Sample with the full T=4000 reverse chain. Evaluate NLL (bits/dim, held-out, full chain) plus FID, CMMD, KID, and density/coverage or precision/recall once those pipelines exist; NLL + FID are the minimum for a first pass.

## Tables

Extend the main image table so each matched endpoint family shows the three-way comparison. For each `(dataset, objective)` and each metric:

```text
linear | geometric-linear | linabar-linear || cosine | geometric-cosine | linabar-cosine
```

Within each endpoint family (linear / cosine), bold the best of the three. This makes the universal-vs-cumulant question directly readable per cell.

## Success criteria

This experiment is informative if it answers:

- Does linear-in-$\bar\alpha$ match or beat geometric on NLL under `L_hybrid` / `L_vlb`, and on which datasets? §6.3 predicts linear-decay carries a smaller constant when the data is closer to Gaussian than its sixth moment suggests, so the datasets where it wins (if any) are themselves informative.
- Does it change the FID picture on CIFAR-10 / ImageNet-64, where geometric was mixed?
- Are the trained-model results consistent with the oracle (Experiment 1) ranking of the same schedules?

Report the result honestly even if linear-in-$\bar\alpha$ loses. A clean "geometric beats linear-in-$\bar\alpha$ on trained models despite the tighter bound" is itself a useful finding about bound tightness versus downstream behavior, and directly informs the §6.3 discussion in the paper.

---

# Run ordering

Use this order.

## Phase 0: Setup and schedule checks

1. Confirm the schedules already in `get_named_beta_schedule` behave as expected -- `geometric`, `geometric_linear`, `geometric_cosine`, and `ours_v2` (= pure linear-decay). Do not reimplement them.
2. Add the new `linabar_linear` / `linabar_cosine` branches (matched-endpoint linear-in-$\bar\alpha$) and a small unit test for them.
3. Reproduce schedule diagnostic plots:
  - `alpha_bar_t`,
  - `z_t`,
  - `r_t`,
  - `Psi(r_t)`.
4. Confirm the geometric and primary linabar schedules match their baseline endpoints. For `linabar_cosine`, this means exact cosine $\bar\alpha_T$ matching with no beta clipping; record its large final beta explicitly.

Exit criteria:

- All schedule tests pass.
- Diagnostic plots match expectations: geometric schedules are straight lines in log odds and have constant `r_t` for `t >= 2`.

## Phase 1: Toy oracle KL

1. Implement GMM posterior formulas.
2. Run Gaussian sanity check.
3. Run 1D symmetric and skewed mixtures.
4. Run 2D grid mixture.
5. Produce tables and plots.

Exit criteria:

- Gaussian sanity check passes.
- Actual bulk KL and proxy plots are available for all four schedules.

## Phase 2: Metrics pipeline

1. Implement or integrate FID, CMMD, KID, and density/coverage or precision/recall.
2. Add feature caching.
3. Validate metrics on a small generated sample directory.
4. Compute metrics for existing sample sets if available.

Exit criteria:

- One `metrics.json` per sample set.
- Aggregated CSV and LaTeX tables generated automatically.

## Phase 3: Curate the existing matched-endpoint results

The matched-endpoint comparison (linear / cosine vs geometric-linear / geometric-cosine) is already trained, sampled, and scored on NLL/FID/TV; no new training or sampling is needed here.

1. Pull the existing NLL/FID/TV numbers from `evaluation_results_full.tsv`.
2. Layer the Phase 2 metrics (CMMD, KID, density/coverage) onto the existing samples if those `.npz` files are still on cluster storage; otherwise re-sample from the existing checkpoints (sampling only, no retraining).
3. Build the cleaned main table with only `L_hybrid` and `L_vlb`; put `L_simple` in an appendix/diagnostic table.
4. Label ImageNet-64 sample counts explicitly; re-sample at one consistent count if 10k/50k are otherwise mixed.

Exit criteria:

- Cleaned main table ready, with the new metrics added wherever samples were available.

## Phase 3b: Linear-in-$\bar\alpha$ schedule (Experiment 5)

1. Add and unit-test `linear_alphabar_betas` (see schedule definitions).
2. Train `linabar-linear` and `linabar-cosine` for all four datasets x three objectives. These are **new** runs; checkpoints do not exist yet. Use the existing-study training config (T=4000, lr 1e-4, batch 128, EMA 0.9999; same architectures and iteration counts as the existing image models) and a fixed, recorded seed.
3. Full-chain sample and evaluate NLL plus the Phase 2 metrics.
4. Extend the main image table to the three-way per-endpoint comparison (baseline / geometric / linear-in-$\bar\alpha$).

Exit criteria:

- Three-way matched-endpoint table for the existing datasets, with `max_beta` and endpoint errors recorded for the `linabar` variants. Any clipped diagnostic variants must be labeled separately and excluded from this table.

## Phase 4: CIFAR-10 NFE sweep

1. Use CIFAR-10 `L_hybrid` checkpoints.
2. Sample with NFE `{25, 50, 100, 250, 1000, 4000}`.
3. Evaluate FID, CMMD, KID, and density/coverage or precision/recall.
4. Produce metric-vs-NFE plots.

Exit criteria:

- NFE curves for all four schedules.
- Sample grids for `NFE=50` and `NFE=250`.

## Phase 5: FFHQ-64

1. Prepare dataset.
2. Run short smoke training for all four schedules.
3. Check sample generation and NLL evaluation.
4. Run full `L_hybrid` training.
5. Generate final samples and metrics.

Exit criteria:

- FFHQ-64 table and sample grids complete.

---

# Configuration matrix

## Main image runs

```text
Datasets:
  MNIST32
  FashionMNIST32
  CIFAR10
  ImageNet64, if available/reusable

Objectives:
  L_hybrid
  L_vlb

Schedules:
  linear
  geometric-linear
  cosine
  geometric-cosine
  linabar-linear      # linear-in-alpha_bar matched to linear endpoints (Experiment 5)
  linabar-cosine      # linear-in-alpha_bar matched to cosine endpoints (Experiment 5)

Note:
  linabar-* are run across all three objectives (incl. L_simple as diagnostic); see Experiment 5.

Sampling:
  full chain, T=4000

Metrics:
  NLL
  FID
  CMMD
  KID
  density/coverage or precision/recall
```

## Optional diagnostic image runs

```text
Objective:
  L_simple

Purpose:
  diagnostic only; do not use as primary schedule-quality evidence.
```

## NFE sweep

```text
Dataset:
  CIFAR10

Objective:
  L_hybrid

Schedules:
  linear
  geometric-linear
  cosine
  geometric-cosine

NFE:
  25, 50, 100, 250, 1000, 4000

Metrics:
  FID
  CMMD
  KID
  density/coverage or precision/recall
```

## New dataset

```text
Dataset:
  FFHQ64

Objective:
  L_hybrid first
  L_vlb optional later

Schedules:
  linear
  geometric-linear
  cosine
  geometric-cosine
```

---

# Final tables and figures to produce

## Table 1: Toy oracle KL totals

Columns:

```text
distribution | T | endpoint family | baseline sum K_bulk | geometric sum K_bulk | relative change | baseline sum Psi | geometric sum Psi
```

Separate linear-matched and cosine-matched endpoint families.

## Figure 1: Toy per-step KL

For representative distributions, plot:

```text
K_t actual versus t
Psi(r_t) versus t
```

Use four schedules in each plot.

## Table 2: Main image results

For `L_hybrid` and `L_vlb` only:

```text
dataset | objective | schedule | NLL | FID | CMMD | KID | density | coverage
```

or pairwise schedule columns if that is easier for the paper. If Experiment 5 is run, extend this to the three-way per-endpoint comparison (baseline / geometric / linear-in-$\bar\alpha$); see Experiment 5.

## Figure 2: Metric deltas

For each dataset/objective, show:

```text
geometric-linear minus linear
geometric-cosine minus cosine
```

for each metric. Lower-is-better metrics should be signed so negative means geometric is better.

## Figure 3: NFE sweep

Four panels:

```text
FID vs NFE
CMMD vs NFE
KID vs NFE
coverage/recall vs NFE
```

## Table 3: FFHQ-64 results

Columns:

```text
schedule | NLL | FID | CMMD | KID | density | coverage
```

## Figure 4: FFHQ-64 sample grids

One grid per schedule, same number of images, same random seed convention.

---

# Reporting conventions

## Naming

Use these exact schedule names in filenames and tables:

```text
linear
glinear
cosine
gcosine
```

In paper text, use:

```text
linear
geometric-linear
cosine
geometric-cosine
```

## Sample counts

Always include the generated sample count in metric metadata. If the count is not 50k, label the metric accordingly.

Examples:

```text
FID-50K
FID-10K
CMMD-50K
KID-10K
```

## Seeds

Even though multiple seeds are not required, every run must record:

```text
training seed
sampling seed
dataset split seed
metric subsampling seed
```

## Checkpoint reuse

If a checkpoint is reused, save a `checkpoint_path.txt` and include:

```text
absolute or relative path
commit hash, if available
training config hash, if available
```

## No silent deviations

If compute constraints force deviations, record them in `notes` fields. Examples:

```text
"ImageNet-64 metrics use 10k generated samples due to compute."
"FFHQ-64 trained for 300k iterations, not 500k."
"CMMD uses ViT-B/32 rather than ViT-L/14 due to GPU memory."
```

---

# Guardrails for interpretation

Use these conclusions only if supported by results:

- “Geometric odds consistently improves likelihood under well-specified objectives.”
- “The perceptual-quality effect is mixed and depends on dataset and endpoint family.”
- “Oracle experiments directly validate the finite-step Gaussianization mechanism.”
- “Reduced-NFE sampling is a practical diagnostic, not the theorem’s primary setting.”

Avoid these claims unless the new data strongly supports them:

- “Geometric odds universally improves FID.”
- “Geometric odds is a new SOTA sampler.”
- “The schedule is optimal for continuous-time score-SDE discretization error.”
- “The hybrid schedule is supported by experiments.”

---

# Agent checklist

Before starting:

- Confirm codebase branch / commit.
- Confirm schedule implementation tests pass.
- Confirm baseline schedule endpoints are computed from actual beta arrays.
- Confirm generated sample image range and preprocessing conventions.
- Confirm metric implementations and feature extractors.

For toy experiments:

- GMM posterior mixture code implemented.
- Gaussian sanity check passes.
- All toy distributions run for `T={50,100,250,1000}`.
- `K_1` saved separately.
- Bulk totals and plots generated.

For image experiments:

- Full-chain samples available for all schedule/objective/dataset combinations.
- NLL computed on held-out data.
- FID/CMMD/KID/density-coverage or precision-recall computed.
- Tables generated with matched-pair bolding.

For NFE sweep:

- CIFAR-10 `L_hybrid` checkpoints found.
- Respaced sampler verified.
- NFE curves generated.

For FFHQ-64:

- Dataset prepared and split recorded.
- Smoke runs complete.
- Full `L_hybrid` runs complete.
- Metrics and sample grids generated.

Final deliverable:

- `results_summary.md`
- `tables/*.csv`
- `tables/*.tex`
- `figures/*.pdf`
- `metrics_json/**/*.json`
- clear notes on any deviations from this plan
