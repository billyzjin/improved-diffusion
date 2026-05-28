# Implementation Plan: Geometric-Odds Diffusion Schedule Experiments

## Purpose

This document is an implementation brief for an agent who will run the next round of experiments for the geometric-odds diffusion schedule paper.

The experiments should support the following core claim:

> For finite-step DDPM-style diffusion, the geometric-odds schedule is the natural endpoint-matched schedule for reducing bulk Gaussianization error. This mechanism should be validated directly on oracle toy problems, then tested in trained image models through likelihood, sample-quality metrics, reduced-NFE sampling, and one additional natural-image dataset.

The selected experiment set is:

1. **Oracle Gaussianization-KL toy experiments**.
2. **Cleaned matched-endpoint image experiments**, without multiple training seeds / standard deviations.
3. **Additional metrics**: CMMD, KID, and fidelity/diversity diagnostics such as precision/recall or density/coverage.
4. **NFE / respacing sweep**.
5. **One additional dataset**: start with **FFHQ-64**.

Do **not** implement the hybrid schedule for now. Also do **not** prioritize COCO, endpoint-grid tuning, class-conditional CIFAR-10, or multiple training seeds in this round.

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

\[
X_t = \sqrt{1-\beta_t}X_{t-1}+\sqrt{\beta_t}Z_t,
\qquad
\bar\alpha_t=\prod_{i=1}^t(1-\beta_i).
\]

The noise-to-signal odds are

\[
z_t=\frac{1-\bar\alpha_t}{\bar\alpha_t}.
\]

For bulk steps `t >= 2`, define

\[
\eta_t=(1-\beta_t)(1-\bar\alpha_{t-1}),
\qquad
r_t=\frac{\beta_t}{\eta_t}
=
\frac{\beta_t}{(1-\beta_t)(1-\bar\alpha_{t-1})}.
\]

The universal per-step proxy is

\[
\Psi(r)=\frac{r^2}{2}-r+\log(1+r).
\]

Use `log1p(r)` in code for numerical stability.

### Geometric-odds schedule

Given:

- number of steps `T`,
- first-step noise `beta_1`,
- terminal signal level `alpha_bar_T`,

compute:

\[
z_1=\frac{\beta_1}{1-\beta_1},
\qquad
z_T=\frac{1-\bar\alpha_T}{\bar\alpha_T},
\]

\[
q=\left(\frac{z_T}{z_1}\right)^{1/(T-1)},
\qquad
z_t=z_1q^{t-1}, \quad t=1,\ldots,T.
\]

Then

\[
\bar\alpha_t=\frac{1}{1+z_t},
\]

and the correct beta formula is

\[
\beta_1 \text{ fixed},
\qquad
\beta_t=\frac{(q-1)z_{t-1}}{1+qz_{t-1}},
\quad t=2,\ldots,T.
\]

Under this schedule, all bulk ratios satisfy

\[
r_2=r_3=\cdots=r_T=q-1.
\]

### Reference implementation

Add this function to the schedule utilities and test it thoroughly.

```python
import numpy as np


def geometric_odds_betas(T: int, beta1: float, alpha_bar_T: float) -> np.ndarray:
    """Return a geometric-odds beta schedule of length T.

    Indexing convention:
        betas[0] = beta_1
        betas[t-1] = beta_t

    The schedule matches beta_1 and alpha_bar_T exactly up to floating-point error.
    """
    if T < 2:
        raise ValueError("T must be at least 2")
    if not (0.0 < beta1 < 1.0):
        raise ValueError("beta1 must be in (0, 1)")
    if not (0.0 < alpha_bar_T < 1.0):
        raise ValueError("alpha_bar_T must be in (0, 1)")

    z1 = beta1 / (1.0 - beta1)
    zT = (1.0 - alpha_bar_T) / alpha_bar_T
    if not (zT > z1):
        raise ValueError("Need zT > z1 so that the chain adds noise overall")

    q = (zT / z1) ** (1.0 / (T - 1))
    z = z1 * q ** np.arange(T, dtype=np.float64)  # z[0] = z_1, ..., z[T-1] = z_T

    betas = np.empty(T, dtype=np.float64)
    betas[0] = beta1
    betas[1:] = ((q - 1.0) * z[:-1]) / (1.0 + q * z[:-1])
    return betas


def alpha_bar_from_betas(betas: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 - betas)


def bulk_ratios(betas: np.ndarray) -> np.ndarray:
    """Return r_t for t=2,...,T as a length T-1 array."""
    alpha_bar = alpha_bar_from_betas(betas)
    alpha_prev = alpha_bar[:-1]  # alpha_bar_{t-1}, t=2,...,T
    beta_t = betas[1:]
    eta_t = (1.0 - beta_t) * (1.0 - alpha_prev)
    return beta_t / eta_t


def psi(r: np.ndarray) -> np.ndarray:
    return 0.5 * r**2 - r + np.log1p(r)
```

### Required tests

Add unit tests before running experiments.

```python
T = 4000
beta1 = 1e-4
alpha_bar_T = 4e-5
betas = geometric_odds_betas(T, beta1, alpha_bar_T)
alpha_bar = alpha_bar_from_betas(betas)
r = bulk_ratios(betas)

assert betas.shape == (T,)
assert np.all(betas > 0) and np.all(betas < 1)
assert abs(betas[0] - beta1) < 1e-12
assert abs(alpha_bar[-1] - alpha_bar_T) / alpha_bar_T < 1e-8
assert np.std(r) / np.mean(r) < 1e-10
```

For matched baselines, never hard-code endpoints except in toy experiments. Compute the baseline `betas`, then set:

```python
beta1_match = baseline_betas[0]
alpha_bar_T_match = np.cumprod(1.0 - baseline_betas)[-1]
geometric_betas = geometric_odds_betas(T, beta1_match, alpha_bar_T_match)
```

---

## Repository / artifact structure

Use or adapt this structure.

```text
experiments/
  schedules/
    schedules.py
    test_schedules.py
    plot_schedules.py
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

\[
K_t=\mathbb E_{X_t}
D_{\mathrm{KL}}
\left(
\mathcal L(X_{t-1}\mid X_t)
\,\|\,
\text{best Gaussian approximation}
\right),
\]

or equivalently after the affine rescaling used in the proof,

\[
K_t=\mathbb E_{X_t}
D_{\mathrm{KL}}
\left(
\mathcal L(S_t\mid X_t)
\,\|\,
\text{moment-matched Gaussian}
\right),
\qquad
S_t=\sqrt{1-\beta_t}X_{t-1}.
\]

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

\[
X_0\mid C=c\sim N(\mu_c,\Sigma_c),
\qquad
P(C=c)=\pi_c.
\]

For diffusion step `t`, define

\[
S_t=\sqrt{\bar\alpha_t}X_0+\sqrt{\eta_t}W,
\qquad
X_t=S_t+\sqrt{\beta_t}Z,
\]

where

\[
\eta_t=(1-\beta_t)(1-\bar\alpha_{t-1}).
\]

Conditional on component `c`,

\[
S_t\mid C=c\sim N(m_c, P_c),
\]

with

\[
m_c=\sqrt{\bar\alpha_t}\mu_c,
\qquad
P_c=\bar\alpha_t\Sigma_c+\eta_t I.
\]

The observation is

\[
X_t\mid C=c\sim N(m_c, P_c+\beta_t I).
\]

Given `X_t = x`, posterior component weights are

\[
w_c(x)\propto \pi_c\,N(x;m_c,P_c+\beta_t I).
\]

The component posterior is Gaussian:

\[
S_t\mid X_t=x,C=c\sim N(\tilde m_c(x),\tilde P_c),
\]

where

\[
K_c=P_c(P_c+\beta_t I)^{-1},
\]

\[
\tilde m_c(x)=m_c+K_c(x-m_c),
\]

\[
\tilde P_c=P_c-P_c(P_c+\beta_t I)^{-1}P_c.
\]

Thus the exact posterior is a mixture:

\[
q_t(s\mid x)=\sum_c w_c(x)N(s;\tilde m_c(x),\tilde P_c).
\]

The KL-best Gaussian is the moment-matched Gaussian:

\[
g_t^\star(s\mid x)=N(m(x),C(x)),
\]

with

\[
m(x)=\sum_c w_c(x)\tilde m_c(x),
\]

\[
C(x)=\sum_c w_c(x)\left[\tilde P_c+
(\tilde m_c(x)-m(x))(\tilde m_c(x)-m(x))^\top\right].
\]

Then estimate

\[
D_{\mathrm{KL}}(q_t(\cdot\mid x)\|g_t^\star(\cdot\mid x))
=
\mathbb E_{s\sim q_t(\cdot\mid x)}
\left[\log q_t(s\mid x)-\log g_t^\star(s\mid x)\right].
\]

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

# Experiment 2: Clean matched-endpoint image experiments

## Goal

Strengthen the current image experiments without adding multiple training seeds.

The existing comparison structure is good: compare each baseline against its geometric-odds counterpart with the same `beta_1` and `alpha_bar_T`, so only the intermediate schedule shape changes.

## Schedules

Use four schedules:

```text
linear
geometric-linear
cosine
geometric-cosine
```

Definitions:

- `linear`: existing DDPM / Improved-Diffusion linear schedule.
- `geometric-linear`: geometric odds with endpoints copied from `linear`.
- `cosine`: existing cosine schedule with the same clipping convention as the baseline implementation.
- `geometric-cosine`: geometric odds with endpoints copied from `cosine`.

Do not add sigmoid / ACS baselines in this round unless all selected experiments are already complete.

## Datasets

Use the current datasets first:

```text
MNIST at 32x32
Fashion-MNIST at 32x32
CIFAR-10 at 32x32
ImageNet-64 at 64x64, if the existing compute pipeline/checkpoints are available
```

## Objectives

Main table:

```text
L_hybrid
L_vlb
```

Optional appendix / diagnostic table:

```text
L_simple
```

Reason: `L_simple` is useful as a diagnostic but is not the cleanest likelihood comparison under schedule changes. The main text should focus on `L_hybrid` and `L_vlb`.

## Training configuration

Use the existing configuration unless there is a strong reason to change it:

```text
T = 4000
learning rate = 1e-4
batch size = 128
EMA = 0.9999
```

Architecture:

```text
32x32 datasets:
  U-Net, 128 base channels
  3 residual blocks per resolution
  attention at 16 and 8
  channel multipliers = (1, 2, 2, 2)

ImageNet-64:
  U-Net, 128 base channels
  3 residual blocks per resolution
  attention at 16 and 8
  channel multipliers = (1, 2, 3, 4)
```

Training length:

```text
MNIST/Fashion-MNIST/CIFAR-10: 500k iterations
ImageNet-64: 200k iterations, or reuse existing checkpoints
```

No multiple seeds are required. Use one fixed training seed and record it.

## Full-chain sampling

Use the full `T=4000` reverse chain for the main image-table results.

Generate:

```text
50k samples for 32x32 datasets
10k or 50k samples for ImageNet-64, depending on compute
```

If using 10k samples for ImageNet-64, label the result as `FID-10K`, `CMMD-10K`, etc. Do not silently compare it to 50k results.

## NLL / bits-dim

Evaluate NLL / bits-dim on held-out test data using the full original diffusion chain, not respaced sampling.

Report:

```text
NLL in bits/dim
FID
CMMD
KID
precision/recall or density/coverage
```

For the main table, use one row per `(dataset, objective)` and columns:

```text
linear
geometric-linear
cosine
geometric-cosine
```

For each metric, bold the better value within each matched pair:

```text
linear vs geometric-linear
cosine vs geometric-cosine
```

## Text interpretation guardrails

Use language like:

> Geometric odds gives consistent NLL improvements under the well-specified objectives. Its effect on perceptual metrics is dataset- and endpoint-dependent.

Avoid claiming universal FID dominance.

---

# Experiment 3: Add CMMD, KID, and fidelity/diversity diagnostics

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

# Experiment 4: NFE / respacing sweep

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

\[
\bar\alpha'_k = \bar\alpha_{s_k},
\qquad
\beta'_k=1-\frac{\bar\alpha'_{k}}{\bar\alpha'_{k-1}}.
\]

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

# Experiment 5: First additional dataset — FFHQ-64

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

# Run ordering

Use this order.

## Phase 0: Setup and schedule tests

1. Implement schedule utilities.
2. Add unit tests for geometric odds.
3. Reproduce schedule diagnostic plots:
   - `alpha_bar_t`,
   - `z_t`,
   - `r_t`,
   - `Psi(r_t)`.
4. Confirm geometric schedules exactly match baseline endpoints.

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

## Phase 3: Clean current image experiments

1. Reuse existing checkpoints where possible.
2. Re-run missing full-chain samples.
3. Evaluate NLL and all metrics.
4. Create cleaned main table with only `L_hybrid` and `L_vlb`.
5. Put `L_simple` in appendix/diagnostic output if available.

Exit criteria:

- Main table ready.
- ImageNet sample count clearly labeled.

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

or pairwise schedule columns if that is easier for the paper.

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

- [ ] Confirm codebase branch / commit.
- [ ] Confirm schedule implementation tests pass.
- [ ] Confirm baseline schedule endpoints are computed from actual beta arrays.
- [ ] Confirm generated sample image range and preprocessing conventions.
- [ ] Confirm metric implementations and feature extractors.

For toy experiments:

- [ ] GMM posterior mixture code implemented.
- [ ] Gaussian sanity check passes.
- [ ] All toy distributions run for `T={50,100,250,1000}`.
- [ ] `K_1` saved separately.
- [ ] Bulk totals and plots generated.

For image experiments:

- [ ] Full-chain samples available for all schedule/objective/dataset combinations.
- [ ] NLL computed on held-out data.
- [ ] FID/CMMD/KID/density-coverage or precision-recall computed.
- [ ] Tables generated with matched-pair bolding.

For NFE sweep:

- [ ] CIFAR-10 `L_hybrid` checkpoints found.
- [ ] Respaced sampler verified.
- [ ] NFE curves generated.

For FFHQ-64:

- [ ] Dataset prepared and split recorded.
- [ ] Smoke runs complete.
- [ ] Full `L_hybrid` runs complete.
- [ ] Metrics and sample grids generated.

Final deliverable:

- [ ] `results_summary.md`
- [ ] `tables/*.csv`
- [ ] `tables/*.tex`
- [ ] `figures/*.pdf`
- [ ] `metrics_json/**/*.json`
- [ ] clear notes on any deviations from this plan
