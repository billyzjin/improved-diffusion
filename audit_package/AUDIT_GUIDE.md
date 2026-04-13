# Code Audit Guide: Geometric-Odds Noise Schedule for Diffusion Models

## Overview

This package contains the implementation of a novel noise schedule for discrete-time diffusion models. The main theoretical contribution is a **geometric-odds schedule** (Theorem 5 in the paper) that minimizes an upper bound on the model error for fixed endpoint parameters. This code implements the schedule, trains diffusion models with it, evaluates NLL (negative log-likelihood), and tunes the endpoint parameters via short probe training runs on a validation split.

The codebase is a fork of [openai/improved-diffusion](https://github.com/openai/improved-diffusion) (Nichol & Dhariwal, 2021). Our additions are confined to a few specific locations described below.

## Key Mathematical Claim

**Theorem (Geometric-Odds Schedule).** For a diffusion model with T steps, fix the first-step noise beta_1 and the terminal signal level alpha_bar_T. Define the odds variable:

    z_t = (1 - alpha_bar_t) / alpha_bar_t

Among all schedules with these endpoints, the model error upper bound is minimized when z_t grows geometrically:

    z_t = z_1 * q^{t-1},    q = (z_T / z_1)^{1/(T-1)}

This yields a constant noise-to-smoothing ratio r_t = q - 1 at every step, and the per-step betas are:

    beta_1 = z_1 / (1 + z_1)
    beta_t = (q - 1) * z_{t-1} / (1 + q * z_{t-1})    for t >= 2

where z_1 = beta_1 / (1 - beta_1) and z_T = (1 - alpha_bar_T) / alpha_bar_T.

## Files and What to Audit

### 1. Schedule Implementation (CRITICAL)

**File:** `improved_diffusion/gaussian_diffusion.py`  
**Function:** `get_named_beta_schedule()` (starts at line 18)  
**What to check:**

- **`geometric_linear` and `geometric_cosine` branches** (~line 68-98): These implement the geometric schedule with endpoints matched to the DDPM linear and cosine schedules respectively. Verify:
  - z_1 = beta_1 / (1 - beta_1) is correct
  - z_T = (1 - alpha_bar_T) / alpha_bar_T is correct
  - q = (z_T / z_1)^{1/(T-1)} matches the theorem
  - The loop correctly computes beta_t = (q-1) * z_{t-1} / (1 + q * z_{t-1})
  - The endpoint extraction for `geometric_linear` matches DDPM's linear schedule endpoints
  - The endpoint extraction for `geometric_cosine` matches Nichol & Dhariwal's cosine schedule endpoints

- **`geometric` branch** (~line 99-118): Generic version accepting user-specified beta_1 and alpha_bar_T. Same formula as above but with explicit parameters. Used for the endpoint probe experiments.

- **`_vb_terms_bpd()` method** (~line 721): Computes per-step VLB terms. At t=0 returns the discretized decoder NLL; at t>0 returns KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t)). This is from the original N&D code, unmodified.

- **`calc_bpd_loop()` method** (~line 849): Iterates from t=T-1 down to t=0, computing VLB at each step. **Note the reverse iteration order** -- the returned `vb` tensor has index 0 = timestep T-1 and index T-1 = timestep 0.

### 2. Discretized Gaussian Log-Likelihood

**File:** `improved_diffusion/losses.py`  
**Function:** `discretized_gaussian_log_likelihood()` (line 50)  
**What to check:**
- Bin width is 2/255 (half-bin = 1/255) for images in [-1, 1]
- Edge handling at x < -0.999 and x > 0.999
- This function is from the original N&D code, unmodified.

### 3. NLL Evaluation

**File:** `scripts/image_nll_no_mpi.py`  
**Function:** `run_bpd_evaluation()` (line 51)  
**What to check:**
- Calls `diffusion.calc_bpd_loop()` correctly
- Averages VLB terms over batches: `np.mean(np.stack(terms), axis=0)` (line 92)
- Saves per-step terms to `vb_terms.npz`, MSE terms to `mse_terms.npz`
- Reports `total_bpd = vb.sum(dim=1) + prior_bpd`
- NaN/Inf check on total_bpd (line 78)

### 4. Per-Step VLB Plotting

**File:** `plot_per_step_vlb.py`  
**What to check:**
- **Array reversal on lines 145 and 193**: `vb = data["arr_0"][::-1]` reverses the array because `calc_bpd_loop` stores results in reverse timestep order. Without this reversal, the x-axis would be inverted.
- Smoothing uses `mode="valid"` with raw edge preservation (not `mode="same"` which would zero-pad boundaries)

### 5. Schedule Comparison Visualization

**File:** `plot_schedule_comparison.py`  
**What to check:**
- `compute_schedule_quantities()` (line 36): Computes alpha_bar, z_t (odds), and r_t = beta_t / eta_t correctly
- Psi(r) = r^2/2 - r + log(1+r) matches the paper's per-step cost function
- The summary statistics (sum Psi, r_t range) are computed correctly

### 6. Model and Diffusion Configuration

**File:** `improved_diffusion/script_util.py`  
**What to check:**
- `model_and_diffusion_defaults()`: Includes `geometric_beta1` and `geometric_alpha_bar_T` parameters (default 0.0)
- These parameters are plumbed through `create_model_and_diffusion()` -> `create_gaussian_diffusion()` -> `get_named_beta_schedule()`

### 7. Experiment Configuration

**File:** `experiment_configs/train_cifar10_no_mpi.slurm`  
**What to check:**
- The case/switch blocks for geometric_linear/geometric_cosine experiments use:
  - `--noise_schedule geometric_linear` (or geometric_cosine)
  - `--learn_sigma True` for hybrid/vlb objectives
  - `--use_kl True` for vlb objective
  - `--dropout 0.3` for all geometric/cosine schedules, `0.1` for linear
  - `--diffusion_steps 4000`
  - `--lr_anneal_steps 500000`
  - Correct LOGDIR paths under `/project_gpfs/bata0/bjin0/`

**File:** `experiment_configs/train_probe_geometric.slurm`  
**What to check:**
- Uses `--noise_schedule geometric` with `--geometric_beta1` and `--geometric_alpha_bar_T` passed from environment
- Trains for 50K steps on a 45K training split
- Evaluates NLL on a 5K validation split (no test data leakage)

**File:** `experiment_configs/evaluate_models_final.slurm`  
**What to check:**
- Schedule detection: `geometric_linear` and `geometric_cosine` patterns are matched BEFORE `linear` and `cosine` (to avoid "geometric_linear" matching the "linear" branch)
- NLL evaluation uses the EMA checkpoint
- FID uses 50K generated samples for 32x32 datasets

## Common Pitfalls to Look For

1. **Off-by-one in schedule indexing**: beta_t is 1-indexed in the paper but 0-indexed in the code. Verify betas[0] = beta_1.

2. **Endpoint extraction**: `geometric_linear` extracts beta_1 and alpha_bar_T by computing the DDPM linear schedule and taking its first beta and cumulative product. `geometric_cosine` does the same with the cosine schedule. Verify these match.

3. **VLB array ordering**: `calc_bpd_loop` iterates t from T-1 down to 0, so the output array has reversed indices. All downstream code must reverse before plotting against timestep t.

4. **Units**: NLL terms are in bits per dimension (divided by d * log(2)). The theoretical bound in the paper is in nats (total). Any conversion between the two should be consistent.

5. **Beta clipping**: All schedules clip betas to max 0.999. Verify this doesn't affect the geometric schedules (it shouldn't -- max beta for our settings is well below 0.999).

## Experiments Run

- **Schedules tested**: linear, cosine, ours (custom), geometric_linear, geometric_cosine, geometric (with custom endpoints)
- **Objectives**: L_simple (fixed variance), L_hybrid (learned variance, rescaled MSE), L_vlb (full VLB)
- **Datasets**: CIFAR-10 (32x32), Fashion-MNIST (32x32), MNIST (32x32), ImageNet-64 (64x64)
- **Diffusion steps**: T = 4000 for all experiments
- **Training**: 500K iterations for 32x32 datasets, 200K for ImageNet-64
- **Endpoint tuning**: Short probe runs (50K steps) on CIFAR-10 with varying beta_1, evaluated on a held-out 5K validation split of the training set (no test data used)
- **Evaluation metrics**: NLL (bits/dim), FID (pytorch-fid), per-step VLB decomposition

## Files NOT Included

### Unmodified from original openai/improved-diffusion:

- `improved_diffusion/unet.py` - U-Net architecture
- `improved_diffusion/image_datasets.py` - Data loading
- `improved_diffusion/train_util_no_mpi.py` - Training loop
- `improved_diffusion/respace.py` - Timestep respacing
- `improved_diffusion/nn.py` - Neural network utilities

### Analogous scripts for other datasets (same structure as the included CIFAR-10 versions):

- `train_fashionmnist_no_mpi.slurm`, `train_mnist_no_mpi.slurm`, `train_imagenet64_no_mpi.slurm` - Training configs for other datasets (same case/switch structure, different data paths and log directories)
- `evaluate_fashionmnist_final.slurm`, `evaluate_mnist_final.slurm`, `evaluate_imagenet64_final.slurm` - Evaluation configs for other datasets (same structure as `evaluate_models_final.slurm`)

### Other plotting scripts (use data from the evaluation pipeline but do not affect results):

- `plot_training_convergence.py` - Training loss curves from progress.csv
- `plot_sample_grids.py` - Visual sample grids from generated images
