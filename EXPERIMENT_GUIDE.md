# Experiment Guide

This document describes the full experiment pipeline for our diffusion model noise schedule comparison paper. It is intended as a reference for anyone (or any AI assistant) working on this project on the Pythia cluster.

## Overview

We compare **4 noise schedules** across **3 training objectives** on **4 datasets**, evaluating with 3 metrics. The goal is to test whether our theoretically derived noise schedules (`ours` and `ours_v2`) outperform the standard `linear` and `cosine` baselines.

### Noise Schedules

| Name | Formula | Source |
|------|---------|--------|
| `linear` | Linearly spaced betas from scaled [0.0001, 0.02] | DDPM (Ho et al. 2020) |
| `cosine` | alpha_bar_t from cosine function | Nichol & Dhariwal 2021 |
| `ours` | Iterative backward recurrence (see code) | Our earlier theoretical analysis |
| `ours_v2` | beta_t = 1/(T - t + 1), giving alpha_bar_t = (T-t)/T | Our improved KL-optimal theory (`diffusion_clean.tex`) |

All schedules are implemented in `improved_diffusion/gaussian_diffusion.py` in the function `get_named_beta_schedule()`.

### Training Objectives

| Short name | Flags | Description |
|------------|-------|-------------|
| `simple` | `--learn_sigma False --use_kl False` | L_simple: MSE loss, fixed variance |
| `hybrid` | `--learn_sigma True --use_kl False --rescale_learned_sigmas True` | L_hybrid: MSE + learned variance (rescaled) |
| `vlb` | `--learn_sigma True --use_kl True --schedule_sampler loss-second-moment` | L_vlb: full variational lower bound with importance sampling |

### Datasets

| Dataset | Image size | Train images | Test/val images | Local data directory | Notes |
|---------|-----------|-------------|----------------|---------------------|-------|
| CIFAR-10 | 32x32 | 50,000 | 10,000 | `cifar_train/`, `cifar_test/` | PNG files, prepared by `datasets/cifar10.py` |
| Fashion-MNIST | 32x32 | 60,000 | 10,000 | `fashion_train/`, `fashion_test/` | PNG files, prepared by `datasets/fashionmnist.py` |
| MNIST | 32x32 | 60,000 | 10,000 | `mnist_train/`, `mnist_test/` | PNG files, prepared by `datasets/mnist.py` |
| ImageNet-64 | 64x64 | ~1.28M | 50,000 | `/project_gpfs/bata0/bjin0/imagenet64_fixed_20260423/train`, `.../val` | Class subfolders, prepared by `datasets/imagenet64.py` via `prepare_imagenet64.slurm` |

CIFAR-10, Fashion-MNIST, and MNIST datasets live in the repo working directory (`/home/bjin0/improved-diffusion/`). ImageNet-64 lives on project storage because of its size.

### Evaluation Metrics

| Metric | Script | Interpretation |
|--------|--------|----------------|
| NLL (bits/dim) | `scripts/image_nll_no_mpi.py` | Log-likelihood on test set; lower is better |
| FID | `scripts/compute_fid_from_npz.py` | Perceptual quality vs training set; lower is better |
| TV | `scripts/compute_tv_from_npz.py` | Total variation of pixel histograms vs test set; lower is better |

### Naming Convention

All experiments follow the pattern `{dataset}_{schedule}_{objective}`:
- Training log directory: `{dataset}_{schedule}_{objective}` (e.g., `cifar10_cosine_vlb`)
- The SLURM `EXPERIMENT` variable uses just `{schedule}_{objective}` (e.g., `cosine_vlb`); the dataset prefix is added by the training script.

## Cluster Environment

- **Cluster**: Pythia (UChicago)
- **SLURM account**: `bata0-external`
- **Partition**: `long_hopper` (H100 GPUs)
- **Project storage**: `/project_gpfs/bata0/bjin0/` (persistent; replaces old `/project_gpfs/bjin0/` which was decommissioned)
- **Repo on cluster**: `/home/bjin0/improved-diffusion/`
- **Python**: `module load python/booth/3.12`

## Storage Layout

All training checkpoints, evaluation outputs, and cached statistics live under `/project_gpfs/bata0/bjin0/`:

```
/project_gpfs/bata0/bjin0/
├── imagenet64_fixed_20260423/         # Corrected ImageNet-64 dataset (PNG images)
│   ├── train/                         #   ~1.28M images in class subfolders
│   └── val/                           #   50K images in class subfolders
│
├── {user}/{job_id}/logs/              # Training outputs (created by SLURM jobs)
│   └── {dataset}_{schedule}_{objective}/
│       ├── log.txt                    # Training log
│       ├── model{step}.pt            # Model checkpoints (every save_interval steps)
│       └── ema_0.9999_{step}.pt      # EMA checkpoints (used for evaluation)
│
├── evaluation_parallel_{timestamp}/   # CIFAR-10 evaluation results
│   ├── cifar10_{schedule}_{objective}/
│   │   ├── nll_results.txt
│   │   ├── fid_results.txt
│   │   ├── tv_results.txt
│   │   ├── samples_50000x32x32x3.npz
│   │   └── ...
│   └── results_summary.txt           # Aggregated summary (created by aggregation script)
│
├── fashionmnist_evaluation_parallel_{timestamp}/
│   └── ...                            # Same structure as CIFAR-10
│
├── mnist_evaluation_parallel_{timestamp}/
│   └── ...                            # Same structure as CIFAR-10
│
├── imagenet64_evaluation_parallel_{timestamp}/
│   └── imagenet64_{schedule}_{objective}/
│       ├── nll_results.txt
│       ├── fid_results.txt
│       ├── tv_results.txt
│       ├── samples.npz               # Note: just "samples.npz" (not sized name)
│       └── ...
│
├── cifar10_train_stats.npz            # Cached FID statistics for CIFAR-10
├── fashionmnist_train_stats.npz       # Cached FID statistics for Fashion-MNIST
├── mnist_train_stats.npz              # Cached FID statistics for MNIST
└── imagenet64_fixed_20260423_train_stats.npz  # Cached FID statistics for corrected ImageNet-64
```

## Model Architecture

All datasets share the same U-Net architecture, differing only in image size and channel multipliers:

| Parameter | CIFAR-10 / Fashion-MNIST / MNIST | ImageNet-64 |
|-----------|----------------------------------|-------------|
| Image size | 32 | 64 |
| Channels | 128 | 128 |
| Res blocks | 3 | 3 |
| Attention heads | 4 | 4 |
| Attention resolutions | 16, 8 | 16, 8 |
| Channel multipliers | (1, 2, 2, 2) | (1, 2, 3, 4) |
| Scale-shift norm | True | True |
| Class-conditional | False | False |

Channel multipliers are hardcoded in `improved_diffusion/script_util.py` based on `image_size`.

### Dropout

- `linear` schedule experiments use dropout = 0.1
- `cosine`, `ours`, and `ours_v2` experiments use dropout = 0.3

### Training Iterations

| Dataset | Iterations | Checkpoint name |
|---------|-----------|----------------|
| CIFAR-10 | 500,000 | `ema_0.9999_500000.pt` |
| Fashion-MNIST | 500,000 | `ema_0.9999_500000.pt` |
| MNIST | 500,000 | `ema_0.9999_500000.pt` |
| ImageNet-64 | 200,000 | `ema_0.9999_200000.pt` |

All use: lr=1e-4, batch_size=128, ema_rate=0.9999, diffusion_steps=4000.

## Pipeline: Step by Step

The pipeline has 5 stages. Each stage must complete before the next begins.

### Stage 0: Dataset Preparation (one-time)

Run these from the repo directory on the cluster. CIFAR-10, Fashion-MNIST, and MNIST are small and can be prepared on a login node:

```bash
python3 datasets/cifar10.py          # creates cifar_train/ and cifar_test/
python3 datasets/fashionmnist.py     # creates fashion_train/ and fashion_test/
python3 datasets/mnist.py            # creates mnist_train/ and mnist_test/
```

ImageNet-64 requires a SLURM job because it processes large NPZ archives:

```bash
bash submit_prepare_imagenet64.sh    # submits prepare_imagenet64.slurm
```

This reads raw `.npz` archives and writes PNG images to `/project_gpfs/bata0/bjin0/imagenet64_fixed_20260423/{train,val}/`.

### Stage 1: Training

Submit training jobs with the `submit_*.sh` scripts. Each script submits multiple independent SLURM jobs (one per experiment) that run in parallel.

**All 9 baseline experiments per dataset** (linear/cosine/ours x simple/hybrid/vlb):

```bash
bash submit_all_cifar10.sh
bash submit_all_fashionmnist.sh
bash submit_all_mnist.sh
bash submit_all_imagenet64.sh
```

**Just the 3 ours_v2 experiments per dataset**:

```bash
bash submit_ours_v2_cifar10.sh
bash submit_ours_v2_fashionmnist.sh
bash submit_ours_v2_imagenet64.sh
```

Note: There are no `submit_ours_v2_mnist.sh` or `submit_ours_v2_mnist_eval.sh` scripts yet. MNIST ours_v2 experiments can be submitted manually or by creating these scripts (follow the CIFAR-10 pattern).

**What the submit scripts do**: They call `sbatch` with `--export=EXPERIMENT={schedule}_{objective}` pointing at the corresponding `train_{dataset}_no_mpi.slurm` script. The SLURM script uses a `case` statement on `$EXPERIMENT` to set the right flags and log directory.

**Training outputs**: Checkpoints are written to `/project_gpfs/bata0/bjin0/{user}/{job_id}/logs/{dataset}_{schedule}_{objective}/`.

Monitor jobs with `squeue -u $USER`.

### Stage 2: Evaluation

After training completes, submit evaluation jobs. Each evaluation job takes one trained model and computes all 3 metrics (NLL, FID, TV).

**All 9 experiments per dataset**:

```bash
bash submit_cifar10_evaluation_jobs.sh
bash submit_fashionmnist_evaluation_jobs.sh
bash submit_mnist_evaluation_jobs.sh
bash submit_imagenet64_evaluation_jobs.sh
```

**Just the 3 ours_v2 experiments**:

```bash
bash submit_ours_v2_cifar10_eval.sh
bash submit_ours_v2_fashionmnist_eval.sh
bash submit_ours_v2_imagenet64_eval.sh
```

**How checkpoint discovery works**: The evaluation SLURM scripts use `find` to locate the EMA checkpoint by experiment name:

```bash
# CIFAR-10/Fashion-MNIST/MNIST (500K steps):
find /project_gpfs/bata0/bjin0 -name "ema_0.9999_500000.pt" -path "*/logs/$EVAL_MODEL_NAME/*"

# ImageNet-64 (200K steps):
find /project_gpfs/bata0/bjin0 -name "ema_0.9999_200000.pt" -path "*/logs/$EVAL_MODEL_NAME/*"
```

If multiple matches exist (e.g., from repeated runs), it takes the newest by modification time.

**Evaluation parameters are inferred from the experiment name** in the evaluation scripts. The order of pattern matching matters -- `ours_v2` must be checked before `ours` since `"ours_v2"` contains `"ours"`:

```bash
if [[ "$exp_name" == *"cosine"* ]]; then
    noise_schedule="cosine"; dropout="0.3"
elif [[ "$exp_name" == *"ours_v2"* ]]; then
    noise_schedule="ours_v2"; dropout="0.3"
elif [[ "$exp_name" == *"ours"* ]]; then
    noise_schedule="ours"; dropout="0.3"
fi
```

**Evaluation outputs**: Results are written to a timestamped directory, e.g., `/project_gpfs/bata0/bjin0/evaluation_parallel_20260401_143022/`.

### Stage 3: Aggregation

After all evaluation jobs for a dataset finish, aggregate the results into a single summary file:

```bash
bash aggregate_evaluation_results.sh $PARENT_EVAL_DIR                  # CIFAR-10
bash aggregate_fashionmnist_evaluation_results.sh $PARENT_EVAL_DIR     # Fashion-MNIST
bash aggregate_mnist_evaluation_results.sh $PARENT_EVAL_DIR            # MNIST
bash aggregate_imagenet64_evaluation_results.sh $PARENT_EVAL_DIR       # ImageNet-64
```

The `$PARENT_EVAL_DIR` is printed by the evaluation submission script when you run it. If omitted, the aggregation script searches for the latest `*evaluation_parallel_*` directory.

Each aggregation script produces a `results_summary.txt` inside the evaluation directory with NLL, FID, and TV scores for every experiment in that directory.

### Stage 4: Plotting

Copy the `results_summary.txt` files to your local machine, then run the plotting scripts:

```bash
# Bar charts grouped by experiment (one chart per metric per dataset)
python3 plot_nll_tv_bars.py

# Bar charts grouped by objective, comparing schedules side-by-side
python3 plot_grouped_by_objective.py
```

Both scripts read from `results.txt` (or a specified results file) and output PNGs to the `plots/` directory.

## Key Files Reference

### Core Library (`improved_diffusion/`)

| File | Purpose |
|------|---------|
| `gaussian_diffusion.py` | Noise schedule definitions, diffusion process, loss computation |
| `script_util.py` | Model/diffusion creation from CLI args, architecture configs |
| `unet.py` | U-Net model architecture |
| `image_datasets.py` | Dataset loading |
| `train_util.py` / `train_util_no_mpi.py` | Training loop |
| `dist_util.py` / `dist_util_no_mpi.py` | Distributed (MPI) / single-GPU utilities |
| `respace.py` | Timestep respacing for fast sampling |
| `losses.py` | Loss functions |
| `resample.py` / `resample_no_mpi.py` | Timestep importance sampling |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `image_train_no_mpi.py` | Training entrypoint (single-GPU, no MPI) |
| `image_sample_no_mpi.py` | Sample generation from trained model |
| `image_nll_no_mpi.py` | NLL (bits/dim) evaluation on test set |
| `compute_fid_from_npz.py` | FID computation from sample .npz vs cached stats |
| `compute_fid_stats_from_dir_recursive.py` | Precompute FID statistics from image directory (handles class subfolders) |
| `compute_tv_from_npz.py` | TV distance computation |

### SLURM Scripts (top-level)

| File | Purpose |
|------|---------|
| `train_{dataset}_no_mpi.slurm` | Training job (parameterized by `$EXPERIMENT`) |
| `evaluate_models_final.slurm` | CIFAR-10 evaluation job |
| `evaluate_fashionmnist_final.slurm` | Fashion-MNIST evaluation job |
| `evaluate_mnist_final.slurm` | MNIST evaluation job |
| `evaluate_imagenet64_final.slurm` | ImageNet-64 evaluation job |
| `prepare_imagenet64.slurm` | ImageNet-64 dataset preparation job |

### Submission Scripts (top-level)

| File | Purpose |
|------|---------|
| `submit_all_{dataset}.sh` | Submit all 9 training jobs for a dataset |
| `submit_ours_v2_{dataset}.sh` | Submit 3 ours_v2 training jobs for a dataset |
| `submit_{dataset}_evaluation_jobs.sh` | Submit all 9 evaluation jobs |
| `submit_ours_v2_{dataset}_eval.sh` | Submit 3 ours_v2 evaluation jobs |

### Aggregation and Plotting

| File | Purpose |
|------|---------|
| `aggregate_evaluation_results.sh` | Aggregate CIFAR-10 evaluation results |
| `aggregate_fashionmnist_evaluation_results.sh` | Aggregate Fashion-MNIST results |
| `aggregate_mnist_evaluation_results.sh` | Aggregate MNIST results |
| `aggregate_imagenet64_evaluation_results.sh` | Aggregate ImageNet-64 results |
| `plot_nll_tv_bars.py` | Bar charts per metric per dataset |
| `plot_grouped_by_objective.py` | Grouped bar charts comparing schedules by objective |

## Important Notes

- **No MPI**: All scripts use the `_no_mpi` variants. The original MPI-based scripts (`image_train.py`, `image_sample.py`, etc.) exist but are not used.
- **NaN safety**: The sampling, NLL, and FID scripts include fail-fast checks for NaN/Inf values to prevent silent data corruption.
- **FID stats caching**: FID reference statistics are computed once per dataset and cached as `.npz` files. If missing, the evaluation script will compute them automatically.
- **Timestep respacing**: Evaluation uses full 4000-step sampling by default. Set `EVAL_TIMESTEP_RESPACING` environment variable to use fewer steps (faster but potentially lower quality).
- **Resume support**: Training scripts support `RESUME_CHECKPOINT` (specific checkpoint path) and `RESUME_LOGDIR` (auto-find latest checkpoint in directory) for resuming interrupted jobs.

## Path Migration Checklist

The project storage moved from `/project_gpfs/bjin0/` to `/project_gpfs/bata0/bjin0/`. The following files contain hardcoded paths that need updating to reflect the new storage location:

- `train_cifar10_no_mpi.slurm` (OPENAI_LOGDIR paths in every case branch)
- `train_fashionmnist_no_mpi.slurm` (same)
- `train_mnist_no_mpi.slurm` (same)
- `train_imagenet64_no_mpi.slurm` (OPENAI_LOGDIR paths and default IMAGENET_*_DIR)
- `evaluate_models_final.slurm` (checkpoint find path, FID stats path, tmp_root)
- `evaluate_fashionmnist_final.slurm` (same)
- `evaluate_mnist_final.slurm` (same)
- `evaluate_imagenet64_final.slurm` (same, plus IMAGENET_*_DIR defaults)
- `submit_cifar10_evaluation_jobs.sh` (PARENT_EVAL_DIR prefix)
- `submit_fashionmnist_evaluation_jobs.sh` (same)
- `submit_mnist_evaluation_jobs.sh` (same)
- `submit_imagenet64_evaluation_jobs.sh` (same)
- `submit_ours_v2_cifar10_eval.sh` (PARENT_EVAL_DIR prefix)
- `submit_ours_v2_fashionmnist_eval.sh` (same)
- `submit_ours_v2_imagenet64_eval.sh` (same, plus IMAGENET_*_DIR defaults)
- `submit_ours_v2_imagenet64.sh` (IMAGENET_*_DIR defaults)
- `submit_all_imagenet64.sh` (IMAGENET_*_DIR defaults)
- `aggregate_evaluation_results.sh` (fallback search path)
- `aggregate_fashionmnist_evaluation_results.sh` (same)
- `aggregate_mnist_evaluation_results.sh` (same)
- `aggregate_imagenet64_evaluation_results.sh` (same)
- `prepare_imagenet64.slurm` (input/output paths)

In all of these files, replace `/project_gpfs/bjin0` with `/project_gpfs/bata0/bjin0`.
