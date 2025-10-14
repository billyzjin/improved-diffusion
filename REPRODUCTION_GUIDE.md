# Guide to Reproducing Paper Results for CIFAR-10

This guide explains how to reproduce the experimental results from Table 2 of the "Improved Denoising Diffusion Probabilistic Models" paper using the provided SLURM scripts.

## Quick Start

The main script `train_cifar10_no_mpi.slurm` can run any of the 5 experiments from the paper by setting the `EXPERIMENT` environment variable.

### Submit Individual Experiments

```bash
# Experiment 1: Linear schedule, L_simple (Best FID: 2.90)
sbatch --export=EXPERIMENT=linear_simple train_cifar10_no_mpi.slurm

# Experiment 2: Linear schedule, L_hybrid
sbatch --export=EXPERIMENT=linear_hybrid train_cifar10_no_mpi.slurm

# Experiment 3: Cosine schedule, L_simple
sbatch --export=EXPERIMENT=cosine_simple train_cifar10_no_mpi.slurm

# Experiment 4: Cosine schedule, L_hybrid
sbatch --export=EXPERIMENT=cosine_hybrid train_cifar10_no_mpi.slurm

# Experiment 5: Cosine schedule, L_vlb (Best NLL: 2.94)
sbatch --export=EXPERIMENT=cosine_vlb train_cifar10_no_mpi.slurm
```

### Submit All Experiments at Once

```bash
# Submit all 5 experiments (they will run in parallel if resources are available)
for exp in linear_simple linear_hybrid cosine_simple cosine_hybrid cosine_vlb; do
    sbatch --export=EXPERIMENT=$exp train_cifar10_no_mpi.slurm
done
```

## Expected Results (from Paper - Table 2)

| Experiment | Schedule | Objective | Dropout | NLL (bits/dim) | FID |
|------------|----------|-----------|---------|----------------|-----|
| linear_simple | Linear | L_simple | 0.1 | 3.37 | **2.90** |
| linear_hybrid | Linear | L_hybrid | 0.1 | 3.26 | 3.07 |
| cosine_simple | Cosine | L_simple | 0.3 | 3.26 | 3.05 |
| cosine_hybrid | Cosine | L_hybrid | 0.3 | 3.17 | 3.19 |
| cosine_vlb | Cosine | L_vlb | 0.3 | **2.94** | 11.47 |

## Training Details

- **Duration**: ~12-24 hours per experiment on H100 GPU
- **Iterations**: 500,000 training steps each
- **Batch Size**: 128
- **Diffusion Steps**: 4,000
- **Model**: UNet with [128, 256, 256, 256] channels, 3 residual blocks per stage
- **Attention**: 4 heads at resolutions 16×16 and 8×8

## Output Locations

Results are saved in separate directories:
```
/home/bjin0/improved-diffusion/logs/
├── cifar10_linear_simple/
├── cifar10_linear_hybrid/
├── cifar10_cosine_simple/
├── cifar10_cosine_hybrid/
└── cifar10_cosine_vlb/
```

Each directory contains:
- Model checkpoints (saved every 10,000 iterations)
- EMA model checkpoints (rate 0.9999)
- Training logs

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check job output (replace JOBID with actual job ID)
tail -f cifar10_paper_JOBID.out

# Check for errors
tail -f cifar10_paper_JOBID.err
```

## Evaluating Results

After training completes, evaluate the models:

### Generate Samples for FID Calculation

```bash
# Generate 50,000 samples (required for FID)
python3 scripts/image_sample.py \
    --model_path logs/cifar10_linear_simple/ema_0.9999_500000.pt \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --attention_resolutions 16,8 \
    --num_heads 4 \
    --use_scale_shift_norm True \
    --diffusion_steps 4000 \
    --noise_schedule linear \
    --num_samples 50000 \
    --batch_size 256 \
    --timestep_respacing 250
```

### Calculate FID

Use your preferred FID calculation tool to compare the generated samples against the CIFAR-10 training set.

### Calculate NLL

```bash
python3 scripts/image_nll.py \
    --model_path logs/cifar10_cosine_vlb/ema_0.9999_500000.pt \
    --data_dir cifar_train \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --attention_resolutions 16,8 \
    --num_heads 4 \
    --use_scale_shift_norm True \
    --diffusion_steps 4000 \
    --noise_schedule cosine \
    --learn_sigma True \
    --rescale_learned_sigmas True \
    --use_kl True
```

## Troubleshooting

### Job Fails Immediately
- Check SLURM error file: `cifar10_paper_JOBID.err`
- Verify CIFAR-10 dataset exists in project directory
- Ensure sufficient disk space in `/home/bjin0/improved-diffusion/logs/`

### Out of Memory
- Reduce batch size: Change `--batch_size 128` to `--batch_size 64`
- Note: This will affect training dynamics and may not match paper exactly

### Job Timeout
- Training might take longer than 24 hours
- Consider using `--lr_anneal_steps 250000` for faster (but less accurate) training

## Notes

- The script automatically applies patches to remove MPI dependencies
- Original files are restored after training completes
- All experiments use single-GPU training (no distributed training)
- Results may vary slightly from paper due to:
  - Hardware differences (H100 vs V100)
  - PyTorch version differences
  - Random seed variations
  - Single-GPU vs multi-GPU training

## Citation

If you use these scripts or reproduce the paper results, please cite:

```bibtex
@inproceedings{nichol2021improved,
  title={Improved denoising diffusion probabilistic models},
  author={Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle={International Conference on Machine Learning},
  pages={8162--8171},
  year={2021},
  organization={PMLR}
}
```

