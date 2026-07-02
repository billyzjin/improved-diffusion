# SLURM Job Scripts for Improved Diffusion (Pythia Cluster)

This directory contains SLURM job scripts for running improved diffusion experiments on the Pythia cluster at Chicago Booth.

## Available Scripts

### 1. CIFAR-10 Training (`train_cifar10.slurm`)
- **Account**: bata0-external
- **Partition**: standard_hopper
- **Resources**: 1 GPU, 16GB RAM, 4 CPUs, 24 hours
- **Dataset**: CIFAR-10 (automatically prepared)
- **Model**: 32x32 images, 128 channels, 3 residual blocks
- **Training**: 4000 diffusion steps, cosine schedule

### 2. ImageNet-64 Training (`train_imagenet64.slurm`)
- **Account**: bata0-external
- **Partition**: long_hopper
- **Resources**: 4 GPUs, 64GB RAM, 16 CPUs, 72 hours
- **Dataset**: ImageNet-64 (you need to download and prepare)
- **Model**: 64x64 images, 128 channels, 3 residual blocks
- **Training**: Distributed training across 4 GPUs

### 3. LSUN Bedroom/Church-64 Training (`submit_image_folder_full_slate.sh`)
- **Account**: bata0-external
- **Partition**: long_hopper for training, standard_l40s for prep
- **Resources**: 1 GPU, 64GB RAM, 8 CPUs for each training job
- **Dataset**: LSUN LMDB source, resized to 64x64 at load time
- **Model**: 64x64 images, 128 channels, 3 residual blocks
- **Training**: image-folder submitter supports linear, cosine, geometric_linear, and geometric_cosine

## How to Use

### 1. Upload to Pythia Cluster
```bash
# Copy scripts to Pythia cluster
scp *.slurm pythia.uchicago.edu:/home/$USER/improved-diffusion/
```

### 2. Submit Jobs on Pythia
```bash
# SSH into Pythia cluster
ssh pythia.uchicago.edu

# Navigate to project directory
cd /home/$USER/improved-diffusion

# Submit CIFAR-10 job
sbatch train_cifar10.slurm

# Submit ImageNet-64 job (after preparing dataset)
sbatch train_imagenet64.slurm

# Prepare LSUN Bedroom-64, then submit the reduced slate
sbatch prepare_lsun_bedroom64.slurm
DATASET=lsun_bedroom64 SCHEDULES=linear,cosine,geometric_linear,geometric_cosine ./submit_image_folder_full_slate.sh

# Or prepare LSUN Church-Outdoor-64, then submit the reduced slate
sbatch prepare_lsun_church64.slurm
DATASET=lsun_church64 SCHEDULES=linear,cosine,geometric_linear,geometric_cosine ./submit_image_folder_full_slate.sh
```

### 3. Monitor Jobs
```bash
# Check job status
squeue -u $USER

# View output
tail -f cifar10_<job_id>.out

# Check for errors
tail -f cifar10_<job_id>.err
```

### 4. Download Results
```bash
# After job completes, download results
rsync -avz your-cluster.edu:/tmp/improved_diffusion_logs/ ./local_results/
```

## Customization

### Adjust Resources
Edit the SLURM directives at the top of each script:
```bash
#SBATCH --gres=gpu:1          # Number of GPUs
#SBATCH --mem=16GB             # Memory
#SBATCH --time=24:00:00        # Time limit
#SBATCH --partition=gpu        # Partition name
```

### Adjust Hyperparameters
Modify the MODEL_FLAGS, DIFFUSION_FLAGS, and TRAIN_FLAGS variables:
```bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

### Memory Optimization
If you run out of memory, add microbatching:
```bash
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 16"
```

## Dataset Preparation

### CIFAR-10
Automatically prepared by the script.

### ImageNet-64
1. Download from: http://www.image-net.org/small/download.php
2. Extract to your data directory
3. Update the `--data_dir` path in the script

### LSUN Bedroom and Church Outdoor
Run:

```bash
sbatch prepare_lsun_bedroom64.slurm
# or
sbatch prepare_lsun_church64.slurm
```

The prep jobs download the official LSUN LMDB zips from `dl.yf.io`, extract them under the matching GPFS source directory, and write `_source_manifest.tsv`. Bedroom uses `/project_gpfs/bata0/bjin0/lsun_bedroom_64x64`; Church Outdoor uses `/project_gpfs/bata0/bjin0/lsun_church_64x64`.

By default, LSUN training and evaluation read the LMDBs directly to avoid consuming millions of GPFS inodes. To materialize PNG folders anyway, submit the prep job with `LSUN_CONVERT_TO_PNG=1`.

After the prep job completes, use:

```bash
DATASET=lsun_bedroom64 ./submit_image_folder_full_slate.sh
# or
DATASET=lsun_church64 ./submit_image_folder_full_slate.sh
```

For the current reduced dataset plan, set `SCHEDULES=linear,cosine,geometric_linear,geometric_cosine`. The default objectives are `simple,hybrid,vlb`; override `OBJECTIVES` if needed.

### Hybrid VB Weight Tuning

The hybrid objective defaults to the original VB weight `0.001`. To run a grid over hybrid weights for datasets supported by `submit_image_folder_full_slate.sh`, use:

```bash
DATASET=celeba64 HYBRID_VB_WEIGHTS=0,1e-4,3e-4,1e-3,3e-3,1e-2 ./submit_hybrid_weight_grid.sh
```

Existing submitters keep their original behavior unless `HYBRID_VB_WEIGHT` or `HYBRID_VB_WEIGHTS` is set.

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 64`
- Add microbatching: `--microbatch 16`
- Reduce model size: `--num_channels 64`

### Long Training Time
- Reduce steps for testing: `--max_steps 10000`
- Use fewer diffusion steps: `--diffusion_steps 1000`

### GPU Issues
- Check GPU availability: `nvidia-smi`
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
