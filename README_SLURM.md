# SLURM Job Scripts for Improved Diffusion (Pythia Cluster)

This directory contains SLURM job scripts for running improved diffusion experiments on the Pythia cluster at Chicago Booth.

## Available Scripts

### 1. CIFAR-10 Training (`train_cifar10.slurm`)
- **Account**: bata0-external
- **Partition**: standard_h100
- **Resources**: 1 GPU, 16GB RAM, 4 CPUs, 24 hours
- **Dataset**: CIFAR-10 (automatically prepared)
- **Model**: 32x32 images, 128 channels, 3 residual blocks
- **Training**: 4000 diffusion steps, cosine schedule

### 2. ImageNet-64 Training (`train_imagenet64.slurm`)
- **Account**: bata0-external
- **Partition**: long_h100
- **Resources**: 4 GPUs, 64GB RAM, 16 CPUs, 72 hours
- **Dataset**: ImageNet-64 (you need to download and prepare)
- **Model**: 64x64 images, 128 channels, 3 residual blocks
- **Training**: Distributed training across 4 GPUs

### 3. LSUN Bedroom Training (`train_lsun.slurm`)
- **Account**: bata0-external
- **Partition**: long_h100
- **Resources**: 4 GPUs, 128GB RAM, 16 CPUs, 120 hours
- **Dataset**: LSUN bedroom (you need to download and prepare)
- **Model**: 256x256 images, 128 channels, 2 residual blocks
- **Training**: Distributed training across 4 GPUs

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

# Submit LSUN job (after preparing dataset)
sbatch train_lsun.slurm
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

### LSUN Bedroom
1. Clone: `git clone https://github.com/fyu/lsun.git`
2. Download: `python3 lsun/download.py bedroom`
3. Convert: `python datasets/lsun_bedroom.py lsun/bedroom_train_lmdb lsun_output`
4. Update the `--data_dir` path in the script

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
