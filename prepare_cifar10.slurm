#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=standard_l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=prepare_cifar10

set -euo pipefail

module load python/booth/3.12
cd /home/bjin0/improved-diffusion
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"

python3 datasets/cifar10.py
