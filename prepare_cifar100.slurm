#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=standard_l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=prepare_cifar100

set -euo pipefail

module load python/booth/3.12
cd /home/bjin0/improved-diffusion
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"

CIFAR100_ROOT=${CIFAR100_ROOT:-/project_gpfs/bata0/bjin0/cifar100_32x32}
CIFAR100_SOURCE_ROOT=${CIFAR100_SOURCE_ROOT:-${CIFAR100_ROOT}/source}
CIFAR100_OVERWRITE=${CIFAR100_OVERWRITE:-0}

overwrite_arg=()
if [ "$CIFAR100_OVERWRITE" = "1" ]; then
    overwrite_arg=(--overwrite)
fi

python3 datasets/cifar100.py \
    --root "$CIFAR100_SOURCE_ROOT" \
    --out_root "$CIFAR100_ROOT" \
    "${overwrite_arg[@]}"
