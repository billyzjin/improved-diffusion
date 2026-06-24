#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=standard_l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=prepare_celeba64

set -euo pipefail

module load python/booth/3.12
cd /home/bjin0/improved-diffusion
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"

if [ -f /home/bjin0/improved-diffusion/.venv/bin/activate ]; then
    source /home/bjin0/improved-diffusion/.venv/bin/activate
fi

CELEBA_ROOT=${CELEBA_ROOT:-/project_gpfs/bata0/bjin0/celeba_64x64}
CELEBA_SOURCE_ROOT=${CELEBA_SOURCE_ROOT:-${CELEBA_ROOT}/source}
CELEBA_OVERWRITE=${CELEBA_OVERWRITE:-0}

overwrite_arg=()
if [ "$CELEBA_OVERWRITE" = "1" ]; then
    overwrite_arg=(--overwrite)
fi

python3 datasets/celeba64.py \
    --root "$CELEBA_SOURCE_ROOT" \
    --out_root "$CELEBA_ROOT" \
    "${overwrite_arg[@]}"
