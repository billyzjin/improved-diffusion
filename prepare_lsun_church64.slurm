#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=standard_l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=prepare_lsun_church64

set -euo pipefail

export LSUN_ROOT=${LSUN_ROOT:-/project_gpfs/bata0/bjin0/lsun_church_64x64}
export LSUN_CATEGORY=${LSUN_CATEGORY:-church_outdoor}
export LSUN_CONVERT_TO_PNG=${LSUN_CONVERT_TO_PNG:-0}

bash /home/bjin0/improved-diffusion/prepare_lsun_bedroom64.slurm
