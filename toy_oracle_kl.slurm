#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=standard_l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=toy_oracle_kl

set -euo pipefail

module load python/booth/3.12

cd /home/bjin0/improved-diffusion

OUTPUT_DIR=${OUTPUT_DIR:-/project_gpfs/bata0/bjin0/toy_oracle_kl_${SLURM_JOB_ID}}
TOY_DISTRIBUTIONS=${TOY_DISTRIBUTIONS:-}
TOY_TIMESTEPS=${TOY_TIMESTEPS:-50 100 250 1000}
TOY_SCHEDULES=${TOY_SCHEDULES:-linear cosine geometric_linear geometric_cosine linabar_linear linabar_cosine}
TOY_N_X_1D=${TOY_N_X_1D:-20000}
TOY_M_POST_1D=${TOY_M_POST_1D:-64}
TOY_N_X_2D=${TOY_N_X_2D:-10000}
TOY_M_POST_2D=${TOY_M_POST_2D:-32}
TOY_SEED=${TOY_SEED:-1234}
TOY_X_CHUNK_SIZE=${TOY_X_CHUNK_SIZE:-1024}

cmd=(
  python3 scripts/run_toy_oracle_kl.py
  --output_dir "$OUTPUT_DIR"
  --timesteps $TOY_TIMESTEPS
  --schedules $TOY_SCHEDULES
  --n_x_1d "$TOY_N_X_1D"
  --m_post_1d "$TOY_M_POST_1D"
  --n_x_2d "$TOY_N_X_2D"
  --m_post_2d "$TOY_M_POST_2D"
  --seed "$TOY_SEED"
  --x_chunk_size "$TOY_X_CHUNK_SIZE"
)

if [ -n "$TOY_DISTRIBUTIONS" ]; then
  cmd+=(--distributions $TOY_DISTRIBUTIONS)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
