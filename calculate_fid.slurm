#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=fid_calc
#SBATCH --output=fid_calc_%j.out
#SBATCH --error=fid_calc_%j.err

echo "=========================================="
echo "FID CALCULATION JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Choose evaluation directory (argument 1) or default to latest run
EVAL_DIR="${1:-/project_gpfs/bjin0/evaluation_parallel_20251206_132203}"
if [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: Evaluation directory not found: $EVAL_DIR"
    exit 1
fi
echo "Using evaluation directory: $EVAL_DIR"

# Load Python
echo "Loading python module..."
module load python/booth/3.12 || { echo "Failed to load python/booth/3.12"; exit 1; }

# Install pytorch-fid locally (cached if already installed)
echo "Ensuring pytorch-fid is installed..."
pip install --user pytorch-fid

# Verify pytorch-fid
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("pytorch_fid")
print("pytorch_fid found:", spec is not None, "->", spec.origin if spec else "missing")
PY

# Run FID calculation
echo "Running calculate_fid_scores.py ..."
cd /home/bjin0/improved-diffusion
python3 calculate_fid_scores.py "$EVAL_DIR"

echo "=========================================="
echo "FID JOB COMPLETE"
echo "Results (if successful) in: $EVAL_DIR/fid_scores.txt"
echo "=========================================="

