#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=standard_hopper
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=tv_calc
#SBATCH --output=tv_calc_%j.out
#SBATCH --error=tv_calc_%j.err

echo "=========================================="
echo "TV CALCULATION JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Usage:
#   sbatch calculate_tv.slurm [evaluation_dir] [real_dir]
# Defaults:
#   evaluation_dir: auto-detect latest /project_gpfs/bjin0/evaluation_parallel_*
#   real_dir: ./cifar_test

# Resolve evaluation directory
if [ -n "$1" ]; then
    EVAL_DIR="$1"
else
    echo "No evaluation directory provided. Searching for the latest evaluation_parallel_* ..."
    EVAL_DIR=$(find /project_gpfs/bjin0 -maxdepth 2 -type d -name "evaluation_parallel_*" 2>/dev/null | sort | tail -1)
fi

REAL_DIR="${2:-./cifar_test}"

if [ -z "$EVAL_DIR" ] || [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: Evaluation directory not found."
    echo "Provide it explicitly: sbatch calculate_tv.slurm /path/to/evaluation_parallel_xxx [real_dir]"
    exit 1
fi
if [ ! -d "$REAL_DIR" ]; then
    echo "ERROR: Real dataset directory not found: $REAL_DIR"
    exit 1
fi

echo "Using evaluation directory: $EVAL_DIR"
echo "Using real dataset directory: $REAL_DIR"

# Load Python
echo "Loading python module..."
module load python/booth/3.12 || { echo "Failed to load python/booth/3.12"; exit 1; }

# Run TV calculation (CPU-based)
echo "Running calculate_tv_distance.py ..."
cd /home/bjin0/improved-diffusion
python3 calculate_tv_distance.py "$EVAL_DIR" --real_dir "$REAL_DIR"

echo "=========================================="
echo "TV JOB COMPLETE"
echo "Results (if successful) in: $EVAL_DIR/tv_results.txt"
echo "=========================================="

