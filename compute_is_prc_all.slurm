#!/bin/bash
#SBATCH --partition=long_hopper
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=bata0-external
#SBATCH --job-name=is_prc_all
#SBATCH --output=slurm_logs/is_prc_all_%j.out
#SBATCH --error=slurm_logs/is_prc_all_%j.err
set -o pipefail

cd /home/bjin0/improved-diffusion

# Baselines (from Jan 2026 evaluation)
BASELINE_DIR="/project_gpfs/bata0/bjin0/evaluation_parallel_20260121_095053"
# Geometric (from Apr 2026 evaluation)
GEOMETRIC_DIR="/project_gpfs/bata0/bjin0/evaluation_parallel_geometric_20260410_231224"

EXPERIMENTS=(
    "$BASELINE_DIR/cifar10_linear_simple"
    "$BASELINE_DIR/cifar10_linear_hybrid"
    "$BASELINE_DIR/cifar10_cosine_simple"
    "$BASELINE_DIR/cifar10_cosine_hybrid"
    "$BASELINE_DIR/cifar10_cosine_vlb"
    "$BASELINE_DIR/cifar10_ours_simple"
    "$BASELINE_DIR/cifar10_ours_hybrid"
    "$GEOMETRIC_DIR/cifar10_geometric_linear_simple"
    "$GEOMETRIC_DIR/cifar10_geometric_linear_hybrid"
    "$GEOMETRIC_DIR/cifar10_geometric_linear_vlb"
    "$GEOMETRIC_DIR/cifar10_geometric_cosine_simple"
    "$GEOMETRIC_DIR/cifar10_geometric_cosine_hybrid"
    "$GEOMETRIC_DIR/cifar10_geometric_cosine_vlb"
)

echo "Computing IS and Precision/Recall for ${#EXPERIMENTS[@]} experiments"
echo ""

for exp_dir in "${EXPERIMENTS[@]}"; do
    exp_name=$(basename "$exp_dir")
    samples=$(ls "$exp_dir"/samples_*.npz 2>/dev/null | head -1)

    if [ -z "$samples" ]; then
        echo "SKIP $exp_name: no samples found"
        continue
    fi

    echo "=== $exp_name ==="
    python3 scripts/compute_is_prc.py \
        --samples_npz "$samples" \
        --real_dir ./cifar_test \
        --output_txt "$exp_dir/is_prc_results.txt" \
        2>&1
    echo ""
done

echo "ALL DONE"
