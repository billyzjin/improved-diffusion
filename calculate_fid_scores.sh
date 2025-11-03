#!/bin/bash

# This script calculates the FID score for the successfully completed experiments.
# It uses the command-line interface of pytorch-fid in a robust way.

set -e # Exit immediately if a command exits with a non-zero status.

# 1. Set up the environment by loading necessary modules.
echo "--- Loading Python and CUDA modules ---"
module load python/booth/3.12
module load cuda/12.8

# 2. Find the most recent parent evaluation directory.
LATEST_EVAL_DIR=$(find /scratch/bjin0 -name "evaluation_parallel_*" -type d 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_EVAL_DIR" ]; then
    echo "ERROR: No 'evaluation_parallel_*' directory was found in /scratch/bjin0"
    exit 1
fi

echo "Calculating FID for results located in: $LATEST_EVAL_DIR"

# 3. Install the required FID calculation tool and its dependencies.
echo ""
echo "--- Installing/Updating pytorch-fid and dependencies ---"
pip3 install --upgrade pytorch-fid scipy

# 4. Set paths for the real dataset and the pre-calculated statistics file.
CIFAR_TRAIN_PATH="./cifar_train"
CIFAR_STATS_FILE="/scratch/bjin0/cifar10_train_stats.npz" # Saved in scratch to persist

if [ ! -d "$CIFAR_TRAIN_PATH" ]; then
    echo "ERROR: Real dataset not found at $CIFAR_TRAIN_PATH"
    echo "Please ensure the 'cifar_train' directory exists in the current folder."
    exit 1
fi

# 5. Pre-calculate statistics for the real CIFAR-10 dataset if they don't exist.
if [ ! -f "$CIFAR_STATS_FILE" ]; then
    echo ""
    echo "--- Pre-calculating statistics for the real CIFAR-10 dataset ---"
    echo "This is a one-time operation. Statistics will be saved to $CIFAR_STATS_FILE"
    # THE CORRECT COMMAND: The tool takes two paths, and --save-stats changes the mode.
    python3 -m pytorch_fid "$CIFAR_TRAIN_PATH" "$CIFAR_STATS_FILE" --save-stats --device cuda
else
    echo "--- Found pre-calculated CIFAR-10 statistics. Skipping calculation. ---"
fi

# 6. Prepare a file to store the final FID results.
FID_RESULTS_FILE="$LATEST_EVAL_DIR/fid_scores.txt"
echo "FID SCORES (lower is better):" > "$FID_RESULTS_FILE"
echo "=============================" >> "$FID_RESULTS_FILE"

# 7. Loop through all experiment sub-directories and calculate FID.
echo ""
echo "--- Calculating FID Scores for Generated Samples ---"
for exp_dir in "$LATEST_EVAL_DIR"/*/; do
    exp_name=$(basename "$exp_dir")
    sample_file_path="${exp_dir}samples_50000x32x32x3.npz"
    
    echo ""
    echo "Processing: $exp_name"
    
    if [ -f "$sample_file_path" ]; then
        echo "    Calculating FID score..."
        # THE CORRECT COMMAND: For comparison, provide the two paths to compare.
        # The output of the command is just the FID score, so we can capture it directly.
        fid_score=$(python3 -m pytorch_fid "$sample_file_path" "$CIFAR_STATS_FILE" --device cuda)
        
        echo "    Done. FID score for $exp_name: $fid_score"
        printf "%-25s: %s\n" "$exp_name" "$fid_score" >> "$FID_RESULTS_FILE"
    else
        echo "    WARNING: Sample file not found at $sample_file_path"
        printf "%-25s: SAMPLES NOT FOUND\n" "$exp_name" >> "$FID_RESULTS_FILE"
    fi
done

echo ""
echo "=========================================="
echo "FID CALCULATION COMPLETE!"
echo "Results summary saved to: $FID_RESULTS_FILE"
echo "=========================================="

# 8. Print the final summary to the console.
echo ""
echo "--- Final FID Results ---"
cat "$FID_RESULTS_FILE"
