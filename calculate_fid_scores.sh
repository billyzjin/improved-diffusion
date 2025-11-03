#!/bin/bash

# This script calculates the FID score for the 6 successfully completed experiments.
# It installs the necessary tools, pre-calculates real dataset statistics,
# and then computes the FID for each set of generated samples.

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
    echo "ERROR: CIFAR-10 training directory not found at '$CIFAR_TRAIN_PATH'"
    echo "Please ensure the dataset is present in your project root before calculating FID."
    exit 1
fi

# 5. Pre-calculate statistics for the real CIFAR-10 training set (if they don't exist).
if [ ! -f "$CIFAR_STATS_FILE" ]; then
    echo ""
    echo "--- Pre-calculating statistics for the real CIFAR-10 dataset ---"
    echo "This is a one-time operation. Statistics will be saved to $CIFAR_STATS_FILE"
    # Corrected Command: Use --save-stats flag with the output file as its argument.
    python3 -m pytorch_fid "$CIFAR_TRAIN_PATH" --save-stats "$CIFAR_STATS_FILE" --device cuda
else
    echo "--- Found pre-calculated CIFAR-10 statistics. Skipping calculation. ---"
fi

# 6. Prepare a file to store the final FID results.
FID_RESULTS_FILE="$LATEST_EVAL_DIR/fid_summary.txt"
echo "FID CALCULATION RESULTS (lower is better)" > "$FID_RESULTS_FILE"
echo "========================================" >> "$FID_RESULTS_FILE"
echo "Date: $(date)" >> "$FID_RESULTS_FILE"
echo "" >> "$FID_RESULTS_FILE"

echo ""
echo "--- Calculating FID for each experiment ---"

# 7. Loop through the experiment sub-directories and calculate FID.
for exp_dir in "$LATEST_EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        
        # Skip the VLB experiment for now as it's being retrained.
        if [[ "$exp_name" == *"vlb"* ]]; then
            echo "--> Skipping $exp_name (this model is being retrained)"
            continue
        fi

        echo "--> Processing $exp_name..."
        
        sample_file_path="${exp_dir}/samples_50000x32x32x3.npz"

        if [ -f "$sample_file_path" ]; then
            echo "    Calculating FID score..."
            # For comparison, the two paths are positional arguments. This is correct.
            fid_score=$(python3 -m pytorch_fid --device cuda "$sample_file_path" "$CIFAR_STATS_FILE")
            
            echo "    Done. FID score for $exp_name: $fid_score"
            printf "%-25s: %s\n" "$exp_name" "$fid_score" >> "$FID_RESULTS_FILE"
        else
            echo "    WARNING: Sample file not found for $exp_name. Cannot calculate FID."
            printf "%-25s: Sample file not found\n" "$exp_name" >> "$FID_RESULTS_FILE"
        fi
    fi
done

# Add paper comparison to the summary file
echo "" >> "$FID_RESULTS_FILE"
echo "PAPER BASELINE COMPARISON (FID):" >> "$FID_RESULTS_FILE"
echo "================================" >> "$FID_RESULTS_FILE"
echo "- linear, L_simple (ours: linear_simple) : 2.90" >> "$FID_RESULTS_FILE"
echo "- linear, L_hybrid (ours: linear_hybrid) : 3.07" >> "$FID_RESULTS_FILE"
echo "- cosine, L_simple (ours: cosine_simple) : 3.05" >> "$FID_RESULTS_FILE"
echo "- cosine, L_hybrid (ours: cosine_hybrid) : 3.19" >> "$FID_RESULTS_FILE"

echo ""
echo "=========================================="
echo "FID CALCULATION COMPLETE!"
echo "Results summary saved to: $FID_RESULTS_FILE"
echo ""
echo "--- Final FID Results Summary ---"
cat "$FID_RESULTS_FILE"
echo "=========================================="
