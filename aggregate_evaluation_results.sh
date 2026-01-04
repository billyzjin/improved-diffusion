#!/bin/bash

# This script aggregates the results from parallel evaluation jobs into a single summary file.
# It should be run AFTER all the parallel Slurm jobs have completed successfully.

# If a path is provided, use it. Otherwise, find the latest evaluation directory automatically.
if [ -n "$1" ]; then
    EVAL_DIR="$1"
    echo "Using provided evaluation directory: $EVAL_DIR"
else
    echo "No directory provided. Searching for the latest one..."
    EVAL_DIR=$(find /project_gpfs/bjin0 -name "evaluation_parallel_*" -type d 2>/dev/null | sort | tail -1)
    
    if [ -z "$EVAL_DIR" ]; then
        echo "ERROR: No 'evaluation_parallel_*' directory was found in /project_gpfs/bjin0"
        exit 1
    fi
    echo "Found latest evaluation directory: $EVAL_DIR"
fi

RESULTS_FILE="$EVAL_DIR/results_summary.txt"

echo "=========================================="
echo "CREATING RESULTS SUMMARY"
echo "Aggregating results from: $EVAL_DIR"
echo "=========================================="

# Create the header for the summary file.
echo "COMPREHENSIVE MODEL EVALUATION RESULTS" > "$RESULTS_FILE"
echo "======================================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "EXPERIMENT TYPES:" >> "$RESULTS_FILE"
echo "=================" >> "$RESULTS_FILE"
echo "- Simple: learn_sigma=False, use_kl=False" >> "$RESULTS_FILE"
echo "- Hybrid: learn_sigma=True, use_kl=False, rescale_learned_sigmas=True" >> "$RESULTS_FILE"
echo "- VLB: learn_sigma=True, use_kl=True, schedule_sampler=loss-second-moment" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "NLL RESULTS (bits/dimension - lower is better):" >> "$RESULTS_FILE"
echo "==============================================" >> "$RESULTS_FILE"

# Extract NLL results from each sub-directory.
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ] && [ -f "$exp_dir/nll_results.txt" ]; then
        exp_name=$(basename "$exp_dir")
        # Extract the final bpd score from the log file.
        # Look for any line with "done ... samples: bpd=" and take the last one
        # Use sed to extract just the number after bpd= and before any subsequent text
        nll_score=$(grep "done .* samples: bpd=" "$exp_dir/nll_results.txt" | tail -1 | sed -E 's/.*bpd=([0-9.]+).*/\1/')
        
        if [ -n "$nll_score" ]; then
            printf "%-25s: %s bits/dimension\n" "$exp_name" "$nll_score" >> "$RESULTS_FILE"
        else
            printf "%-25s: ERROR extracting NLL\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "FID RESULTS (lower is better):" >> "$RESULTS_FILE"
echo "==============================" >> "$RESULTS_FILE"

# Extract FID results from each sub-directory.
# Each eval job writes a single number to fid_results.txt.
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/fid_results.txt" ]; then
            fid_score=$(tail -1 "$exp_dir/fid_results.txt" | tr -d '\r\n' | awk '{print $1}')
            if [ -n "$fid_score" ]; then
                printf "%-25s: %s\n" "$exp_name" "$fid_score" >> "$RESULTS_FILE"
            else
                printf "%-25s: ERROR extracting FID\n" "$exp_name" >> "$RESULTS_FILE"
            fi
        else
            printf "%-25s: FID not found\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "TOTAL VARIATION (TV) RESULTS (lower is better):" >> "$RESULTS_FILE"
echo "===============================================" >> "$RESULTS_FILE"

for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/tv_results.txt" ]; then
            tv_score=$(tail -1 "$exp_dir/tv_results.txt" | tr -d '\r\n' | awk '{print $1}')
            if [ -n "$tv_score" ]; then
                printf "%-25s: %s\n" "$exp_name" "$tv_score" >> "$RESULTS_FILE"
            else
                printf "%-25s: ERROR extracting TV\n" "$exp_name" >> "$RESULTS_FILE"
            fi
        else
            printf "%-25s: TV not found\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "SAMPLE GENERATION STATUS (for FID calculation):" >> "$RESULTS_FILE"
echo "==============================================" >> "$RESULTS_FILE"

# Check the status of the generated sample files in each sub-directory.
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/samples_50000x32x32x3.npz" ]; then
            printf "%-25s: Samples generated successfully\n" "$exp_name" >> "$RESULTS_FILE"
        else
            printf "%-25s: No samples found\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "PAPER BASELINE COMPARISON:" >> "$RESULTS_FILE"
echo "=========================" >> "$RESULTS_FILE"
echo "From Table 2 of the paper:" >> "$RESULTS_FILE"
echo "- linear, L_simple (ours: linear_simple) : 3.37 bpd" >> "$RESULTS_FILE"
echo "- linear, L_hybrid (ours: linear_hybrid) : 3.26 bpd" >> "$RESULTS_FILE"
echo "- cosine, L_simple (ours: cosine_simple) : 3.26 bpd" >> "$RESULTS_FILE"
echo "- cosine, L_hybrid (ours: cosine_hybrid) : 3.17 bpd" >> "$RESULTS_FILE"
echo "- cosine, L_vlb    (ours: cosine_vlb)    : 2.94 bpd" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Your custom 'ours' schedule can be compared against these baselines." >> "$RESULTS_FILE"

echo "=========================================="
echo "AGGREGATION COMPLETE!"
echo "Results summary saved to: $RESULTS_FILE"
echo ""
echo "To view the final results, run:"
echo "  cat $RESULTS_FILE"
echo "=========================================="
