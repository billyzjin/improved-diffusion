#!/bin/bash

# Aggregates results from parallel Fashion-MNIST evaluation jobs into a summary file.

if [ -n "$1" ]; then
    EVAL_DIR="$1"
    echo "Using provided evaluation directory: $EVAL_DIR"
else
    echo "No directory provided. Searching for the latest one..."
    # Avoid a slow recursive `find` over /project_gpfs (ImageNet contains millions of files).
    EVAL_DIR=$(ls -1d /project_gpfs/bata0/bjin0/fashion_evaluation_parallel_* 2>/dev/null | sort | tail -1)
    if [ -z "$EVAL_DIR" ]; then
        echo "ERROR: No 'fashion_evaluation_parallel_*' directory was found in /project_gpfs/bata0/bjin0"
        exit 1
    fi
    echo "Found latest evaluation directory: $EVAL_DIR"
fi

RESULTS_FILE="$EVAL_DIR/results_summary.txt"

echo "=========================================="
echo "CREATING FASHION-MNIST RESULTS SUMMARY"
echo "Aggregating results from: $EVAL_DIR"
echo "=========================================="

echo "FASHION-MNIST MODEL EVALUATION RESULTS" > "$RESULTS_FILE"
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

for exp_dir in "$EVAL_DIR"/fashionmnist_*; do
    if [ -d "$exp_dir" ] && [ -f "$exp_dir/nll_results.txt" ]; then
        exp_name=$(basename "$exp_dir")
        nll_score=$(grep "done .* samples: bpd=" "$exp_dir/nll_results.txt" | tail -1 | sed -E 's/.*bpd=([0-9.]+).*/\1/')
        if [ -n "$nll_score" ]; then
            printf "%-30s: %s bits/dimension\n" "$exp_name" "$nll_score" >> "$RESULTS_FILE"
        else
            printf "%-30s: ERROR extracting NLL\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "FID RESULTS (lower is better):" >> "$RESULTS_FILE"
echo "==============================" >> "$RESULTS_FILE"

for exp_dir in "$EVAL_DIR"/fashionmnist_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/fid_results.txt" ]; then
            fid_score=$(tail -1 "$exp_dir/fid_results.txt" | tr -d '\r\n' | awk '{print $1}')
            if [ -n "$fid_score" ]; then
                printf "%-30s: %s\n" "$exp_name" "$fid_score" >> "$RESULTS_FILE"
            else
                printf "%-30s: ERROR extracting FID\n" "$exp_name" >> "$RESULTS_FILE"
            fi
        else
            printf "%-30s: FID not found\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "TOTAL VARIATION (TV) RESULTS (lower is better):" >> "$RESULTS_FILE"
echo "===============================================" >> "$RESULTS_FILE"

for exp_dir in "$EVAL_DIR"/fashionmnist_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/tv_results.txt" ]; then
            tv_score=$(tail -1 "$exp_dir/tv_results.txt" | tr -d '\r\n' | awk '{print $1}')
            if [ -n "$tv_score" ]; then
                printf "%-30s: %s\n" "$exp_name" "$tv_score" >> "$RESULTS_FILE"
            else
                printf "%-30s: ERROR extracting TV\n" "$exp_name" >> "$RESULTS_FILE"
            fi
        else
            printf "%-30s: TV not found\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "SAMPLE GENERATION STATUS:" >> "$RESULTS_FILE"
echo "========================" >> "$RESULTS_FILE"

for exp_dir in "$EVAL_DIR"/fashionmnist_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/samples_50000x32x32x3.npz" ]; then
            printf "%-30s: Samples generated successfully\n" "$exp_name" >> "$RESULTS_FILE"
        else
            printf "%-30s: No samples found\n" "$exp_name" >> "$RESULTS_FILE"
        fi
    fi
done

echo "=========================================="
echo "AGGREGATION COMPLETE!"
echo "Results summary saved to: $RESULTS_FILE"
echo "=========================================="


