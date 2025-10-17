#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <evaluation_directory>"
    echo "Example: $0 /scratch/bjin0/evaluation_20241201_143022"
    exit 1
fi

EVAL_DIR="$1"

if [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: Directory not found: $EVAL_DIR"
    exit 1
fi

echo "=========================================="
echo "COMPARING EVALUATION RESULTS"
echo "=========================================="

# Create results summary
RESULTS_FILE="$EVAL_DIR/results_comparison.txt"
echo "DIFFUSION MODEL EVALUATION RESULTS" > "$RESULTS_FILE"
echo "=================================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "NLL RESULTS (bits/dimension - lower is better):" >> "$RESULTS_FILE"
echo "==============================================" >> "$RESULTS_FILE"

# Extract NLL results
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ] && [ -f "$exp_dir/nll_results.txt" ]; then
        exp_name=$(basename "$exp_dir")
        nll_score=$(grep "done 10000 samples: bpd=" "$exp_dir/nll_results.txt" | tail -1 | awk '{print $4}')
        
        if [ -n "$nll_score" ]; then
            echo "$exp_name: $nll_score bits/dimension" >> "$RESULTS_FILE"
            echo "$exp_name: $nll_score bits/dimension"
        else
            echo "$exp_name: ERROR extracting NLL" >> "$RESULTS_FILE"
            echo "$exp_name: ERROR extracting NLL"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "SAMPLE GENERATION STATUS:" >> "$RESULTS_FILE"
echo "========================" >> "$RESULTS_FILE"

# Check sample generation status
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/samples_50000x32x32x3.npz" ]; then
            echo "$exp_name: Samples generated successfully" >> "$RESULTS_FILE"
            echo "$exp_name: Samples generated successfully"
        else
            echo "$exp_name: No samples found" >> "$RESULTS_FILE"
            echo "$exp_name: No samples found"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "PAPER BASELINE COMPARISON:" >> "$RESULTS_FILE"
echo "=========================" >> "$RESULTS_FILE"
echo "From Table 2 of the paper:" >> "$RESULTS_FILE"
echo "- Linear Simple: ~3.17 bits/dimension" >> "$RESULTS_FILE"
echo "- Cosine Simple: ~3.17 bits/dimension" >> "$RESULTS_FILE"
echo "- Linear Hybrid: ~3.11 bits/dimension" >> "$RESULTS_FILE"
echo "- Cosine Hybrid: ~3.11 bits/dimension" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Your custom 'ours' schedule should be compared against these baselines." >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "RESULTS COMPARISON COMPLETE!"
echo "=========================================="
echo "Full results saved to: $RESULTS_FILE"
echo ""
echo "SUMMARY:"
cat "$RESULTS_FILE"
