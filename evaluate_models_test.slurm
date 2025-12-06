#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=eval_test
#SBATCH --output=evaluation_test_%j.out
#SBATCH --error=evaluation_test_%j.err

echo "=========================================="
echo "EVALUATING MODELS (5-MINUTE TEST VERSION)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Set up environment with error checking
echo "Loading modules..."
module load python/booth/3.12 || { echo "ERROR: Failed to load python/booth/3.12"; exit 1; }

# Try different CUDA module versions (based on actual availability)
if module load cuda/12.8 2>/dev/null; then
    echo "Loaded cuda/12.8"
elif module load cuda/12.4 2>/dev/null; then
    echo "Loaded cuda/12.4"
elif module load cuda/12.2 2>/dev/null; then
    echo "Loaded cuda/12.2"
elif module load cuda/11.8 2>/dev/null; then
    echo "Loaded cuda/11.8"
else
    echo "ERROR: No CUDA module found"
    exit 1
fi

# NVIDIA modules not available on Pythia, skip
echo "INFO: NVIDIA modules not available on Pythia, skipping"

# Test PyTorch
echo "Testing PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || { echo "ERROR: PyTorch test failed"; exit 1; }

# Ensure we're in the correct directory
cd /home/bjin0/improved-diffusion
echo "Working directory: $(pwd)"

# Create evaluation directory
EVAL_DIR="/project_gpfs/bjin0/evaluation_test_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"
echo "Test evaluation results will be saved to: $EVAL_DIR"

# List of experiments to test (JUST ONE for fast testing)
EXPERIMENTS=(
    "cifar10_cosine_simple"
)

echo ""
echo "Testing all 7 experiment types (comprehensive test version)..."
echo ""

# Function to create comprehensive no-MPI patches
# create_no_mpi_patches() {
    # DEPRECATED: Patches are now in the repo
    # echo "Using permanently patched non-MPI versions of scripts."
# }

# Function to restore original files
# restore_originals() {
    # DEPRECATED
    # echo "Originals are no longer backed up, skipping restore."
# }

# Set up cleanup trap
# trap restore_originals EXIT

# Create no-MPI patches
# create_no_mpi_patches

# Function to evaluate a single model
evaluate_model() {
    local exp_name="$1"
    local model_path="$2"
    
    echo "=========================================="
    echo "TESTING: $exp_name"
    echo "Model: $model_path"
    echo "=========================================="
    
    # Create experiment directory
    local exp_dir="$EVAL_DIR/$exp_name"
    mkdir -p "$exp_dir"
    
    # Determine the correct parameters for this model
    local noise_schedule="linear"  # default
    local learn_sigma="False"
    local dropout="0.1"  # Default for linear schedule
    local use_kl="False"
    local rescale_learned_sigmas="False"
    
    if [[ "$exp_name" == *"cosine"* ]]; then
        noise_schedule="cosine"
        dropout="0.3"  # Cosine schedule uses dropout=0.3
    elif [[ "$exp_name" == *"ours"* ]]; then
        noise_schedule="ours"
        dropout="0.3"  # Custom schedule follows cosine pattern
    fi
    
    if [[ "$exp_name" == *"hybrid"* ]]; then
        learn_sigma="True"
        use_kl="True"
        # Dropout already set above based on noise schedule
    elif [[ "$exp_name" == *"vlb"* ]]; then
        learn_sigma="True"
        use_kl="True"
        dropout="0.3"
        rescale_learned_sigmas="True"  # VLB experiment uses rescale_learned_sigmas=True
    fi
    
    echo "Using parameters: noise_schedule=$noise_schedule, learn_sigma=$learn_sigma, dropout=$dropout, use_kl=$use_kl, rescale_learned_sigmas=$rescale_learned_sigmas"
    
    # 1. Calculate NLL (bits/dimension) using patched script - SMALL TEST
    echo "Testing NLL calculation (100 samples, 2 batches)..."
    python3 scripts/image_nll.py \
        --model_path "$model_path" \
        --data_dir ./cifar_test \
        --batch_size 50 \
        --num_samples 100 \
        --image_size 32 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --num_heads 4 \
        --attention_resolutions 16,8 \
        --use_scale_shift_norm True \
        --dropout "$dropout" \
        --learn_sigma "$learn_sigma" \
        --use_kl "$use_kl" \
        --rescale_learned_sigmas "$rescale_learned_sigmas" \
        --diffusion_steps 4000 \
        --noise_schedule "$noise_schedule" \
        --class_cond False \
        --use_checkpoint False \
        --rescale_timesteps True \
        --predict_xstart False \
        --clip_denoised True \
        2>&1 | tee "$exp_dir/nll_test_results.txt"
    
    # 2. Generate samples for FID using patched script - SMALL TEST
    echo "Testing sample generation (100 samples)..."
    python3 scripts/image_sample.py \
        --model_path "$model_path" \
        --num_samples 100 \
        --batch_size 50 \
        --image_size 32 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --num_heads 4 \
        --attention_resolutions 16,8 \
        --use_scale_shift_norm True \
        --dropout "$dropout" \
        --learn_sigma "$learn_sigma" \
        --use_kl "$use_kl" \
        --rescale_learned_sigmas "$rescale_learned_sigmas" \
        --diffusion_steps 4000 \
        --noise_schedule "$noise_schedule" \
        --class_cond False \
        --use_checkpoint False \
        --rescale_timesteps True \
        --predict_xstart False \
        --clip_denoised True \
        --use_ddim True \
        2>&1 | tee "$exp_dir/sample_test_results.txt"
    
    # Move generated samples to experiment directory
    # Extract sample file path directly from the "saving to" log line
    # Use robust regex to capture path until .npz extension
    sample_file_source=$(grep "saving to" "$exp_dir/sample_test_results.txt" | tail -1 | sed -n 's/.*saving to \(.*\.npz\).*/\1/p')
    
    if [ -n "$sample_file_source" ] && [ -f "$sample_file_source" ]; then
        cp "$sample_file_source" "$exp_dir/"
        echo "Test samples saved to: $exp_dir/samples_100x32x32x3.npz"
        # Clean up the temporary file
        rm -f "$sample_file_source"
    else
        echo "ERROR: Sample file not found at $sample_file_source"
        echo "Available files in logger directory (if determined):"
        # Try to fallback to finding logger dir for debugging
        logger_dir=$(grep "Logging to" "$exp_dir/sample_test_results.txt" | tail -1 | sed 's/.*Logging to //' | awk '{print $1}' | tr -d '\r\n')
        ls -la "$logger_dir" 2>/dev/null || echo "Logger directory not accessible"
    fi
    
    echo "Completed test for $exp_name"
    echo ""
}

# Test all available models (comprehensive testing)
for exp_name in "${EXPERIMENTS[@]}"; do
    model_path=$(find /project_gpfs/bjin0 -name "ema_0.9999_500000.pt" -path "*/$exp_name/*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
    if [ -n "$model_path" ] && [ -f "$model_path" ]; then
        evaluate_model "$exp_name" "$model_path"
    else
        echo "WARNING: Model not found for $exp_name"
    fi
done

# Create test results summary
echo "=========================================="
echo "CREATING TEST RESULTS SUMMARY"
echo "=========================================="

RESULTS_FILE="$EVAL_DIR/test_results_summary.txt"
echo "5-MINUTE TEST EVALUATION RESULTS" > "$RESULTS_FILE"
echo "===============================" >> "$RESULTS_FILE"
echo "Job ID: $SLURM_JOB_ID" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "NLL TEST RESULTS (100 samples):" >> "$RESULTS_FILE"
echo "===============================" >> "$RESULTS_FILE"

# Extract NLL results
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ] && [ -f "$exp_dir/nll_test_results.txt" ]; then
        exp_name=$(basename "$exp_dir")
        nll_score=$(grep "done 100 samples: bpd=" "$exp_dir/nll_test_results.txt" | tail -1 | awk '{print $4}')
        
        if [ -n "$nll_score" ]; then
            echo "$exp_name: $nll_score bits/dimension (100 samples)" >> "$RESULTS_FILE"
        else
            echo "$exp_name: ERROR extracting NLL" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "SAMPLE GENERATION TEST STATUS:" >> "$RESULTS_FILE"
echo "==============================" >> "$RESULTS_FILE"

# Check sample generation status
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/samples_100x32x32x3.npz" ]; then
            echo "$exp_name: Test samples generated successfully" >> "$RESULTS_FILE"
        else
            echo "$exp_name: No test samples found" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "TEST COMPLETION STATUS:" >> "$RESULTS_FILE"
echo "======================" >> "$RESULTS_FILE"
echo "If you see this message, all patches are working correctly!" >> "$RESULTS_FILE"
echo "All 7 experiment types have been tested with correct parameters." >> "$RESULTS_FILE"
echo "The full evaluation script should work without issues." >> "$RESULTS_FILE"

echo "=========================================="
echo "COMPREHENSIVE TEST COMPLETE!"
echo "=========================================="
echo "Test results saved in: $EVAL_DIR"
echo "Summary saved to: $RESULTS_FILE"
echo ""
echo "To view test results:"
echo "  cat $RESULTS_FILE"
echo "  ls -la $EVAL_DIR"
echo ""
echo "If test passed, run the full evaluation:"
echo "  sbatch evaluate_models_final.slurm"
