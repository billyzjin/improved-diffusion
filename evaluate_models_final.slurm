#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=eval_single_model

# NOTE: The --output and --error flags are now set by the submit_evaluation_jobs.sh script.

echo "=========================================="
echo "EVALUATING SINGLE MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Model to evaluate: $EVAL_MODEL_NAME"
echo "Saving results to: $PARENT_EVAL_DIR"
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

# The parent evaluation directory is now passed in as an environment variable
EVAL_DIR="$PARENT_EVAL_DIR"
echo "Evaluation results will be saved to sub-directories within: $EVAL_DIR"

# List of all experiments to evaluate (comprehensive)
# NOTE: This is no longer used. The model to evaluate is passed via $EVAL_MODEL_NAME
# EXPERIMENTS=(
#     "cifar10_ours_simple"
#     "cifar10_linear_simple" 
#     "cifar10_cosine_simple"
#     "cifar10_linear_hybrid"
#     "cifar10_cosine_hybrid"
#     "cifar10_cosine_vlb"
#     "cifar10_ours_hybrid"
# )

echo ""
echo "Evaluating 1 model: $EVAL_MODEL_NAME"
echo ""

# Function to create comprehensive no-MPI patches
create_no_mpi_patches() {
    # Using permanently patched non-MPI versions of scripts.
    echo "Using permanently patched non-MPI versions of scripts."
}

# Function to restore original files
restore_originals() {
    # NOTE: This function is now deprecated as we are using permanent patches.
    echo "Originals are no longer backed up, skipping restore."
}

# Set up cleanup trap
trap restore_originals EXIT

# Create no-MPI patches - DEPRECATED
# create_no_mpi_patches


# Function to evaluate a single model
evaluate_model() {
    local exp_name="$1"
    local model_path="$2"
    
    echo "=========================================="
    echo "EVALUATING: $exp_name"
    echo "Model: $model_path"
    echo "Start Time: $(date)"
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
    
    # 1. Calculate NLL (bits/dimension)
    echo "Calculating NLL..."
    python3 scripts/image_nll.py \
        --model_path "$model_path" \
        --data_dir ./cifar_test \
        --batch_size 128 \
        --num_samples 10000 \
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
        2>&1 | tee "$exp_dir/nll_results.txt"
    
    # 2. Generate samples for FID
    echo "Generating samples for FID..."
    python3 scripts/image_sample.py \
        --model_path "$model_path" \
        --num_samples 50000 \
        --batch_size 128 \
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
        --use_ddim False \
        2>&1 | tee "$exp_dir/sample_results.txt"
    
    # Move generated samples to experiment directory
    # Extract sample file path directly from the "saving to" log line
    # Use sed to robustly extract the path ending in .npz, ignoring any trailing text attached to the line
    sample_file_source=$(grep "saving to" "$exp_dir/sample_results.txt" | tail -1 | sed -n 's/.*saving to \(.*\.npz\).*/\1/p')
    
    if [ -n "$sample_file_source" ] && [ -f "$sample_file_source" ]; then
        cp "$sample_file_source" "$exp_dir/"
        echo "Samples saved to: $exp_dir/samples_50000x32x32x3.npz"
        # Clean up the temporary file
        rm -f "$sample_file_source"
    else
        echo "ERROR: Sample file not found at $sample_file_source"
        echo "Available files in logger directory (if determined):"
        # Try to fallback to finding logger dir for debugging
        logger_dir=$(grep "Logging to" "$exp_dir/sample_results.txt" | tail -1 | sed 's/.*Logging to //' | awk '{print $1}' | tr -d '\r\n')
        ls -la "$logger_dir" 2>/dev/null || echo "Logger directory not accessible"
    fi
    
    echo "Completed evaluation for $exp_name"
    echo "End Time: $(date)"
    echo ""
}

# Evaluate the single model passed from the submission script
model_path=$(find /project_gpfs/bjin0 -name "ema_0.9999_500000.pt" -path "*/logs/$EVAL_MODEL_NAME/*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
    if [ -n "$model_path" ] && [ -f "$model_path" ]; then
    evaluate_model "$EVAL_MODEL_NAME" "$model_path"
    else
    echo "FATAL: Model not found for $EVAL_MODEL_NAME"
    exit 1
fi

# The results summary is now created by a separate script after all jobs are complete.

echo "=========================================="
echo "SINGLE MODEL EVALUATION COMPLETE!"
echo "=========================================="
echo "Results saved in: $EVAL_DIR/$EVAL_MODEL_NAME"
echo ""
echo "After all jobs are finished, run the aggregation script to get the final summary."
