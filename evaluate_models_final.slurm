#!/bin/bash
set -o pipefail
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=eval_single_model

# NOTE: The --output and --error flags are now set by the submit_cifar10_evaluation_jobs.sh script.

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

echo "Using clean in-repo no-MPI entrypoints (no runtime patching)."


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
    
    # Determine the correct parameters for this model (match paper definitions)
    local noise_schedule="linear"  # default
    local learn_sigma="False"
    local dropout="0.1"  # Default for linear schedule
    local use_kl="False"
    local rescale_learned_sigmas="False"
    
    if [[ "$exp_name" == *"geometric_linear"* ]]; then
        noise_schedule="geometric_linear"
        dropout="0.3"
    elif [[ "$exp_name" == *"geometric_cosine"* ]]; then
        noise_schedule="geometric_cosine"
        dropout="0.3"
    elif [[ "$exp_name" == *"cosine"* ]]; then
        noise_schedule="cosine"
        dropout="0.3"  # Cosine schedule uses dropout=0.3
    elif [[ "$exp_name" == *"ours_v2"* ]]; then
        noise_schedule="ours_v2"
        dropout="0.3"  # Optimal schedule follows cosine dropout
    elif [[ "$exp_name" == *"ours"* ]]; then
        noise_schedule="ours"
        dropout="0.3"  # Custom schedule follows cosine pattern
    fi
    
    if [[ "$exp_name" == *"hybrid"* ]]; then
        # L_hybrid: learn_sigma=True, loss_type=RESCALED_MSE (no KL),
        # which requires rescale_learned_sigmas=True.
        learn_sigma="True"
        use_kl="False"
        rescale_learned_sigmas="True"
    elif [[ "$exp_name" == *"vlb"* ]]; then
        learn_sigma="True"
        use_kl="True"
        # When use_kl=True, loss_type is KL/RESCALED_KL; rescale_learned_sigmas is ignored.
        rescale_learned_sigmas="True"
    fi
    
    echo "Using parameters: noise_schedule=$noise_schedule, learn_sigma=$learn_sigma, dropout=$dropout, use_kl=$use_kl, rescale_learned_sigmas=$rescale_learned_sigmas"
    
    # Use a stable logger directory inside the experiment dir, so artifacts land in one place.
    export OPENAI_LOGDIR="$exp_dir/logger"
    export OPENAI_LOG_FORMAT="stdout,log"
    mkdir -p "$OPENAI_LOGDIR"
    
    # 1. Calculate NLL (bits/dimension)
    if [ "${SKIP_NLL:-0}" = "1" ]; then
        echo "Skipping NLL (SKIP_NLL=1)"
    else
    echo "Calculating NLL..."
    python3 scripts/image_nll_no_mpi.py \
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
    fi
    
    # 2. Generate samples for FID.
    # IMPORTANT: Table 2 FID in the paper is reported for the trained models; using
    # an aggressively respaced sampler (e.g. 250 steps) can significantly degrade
    # FID for fixed-sigma (L_simple) models. Therefore, default is full sampling
    # (no timestep_respacing). You can override for fast-sampling experiments by
    # exporting EVAL_TIMESTEP_RESPACING (e.g. 250) before submitting jobs.
    echo "Generating samples for FID..."
    RESPACING_FLAG=""
    if [ -n "${EVAL_TIMESTEP_RESPACING:-}" ]; then
        RESPACING_FLAG="--timestep_respacing ${EVAL_TIMESTEP_RESPACING}"
        echo "Using timestep respacing for sampling: ${EVAL_TIMESTEP_RESPACING}"
    else
        echo "Using full sampling (no timestep respacing)."
    fi
    python3 scripts/image_sample_no_mpi.py \
        --model_path "$model_path" \
        --num_samples 50000 \
        --batch_size 256 \
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
        $RESPACING_FLAG \
        --class_cond False \
        --use_checkpoint False \
        --rescale_timesteps True \
        --predict_xstart False \
        --clip_denoised True \
        --use_ddim False \
        2>&1 | tee "$exp_dir/sample_results.txt"
    
    # Copy generated samples into the experiment directory with a stable name.
    sample_file_source=$(ls -1 "$OPENAI_LOGDIR"/samples_*.npz 2>/dev/null | head -1)
    if [ -n "$sample_file_source" ] && [ -f "$sample_file_source" ]; then
        cp "$sample_file_source" "$exp_dir/samples_50000x32x32x3.npz"
        echo "Samples saved to: $exp_dir/samples_50000x32x32x3.npz"
    else
        echo "ERROR: Sample file not found in $OPENAI_LOGDIR"
        ls -la "$OPENAI_LOGDIR" 2>/dev/null || true
    fi

    # 3. Compute FID (paper metric) using GPU in this eval job, and write a single number.
    echo "Computing FID..."
    # Ensure pytorch-fid is available.
    if ! python3 -c "import importlib.util; spec=importlib.util.find_spec('pytorch_fid'); print('pytorch_fid found:', spec is not None)" >/dev/null 2>&1; then
        echo "ERROR: pytorch_fid is not installed in this environment."
        echo "Please install it in your cluster environment (e.g. pip install pytorch-fid)."
        return 1
    fi

    # Ensure CIFAR-10 train stats exist (compute once into scratch if missing).
    CIFAR_STATS_FILE="/project_gpfs/bata0/bjin0/cifar10_train_stats.npz"
    if [ ! -f "$CIFAR_STATS_FILE" ]; then
        echo "Precomputing CIFAR-10 train stats to: $CIFAR_STATS_FILE"
        python3 -m pytorch_fid ./cifar_train "$CIFAR_STATS_FILE" --save-stats --device cuda 2>&1 | tee "$exp_dir/fid_stats_build.txt"
    fi

    python3 scripts/compute_fid_from_npz.py \
        --samples_npz "$exp_dir/samples_50000x32x32x3.npz" \
        --stats_npz "$CIFAR_STATS_FILE" \
        --device cuda \
        --tmp_root "/project_gpfs/bata0/bjin0" \
        --output_txt "$exp_dir/fid_results.txt" \
        2>&1 | tee "$exp_dir/fid_compute_log.txt"

    # 4. Compute Total Variation (TV) distance vs real CIFAR-10 test set (pixel histogram TV).
    echo "Computing TV distance..."
    python3 scripts/compute_tv_from_npz.py \
        --samples_npz "$exp_dir/samples_50000x32x32x3.npz" \
        --real_dir ./cifar_test \
        --output_txt "$exp_dir/tv_results.txt" \
        2>&1 | tee "$exp_dir/tv_compute_log.txt"
    
    echo "Completed evaluation for $exp_name"
    echo "End Time: $(date)"
    echo ""
}

# Evaluate the single model passed from the submission script
model_path=$(find /project_gpfs/bata0/bjin0 -name "ema_0.9999_500000.pt" -path "*/logs/$EVAL_MODEL_NAME/*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
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
