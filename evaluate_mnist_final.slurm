#!/bin/bash
set -o pipefail
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=eval_mnist_single_model

# NOTE: The --output and --error flags are set by the submit script.

echo "=========================================="
echo "EVALUATING SINGLE MNIST MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Model to evaluate: $EVAL_MODEL_NAME"
echo "Saving results to: $PARENT_EVAL_DIR"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

echo "Loading modules..."
module load python/booth/3.12 || { echo "ERROR: Failed to load python/booth/3.12"; exit 1; }

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

echo "Testing PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || { echo "ERROR: PyTorch test failed"; exit 1; }

cd /home/bjin0/improved-diffusion
echo "Working directory: $(pwd)"

EVAL_DIR="$PARENT_EVAL_DIR"
echo "Evaluation results will be saved to sub-directories within: $EVAL_DIR"

echo "Using clean in-repo no-MPI entrypoints (no runtime patching)."

evaluate_model() {
    local exp_name="$1"
    local model_path="$2"

    echo "=========================================="
    echo "EVALUATING: $exp_name"
    echo "Model: $model_path"
    echo "Start Time: $(date)"
    echo "=========================================="

    local exp_dir="$EVAL_DIR/$exp_name"
    mkdir -p "$exp_dir"

    local noise_schedule="linear"
    local learn_sigma="False"
    local dropout="0.1"
    local use_kl="False"
    local rescale_learned_sigmas="False"

    if [[ "$exp_name" == *"linabar_linear"* ]]; then
        noise_schedule="linabar_linear"
        dropout="0.3"
    elif [[ "$exp_name" == *"linabar_cosine"* ]]; then
        noise_schedule="linabar_cosine"
        dropout="0.3"
    elif [[ "$exp_name" == *"geometric_linear"* ]]; then
        noise_schedule="geometric_linear"
        dropout="0.3"
    elif [[ "$exp_name" == *"geometric_cosine"* ]]; then
        noise_schedule="geometric_cosine"
        dropout="0.3"
    elif [[ "$exp_name" == *"cosine"* ]]; then
        noise_schedule="cosine"
        dropout="0.3"
    elif [[ "$exp_name" == *"ours_v2"* ]]; then
        noise_schedule="ours_v2"
        dropout="0.3"
    elif [[ "$exp_name" == *"ours"* ]]; then
        noise_schedule="ours"
        dropout="0.3"
    fi

    if [[ "$exp_name" == *"hybrid"* ]]; then
        learn_sigma="True"
        use_kl="False"
        rescale_learned_sigmas="True"
    elif [[ "$exp_name" == *"vlb"* ]]; then
        learn_sigma="True"
        use_kl="True"
        rescale_learned_sigmas="True"
    fi

    echo "Using parameters: noise_schedule=$noise_schedule, learn_sigma=$learn_sigma, dropout=$dropout, use_kl=$use_kl, rescale_learned_sigmas=$rescale_learned_sigmas"

    export OPENAI_LOGDIR="$exp_dir/logger"
    export OPENAI_LOG_FORMAT="stdout,log"
    mkdir -p "$OPENAI_LOGDIR"

    if [ "${SKIP_NLL:-0}" = "1" ]; then
        echo "Skipping NLL (SKIP_NLL=1)"
    else
    echo "Calculating NLL..."
    python3 scripts/image_nll_no_mpi.py \
        --model_path "$model_path" \
        --data_dir ./mnist_test \
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

    sample_file_source=$(ls -1 "$OPENAI_LOGDIR"/samples_*.npz 2>/dev/null | head -1)
    if [ -n "$sample_file_source" ] && [ -f "$sample_file_source" ]; then
        cp "$sample_file_source" "$exp_dir/samples_50000x32x32x3.npz"
        echo "Samples saved to: $exp_dir/samples_50000x32x32x3.npz"
    else
        echo "ERROR: Sample file not found in $OPENAI_LOGDIR"
        ls -la "$OPENAI_LOGDIR" 2>/dev/null || true
    fi

    echo "Computing FID..."
    if ! python3 -c "import importlib.util; spec=importlib.util.find_spec('pytorch_fid'); print('pytorch_fid found:', spec is not None)" >/dev/null 2>&1; then
        echo "ERROR: pytorch_fid is not installed in this environment."
        echo "Please install it in your cluster environment (e.g. pip install pytorch-fid)."
        return 1
    fi

    MNIST_STATS_FILE="/project_gpfs/bata0/bjin0/mnist_train_stats.npz"
    if [ ! -f "$MNIST_STATS_FILE" ]; then
        echo "Precomputing MNIST train stats to: $MNIST_STATS_FILE"
        python3 -m pytorch_fid ./mnist_train "$MNIST_STATS_FILE" --save-stats --device cuda 2>&1 | tee "$exp_dir/fid_stats_build.txt"
    fi

    python3 scripts/compute_fid_from_npz.py \
        --samples_npz "$exp_dir/samples_50000x32x32x3.npz" \
        --stats_npz "$MNIST_STATS_FILE" \
        --device cuda \
        --tmp_root "/project_gpfs/bata0/bjin0" \
        --output_txt "$exp_dir/fid_results.txt" \
        2>&1 | tee "$exp_dir/fid_compute_log.txt"

    if [ "${SKIP_TV:-0}" = "1" ]; then
        echo "Skipping TV distance (SKIP_TV=1)"
    else
        echo "Computing TV distance..."
        python3 scripts/compute_tv_from_npz.py \
            --samples_npz "$exp_dir/samples_50000x32x32x3.npz" \
            --real_dir ./mnist_test \
            --output_txt "$exp_dir/tv_results.txt" \
            2>&1 | tee "$exp_dir/tv_compute_log.txt"
    fi

    echo "Completed evaluation for $exp_name"
    echo "End Time: $(date)"
    echo ""
}

model_path=${MODEL_PATH:-}
if [ -z "$model_path" ]; then
    model_path=$(find /project_gpfs/bata0/bjin0 -name "ema_0.9999_500000.pt" -path "*/logs/$EVAL_MODEL_NAME/*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
fi

if [ -n "$model_path" ] && [ -f "$model_path" ]; then
    evaluate_model "$EVAL_MODEL_NAME" "$model_path"
else
    echo "FATAL: Model not found for $EVAL_MODEL_NAME"
    exit 1
fi

echo "=========================================="
echo "SINGLE MODEL EVALUATION COMPLETE!"
echo "=========================================="
echo "Results saved in: $EVAL_DIR/$EVAL_MODEL_NAME"

