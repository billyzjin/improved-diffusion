#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_models
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err

echo "=========================================="
echo "EVALUATING MODELS USING SLURM JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Set up environment
module load python/booth/3.12
module load cuda/12.6
module load nvidia/24.11

# Create evaluation directory
EVAL_DIR="/scratch/bjin0/evaluation_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"
echo "Evaluation results will be saved to: $EVAL_DIR"

# List of experiments to evaluate
EXPERIMENTS=(
    "cifar10_ours_simple"
    "cifar10_linear_simple" 
    "cifar10_cosine_simple"
    "cifar10_linear_hybrid"
    "cifar10_cosine_hybrid"
    "cifar10_cosine_vlb"
)

echo ""
echo "Evaluating ${#EXPERIMENTS[@]} models using existing codebase tools..."
echo ""

# Function to evaluate a single model
evaluate_model() {
    local exp_name="$1"
    local model_path="$2"
    
    echo "=========================================="
    echo "EVALUATING: $exp_name"
    echo "Model: $model_path"
    echo "=========================================="
    
    # Create experiment directory
    local exp_dir="$EVAL_DIR/$exp_name"
    mkdir -p "$exp_dir"
    
    # Determine the correct noise schedule for this model
    local noise_schedule="linear"  # default
    if [[ "$exp_name" == *"cosine"* ]]; then
        noise_schedule="cosine"
    elif [[ "$exp_name" == *"ours"* ]]; then
        noise_schedule="ours"
    fi
    
    # Determine other parameters based on experiment name
    local learn_sigma="False"
    local dropout="0.0"
    if [[ "$exp_name" == *"hybrid"* ]]; then
        learn_sigma="True"
        if [[ "$exp_name" == *"cosine"* ]]; then
            dropout="0.3"
        else
            dropout="0.1"
        fi
    fi
    
    echo "Using parameters: noise_schedule=$noise_schedule, learn_sigma=$learn_sigma, dropout=$dropout"
    
    # 1. Calculate NLL (bits/dimension) using existing script
    echo "Calculating NLL using scripts/image_nll.py..."
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
        --rescale_learned_sigmas False \
        --diffusion_steps 4000 \
        --noise_schedule "$noise_schedule" \
        --class_cond False \
        --use_checkpoint False \
        --rescale_timesteps True \
        --use_kl False \
        --predict_xstart False \
        --clip_denoised True \
        2>&1 | tee "$exp_dir/nll_results.txt"
    
    # 2. Generate samples for FID using existing script
    echo "Generating samples using scripts/image_sample.py..."
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
        --rescale_learned_sigmas False \
        --diffusion_steps 4000 \
        --noise_schedule "$noise_schedule" \
        --class_cond False \
        --use_checkpoint False \
        --rescale_timesteps True \
        --use_kl False \
        --predict_xstart False \
        --clip_denoised True \
        --use_ddim True \
        2>&1 | tee "$exp_dir/sample_results.txt"
    
    # Move generated samples to experiment directory
    if [ -f "samples_50000x32x32x3.npz" ]; then
        mv "samples_50000x32x32x3.npz" "$exp_dir/"
        echo "Samples saved to: $exp_dir/samples_50000x32x32x3.npz"
    fi
    
    echo "Completed evaluation for $exp_name"
    echo ""
}

# Evaluate each model
for exp_name in "${EXPERIMENTS[@]}"; do
    model_path=$(find /scratch/bjin0 -name "model500000.pt" -path "*/logs/$exp_name/*" 2>/dev/null | head -1)
    
    if [ -n "$model_path" ] && [ -f "$model_path" ]; then
        evaluate_model "$exp_name" "$model_path"
    else
        echo "WARNING: Model not found for $exp_name"
    fi
done

# Create results summary
echo "=========================================="
echo "CREATING RESULTS SUMMARY"
echo "=========================================="

RESULTS_FILE="$EVAL_DIR/results_summary.txt"
echo "DIFFUSION MODEL EVALUATION RESULTS" > "$RESULTS_FILE"
echo "=================================" >> "$RESULTS_FILE"
echo "Job ID: $SLURM_JOB_ID" >> "$RESULTS_FILE"
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
        else
            echo "$exp_name: ERROR extracting NLL" >> "$RESULTS_FILE"
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
        else
            echo "$exp_name: No samples found" >> "$RESULTS_FILE"
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

echo "=========================================="
echo "EVALUATION COMPLETE!"
echo "=========================================="
echo "Results saved in: $EVAL_DIR"
echo "Summary saved to: $RESULTS_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULTS_FILE"
echo "  ls -la $EVAL_DIR"
echo ""
echo "For FID calculation, install pytorch-fid and run it on the generated samples"
