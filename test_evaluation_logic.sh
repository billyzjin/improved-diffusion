#!/bin/bash

echo "=========================================="
echo "TESTING EVALUATION SCRIPT LOGIC"
echo "=========================================="

# Test the parameter logic from evaluate_models_final.slurm
test_parameters() {
    local exp_name="$1"
    
    echo "Testing experiment: $exp_name"
    
    # Determine the correct parameters for this model (copied from evaluate_models_final.slurm)
    local noise_schedule="linear"  # default
    local learn_sigma="False"
    local dropout="0.1"  # Default for linear schedule
    local use_kl="False"
    local schedule_sampler="uniform"
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
        schedule_sampler="loss-second-moment"
        # Dropout already set above based on noise schedule
    elif [[ "$exp_name" == *"vlb"* ]]; then
        learn_sigma="True"
        use_kl="True"
        schedule_sampler="loss-second-moment"
        dropout="0.3"
        rescale_learned_sigmas="True"  # VLB experiment uses rescale_learned_sigmas=True
    fi
    
    echo "  Parameters: noise_schedule=$noise_schedule, learn_sigma=$learn_sigma, dropout=$dropout, use_kl=$use_kl, schedule_sampler=$schedule_sampler, rescale_learned_sigmas=$rescale_learned_sigmas"
    echo ""
}

# Test all experiments
echo "Testing parameter logic for all experiments:"
echo ""

test_parameters "cifar10_ours_simple"
test_parameters "cifar10_linear_simple" 
test_parameters "cifar10_cosine_simple"
test_parameters "cifar10_linear_hybrid"
test_parameters "cifar10_cosine_hybrid"
test_parameters "cifar10_cosine_vlb"
test_parameters "cifar10_ours_hybrid"

echo "=========================================="
echo "PARAMETER LOGIC TEST COMPLETE"
echo "=========================================="

# Test model path finding logic
echo ""
echo "Testing model path finding logic:"
echo "Looking for models with pattern: ema_0.9999_500000.pt"
echo ""

# Check if any models exist
for exp in cifar10_ours_simple cifar10_linear_simple cifar10_cosine_simple cifar10_linear_hybrid cifar10_cosine_hybrid cifar10_cosine_vlb cifar10_ours_hybrid; do
    model_path=$(find /scratch/bjin0 -name "ema_0.9999_500000.pt" -path "*/logs/$exp/*" 2>/dev/null | head -1)
    if [ -n "$model_path" ] && [ -f "$model_path" ]; then
        echo "✅ $exp: $model_path"
    else
        echo "❌ $exp: Model not found"
    fi
done

echo ""
echo "=========================================="
echo "EVALUATION SCRIPT LOGIC TEST COMPLETE"
echo "=========================================="
