#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_robust
#SBATCH --output=evaluation_robust_%j.out
#SBATCH --error=evaluation_robust_%j.err

echo "=========================================="
echo "EVALUATING MODELS (ROBUST VERSION)"
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

# Create evaluation directory
EVAL_DIR="/scratch/bjin0/evaluation_robust_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"
echo "Evaluation results will be saved to: $EVAL_DIR"

# List of key experiments to evaluate
EXPERIMENTS=(
    "cifar10_ours_simple"
    "cifar10_linear_simple" 
    "cifar10_cosine_simple"
)

echo ""
echo "Evaluating ${#EXPERIMENTS[@]} key models (robust version)..."
echo ""

# Function to create comprehensive no-MPI patches
create_no_mpi_patches() {
    echo "Creating comprehensive no-MPI patches for evaluation..."
    
    # Create dist_util_no_mpi.py
    cat > improved_diffusion/dist_util_no_mpi.py << 'EOF'
import os
import torch

def setup_dist():
    pass

def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_state_dict(path, map_location=None):
    return torch.load(path, map_location=map_location)

def sync_params(params):
    pass

def get_world_size():
    return 1

def get_rank():
    return 0

def is_main_process():
    return True

def barrier():
    pass

def broadcast(tensor, src=0):
    return tensor

def all_gather(tensor):
    return [tensor]

def all_reduce(tensor, op=None):
    return tensor
EOF

    # Create image_datasets_no_mpi.py with deterministic support
    cat > improved_diffusion/image_datasets_no_mpi.py << 'EOF'
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size, class_cond=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.class_cond = class_cond
        
        # Get list of image files
        self.image_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        # Create dummy labels if class_cond is True
        if class_cond:
            self.labels = np.random.randint(0, 10, len(self.image_files))
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) * 2.0  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        if self.class_cond:
            return image, self.labels[idx]
        else:
            return image

def load_data(data_dir, batch_size, image_size, class_cond=False, deterministic=False):
    dataset = ImageDataset(data_dir, image_size, class_cond)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=not deterministic,
        num_workers=1,
        pin_memory=True
    )
    
    # Return infinite iterator
    while True:
        for batch in dataloader:
            if class_cond:
                images, labels = batch
                yield images, {"y": labels}
            else:
                yield batch, {}
EOF

    # Create a patched version of image_sample.py
    cat > scripts/image_sample_no_mpi.py << 'EOF'
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # No-MPI version: just use the sample directly
        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            all_labels.extend([classes.cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    else:
        label_arr = None

    shape_str = "x".join(map(str, arr.shape))
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
EOF

    # Backup original files
    cp improved_diffusion/dist_util.py improved_diffusion/dist_util_original.py
    cp improved_diffusion/image_datasets.py improved_diffusion/image_datasets_original.py
    
    # Replace with no-MPI versions
    cp improved_diffusion/dist_util_no_mpi.py improved_diffusion/dist_util.py
    cp improved_diffusion/image_datasets_no_mpi.py improved_diffusion/image_datasets.py
    
    echo "Comprehensive no-MPI patches created and applied"
}

# Function to restore original files
restore_originals() {
    echo "Restoring original files..."
    if [ -f "improved_diffusion/dist_util_original.py" ]; then
        cp improved_diffusion/dist_util_original.py improved_diffusion/dist_util.py
        rm improved_diffusion/dist_util_original.py
    fi
    if [ -f "improved_diffusion/image_datasets_original.py" ]; then
        cp improved_diffusion/image_datasets_original.py improved_diffusion/image_datasets.py
        rm improved_diffusion/image_datasets_original.py
    fi
    rm -f improved_diffusion/dist_util_no_mpi.py
    rm -f improved_diffusion/image_datasets_no_mpi.py
    rm -f scripts/image_sample_no_mpi.py
}

# Set up cleanup trap
trap restore_originals EXIT

# Create no-MPI patches
create_no_mpi_patches

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
    
    echo "Using parameters: noise_schedule=$noise_schedule, learn_sigma=False, dropout=0.0"
    
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
        --dropout 0.0 \
        --learn_sigma False \
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
    
    # 2. Generate samples for FID using patched script
    echo "Generating samples using patched image_sample_no_mpi.py..."
    python3 scripts/image_sample_no_mpi.py \
        --model_path "$model_path" \
        --num_samples 50000 \
        --batch_size 128 \
        --image_size 32 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --num_heads 4 \
        --attention_resolutions 16,8 \
        --use_scale_shift_norm True \
        --dropout 0.0 \
        --learn_sigma False \
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
echo "ROBUST MODEL EVALUATION RESULTS" > "$RESULTS_FILE"
echo "===============================" >> "$RESULTS_FILE"
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
echo "" >> "$RESULTS_FILE"
echo "Your custom 'ours' schedule should be compared against these baselines." >> "$RESULTS_FILE"

echo "=========================================="
echo "ROBUST MODEL EVALUATION COMPLETE!"
echo "=========================================="
echo "Results saved in: $EVAL_DIR"
echo "Summary saved to: $RESULTS_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULTS_FILE"
echo "  ls -la $EVAL_DIR"
