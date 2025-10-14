#!/bin/bash

echo "=========================================="
echo "SETTING UP PYTORCH ENVIRONMENT ON ALCF"
echo "=========================================="

# Load Python module
echo "Loading Python module..."
module load cray-python/3.11.7

# Check if environment already exists
if [ -d "$HOME/pytorch_env" ]; then
    echo "PyTorch environment already exists at ~/pytorch_env"
    echo "Do you want to recreate it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf ~/pytorch_env
    else
        echo "Using existing environment..."
        source ~/pytorch_env/bin/activate
        echo "Environment activated. You can now use PyTorch in your PBS jobs."
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment at ~/pytorch_env..."
python3 -m venv ~/pytorch_env

# Activate environment
echo "Activating environment..."
source ~/pytorch_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo "Installing PyTorch (this may take 5-10 minutes)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages required for diffusion models
echo "Installing additional packages..."
pip install numpy pillow tqdm blobfile

# Test installation
echo "Testing all package installations..."
python3 -c "
import torch
import torchvision
import numpy as np
import PIL
import tqdm
import blobfile as bf

print('=== Package Versions ===')
print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'PIL version: {PIL.__version__}')
print(f'TQDM version: {tqdm.__version__}')
print(f'Blobfile version: {bf.__version__}')

print('\n=== CUDA Test ===')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    
    # Test GPU computation
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print(f'GPU test successful! Result shape: {y.shape}')

print('\n=== All packages installed successfully! ===')
"

echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo "Your PyTorch environment is at: ~/pytorch_env"
echo ""
echo "To use it in PBS jobs, add this to your script:"
echo "  source ~/pytorch_env/bin/activate"
echo ""
echo "Example PBS script:"
echo "  #!/bin/bash -l"
echo "  #PBS -A Brownian_bandits"
echo "  #PBS -l select=1:system=polaris"
echo "  #PBS -l walltime=0:30:00"
echo "  #PBS -l filesystems=home:eagle"
echo "  #PBS -q debug"
echo ""
echo "  module load cray-python/3.11.7"
echo "  module load cuda/12.6"
echo "  module load nvidia/24.11"
echo "  source ~/pytorch_env/bin/activate"
echo "  python3 your_script.py"
echo "=========================================="
