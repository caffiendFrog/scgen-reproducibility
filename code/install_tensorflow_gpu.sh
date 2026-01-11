#!/bin/bash
# Script to install TensorFlow with GPU support on AWS SageMaker
# This should be run in your conda environment

echo "============================================================"
echo "Installing TensorFlow with GPU Support"
echo "============================================================"

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "WARNING: Not in a conda environment. Activate your environment first:"
    echo "  conda activate scgen-repro-env"
    exit 1
fi

echo "Current conda environment: $CONDA_DEFAULT_ENV"

# Check TensorFlow version
echo ""
echo "Current TensorFlow installation:"
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>&1

# For TensorFlow 2.x, GPU support is included but needs CUDA/cuDNN
# On SageMaker, CUDA should be pre-installed, but libraries might not be linked

echo ""
echo "============================================================"
echo "Option 1: Install tensorflow-gpu (recommended for TF 1.x)"
echo "============================================================"
echo "If you need TensorFlow 1.x with GPU:"
echo "  conda install -c conda-forge tensorflow-gpu=1.15"
echo ""
echo "============================================================"
echo "Option 2: Fix CUDA library paths for TF 2.x"
echo "============================================================"
echo "For TensorFlow 2.x, set environment variables:"
echo ""
echo "Add to ~/.bashrc or run before training:"
echo ""
echo "  # Find CUDA installation"
echo "  export CUDA_HOME=/usr/local/cuda"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo ""
echo "============================================================"
echo "Option 3: Reinstall TensorFlow with GPU support"
echo "============================================================"
echo "Uninstall current TensorFlow:"
echo "  pip uninstall tensorflow tensorflow-gpu"
echo ""
echo "Install TensorFlow with GPU (for TF 2.x, GPU is included):"
echo "  pip install tensorflow[and-cuda]"
echo ""
echo "OR for specific version:"
echo "  pip install tensorflow==2.15.0  # Known to work well with CUDA 11.8"
echo ""

# Check for CUDA
echo "============================================================"
echo "Checking CUDA installation..."
echo "============================================================"

if [ -d "/usr/local/cuda" ]; then
    echo "✓ Found CUDA at /usr/local/cuda"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        echo "CUDA version:"
        cat /usr/local/cuda/version.txt
    fi
else
    echo "✗ CUDA not found at /usr/local/cuda"
    echo "  On SageMaker GPU instances, CUDA should be pre-installed"
    echo "  Check: ls -la /usr/local/ | grep cuda"
fi

# Check nvidia-smi
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "✗ nvidia-smi not found"
fi

echo ""
echo "============================================================"
echo "Quick fix - Set environment variables:"
echo "============================================================"
echo "Run these commands:"
echo ""
echo "  export CUDA_HOME=/usr/local/cuda"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo ""
echo "Then test:"
echo "  python code/check_tensorflow_gpu.py"

