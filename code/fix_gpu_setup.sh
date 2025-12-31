#!/bin/bash
# Script to fix GPU setup on AWS SageMaker
# This script checks and installs the necessary components for GPU support

echo "============================================================"
echo "GPU Setup Fix for AWS SageMaker"
echo "============================================================"

# Check if we're on a GPU instance
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "✗ nvidia-smi not found - you may not be on a GPU instance"
    exit 1
fi

# Check CUDA
echo ""
echo "Checking CUDA installation..."
if [ -d "/usr/local/cuda" ]; then
    echo "✓ CUDA directory found at /usr/local/cuda"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        cat /usr/local/cuda/version.txt
    fi
else
    echo "✗ CUDA directory not found"
    echo "  CUDA should be at /usr/local/cuda on SageMaker GPU instances"
fi

# Check for CUDA in common locations
echo ""
echo "Searching for CUDA libraries..."
find /usr/local -name "libcudart.so*" 2>/dev/null | head -5
find /usr -name "libcudart.so*" 2>/dev/null | head -5

# Check TensorFlow installation
echo ""
echo "Checking TensorFlow installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'TensorFlow location: {tf.__file__}')" 2>&1

# Check if tensorflow-gpu is needed
echo ""
echo "Checking if tensorflow-gpu package is installed..."
python -c "import tensorflow as tf; from tensorflow.python.client import device_lib; devices = device_lib.list_local_devices(); gpus = [d for d in devices if d.device_type == 'GPU']; print(f'GPUs detected: {len(gpus)}'); [print(f'  - {g.name}') for g in gpus]" 2>&1

echo ""
echo "============================================================"
echo "Recommendations:"
echo "============================================================"
echo "1. If no GPUs detected, you may need to install tensorflow-gpu:"
echo "   conda install tensorflow-gpu"
echo "   OR"
echo "   pip install tensorflow-gpu"
echo ""
echo "2. For TensorFlow 2.x, GPU support is included in tensorflow package"
echo "   but you need compatible CUDA/cuDNN versions"
echo ""
echo "3. Set environment variables if CUDA is in non-standard location:"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo "   export CUDA_HOME=/usr/local/cuda"
echo ""
echo "4. On SageMaker, ensure you're using a GPU instance type (e.g., ml.p3.2xlarge)"

