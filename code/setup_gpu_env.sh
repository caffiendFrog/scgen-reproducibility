#!/bin/bash
# Quick setup script to configure GPU environment for TensorFlow on SageMaker
# Run this before training: source code/setup_gpu_env.sh

echo "Setting up GPU environment for TensorFlow..."

# Find CUDA installation (common locations on SageMaker)
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-11"
    "/usr/local/cuda-12"
    "/opt/cuda"
)

CUDA_HOME=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        CUDA_HOME="$path"
        echo "Found CUDA at: $CUDA_HOME"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "WARNING: CUDA installation not found in standard locations"
    echo "Searching for CUDA libraries..."
    find /usr/local -name "libcudart.so*" 2>/dev/null | head -1 | while read lib; do
        CUDA_HOME=$(dirname $(dirname "$lib"))
        echo "Found CUDA libraries at: $CUDA_HOME"
    done
fi

if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME="$CUDA_HOME"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
    
    echo "✓ Environment variables set:"
    echo "  CUDA_HOME=$CUDA_HOME"
    echo "  LD_LIBRARY_PATH includes: $CUDA_HOME/lib64"
    echo ""
    echo "Test GPU detection:"
    echo "  python code/check_tensorflow_gpu.py"
else
    echo "✗ Could not find CUDA installation"
    echo "  You may need to install CUDA or use a GPU instance"
fi

