#!/bin/bash
# Comprehensive setup script for scgen-reproducibility environment
# This script creates the conda environment and registers the Jupyter kernel

set -e  # Exit on error

echo "========================================="
echo "scgen-reproducibility Environment Setup"
echo "========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create conda environment
echo "Step 1: Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment and register kernel
echo ""
echo "Step 2: Registering Jupyter kernel..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scgen-repro-env

# Register the kernel
python -m ipykernel install --user --name scgen-repro-env --display-name "Python (scgen-repro-env)"

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate scgen-repro-env"
echo ""
echo "To start Jupyter, run:"
echo "  jupyter notebook"
echo ""
echo "The kernel 'Python (scgen-repro-env)' is now available in Jupyter notebooks."
echo ""

