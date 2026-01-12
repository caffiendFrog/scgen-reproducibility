#!/bin/bash
# Bootstrap script for macOS to create and activate the scgen-reproducibility conda environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Creating conda environment from environment.yml..."
cd "$REPO_ROOT"

# Create or update the environment
if conda env list | grep -q "^scgen-repro-env "; then
    echo "Environment 'scgen-repro-env' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "Creating new environment 'scgen-repro-env'..."
    conda env create -f environment.yml
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate scgen-repro-env"
echo ""
echo "Or use the activation helper:"
echo "  source scripts/activate_env_mac.sh"
