#!/bin/bash
# Create symlink from Jupyter Notebooks/scgen to code/scgen (Linux)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NOTEBOOKS_SCGEN="$REPO_ROOT/Jupyter Notebooks/scgen"
CODE_SCGEN="$REPO_ROOT/code/scgen"

cd "$REPO_ROOT"

# Remove existing duplicate directory if it exists
if [ -d "$NOTEBOOKS_SCGEN" ] && [ ! -L "$NOTEBOOKS_SCGEN" ]; then
    echo "Removing duplicate scgen directory..."
    rm -rf "$NOTEBOOKS_SCGEN"
fi

# Create symlink if it doesn't exist
if [ ! -e "$NOTEBOOKS_SCGEN" ]; then
    echo "Creating symlink from 'Jupyter Notebooks/scgen' to 'code/scgen'..."
    ln -s "../code/scgen" "$NOTEBOOKS_SCGEN"
    echo "Symlink created successfully!"
else
    echo "Symlink or directory already exists at 'Jupyter Notebooks/scgen'"
    echo "Skipping symlink creation."
fi

# Add repo code/ to Python path for this conda env (so "import scgen" works in any cwd, e.g. Jupyter)
if [ -n "$CONDA_PREFIX" ] && [ -d "$REPO_ROOT/code" ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    PTH_FILE="$SITE_PACKAGES/scgen-repro.pth"
    echo "$REPO_ROOT/code" > "$PTH_FILE"
    echo "Added $REPO_ROOT/code to Python path ($PTH_FILE). Restart the kernel or open a new terminal for 'import scgen' to work."
else
    echo "Tip: activate scgen-repro-env and run this script again to make 'import scgen' work in notebooks from any directory."
fi
