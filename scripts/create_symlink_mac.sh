#!/bin/bash
# Create symlink from Jupyter Notebooks/scgen to code/scgen on macOS

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
