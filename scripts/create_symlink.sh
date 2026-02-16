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

# Add repo code/ to Python path and env setup for this conda env (all notebooks/kernels)
if [ -n "$CONDA_PREFIX" ] && [ -d "$REPO_ROOT/code" ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    PTH_FILE="$SITE_PACKAGES/scgen-repro.pth"
    # Run at Python startup: R_HOME for rpy2, and cwd for Jupyter so ../data etc. resolve
    python -c "
import os
pth_dir = '''$SITE_PACKAGES'''
code = '''# Env setup for scgen-repro: R_HOME for rpy2, cwd for Jupyter notebooks.
import os as _os
import sys as _sys
if \"R_HOME\" not in _os.environ:
    _r = _os.path.join(_sys.prefix, \"lib\", \"R\")
    if _os.path.isdir(_r):
        _os.environ[\"R_HOME\"] = _r
# In Jupyter, set cwd to repo Jupyter Notebooks so paths like ../data resolve
try:
    _ip = get_ipython()
    if _ip is not None:
        _code_path = next((p for p in _sys.path if p.endswith(\"code\")), None)
        if _code_path:
            _repo = _os.path.dirname(_code_path)
            _nb_dir = _os.path.join(_repo, \"Jupyter Notebooks\")
            if _os.path.isdir(_nb_dir):
                _os.chdir(_nb_dir)
except NameError:
    pass
'''
with open(os.path.join(pth_dir, 'scgen_repro_env.py'), 'w') as f:
    f.write(code)
"
    printf '%s\n%s\n' "$REPO_ROOT/code" "import scgen_repro_env" > "$PTH_FILE"
    echo "Added repo code/ to path and R_HOME setup ($PTH_FILE). Restart the kernel or open a new terminal for changes to apply."
else
    echo "Tip: activate scgen-repro-env and run this script again to make 'import scgen' and rpy2 (R_HOME) work in notebooks."
fi
