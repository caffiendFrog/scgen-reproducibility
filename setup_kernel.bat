@echo off
REM Script to register the conda environment as a Jupyter kernel (Windows)
REM Run this after activating the conda environment

REM Activate the conda environment (if not already activated)
REM call conda activate scgen-repro-env

REM Install ipykernel if not already installed (should be in environment.yml)
REM pip install ipykernel

REM Register the kernel
python -m ipykernel install --user --name scgen-repro-env --display-name "Python (scgen-repro-env)"

echo Kernel 'scgen-repro-env' has been registered successfully!
echo You can now select it as a kernel in Jupyter notebooks.

pause

