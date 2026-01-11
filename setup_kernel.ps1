# PowerShell script to register the conda environment as a Jupyter kernel
# Run this after activating the conda environment

# Activate the conda environment (if not already activated)
# conda activate scgen-repro-env

# Install ipykernel if not already installed (should be in environment.yml)
# pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name scgen-repro-env --display-name "Python (scgen-repro-env)"

Write-Host "Kernel 'scgen-repro-env' has been registered successfully!" -ForegroundColor Green
Write-Host "You can now select it as a kernel in Jupyter notebooks." -ForegroundColor Green

