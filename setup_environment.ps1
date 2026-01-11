# Comprehensive setup script for scgen-reproducibility environment (PowerShell)
# This script creates the conda environment and registers the Jupyter kernel

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "scgen-reproducibility Environment Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is installed
try {
    $condaVersion = conda --version 2>&1
    Write-Host "Found conda: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: conda is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda first" -ForegroundColor Red
    exit 1
}

# Create conda environment
Write-Host "Step 1: Creating conda environment from environment.yml..." -ForegroundColor Yellow
conda env create -f environment.yml

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create conda environment" -ForegroundColor Red
    exit 1
}

# Activate environment and register kernel
Write-Host ""
Write-Host "Step 2: Registering Jupyter kernel..." -ForegroundColor Yellow

# Initialize conda for PowerShell
(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression

# Activate the environment
conda activate scgen-repro-env

# Register the kernel
python -m ipykernel install --user --name scgen-repro-env --display-name "Python (scgen-repro-env)"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to register Jupyter kernel" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Cyan
Write-Host "  conda activate scgen-repro-env" -ForegroundColor White
Write-Host ""
Write-Host "To start Jupyter, run:" -ForegroundColor Cyan
Write-Host "  jupyter notebook" -ForegroundColor White
Write-Host ""
Write-Host "The kernel 'Python (scgen-repro-env)' is now available in Jupyter notebooks." -ForegroundColor Green
Write-Host ""

