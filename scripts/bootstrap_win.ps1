# Bootstrap script for Windows to create and activate the scgen-reproducibility conda environment

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host "Creating conda environment from environment.yml..." -ForegroundColor Green
Set-Location $RepoRoot

# Check if environment already exists
$envExists = conda env list | Select-String -Pattern "^scgen-repor-env "

if ($envExists) {
    Write-Host "Environment 'scgen-repor-env' already exists. Updating..." -ForegroundColor Yellow
    conda env update -f environment.yml --prune
} else {
    Write-Host "Creating new environment 'scgen-repor-env'..." -ForegroundColor Green
    conda env create -f environment.yml
}

Write-Host ""
Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Cyan
Write-Host "  conda activate scgen-repor-env"
Write-Host ""
Write-Host "Or use the activation helper:" -ForegroundColor Cyan
Write-Host "  .\scripts\activate_env_win.ps1"
