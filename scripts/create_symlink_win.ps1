# Create symlink from Jupyter Notebooks/scgen to code/scgen on Windows
# Attempts symbolic link first, falls back to directory junction if needed

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

$NotebooksScgen = Join-Path $RepoRoot "Jupyter Notebooks\scgen"
$CodeScgen = Join-Path $RepoRoot "code\scgen"

Set-Location $RepoRoot

# Remove existing duplicate directory if it exists (but not if it's already a link)
if (Test-Path $NotebooksScgen) {
    $item = Get-Item $NotebooksScgen -ErrorAction SilentlyContinue
    if ($item -and $item.LinkType -eq $null) {
        Write-Host "Removing duplicate scgen directory..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $NotebooksScgen
    } elseif ($item -and $item.LinkType -ne $null) {
        Write-Host "Symlink already exists. Skipping creation." -ForegroundColor Green
        exit 0
    }
}

# Try to create symbolic link
Write-Host "Attempting to create symbolic link..." -ForegroundColor Green
try {
    $targetPath = Resolve-Path $CodeScgen
    $linkPath = $NotebooksScgen
    $relativePath = "..\code\scgen"
    
    # Try symbolic link first (requires Developer Mode or admin on Windows)
    New-Item -ItemType SymbolicLink -Path $linkPath -Target $relativePath -Force | Out-Null
    Write-Host "Symbolic link created successfully!" -ForegroundColor Green
} catch {
    Write-Host "Symbolic link creation failed (may require Developer Mode or admin privileges)." -ForegroundColor Yellow
    Write-Host "Attempting directory junction as fallback..." -ForegroundColor Yellow
    
    try {
        # Fallback to directory junction (doesn't require special privileges)
        $targetPath = Resolve-Path $CodeScgen
        cmd /c mklink /J "$NotebooksScgen" "$targetPath" | Out-Null
        Write-Host "Directory junction created successfully!" -ForegroundColor Green
        Write-Host "Note: Using directory junction instead of symbolic link." -ForegroundColor Cyan
    } catch {
        Write-Host "Failed to create directory junction. Error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Manual steps:" -ForegroundColor Yellow
        Write-Host "1. Delete 'Jupyter Notebooks\scgen' directory if it exists" -ForegroundColor Yellow
        Write-Host "2. Open Command Prompt as Administrator" -ForegroundColor Yellow
        Write-Host "3. Run: mklink /D `"$NotebooksScgen`" `"$targetPath`"" -ForegroundColor Yellow
        exit 1
    }
}
