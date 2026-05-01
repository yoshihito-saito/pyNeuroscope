param(
    [switch]$InstallPyInstaller
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($InstallPyInstaller) {
    powershell -ExecutionPolicy Bypass -File .\tools\build_windows_app.ps1 -Zip -InstallPyInstaller
} else {
    powershell -ExecutionPolicy Bypass -File .\tools\build_windows_app.ps1 -Zip
}

$zipPath = Join-Path $RepoRoot "pyNeuroscope-Windows.zip"
$installerSource = Join-Path $RepoRoot "dist\pyNeuroscope-Setup.exe"
$installerTarget = Join-Path $RepoRoot "pyNeuroscope-Setup.exe"

python -m PyInstaller `
    --clean `
    --noconfirm `
    --onefile `
    --windowed `
    --name pyNeuroscope-Setup `
    --add-data "$zipPath;." `
    tools\install_pyneuroscope.py

Copy-Item -LiteralPath $installerSource -Destination $installerTarget -Force

Write-Host ""
Write-Host "Built installer:"
Write-Host "  $installerTarget"
Write-Host ""
Write-Host "Running this installer will install pyNeuroscope to:"
Write-Host "  %LOCALAPPDATA%\Programs\pyNeuroscope"
Write-Host "and create:"
Write-Host "  Desktop\pyNeuroscope.lnk"
