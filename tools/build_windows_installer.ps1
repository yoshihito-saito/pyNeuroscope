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
if ($LASTEXITCODE -ne 0) { throw "App packaging failed" }

$zipPath = Join-Path $RepoRoot "pyNeuroscope-Windows.zip"
$installerSource = Join-Path $RepoRoot "dist\pyNeuroscope-Setup.exe"
$installerTarget = Join-Path $RepoRoot "pyNeuroscope-Setup.exe"

python -m PyInstaller `
    --clean `
    --noconfirm `
    --onefile `
    --windowed `
    --name pyNeuroscope-Setup `
    --icon logo\logo.ico `
    --version-file tools\windows_version_info.txt `
    --add-data "$zipPath;." `
    tools\install_pyneuroscope.py
if ($LASTEXITCODE -ne 0) { throw "Installer build failed" }

Copy-Item -LiteralPath $installerSource -Destination $installerTarget -Force

Write-Host ""
Write-Host "Built installer:"
Write-Host "  $installerTarget"
Write-Host ""
Write-Host "Running this installer will install pyNeuroscope to:"
Write-Host "  a folder selected at install time"
Write-Host "or, with --quiet:"
Write-Host "  %LOCALAPPDATA%\Programs\pyNeuroscope"
Write-Host "and create:"
Write-Host "  Desktop\pyNeuroscope.lnk"
