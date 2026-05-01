param(
    [switch]$InstallPyInstaller,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($InstallPyInstaller) {
    python -m pip install pyinstaller
}

python -m PyInstaller --clean --noconfirm pyneuroscope.spec

$appDir = Join-Path $RepoRoot "dist\pyNeuroscope"
$probeXmlSource = Join-Path $RepoRoot "probe_xmls"
$probeXmlTarget = Join-Path $appDir "probe_xmls"

if (Test-Path $probeXmlSource) {
    if (Test-Path $probeXmlTarget) {
        Remove-Item -LiteralPath $probeXmlTarget -Recurse -Force
    }
    Copy-Item -LiteralPath $probeXmlSource -Destination $probeXmlTarget -Recurse
}

Write-Host ""
Write-Host "Built app:"
Write-Host "  $appDir\pyNeuroscope.exe"
Write-Host ""
Write-Host "To distribute, copy the whole folder:"
Write-Host "  $appDir"

if (Test-Path $probeXmlTarget) {
    Write-Host ""
    Write-Host "Included related probe XML folder:"
    Write-Host "  $probeXmlTarget"
}

if ($Zip) {
    $zipPath = Join-Path $RepoRoot "pyNeuroscope-Windows.zip"
    Compress-Archive -Path $appDir -DestinationPath $zipPath -Force
    Write-Host ""
    Write-Host "Created zip:"
    Write-Host "  $zipPath"
}
