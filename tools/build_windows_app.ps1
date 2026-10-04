param(
    [switch]$InstallPyInstaller,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($InstallPyInstaller) {
    python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller installation failed" }
}

python -m PyInstaller --clean --noconfirm pyneuroscope.spec
if ($LASTEXITCODE -ne 0) { throw "App build failed" }

$appDir = Join-Path $RepoRoot "dist\pyNeuroscope"
$probeXmlSource = Join-Path $RepoRoot "probe_xmls"
$probeXmlTarget = Join-Path $appDir "probe_xmls"
$probeGeometrySource = Join-Path $RepoRoot "probe_geometry"
$probeGeometryTarget = Join-Path $appDir "probe_geometry"

function Remove-BundledDataFolder([string]$Target) {
    $absoluteTarget = [IO.Path]::GetFullPath($Target)
    $allowedRoot = [IO.Path]::GetFullPath($appDir).TrimEnd('\') + '\'
    if (-not $absoluteTarget.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Bundled data target is outside the app directory: $absoluteTarget"
    }
    if (Test-Path -LiteralPath $absoluteTarget) {
        if ((Get-Item -LiteralPath $absoluteTarget).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to replace a linked bundled data folder: $absoluteTarget"
        }
        Remove-Item -LiteralPath $absoluteTarget -Recurse -Force
    }
}

if (Test-Path $probeXmlSource) {
    Remove-BundledDataFolder $probeXmlTarget
    Copy-Item -LiteralPath $probeXmlSource -Destination $probeXmlTarget -Recurse
}

if (Test-Path $probeGeometrySource) {
    Remove-BundledDataFolder $probeGeometryTarget
    Copy-Item -LiteralPath $probeGeometrySource -Destination $probeGeometryTarget -Recurse
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

if (Test-Path $probeGeometryTarget) {
    Write-Host ""
    Write-Host "Included probe geometry folder:"
    Write-Host "  $probeGeometryTarget"
}

if ($Zip) {
    $zipPath = Join-Path $RepoRoot "pyNeuroscope-Windows.zip"
    python -m zipfile -c $zipPath $appDir
    if ($LASTEXITCODE -ne 0) { throw "ZIP packaging failed" }
    Write-Host ""
    Write-Host "Created zip:"
    Write-Host "  $zipPath"
}
