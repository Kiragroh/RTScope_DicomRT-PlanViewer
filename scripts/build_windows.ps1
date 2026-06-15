$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Version = $env:RTSCOPE_VERSION
if (-not $Version) {
    $Version = (git describe --tags --always --dirty) -replace '^v', ''
}

python -m PyInstaller --noconfirm --clean RTScopePlanEvalViewer.spec

$ReleaseDir = Join-Path $Root "release"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

$ZipPath = Join-Path $ReleaseDir "RTScopePlanEvalViewer-v$Version-windows-x64.zip"
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Compress-Archive -Path (Join-Path $Root "dist\RTScopePlanEvalViewer\*") -DestinationPath $ZipPath -Force
Write-Host $ZipPath
