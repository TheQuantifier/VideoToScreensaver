$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Building app bundle..." -ForegroundColor Cyan
python build_app.py

$possibleIscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
)

$iscc = $possibleIscc | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup not found. Install from https://jrsoftware.org/isdl.php"
}

Write-Host "Compiling installer..." -ForegroundColor Cyan
& $iscc "installer\VideoToScreensaver.iss"

Write-Host "Done. Installer at release\VideoToScreensaver-Setup.exe" -ForegroundColor Green
