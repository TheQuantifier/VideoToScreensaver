$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$pythonCandidates = @(
    (Join-Path $root ".venv\Scripts\python.exe"),
    "python",
    "py"
)

$pythonCmd = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -like "*.exe") {
        if (Test-Path $candidate) {
            $pythonCmd = $candidate
            break
        }
        continue
    }

    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $pythonCmd = $candidate
        break
    }
}

if (-not $pythonCmd) {
    throw "Python was not found. Create .venv or install Python and add it to PATH."
}

Write-Host "Launching VideoToScreensaver UI..." -ForegroundColor Cyan

try {
    & $pythonCmd "src\app.py"
}
catch {
    $message = $_.Exception.Message
    if ($message -match "No module named") {
        Write-Host "Missing dependencies. Install them with:" -ForegroundColor Yellow
        Write-Host "  python -m pip install -r requirements.txt" -ForegroundColor Yellow
    }
    throw
}
