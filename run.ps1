Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Voice Clone Trainer v1.0" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

# Find Python
Write-Host "[1/3] Finding Python..." -ForegroundColor Yellow
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "       $($py.Source)" -ForegroundColor Green

# Venv
$venv = "$env:TEMP\vcenv"
Write-Host "[2/3] Venv: $venv" -ForegroundColor Yellow
if (-not (Test-Path "$venv\Scripts\activate.ps1")) {
    Write-Host "       Creating..." -ForegroundColor Gray
    & python -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] venv failed" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
& "$venv\Scripts\Activate.ps1"

# Deps
Write-Host "[3/3] Dependencies..." -ForegroundColor Yellow
$flaskOk = python -c "import flask" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "       Installing..." -ForegroundColor Gray
    pip install -r requirements.txt
}

if (-not (Test-Path "uploads")) { New-Item -ItemType Directory "uploads" | Out-Null }
if (-not (Test-Path "models")) { New-Item -ItemType Directory "models" | Out-Null }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  URL: http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "  Log: server.log" -ForegroundColor Green
Write-Host "  Ctrl+C to stop" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Open browser
Start-Job -ScriptBlock { Start-Sleep 3; Start-Process "http://127.0.0.1:5000" } | Out-Null

# Run server
try {
    python server.py
} catch {
    Write-Host "[ERROR] $_" -ForegroundColor Red
}
Write-Host ""
Write-Host "Server stopped." -ForegroundColor Red
if (Test-Path "server.log") {
    Write-Host "--- Last 20 lines of server.log ---" -ForegroundColor Yellow
    Get-Content "server.log" -Tail 20
}
Read-Host "Press Enter to exit"
