# One-command launcher for the SYNGEN web demo (Windows PowerShell).
#   .\run_demo.ps1            build the UI if needed, start the API, open the browser
#   .\run_demo.ps1 -Rebuild   force a fresh UI build
param([switch]$Rebuild, [int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if ($Rebuild -or -not (Test-Path 'frontend/dist/index.html')) {
    Write-Host 'Building the frontend...' -ForegroundColor Cyan
    Push-Location frontend
    if (-not (Test-Path 'node_modules')) { npm install }
    npm run build
    Pop-Location
}

$env:PORT = "$Port"
Start-Job -ScriptBlock { param($p) Start-Sleep -Seconds 3; Start-Process "http://localhost:$p" } -ArgumentList $Port | Out-Null
python server/app.py
