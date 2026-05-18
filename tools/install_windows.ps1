# PowerShell script for automated environment setup on Windows using Chocolatey
$ErrorActionPreference = "Stop"

Write-Host "Starting automated installation for Windows..." -ForegroundColor Cyan

# 1. Check and install Chocolatey
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not detected. Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}
else {
    Write-Host "Chocolatey is already installed." -ForegroundColor Green
}

# Reload environment variables in current session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# 2. Install system dependencies
Write-Host "Installing required tools..." -ForegroundColor Cyan

# Python 3.12 (or newer if available)
Write-Host "Installing Python 3..." -ForegroundColor Yellow
choco install python3 --version=3.12.2 -y --skip-automated-dependency-resolution

# Quarto CLI
Write-Host "Installing Quarto CLI..." -ForegroundColor Yellow
choco install quarto -y

# Typst (fast PDF engine)
Write-Host "Installing Typst..." -ForegroundColor Yellow
choco install typst -y

# Go-Task (Taskfile)
Write-Host "Installing Go-Task..." -ForegroundColor Yellow
choco install go-task -y

Write-Host "Installation completed successfully." -ForegroundColor Green
Write-Host "IMPORTANT: close and reopen your terminal to apply global PATH changes." -ForegroundColor Yellow
