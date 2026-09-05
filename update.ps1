param(
    [switch]$Setup,
    [string]$RepoUrl = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step($msg) {
    Write-Host "[Second Brain] $msg" -ForegroundColor Cyan
}

function Fail($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
    exit 1
}

# Files/data that must never be overwritten by updates.
$protected = @(
    ".env",
    "data",
    "memory.db",
    "second_brain.db"
)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "Git is not installed. Install Git for Windows first."
}

if ($Setup) {
    if ([string]::IsNullOrWhiteSpace($RepoUrl)) {
        Fail "Run setup with: .\update.ps1 -Setup -RepoUrl https://github.com/OWNER/REPO.git"
    }

    if (-not (Test-Path ".git")) {
        Write-Step "Initializing Git..."
        git init | Out-Host
    }

    $origin = git remote get-url origin 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($origin)) {
        Write-Step "Adding remote origin..."
        git remote add origin $RepoUrl
    } else {
        Write-Step "Updating remote origin..."
        git remote set-url origin $RepoUrl
    }

    Write-Step "Updater setup complete."
    Write-Host "From now on, run: .\update.ps1"
    exit 0
}

if (-not (Test-Path ".git")) {
    Fail "Updater is not linked to GitHub yet. One-time setup is required."
}

$origin = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($origin)) {
    Fail "GitHub remote is missing. Run .\update.ps1 -Setup -RepoUrl <repo-url>"
}

Write-Step "Checking for local code changes..."
$changes = git status --porcelain --untracked-files=no

if ($changes) {
    Write-Step "Saving local code changes temporarily..."
    git stash push -m "second-brain-auto-update" | Out-Host
    $stashed = $true
} else {
    $stashed = $false
}

try {
    Write-Step "Downloading latest version..."
    git fetch origin | Out-Host

    $branch = git branch --show-current
    if ([string]::IsNullOrWhiteSpace($branch)) {
        $branch = "main"
    }

    Write-Step "Updating branch: $branch"
    git pull --ff-only origin $branch | Out-Host

    Write-Step "Update complete."

    if (Test-Path ".venv\Scripts\python.exe") {
        Write-Step "Updating Python dependencies..."
        & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt | Out-Host
    } else {
        Write-Host "Python environment not found. run.ps1 will create/update it on next launch."
    }
}
finally {
    if ($stashed) {
        Write-Step "Restoring your local code changes..."
        git stash pop | Out-Host
    }
}

Write-Host ""
Write-Host "Done. Start Second Brain with: .\run.ps1" -ForegroundColor Green
