$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Load local settings before startup decisions. Secrets remain in .env, which is gitignored.
if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
      [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
  }
}

if (!(Test-Path .venv)) {
  py -3.11 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt

# SearXNG gives the Second Brain a real general-web search layer. It is optional:
# failure to start it does not block the app because Google News + archive retrieval remain available.
$searxEnabled = if ($env:SEARXNG_ENABLED) { $env:SEARXNG_ENABLED.ToLower() -eq "true" } else { $true }
$searxAutoStart = if ($env:SEARXNG_AUTO_START) { $env:SEARXNG_AUTO_START.ToLower() -eq "true" } else { $true }
if ($searxEnabled -and $searxAutoStart -and (Get-Command docker -ErrorAction SilentlyContinue)) {
  try {
    docker info *> $null
    docker compose -f docker-compose.searxng.yml up -d *> $null
    Write-Host "[SEARXNG] local web research backend requested on 127.0.0.1:8080"
  } catch {
    Write-Host "[SEARXNG] could not auto-start; continuing with fallback research sources." -ForegroundColor Yellow
  }
}

# Bind on all local interfaces so Docker/n8n can reach the Second Brain via
# host.docker.internal:8765. Do not expose/forward this port to the public Internet.
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8765
