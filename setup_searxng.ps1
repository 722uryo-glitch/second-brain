$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Host "[SEARXNG] Docker command not found. Install/start Docker Desktop, then run this again." -ForegroundColor Yellow
  exit 1
}

try {
  docker info *> $null
} catch {
  Write-Host "[SEARXNG] Docker Desktop is not running." -ForegroundColor Yellow
  exit 1
}

Write-Host "[SEARXNG] Starting local metasearch service..."
docker compose -f docker-compose.searxng.yml up -d

$ok = $false
for ($i = 0; $i -lt 30; $i++) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:8080/search?q=test&format=json"
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch {}
  Start-Sleep -Seconds 1
}

if ($ok) {
  Write-Host "[SEARXNG] Ready: http://127.0.0.1:8080" -ForegroundColor Green
} else {
  Write-Host "[SEARXNG] Container started but API health check did not pass yet. Check: docker compose -f docker-compose.searxng.yml logs" -ForegroundColor Yellow
}
