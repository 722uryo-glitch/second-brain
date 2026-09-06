$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/5] Checking Docker..."
docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
  throw "Docker is not running. Open Docker Desktop and try again."
}

Write-Host "[2/5] Starting n8n..."
docker compose -f docker-compose.n8n.yml up -d
if ($LASTEXITCODE -ne 0) {
  throw "Could not start n8n."
}

Write-Host "[3/5] Waiting for n8n container..."
$containerId = ""
for ($i = 0; $i -lt 30; $i++) {
  $containerId = (docker compose -f docker-compose.n8n.yml ps -q n8n).Trim()
  if ($containerId) { break }
  Start-Sleep -Seconds 1
}
if (-not $containerId) {
  throw "n8n container did not start. Open Docker Desktop and try again."
}

# Give n8n a few seconds to finish booting before using its CLI.
Start-Sleep -Seconds 4

Write-Host "[4/5] Testing Second Brain from inside Docker..."
# Use single quotes inside JavaScript. PowerShell/Docker on Windows can strip
# nested double quotes from node -e arguments.
$testScript = "fetch('http://host.docker.internal:8765/api/health').then(async r => { console.log(await r.text()); if (!r.ok) process.exit(2); }).catch(e => { console.error(e.message); process.exit(1); });"
docker exec $containerId node -e $testScript
if ($LASTEXITCODE -ne 0) {
  throw "n8n cannot reach Second Brain. Make sure .\run.ps1 is running, then retry."
}

Write-Host "[5/5] Importing Second Brain workflow into n8n..."
docker cp ".\n8n\second-brain-reflect.json" "${containerId}:/tmp/second-brain-reflect.json" | Out-Null
if ($LASTEXITCODE -ne 0) {
  throw "Could not copy the workflow into the n8n container."
}

docker exec $containerId n8n import:workflow --input=/tmp/second-brain-reflect.json
if ($LASTEXITCODE -ne 0) {
  throw "n8n workflow import failed."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Second Brain: http://127.0.0.1:8765"
Write-Host "n8n:          http://127.0.0.1:5678"
Write-Host ""
Write-Host "Open n8n and look for: Second Brain - Reflect Test"
