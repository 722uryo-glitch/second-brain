$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Second Brain V1 setup"
Write-Host ""

if (!(Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Created .env from .env.example"
} else {
  Write-Host ".env already exists; preserving existing secrets/settings."
}

$required = @{
  "EXTERNAL_COLLECTION_ENABLED"="true"
  "EXTERNAL_COLLECTION_INTERVAL_MINUTES"="15"
  "EXTERNAL_ITEMS_PER_FEED"="100"
  "EXTERNAL_CONCURRENCY"="16"
  "DOCUMENT_FETCH_ENABLED"="true"
  "DOCUMENT_FETCH_BATCH_SIZE"="120"
  "DOCUMENT_FETCH_CONCURRENCY"="12"
  "FACTCHECK_ENABLED"="true"
  "FACTCHECK_INTERVAL_SECONDS"="5"
  "FACTCHECK_BATCH_SIZE"="30"
  "FACTCHECK_MAX_BATCH_SIZE"="80"
  "GDELT_ENABLED"="true"
  "GITHUB_ENABLED"="true"
  "GITHUB_EVENT_PAGES"="5"
  "BLUESKY_ENABLED"="true"
  "REDDIT_ENABLED"="true"
  "MASTODON_ENABLED"="true"
  "X_ENABLED"="true"
  "OBSIDIAN_ENABLED"="true"
  "OBSIDIAN_VAULT_PATH"="obsidian_vault"
  "OBSIDIAN_EXPORT_INTERVAL_MINUTES"="30"
  "OBSIDIAN_MAX_CLAIMS"="300"
  "OBSIDIAN_MAX_EXTERNAL"="500"
  "OBSIDIAN_MAX_MEMORIES"="200"
}

$lines = Get-Content ".env"
foreach ($key in $required.Keys) {
  if (-not ($lines | Where-Object { $_ -match "^$([regex]::Escape($key))=" })) {
    Add-Content ".env" "$key=$($required[$key])"
  }
}

if (!(Test-Path "obsidian_vault")) {
  New-Item -ItemType Directory -Path "obsidian_vault" | Out-Null
}

Write-Host ""
Write-Host "V1 collectors enabled:"
Write-Host "  Google News global editions"
Write-Host "  GDELT"
Write-Host "  GitHub public events + repository search"
Write-Host "  Bluesky"
Write-Host "  Reddit RSS search"
Write-Host "  Mastodon hashtag feeds"
Write-Host "  Primary-source feeds"
Write-Host "  X adapter (requires X_BEARER_TOKEN)"
Write-Host "  Full-document fetch + adaptive fact-check queue"
Write-Host "  Obsidian export"
Write-Host ""
Write-Host "Optional: add GITHUB_TOKEN and X_BEARER_TOKEN to .env for higher GitHub limits and X collection."
Write-Host ""
Write-Host "Next: .\run.ps1"
