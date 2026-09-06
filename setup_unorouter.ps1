param(
  [string]$ApiKey = "",
  [switch]$EnableCloudChat
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (!(Test-Path .env)) {
  Copy-Item .env.example .env
  Write-Host "Created .env from .env.example"
}

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
  Write-Host "Open UnoRouter and create/copy an API key: https://unorouter.com/en"
  $ApiKey = Read-Host "Paste UNOROUTER_API_KEY"
}

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
  throw "UNOROUTER_API_KEY cannot be empty."
}

$content = Get-Content .env -Raw

function Set-EnvValue([string]$Name, [string]$Value) {
  $script:content = $script:content -replace "(?m)^$([regex]::Escape($Name))=.*$", "$Name=$Value"
  if ($script:content -notmatch "(?m)^$([regex]::Escape($Name))=") {
    $script:content += "`r`n$Name=$Value`r`n"
  }
}

Set-EnvValue "UNOROUTER_ENABLED" "true"
Set-EnvValue "UNOROUTER_API_KEY" $ApiKey.Trim()

if ($EnableCloudChat) {
  Set-EnvValue "UNOROUTER_PRIVATE_CHAT" "true"
  Write-Host "Personal chat + recalled memories MAY be sent to UnoRouter."
} else {
  Set-EnvValue "UNOROUTER_PRIVATE_CHAT" "false"
  Write-Host "Personal chat remains local. Public fact-check work will use UnoRouter."
}

Set-Content .env $content -Encoding UTF8

Write-Host ""
Write-Host "UnoRouter configured."
Write-Host "Restart Second Brain with: .\run.ps1"
Write-Host "Router status after startup: http://127.0.0.1:8765/api/ai-router/status"
Write-Host ""
Write-Host "To also use stronger cloud models for normal chat, run:"
Write-Host "  .\setup_unorouter.ps1 -EnableCloudChat"
