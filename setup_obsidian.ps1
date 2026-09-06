$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/4] Checking Obsidian..."
$obsidian = Get-Command obsidian -ErrorAction SilentlyContinue
if (-not $obsidian) {
  $installed = Get-ItemProperty HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\* -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -like "Obsidian*" } |
    Select-Object -First 1

  if (-not $installed) {
    Write-Host "Obsidian not found. Installing with winget..."
    winget install --id Obsidian.Obsidian -e --accept-source-agreements --accept-package-agreements
  } else {
    Write-Host "Obsidian is already installed."
  }
} else {
  Write-Host "Obsidian is already installed."
}

Write-Host "[2/4] Creating Second Brain vault..."
$vault = Join-Path $PSScriptRoot "obsidian_vault"
New-Item -ItemType Directory -Force -Path $vault | Out-Null

Write-Host "[3/4] Triggering first export if Second Brain is running..."
try {
  $result = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8765/api/obsidian/export" -TimeoutSec 30
  Write-Host "Exported to: $($result.vault)"
} catch {
  Write-Host "Second Brain is not reachable yet. The vault will export automatically after .\run.ps1 starts."
}

Write-Host "[4/4] Done."
Write-Host "Vault path: $vault"
Write-Host ""
Write-Host "In Obsidian choose: Open folder as vault"
Write-Host "and select the folder above."
Start-Process explorer.exe $vault
