$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (!(Test-Path .venv)) {
  py -3.11 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt

if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
      [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
  }
}

# Bind on all local interfaces so Docker/n8n can reach the Second Brain via
# host.docker.internal:8765. The app is still only exposed on the user's PC
# unless the OS/network explicitly forwards that port.
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8765
