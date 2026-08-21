$ErrorActionPreference = "Stop"
uv lock
Get-ChildItem workers -Directory | ForEach-Object { uv lock --project $_.FullName }
if (Test-Path vendor/Irodori-TTS/pyproject.toml) { uv lock --project vendor/Irodori-TTS }
Write-Host "All local uv lockfiles refreshed."
