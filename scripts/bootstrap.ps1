$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found on PATH. Install uv first, then rerun this script."
}

uv sync
uv run persona doctor

Write-Host "PersonaVoice core environment is ready."
Write-Host "Create a persona with: uv run persona init <name> --authorized"
