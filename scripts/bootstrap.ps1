$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv is required. Install uv first: https://docs.astral.sh/uv/"
}
uv sync
uv run persona doctor
Write-Host "Root environment is ready. Next: uv run persona setup"
