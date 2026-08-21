$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv is required. Install uv first: https://docs.astral.sh/uv/"
}
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw "uv sync --locked failed with exit code $LASTEXITCODE" }
uv run --locked persona doctor
if ($LASTEXITCODE -ne 0) { throw "persona doctor failed with exit code $LASTEXITCODE" }
Write-Host "Root environment is ready. Next: uv run --locked persona setup"
