$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([string]$Label) {
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit code $LASTEXITCODE"
  }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv is required. Install uv first: https://docs.astral.sh/uv/"
}

uv sync --locked
Assert-NativeSuccess "uv sync --locked"

# Do not install or mutate machine-wide FFmpeg here. On Windows, `persona setup`
# materializes the audited shared FFmpeg runtime inside gitignored `.runtime/tools`.
# Linux/macOS keep their explicit system/override FFmpeg contract.
uv run --locked persona doctor
Assert-NativeSuccess "persona doctor"
Write-Host "Root environment is ready. Next: uv run --locked persona setup --backend auto"
