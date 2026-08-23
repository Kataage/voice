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

# Bootstrap only verifies the locked root environment and CLI entrypoint. Do not
# require or mutate FFmpeg here: on Windows, `persona setup` materializes the
# audited shared runtime inside gitignored `.runtime/tools`; Linux/macOS validate
# their explicit system/override runtime during setup.
uv run --locked persona --help *> $null
Assert-NativeSuccess "persona CLI smoke test"
Write-Host "Root environment is ready. Next: uv run --locked persona setup --backend auto"
