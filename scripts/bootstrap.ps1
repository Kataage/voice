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

# TorchCodec 0.10 supports FFmpeg 4-8 and requires shared FFmpeg DLLs on
# Windows. A plain static ffmpeg.exe is not sufficient. Probe PersonaVoice's
# audited runtime resolver first so an already valid install is always reused.
uv run --locked python -c "from personavoice.runtime_dependencies import require_ffmpeg_runtime; r=require_ffmpeg_runtime(); print('FFmpeg runtime:', r.bin_dir, 'major', r.version_major)"
$FfmpegReady = ($LASTEXITCODE -eq 0)

if (-not $FfmpegReady) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "A shared FFmpeg 4-8 runtime is required for TorchCodec. WinGet was not found, so install Gyan.FFmpeg.Shared 8.1.1 manually and rerun bootstrap."
  }

  Write-Host "Installing audited shared FFmpeg 8.1.1 for TorchCodec..."
  winget install --id Gyan.FFmpeg.Shared --exact --version 8.1.1 --accept-package-agreements --accept-source-agreements --disable-interactivity
  Assert-NativeSuccess "WinGet shared FFmpeg install"

  # The current shell may not receive WinGet's PATH update. PersonaVoice also
  # scans the WinGet package directory directly, so this succeeds without asking
  # the user to restart PowerShell.
  uv run --locked python -c "from personavoice.runtime_dependencies import require_ffmpeg_runtime; r=require_ffmpeg_runtime(); print('FFmpeg runtime:', r.bin_dir, 'major', r.version_major)"
  Assert-NativeSuccess "shared FFmpeg runtime verification"
}

uv run --locked persona doctor
Assert-NativeSuccess "persona doctor"
Write-Host "Root environment is ready. Next: uv run --locked persona setup"
