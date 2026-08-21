#!/usr/bin/env sh
set -eu
command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync
uv run persona doctor
echo "Root environment is ready. Next: uv run persona setup"
