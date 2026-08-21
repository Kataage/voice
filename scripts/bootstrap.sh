#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH. Install uv first, then rerun this script." >&2
  exit 1
fi

uv sync
uv run persona doctor

echo "PersonaVoice core environment is ready."
echo "Create a persona with: uv run persona init <name> --authorized"
