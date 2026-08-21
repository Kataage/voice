#!/usr/bin/env sh
set -eu
uv lock
for d in workers/*; do [ -f "$d/pyproject.toml" ] && uv lock --project "$d"; done
[ -f vendor/Irodori-TTS/pyproject.toml ] && uv lock --project vendor/Irodori-TTS || true
echo "All local uv lockfiles refreshed."
