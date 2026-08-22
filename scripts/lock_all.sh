#!/usr/bin/env sh
set -eu

IRODORI_REVISION="8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
IRODORI_DIR="vendor/Irodori-TTS"
MANAGED_IRODORI_PROJECT="locks/Irodori-TTS.pyproject.toml"
MANAGED_IRODORI_LOCK="locks/Irodori-TTS.uv.lock"

uv lock
for d in workers/*; do
  [ -f "$d/pyproject.toml" ] && uv lock --project "$d"
done

if [ -f "$IRODORI_DIR/pyproject.toml" ]; then
  if [ ! -f "$MANAGED_IRODORI_PROJECT" ]; then
    echo "Managed Irodori project overlay is missing: $MANAGED_IRODORI_PROJECT" >&2
    exit 1
  fi

  head_revision="$(git -C "$IRODORI_DIR" rev-parse HEAD)"
  if [ "$head_revision" != "$IRODORI_REVISION" ]; then
    echo "Irodori checkout is not at the audited revision: $head_revision" >&2
    exit 1
  fi
  if [ -n "$(git -C "$IRODORI_DIR" status --porcelain)" ]; then
    echo "Irodori checkout has local changes; refusing to generate a managed lock." >&2
    exit 1
  fi

  mkdir -p locks
  vendor_project="$IRODORI_DIR/pyproject.toml"
  vendor_lock="$IRODORI_DIR/uv.lock"
  project_backup="$(mktemp)"
  lock_backup="$(mktemp)"
  cp "$vendor_project" "$project_backup"
  had_lock=0
  if [ -f "$vendor_lock" ]; then
    cp "$vendor_lock" "$lock_backup"
    had_lock=1
  fi

  restore_vendor_setup_files() {
    cp "$project_backup" "$vendor_project"
    if [ "$had_lock" -eq 1 ]; then
      cp "$lock_backup" "$vendor_lock"
    else
      rm -f "$vendor_lock"
    fi
    rm -f "$project_backup" "$lock_backup"
  }
  trap restore_vendor_setup_files EXIT HUP INT TERM

  cp "$MANAGED_IRODORI_PROJECT" "$vendor_project"
  uv lock --project "$IRODORI_DIR"
  cp "$vendor_lock" "$MANAGED_IRODORI_LOCK"
  restore_vendor_setup_files
  trap - EXIT HUP INT TERM
else
  echo "Irodori vendor checkout is absent; managed Irodori lock was left unchanged."
fi

echo "All audited uv lockfiles refreshed."
