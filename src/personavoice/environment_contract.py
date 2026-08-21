from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

WORKER_NAMES = ("asr", "diarization", "sense", "lfm", "seed_vc")
ENVIRONMENT_CONTRACT_SCHEMA = 1


def _sha256(path: Path) -> str:
    try:
        if not path.is_file():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "unreadable"


def environment_contract(repo_root: Path) -> dict[str, Any]:
    """Fingerprint every dependency declaration used by a completed setup.

    `persona setup` records this value only after all isolated environments sync
    successfully. `persona doctor` recomputes it from the current checkout. A
    mismatch therefore means the installed environments belong to a different
    repository dependency generation and must be resynced before offline-ready
    can be claimed.
    """

    workers = {}
    for name in WORKER_NAMES:
        project = repo_root / "workers" / name
        workers[name] = {
            "pyproject_sha256": _sha256(project / "pyproject.toml"),
            "lock_sha256": _sha256(project / "uv.lock"),
        }
    return {
        "schema": ENVIRONMENT_CONTRACT_SCHEMA,
        "root": {
            "pyproject_sha256": _sha256(repo_root / "pyproject.toml"),
            "lock_sha256": _sha256(repo_root / "uv.lock"),
        },
        "irodori": {
            "managed_lock_sha256": _sha256(repo_root / "locks" / "Irodori-TTS.uv.lock"),
        },
        "workers": workers,
    }


def environment_contract_status(repo_root: Path, recorded: Any) -> dict[str, Any]:
    current = environment_contract(repo_root)
    valid_recorded = recorded if isinstance(recorded, dict) else {}
    return {
        "ok": valid_recorded == current,
        "recorded": valid_recorded,
        "current": current,
        "error": None if valid_recorded == current else (
            "installed environments were created for a different dependency contract; "
            "run `persona setup` to resync the audited local environments"
        ),
    }
