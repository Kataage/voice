from __future__ import annotations

import hashlib
import json
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
    successfully. Consumers recompute it from the current checkout. A mismatch
    means the installed environments belong to a different repository dependency
    generation and must be resynced before they can be used safely.
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
        "error": None
        if valid_recorded == current
        else (
            "installed environments were created for a different dependency contract; "
            "run `persona setup` to resync the audited local environments"
        ),
    }


def require_current_environment(repo_root: Path) -> dict[str, Any]:
    """Return setup state only when it matches the current dependency contract.

    Runtime code must not silently execute an older `.venv` after the repository,
    worker projects, or audited lockfiles change. This check is intentionally
    independent from `persona doctor` so direct prepare/train/inference commands
    fail closed even when the user did not run doctor first.
    """

    setup_path = repo_root / ".runtime" / "setup.json"
    if not setup_path.is_file():
        raise FileNotFoundError(
            "PersonaVoice setup state is missing. Run `persona setup` before model work."
        )
    try:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"PersonaVoice setup state is unreadable: {setup_path}. Re-run `persona setup`."
        ) from exc
    if not isinstance(setup, dict):
        raise RuntimeError(
            f"PersonaVoice setup state has an invalid format: {setup_path}. Re-run `persona setup`."
        )

    status = environment_contract_status(repo_root, setup.get("environment_contract"))
    if not status["ok"]:
        raise RuntimeError(
            "PersonaVoice local environments are stale for the current repository dependency "
            "contract. Run `persona setup` to resync the audited environments before model work."
        )
    return setup
