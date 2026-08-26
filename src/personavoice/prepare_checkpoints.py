from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from personavoice.worker_contracts import PREPARE_CACHE_VALIDATORS

CHECKPOINT_SCHEMA = 1
PROGRESS_SCHEMA = 1
_SAFE_ITEM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROGRESS_FILE = "progress.json"


def checkpoint_dir(cache_dir: Path) -> Path:
    return cache_dir / ".checkpoints"


def _safe_item_id(item_id: str) -> str:
    value = str(item_id)
    if value == "progress" or not _SAFE_ITEM_ID.fullmatch(value):
        raise ValueError(f"Unsafe prepare checkpoint item id: {value!r}")
    return value


def checkpoint_path(directory: Path, item_id: str) -> Path:
    return directory / f"{_safe_item_id(item_id)}.json"


def recover_checkpoint(directory: Path, item_id: str, kind: str) -> dict[str, Any] | None:
    """Return one semantically valid worker checkpoint, deleting invalid candidates.

    Worker-written files are never trusted merely because they are valid JSON.
    The central worker contract remains the authority before a partial result can
    be promoted into the normal prepare cache after a crash or forced shutdown.
    """

    validator = PREPARE_CACHE_VALIDATORS.get(kind)
    if validator is None:
        raise ValueError(f"Unsupported prepare checkpoint kind: {kind!r}")
    path = checkpoint_path(directory, item_id)
    if not path.is_file():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    valid_envelope = (
        isinstance(envelope, dict)
        and envelope.get("schema") == CHECKPOINT_SCHEMA
        and envelope.get("id") == str(item_id)
        and "result" in envelope
    )
    if not valid_envelope:
        path.unlink(missing_ok=True)
        return None
    result = envelope["result"]
    try:
        valid_result = bool(validator(result))
    except (TypeError, ValueError, OverflowError):
        valid_result = False
    if not valid_result:
        path.unlink(missing_ok=True)
        return None
    return result


def discard_checkpoint(directory: Path, item_id: str) -> None:
    checkpoint_path(directory, item_id).unlink(missing_ok=True)


def cleanup_checkpoint_dir(directory: Path) -> None:
    """Remove a completed batch's transient checkpoint/progress directory."""

    shutil.rmtree(directory, ignore_errors=True)


def _progress_value(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != PROGRESS_SCHEMA:
        return None
    completed = value.get("completed")
    total = value.get("total")
    failed = value.get("failed")
    if (
        not isinstance(completed, int)
        or isinstance(completed, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or not isinstance(failed, int)
        or isinstance(failed, bool)
        or completed < 0
        or total < 0
        or failed < 0
        or completed > total
        or failed > completed
    ):
        return None
    state = value.get("state")
    if state not in {"running", "finished"}:
        return None
    for key in ("worker", "command", "phase"):
        if not isinstance(value.get(key), str) or not value[key]:
            return None
    current_id = value.get("current_id")
    if current_id is not None and not isinstance(current_id, str):
        return None
    return value


def prepare_batch_progress(persona_root: Path) -> dict[str, dict[str, Any]]:
    """Expose inexpensive live/stale batch metadata for `persona status`.

    Progress files are advisory only. Stage OS locks remain the liveness source of
    truth, and checkpoint data still requires semantic validation before reuse.
    """

    output: dict[str, dict[str, Any]] = {}
    cache_roots = [persona_root / "cache"]
    generations = persona_root / "generations" / "prepare"
    if generations.is_dir():
        cache_roots.extend(
            path / "cache"
            for path in sorted(generations.iterdir())
            if path.is_dir() and re.fullmatch(r"pl-[0-9a-f]{32}", path.name)
        )
    for cache_root in cache_roots:
        for kind in ("identity", "asr", "diarization", "sense", "alignment"):
            directory = checkpoint_dir(cache_root / kind)
            progress = _progress_value(directory / _PROGRESS_FILE)
            if progress is None:
                continue
            checkpointed = sum(
                1
                for path in directory.glob("*.json")
                if path.name != _PROGRESS_FILE and path.is_file()
            )
            key = kind if cache_root == persona_root / "cache" else f"{cache_root.parent.name}:{kind}"
            output[key] = {**progress, "checkpointed_successes": checkpointed}
    return output
