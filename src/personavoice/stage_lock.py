from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from personavoice.process_lock import (
    ProcessLockUnavailable,
    exclusive_process_lock,
    process_lock_held,
)

_STAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class StageLockError(RuntimeError):
    """Raised when another process is already executing the same persona stage."""


def stage_lock_path(persona_root: Path, name: str) -> Path:
    if not _STAGE_NAME.fullmatch(name):
        raise ValueError(f"Unsafe stage name: {name!r}")
    return persona_root / ".runtime" / "stage-locks" / f"{name}.lock"


@contextmanager
def stage_lock(persona_root: Path, name: str) -> Iterator[Path]:
    path = stage_lock_path(persona_root, name)
    try:
        with exclusive_process_lock(path) as locked:
            yield locked
    except ProcessLockUnavailable as exc:
        raise StageLockError(
            f"Another PersonaVoice `{name}` process is already running for {persona_root.name}. "
            "Wait for it to finish; use `persona status` to inspect the active run."
        ) from exc


def stage_lock_held(persona_root: Path, name: str) -> bool:
    return process_lock_held(stage_lock_path(persona_root, name))
