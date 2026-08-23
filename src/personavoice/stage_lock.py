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
    """Raised when another process is already mutating the same persona."""


def stage_lock_path(persona_root: Path, name: str) -> Path:
    if not _STAGE_NAME.fullmatch(name):
        raise ValueError(f"Unsafe stage name: {name!r}")
    return persona_root / ".runtime" / "stage-locks" / f"{name}.lock"


def persona_mutation_lock_path(persona_root: Path) -> Path:
    return persona_root / ".runtime" / "persona-mutation.lock"


@contextmanager
def stage_lock(persona_root: Path, name: str) -> Iterator[Path]:
    """Serialize all mutating stages while retaining per-stage liveness locks.

    The persona-wide lock prevents prepare and train from running concurrently
    against the same datasets/models. The stage-specific lock remains held for
    the whole stage so status can distinguish an active run from a stale
    ``status=running`` record after a crash.
    """

    path = stage_lock_path(persona_root, name)
    mutation_path = persona_mutation_lock_path(persona_root)
    try:
        with exclusive_process_lock(mutation_path), exclusive_process_lock(path) as locked:
            yield locked
    except ProcessLockUnavailable as exc:
        raise StageLockError(
            f"Another PersonaVoice prepare/train process is already running for "
            f"{persona_root.name}. Wait for it to finish; use `persona status` to inspect "
            "the active run."
        ) from exc


def stage_lock_held(persona_root: Path, name: str) -> bool:
    """Probe only the named stage lock; the persona-wide lock is not a liveness signal."""

    return process_lock_held(stage_lock_path(persona_root, name))
