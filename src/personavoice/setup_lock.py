from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from personavoice.process_lock import ProcessLockUnavailable, exclusive_process_lock


class SetupLockError(RuntimeError):
    """Raised when another PersonaVoice setup session owns the repository lock."""


@contextmanager
def setup_lock(repo_root: Path) -> Iterator[Path]:
    """Hold a crash-safe, process-scoped exclusive lock for one setup session."""

    path = repo_root / ".runtime" / "setup.lock"
    try:
        with exclusive_process_lock(path) as locked:
            yield locked
    except ProcessLockUnavailable as exc:
        raise SetupLockError(
            "Another `persona setup` process is already running for this repository. "
            "Wait for it to finish before starting another setup."
        ) from exc
