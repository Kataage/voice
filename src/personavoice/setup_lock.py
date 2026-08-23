from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class SetupLockError(RuntimeError):
    """Raised when another PersonaVoice setup session owns the repository lock."""


def _lock_posix(handle: BinaryIO) -> None:
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SetupLockError(
            "Another `persona setup` process is already running for this repository. "
            "Wait for it to finish before starting another setup."
        ) from exc


def _unlock_posix(handle: BinaryIO) -> None:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_windows(handle: BinaryIO) -> None:
    import msvcrt

    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        raise SetupLockError(
            "Another `persona setup` process is already running for this repository. "
            "Wait for it to finish before starting another setup."
        ) from exc


def _unlock_windows(handle: BinaryIO) -> None:
    import msvcrt

    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def setup_lock(repo_root: Path) -> Iterator[Path]:
    """Hold a crash-safe, process-scoped exclusive lock for one setup session.

    The lock file itself may remain in `.runtime`, but the OS lock is released
    automatically when the process exits, including after a crash. This avoids
    stale PID/marker heuristics while preventing concurrent environment sync,
    model materialization, and setup-state publication in the same repository.
    """

    runtime = repo_root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / "setup.lock"
    with path.open("a+b") as handle:
        # Windows byte-range locking requires the locked byte to exist. Keeping
        # one sentinel byte is harmless; `.runtime` is gitignored.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())

        if os.name == "nt":
            _lock_windows(handle)
            unlock = _unlock_windows
        else:
            _lock_posix(handle)
            unlock = _unlock_posix
        try:
            yield path
        finally:
            unlock(handle)
