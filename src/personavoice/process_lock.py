from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class ProcessLockUnavailable(RuntimeError):
    """Raised when another process owns a PersonaVoice OS file lock."""


def _lock_posix(handle: BinaryIO) -> None:
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ProcessLockUnavailable from exc


def _unlock_posix(handle: BinaryIO) -> None:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_windows(handle: BinaryIO) -> None:
    import msvcrt

    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        raise ProcessLockUnavailable from exc


def _unlock_windows(handle: BinaryIO) -> None:
    import msvcrt

    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _ensure_sentinel(handle: BinaryIO) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())


def _lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        _lock_windows(handle)
    else:
        _lock_posix(handle)


def _unlock(handle: BinaryIO) -> None:
    if os.name == "nt":
        _unlock_windows(handle)
    else:
        _unlock_posix(handle)


@contextmanager
def exclusive_process_lock(path: Path) -> Iterator[Path]:
    """Hold a crash-safe, non-blocking process-scoped exclusive file lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        _ensure_sentinel(handle)
        _lock(handle)
        try:
            yield path
        finally:
            _unlock(handle)


def process_lock_held(path: Path) -> bool:
    """Return whether an active process currently owns ``path``.

    Missing lock files are not created by the probe. A stale file therefore
    becomes inactive as soon as the owning process exits, while active OS locks
    remain authoritative without PID timeout/reuse heuristics.
    """

    try:
        handle = path.open("r+b")
    except FileNotFoundError:
        return False
    with handle:
        try:
            _lock(handle)
        except ProcessLockUnavailable:
            return True
        else:
            _unlock(handle)
            return False
