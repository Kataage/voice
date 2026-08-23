from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one patch anchor in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")


write(
    "src/personavoice/process_lock.py",
    '''from __future__ import annotations

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
        handle.write(b"\\0")
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
''',
)

write(
    "src/personavoice/stage_lock.py",
    '''from __future__ import annotations

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
''',
)

write(
    "src/personavoice/setup_lock.py",
    '''from __future__ import annotations

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
''',
)

replace_once(
    "src/personavoice/state.py",
    "import math\nimport shutil\nimport sqlite3\n",
    "import math\nimport os\nimport shutil\nimport socket\nimport sqlite3\n",
)
replace_once(
    "src/personavoice/state.py",
    "from typing import Any\n",
    "from typing import Any\nfrom uuid import uuid4\n",
)
replace_once(
    "src/personavoice/state.py",
    "from personavoice.model_assets import (\n",
    "from personavoice.model_assets import (\n",
)
replace_once(
    "src/personavoice/state.py",
    "from personavoice.worker_contracts import purge_invalid_prepare_caches\n",
    "from personavoice.stage_lock import stage_lock\n"
    "from personavoice.worker_contracts import purge_invalid_prepare_caches\n",
)

old_running = '''    @contextmanager
    def running(
        self,
        name: str,
        fingerprint: str,
        *,
        force: bool = False,
    ) -> Iterator[dict[str, Any]]:
        state = self.load()
        stage = state.setdefault("stages", {}).setdefault(name, {})
        if name == "prepare":
            old_fingerprint = stage.get("fingerprint")
            old_policy = stage.get("cache_policy_version")
            must_invalidate = (
                force
                or old_policy != PREPARE_CACHE_POLICY_VERSION
                or (old_fingerprint is not None and old_fingerprint != fingerprint)
            )
            if must_invalidate:
                self._invalidate_prepare_derived()
            purge_invalid_prepare_caches(self.path.parent)

        stage.update(
            {
                "status": "running",
                "fingerprint": fingerprint,
                "started_at": _now(),
                "finished_at": None,
                "error": None,
            }
        )
        if name == "prepare":
            stage["cache_policy_version"] = PREPARE_CACHE_POLICY_VERSION
        self.save(state)
        try:
            yield stage
        except Exception as exc:
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            stage.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "error": str(exc),
                }
            )
            self.save(state)
            raise
        else:
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            stage.update(
                {
                    "status": "complete",
                    "finished_at": _now(),
                    "error": None,
                }
            )
            self.save(state)
'''
new_running = '''    @contextmanager
    def running(
        self,
        name: str,
        fingerprint: str,
        *,
        force: bool = False,
    ) -> Iterator[dict[str, Any]]:
        # The OS lock is the source of truth for liveness. It is released by the
        # kernel on normal exit, exceptions, crashes, and forced process death,
        # so long-running jobs never depend on arbitrary stale timeouts or PID
        # reuse heuristics. Acquire before mutating state/cache so a second
        # process cannot invalidate artifacts owned by the active stage.
        with stage_lock(self.path.parent, name):
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            if name == "prepare":
                old_fingerprint = stage.get("fingerprint")
                old_policy = stage.get("cache_policy_version")
                must_invalidate = (
                    force
                    or old_policy != PREPARE_CACHE_POLICY_VERSION
                    or (old_fingerprint is not None and old_fingerprint != fingerprint)
                )
                if must_invalidate:
                    self._invalidate_prepare_derived()
                purge_invalid_prepare_caches(self.path.parent)

            try:
                hostname = socket.gethostname()
            except OSError:
                hostname = None
            stage.update(
                {
                    "status": "running",
                    "fingerprint": fingerprint,
                    "started_at": _now(),
                    "finished_at": None,
                    "error": None,
                    "runner": {
                        "lock_protocol": 1,
                        "run_id": uuid4().hex,
                        "pid": os.getpid(),
                        "hostname": hostname,
                    },
                }
            )
            if name == "prepare":
                stage["cache_policy_version"] = PREPARE_CACHE_POLICY_VERSION
            self.save(state)
            try:
                yield stage
            except Exception as exc:
                state = self.load()
                stage = state.setdefault("stages", {}).setdefault(name, {})
                stage.update(
                    {
                        "status": "failed",
                        "finished_at": _now(),
                        "error": str(exc),
                    }
                )
                self.save(state)
                raise
            else:
                state = self.load()
                stage = state.setdefault("stages", {}).setdefault(name, {})
                stage.update(
                    {
                        "status": "complete",
                        "finished_at": _now(),
                        "error": None,
                    }
                )
                self.save(state)
'''
replace_once("src/personavoice/state.py", old_running, new_running)

replace_once(
    "src/personavoice/status.py",
    "from personavoice.state import StateStore\n",
    "from personavoice.stage_lock import stage_lock_held\nfrom personavoice.state import StateStore\n",
)
old_stage_audit = '''def _stage_audit(
    store: StateStore,
    state: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    stages = state.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    stage = stages.get(name)
    stage = stage if isinstance(stage, dict) else {}
    recorded_fingerprint = stage.get("fingerprint")
    artifact_complete = False
    if isinstance(recorded_fingerprint, str) and recorded_fingerprint:
        try:
            artifact_complete = store.is_complete(name, recorded_fingerprint)
        except (OSError, ValueError, TypeError):
            artifact_complete = False
    return {
        "recorded_status": stage.get("status"),
        "recorded_fingerprint": recorded_fingerprint,
        "artifact_complete": artifact_complete,
        "current_fingerprint": None,
        "fingerprint_current": None,
        "current_complete": None,
    }
'''
new_stage_audit = '''def _stage_audit(
    store: StateStore,
    state: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    stages = state.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    stage = stages.get(name)
    stage = stage if isinstance(stage, dict) else {}
    recorded_status = stage.get("status")
    recorded_fingerprint = stage.get("fingerprint")
    artifact_complete = False
    if isinstance(recorded_fingerprint, str) and recorded_fingerprint:
        try:
            artifact_complete = store.is_complete(name, recorded_fingerprint)
        except (OSError, ValueError, TypeError):
            artifact_complete = False

    runner = stage.get("runner") if isinstance(stage.get("runner"), dict) else None
    protocol_known = bool(runner and runner.get("lock_protocol") == 1)
    lock_held: bool | None = None
    activity_error: str | None = None
    if recorded_status == "running" and protocol_known:
        try:
            lock_held = stage_lock_held(store.path.parent, name)
        except (OSError, RuntimeError, ValueError) as exc:
            activity_error = f"{type(exc).__name__}: {exc}"

    activity_known = recorded_status == "running" and protocol_known and activity_error is None
    running_active = lock_held is True if activity_known else None
    stale_running = lock_held is False if activity_known else None
    recovery_hint = None
    if stale_running is True:
        recovery_hint = (
            f"The previous {name} process no longer owns its OS lock. Rerun the same command "
            "without --force to resume from valid caches/artifacts."
        )

    return {
        "recorded_status": recorded_status,
        "recorded_fingerprint": recorded_fingerprint,
        "artifact_complete": artifact_complete,
        "runner": runner,
        "activity_known": activity_known,
        "run_lock_held": lock_held,
        "running_active": running_active,
        "stale_running": stale_running,
        "activity_error": activity_error,
        "recovery_hint": recovery_hint,
        "current_fingerprint": None,
        "fingerprint_current": None,
        "current_complete": None,
    }
'''
replace_once("src/personavoice/status.py", old_stage_audit, new_stage_audit)

replace_once(
    "src/personavoice/cli.py",
    "from personavoice.setup_lock import SetupLockError, setup_lock\n",
    "from personavoice.setup_lock import SetupLockError, setup_lock\n"
    "from personavoice.stage_lock import StageLockError\n",
)
replace_once(
    "src/personavoice/cli.py",
    '''@app.command()
def prepare(name: str, force: bool = False) -> None:
    root, paths, cfg = _load(name)
    _print(prepare_persona(root, paths, cfg, force=force))


@app.command()
def train(name: str, force: bool = False) -> None:
    root, paths, cfg = _load(name)
    _print(train_persona(root, paths, cfg, force=force))


@app.command()
def build(
    name: str,
    force: bool = False,
    evaluate_after: bool = typer.Option(True, "--eval/--no-eval"),
) -> None:
    """One-command prepare + train + evaluation."""
    root, paths, cfg = _load(name)
    result = {"prepare": prepare_persona(root, paths, cfg, force=force)}
    result["train"] = train_persona(root, paths, cfg, force=force)
    if evaluate_after:
        result["evaluation"] = evaluate(root, paths, cfg)["summary"]
    _print(result)
''',
    '''@app.command()
def prepare(name: str, force: bool = False) -> None:
    root, paths, cfg = _load(name)
    try:
        result = prepare_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    _print(result)


@app.command()
def train(name: str, force: bool = False) -> None:
    root, paths, cfg = _load(name)
    try:
        result = train_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    _print(result)


@app.command()
def build(
    name: str,
    force: bool = False,
    evaluate_after: bool = typer.Option(True, "--eval/--no-eval"),
) -> None:
    """One-command prepare + train + evaluation."""
    root, paths, cfg = _load(name)
    try:
        result = {"prepare": prepare_persona(root, paths, cfg, force=force)}
        result["train"] = train_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    if evaluate_after:
        result["evaluation"] = evaluate(root, paths, cfg)["summary"]
    _print(result)
''',
)

write(
    "tests/test_stage_runtime_lock.py",
    '''from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.config import PersonaConfig
from personavoice.process_lock import exclusive_process_lock, process_lock_held
from personavoice.project import init_persona
from personavoice.setup_lock import SetupLockError, setup_lock
from personavoice.stage_lock import StageLockError
from personavoice.state import StateStore
from personavoice.status import persona_status


def test_process_lock_probe_tracks_live_owner(tmp_path: Path):
    path = tmp_path / "runtime.lock"
    assert process_lock_held(path) is False
    with exclusive_process_lock(path):
        assert process_lock_held(path) is True
    assert process_lock_held(path) is False


def test_setup_lock_still_rejects_parallel_setup(tmp_path: Path):
    with setup_lock(tmp_path):
        with pytest.raises(SetupLockError, match="already running"):
            with setup_lock(tmp_path):
                pass


def _persona(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    return paths, PersonaConfig.load(paths.config), StateStore(paths.state)


def test_running_stage_reports_active_os_lock(tmp_path: Path):
    paths, cfg, store = _persona(tmp_path)
    with store.running("prepare", "fingerprint"):
        report = persona_status(tmp_path, paths, cfg)
        audit = report["audit"]["prepare"]
        assert audit["activity_known"] is True
        assert audit["run_lock_held"] is True
        assert audit["running_active"] is True
        assert audit["stale_running"] is False
        assert audit["runner"]["lock_protocol"] == 1
        assert isinstance(audit["runner"]["run_id"], str)
        assert isinstance(audit["runner"]["pid"], int)


def test_stale_running_is_detected_without_timeout(tmp_path: Path):
    paths, cfg, store = _persona(tmp_path)
    with store.running("prepare", "fingerprint"):
        pass
    state = store.load()
    stage = state["stages"]["prepare"]
    stage["status"] = "running"
    store.save(state)

    report = persona_status(tmp_path, paths, cfg)
    audit = report["audit"]["prepare"]
    assert audit["activity_known"] is True
    assert audit["run_lock_held"] is False
    assert audit["running_active"] is False
    assert audit["stale_running"] is True
    assert "without --force" in audit["recovery_hint"]


def test_legacy_running_without_lock_protocol_is_not_false_stale(tmp_path: Path):
    paths, cfg, store = _persona(tmp_path)
    state = store.load()
    state.setdefault("stages", {})["prepare"] = {
        "status": "running",
        "fingerprint": "legacy",
    }
    store.save(state)

    audit = persona_status(tmp_path, paths, cfg)["audit"]["prepare"]
    assert audit["activity_known"] is False
    assert audit["running_active"] is None
    assert audit["stale_running"] is None


def test_parallel_same_stage_is_rejected_before_state_mutation(tmp_path: Path):
    paths, _cfg, first = _persona(tmp_path)
    second = StateStore(paths.state)
    with first.running("prepare", "first"):
        before = first.load()["stages"]["prepare"]["runner"]["run_id"]
        with pytest.raises(StageLockError, match="already running"):
            with second.running("prepare", "second"):
                pass
        current = first.load()["stages"]["prepare"]
        assert current["fingerprint"] == "first"
        assert current["runner"]["run_id"] == before
''',
)

print("stage runtime lock integration patch applied")
