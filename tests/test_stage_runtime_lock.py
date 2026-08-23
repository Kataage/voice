from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.config import PersonaConfig
from personavoice.process_lock import exclusive_process_lock, process_lock_held
from personavoice.project import init_persona
from personavoice.setup_lock import SetupLockError, setup_lock
from personavoice.stage_lock import StageLockError, stage_lock_path
from personavoice.state import StateStore
from personavoice.status import persona_status


def test_process_lock_probe_tracks_live_owner(tmp_path: Path):
    path = tmp_path / "runtime.lock"
    assert process_lock_held(path) is False
    with exclusive_process_lock(path):
        assert process_lock_held(path) is True
    assert process_lock_held(path) is False


def test_setup_lock_still_rejects_parallel_setup(tmp_path: Path):
    with (
        setup_lock(tmp_path),
        pytest.raises(SetupLockError, match="already running"),
        setup_lock(tmp_path),
    ):
        pass


@pytest.mark.parametrize("name", ["../prepare", "prepare/other", "", ".prepare", "prepare lock"])
def test_stage_lock_path_rejects_unsafe_names(tmp_path: Path, name: str):
    with pytest.raises(ValueError, match="Unsafe stage name"):
        stage_lock_path(tmp_path, name)


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
        with pytest.raises(StageLockError, match="already running"), second.running(
            "prepare", "second"
        ):
            pass
        current = first.load()["stages"]["prepare"]
        assert current["fingerprint"] == "first"
        assert current["runner"]["run_id"] == before
