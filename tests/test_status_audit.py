from __future__ import annotations

import json
from pathlib import Path

from personavoice import status as status_module
from personavoice.config import PersonaConfig
from personavoice.pipeline import _prepare_fingerprint
from personavoice.project import init_persona
from personavoice.state import StateStore
from personavoice.status import persona_status


def test_status_reports_unreadable_state_without_crashing(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    paths.state.write_text("{broken", encoding="utf-8")

    result = persona_status(tmp_path, paths, cfg)

    assert result["state"] is None
    assert result["audit"]["state_readable"] is False
    assert "JSONDecodeError" in result["audit"]["state_error"]


def test_status_exposes_incomplete_recorded_stage_artifacts(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    state["stages"] = {
        "prepare": {
            "status": "complete",
            "fingerprint": "recorded-prepare",
            "cache_policy_version": "stale-policy",
            "result": {},
        },
        "train": {
            "status": "complete",
            "fingerprint": "recorded-train",
            "result": {},
        },
    }
    StateStore(paths.state).save(state)

    result = persona_status(tmp_path, paths, cfg)

    assert result["audit"]["state_readable"] is True
    assert result["audit"]["inputs_verified"] is False
    assert result["audit"]["prepare"]["recorded_status"] == "complete"
    assert result["audit"]["prepare"]["artifact_complete"] is False
    assert result["audit"]["train"]["artifact_complete"] is False
    assert result["audit"]["prepare"]["current_complete"] is None
    assert result["audit"]["train"]["blocked_by_prepare"] is None


def test_status_verify_detects_stale_prepare_fingerprint(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    source = paths.raw / "source.wav"
    source.write_bytes(b"before")
    recorded = _prepare_fingerprint(paths, cfg)

    state = json.loads(paths.state.read_text(encoding="utf-8"))
    state["stages"] = {
        "prepare": {
            "status": "complete",
            "fingerprint": recorded,
            "cache_policy_version": "stale-policy",
            "result": {},
        }
    }
    StateStore(paths.state).save(state)
    source.write_bytes(b"after")

    result = persona_status(tmp_path, paths, cfg, verify_inputs=True)

    prepare = result["audit"]["prepare"]
    train = result["audit"]["train"]
    assert result["audit"]["inputs_verified"] is True
    assert prepare["current_fingerprint"] != recorded
    assert prepare["fingerprint_current"] is False
    assert prepare["current_complete"] is False
    assert train["blocked_by_prepare"] is True
    assert train["current_complete"] is False


def test_status_never_reports_train_current_when_prepare_is_stale(
    tmp_path: Path,
    monkeypatch,
):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    state["stages"] = {
        "prepare": {
            "status": "complete",
            "fingerprint": "old-prepare",
            "cache_policy_version": "stale-policy",
            "result": {},
        },
        "train": {
            "status": "complete",
            "fingerprint": "same-train",
            "result": {},
        },
    }
    StateStore(paths.state).save(state)

    monkeypatch.setattr(status_module, "_prepare_fingerprint", lambda _paths, _cfg: "new-prepare")
    monkeypatch.setattr(status_module, "_training_fingerprint", lambda _paths, _cfg: "same-train")

    original = StateStore.is_complete

    def fake_complete(self, name: str, fingerprint: str) -> bool:
        if name == "train" and fingerprint == "same-train":
            return True
        return original(self, name, fingerprint)

    monkeypatch.setattr(StateStore, "is_complete", fake_complete)
    result = persona_status(tmp_path, paths, cfg, verify_inputs=True)

    train = result["audit"]["train"]
    assert train["fingerprint_current"] is True
    assert train["blocked_by_prepare"] is True
    assert train["current_complete"] is False
