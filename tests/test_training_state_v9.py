from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.project import init_persona
from personavoice.state import StateStore


def _result(fingerprint: str) -> dict:
    return {
        "train_schema": 9,
        "fingerprint": fingerprint,
        "plan_fingerprint": "a" * 64,
        "executor": {"kind": "local", "remote_state": None},
        "families": {
            "irodori": {"enabled": False, "method": "full", "artifact": None},
            "lfm": {"enabled": False, "method": "full", "artifact": None},
            "seed-vc": {"enabled": False, "method": "finetune", "artifact": None},
        },
        "download_verified": True,
        "quality_gate": {"passed": True, "checks": []},
    }


def test_trained_candidate_is_not_complete_until_published(tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)
    fingerprint = "request-v9"

    with store.running("train", fingerprint, success_status="trained"):
        store.set_result("train", _result(fingerprint))

    assert store.is_trained(fingerprint) is True
    assert store.is_complete("train", fingerprint) is False

    store.set_status("train", "complete")
    assert store.is_complete("train", fingerprint) is True


def test_training_state_rejects_secret_named_fields(tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    result = _result("request-v9")
    result["executor"]["modal_token_secret"] = "do-not-store"

    with pytest.raises(ValueError, match="may not contain"):
        StateStore(paths.state).set_result("train", result)


def test_state_rejects_known_secret_values_in_result_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf-state-sentinel-value"
    monkeypatch.setenv("HF_TOKEN", secret)
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)

    with pytest.raises(ValueError, match="may not contain"):
        store.set_result("train", {"diagnostic": f"download rejected {secret}"})
    with pytest.raises(ValueError, match="may not contain"):
        store.set_progress("train", {"diagnostic": f"worker rejected {secret}"})

    assert secret not in paths.state.read_text(encoding="utf-8")


def test_state_redacts_all_known_process_secrets_from_failures_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "HF_TOKEN": "hf-state-secret-value",
        "MODAL_TOKEN_ID": "modal-state-id-value",
        "MODAL_TOKEN_SECRET": "modal-state-secret-value",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)

    diagnostic = " / ".join(secrets.values())
    with pytest.raises(RuntimeError), store.running("train", "request-v9"):
        raise RuntimeError(f"remote failure: {diagnostic}")

    state = json.loads(paths.state.read_text(encoding="utf-8"))
    assert state["stages"]["train"]["error"] == (
        "remote failure: [redacted] / [redacted] / [redacted]"
    )
    assert not any(value in paths.state.read_text(encoding="utf-8") for value in secrets.values())

    store.set_status("train", "failed", error=f"retry failed: {diagnostic}")
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    assert state["stages"]["train"]["error"] == (
        "retry failed: [redacted] / [redacted] / [redacted]"
    )
    assert not any(value in paths.state.read_text(encoding="utf-8") for value in secrets.values())
