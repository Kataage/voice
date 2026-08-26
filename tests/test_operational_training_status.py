from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from personavoice import doctor as doctor_module
from personavoice import status as status_module
from personavoice.config import PersonaConfig, TrainingConfig
from personavoice.executors import LocalResources
from personavoice.project import init_persona
from personavoice.state import StateStore
from personavoice.status import persona_status


def test_doctor_local_training_preflight_uses_full_defaults(monkeypatch, tmp_path: Path) -> None:
    setup = {"irodori_backend": "cu128"}
    monkeypatch.setattr(doctor_module, "_setup_state", lambda _root: setup)
    monkeypatch.setattr(
        doctor_module,
        "require_current_environment",
        lambda _root: setup,
    )
    monkeypatch.setattr(
        doctor_module,
        "inspect_local_resources",
        lambda *_args, **_kwargs: LocalResources(
            backend="cu128",
            setup_current=True,
            gpu_total_mib=32 * 1024,
            gpu_free_mib=30 * 1024,
            ram_available_bytes=64 * 1024**3,
            disk_free_bytes=100 * 1024**3,
        ),
    )

    result = doctor_module.training_preflight_status(tmp_path, TrainingConfig())

    assert result["ok"] is True
    assert result["requested_full_families"] == ["irodori", "lfm"]
    assert result["resources"]["backend"] == "cu128"
    assert result["failures"] == []


def test_doctor_modal_readiness_never_returns_auth_values(monkeypatch) -> None:
    secret = "modal-secret-must-not-appear"
    monkeypatch.setattr(doctor_module, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        doctor_module,
        "detect_modal_auth",
        lambda: SimpleNamespace(configured=True, source="environment", value=secret),
    )

    result = doctor_module.modal_readiness_status()

    assert result == {
        "ready": True,
        "sdk_installed": True,
        "auth_configured": True,
        "auth_source": "environment",
        "network_probe_performed": False,
    }
    assert secret not in repr(result)


def test_secret_filter_omits_values_but_preserves_remote_call_id(monkeypatch) -> None:
    secret = "must-never-be-visible"
    monkeypatch.setenv("HF_TOKEN", secret)
    unsafe = {
        "modal_token_secret": secret,
        "diagnostic": f"worker echoed {secret}",
        "nested": {
            "authorization_header": secret,
            "github_token": secret,
            "password": secret,
            "call_id": "fc-123",
        },
    }

    for sanitizer in (
        doctor_module._without_secret_values,
        status_module._without_secret_values,
    ):
        filtered = sanitizer(unsafe)
        assert secret not in repr(filtered)
        assert filtered == {
            "diagnostic": "[redacted]",
            "nested": {"call_id": "fc-123"},
        }


def _training_state(paths, *, published: bool) -> dict:
    fingerprint = "f" * 64
    plan_fingerprint = "p" * 64
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    state["modal_token_secret"] = "state-secret-must-not-appear"
    state["stages"] = {
        "train": {
            "status": "complete" if published else "trained",
            "fingerprint": fingerprint,
            "progress": {
                "executor": "modal",
                "remote_state": "running" if not published else "complete",
                "model": "irodori,lfm",
                "step": 2000,
                "checkpoint": "irodori/checkpoint-2000",
                "submission": {
                    "call_id": "fc-123",
                    "plan_fingerprint": plan_fingerprint,
                    "modal_token_secret": "nested-secret-must-not-appear",
                    "bundle_audit": {
                        "schema_version": 1,
                        "fingerprint": "a" * 64,
                        "file_count": 3,
                        "total_bytes": 4096,
                        "files": [
                            {
                                "path": "contracts/training-plan.json",
                                "role": "training-plan",
                                "sha256": "b" * 64,
                                "size": 1024,
                            }
                        ],
                    },
                },
            },
            "result": {
                "train_schema": 9,
                "fingerprint": fingerprint,
                "plan_fingerprint": plan_fingerprint,
                "executor": {
                    "kind": "modal",
                    "local_preflight": {"ok": False, "failures": []},
                },
                "families": {
                    "irodori": {
                        "enabled": True,
                        "method": "full",
                        "artifact": "models/.candidates/irodori/artifact",
                    },
                    "lfm": {
                        "enabled": True,
                        "method": "full",
                        "artifact": "models/.candidates/lfm/artifact",
                    },
                    "seed-vc": {
                        "enabled": False,
                        "method": "finetune",
                        "artifact": None,
                    },
                },
                "download_verified": True,
                "quality_gate": {
                    "passed": published,
                    "pending_local_evaluation": not published,
                },
            },
        }
    }
    paths.state.write_text(json.dumps(state), encoding="utf-8")
    return state


def test_status_reports_plan_remote_progress_candidate_and_publication_without_secrets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    _training_state(paths, published=False)
    monkeypatch.setattr(
        status_module,
        "training_preflight_status",
        lambda *_args, **_kwargs: {"ok": False, "failures": ["gpu"]},
    )
    monkeypatch.setattr(
        status_module,
        "modal_readiness_status",
        lambda: {
            "ready": True,
            "sdk_installed": True,
            "auth_configured": True,
            "auth_source": "environment",
            "network_probe_performed": False,
        },
    )
    monkeypatch.setattr(StateStore, "is_trained", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(StateStore, "is_complete", lambda *_args, **_kwargs: False)

    result = persona_status(tmp_path, paths, cfg)
    operation = result["audit"]["train"]["operation"]

    assert operation["executor"] == "modal"
    assert operation["remote_call_id"] == "fc-123"
    assert operation["remote_state"] == "running"
    assert operation["step"] == 2000
    assert operation["checkpoint"] == "irodori/checkpoint-2000"
    assert operation["bundle_audit"]["file_count"] == 3
    assert operation["bundle_audit"]["files"][0]["path"] == (
        "contracts/training-plan.json"
    )
    assert operation["plan_fingerprint"] == "p" * 64
    assert operation["candidate_complete"] is True
    assert operation["published_complete"] is False
    assert operation["families"]["irodori"]["candidate_ready"] is True
    assert operation["families"]["irodori"]["published"] is False
    assert result["audit"]["local_training_preflight"]["ok"] is False
    assert result["audit"]["modal"]["auth_configured"] is True
    assert "state-secret-must-not-appear" not in repr(result)
    assert "nested-secret-must-not-appear" not in repr(result)
    assert "modal_token_secret" not in repr(result)


def test_status_distinguishes_quality_published_artifacts(monkeypatch, tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    _training_state(paths, published=True)
    monkeypatch.setattr(
        status_module,
        "training_preflight_status",
        lambda *_args, **_kwargs: {"ok": True, "failures": []},
    )
    monkeypatch.setattr(
        status_module,
        "modal_readiness_status",
        lambda: {"ready": False, "network_probe_performed": False},
    )
    monkeypatch.setattr(StateStore, "is_trained", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(StateStore, "is_complete", lambda *_args, **_kwargs: True)

    operation = persona_status(tmp_path, paths, cfg)["audit"]["train"]["operation"]

    assert operation["candidate_complete"] is True
    assert operation["published_complete"] is True
    assert operation["quality_gate"]["passed"] is True
    assert operation["families"]["lfm"]["published"] is True
