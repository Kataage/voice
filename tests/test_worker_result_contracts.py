from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from personavoice import workers
from personavoice.model_assets import SEED_VC_SOURCE_REVISION
from personavoice.worker_contracts import (
    purge_invalid_prepare_caches,
    valid_asr_result,
    valid_diarization_result,
    valid_embedding_result,
    valid_sense_result,
    validate_worker_response,
)


def _valid_asr() -> dict:
    return {
        "language": "ja",
        "language_probability": 0.99,
        "duration": 1.0,
        "segments": [],
    }


def _valid_diarization() -> dict:
    return {
        "turns": [],
        "exclusive_turns": [],
        "speaker_embeddings": {},
    }


def _valid_sense() -> dict:
    return {
        "raw": "",
        "emotion": "UNKNOWN",
        "events": [],
        "tags": [],
    }


def test_prepare_result_validators_accept_minimal_valid_outputs_and_reject_empty_dicts():
    assert valid_asr_result(_valid_asr())
    assert valid_diarization_result(_valid_diarization())
    assert valid_embedding_result({"embedding": [0.1, -0.2, 0.3]})
    assert valid_sense_result(_valid_sense())

    assert not valid_asr_result({})
    assert not valid_diarization_result({})
    assert not valid_embedding_result({})
    assert not valid_sense_result({})


def test_prepare_cache_purge_removes_semantically_invalid_json_only(tmp_path: Path):
    cache = tmp_path / "cache"
    valid_values = {
        "asr": _valid_asr(),
        "diarization": _valid_diarization(),
        "identity": {"embedding": [0.1, 0.2]},
        "sense": _valid_sense(),
    }
    for name, value in valid_values.items():
        directory = cache / name
        directory.mkdir(parents=True)
        (directory / "valid.json").write_text(json.dumps(value), encoding="utf-8")
        (directory / "empty.json").write_text("{}", encoding="utf-8")
        (directory / "truncated.json").write_text("{broken", encoding="utf-8")

    removed = purge_invalid_prepare_caches(tmp_path)

    assert len(removed) == 4
    for name in valid_values:
        directory = cache / name
        assert (directory / "valid.json").is_file()
        assert not (directory / "empty.json").exists()
        # Syntax corruption is preserved at this stage so a same-fingerprint
        # resume does not eagerly discard expensive cache state. The existing
        # per-cache pipeline reader removes this file when it is actually read.
        assert (directory / "truncated.json").is_file()


def test_worker_batch_contract_rejects_invalid_success_payload():
    with pytest.raises(RuntimeError, match="invalid response schema"):
        validate_worker_response(
            "asr",
            "batch_transcribe",
            {"results": [{"id": "source", "ok": True, "result": {}}]},
        )

    validate_worker_response(
        "asr",
        "batch_transcribe",
        {"results": [{"id": "source", "ok": True, "result": _valid_asr()}]},
    )


def test_worker_batch_contract_rejects_duplicate_ids_and_invalid_error_rows():
    with pytest.raises(RuntimeError, match="invalid response schema"):
        validate_worker_response(
            "sense",
            "batch_analyze",
            {
                "results": [
                    {"id": "same", "ok": True, "result": _valid_sense()},
                    {"id": "same", "ok": True, "result": _valid_sense()},
                ]
            },
        )
    with pytest.raises(RuntimeError, match="invalid response schema"):
        validate_worker_response(
            "sense",
            "batch_analyze",
            {"results": [{"id": "x", "ok": False, "error": ""}]},
        )


def test_worker_call_rejects_invalid_subprocess_result_before_return(tmp_path: Path, monkeypatch):
    project = tmp_path / "workers" / "asr"
    project.mkdir(parents=True)
    instance = workers.Worker(name="asr", project_dir=project)

    monkeypatch.setattr(
        workers,
        "require_current_environment",
        lambda _root, **_kwargs: {"irodori_backend": "cpu"},
    )
    monkeypatch.setattr(
        workers,
        "run_json",
        lambda *_args, **_kwargs: {
            "results": [{"id": "source", "ok": True, "result": {}}]
        },
    )

    with pytest.raises(RuntimeError, match="invalid response schema"):
        instance.call(tmp_path, "batch_transcribe", {"items": []})

    requests = tmp_path / ".runtime" / "requests"
    assert not list(requests.glob("*.json"))


def _seed_vendor(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "seed-vc"
    (vendor / ".git").mkdir(parents=True)
    (vendor / "inference_v2.py").write_text("# pinned\n", encoding="utf-8")
    return vendor


def test_seed_vc_vendor_preflight_rejects_wrong_head_and_untracked_files(
    tmp_path: Path,
    monkeypatch,
):
    vendor = _seed_vendor(tmp_path)

    def clean_run(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=SEED_VC_SOURCE_REVISION + "\n")
        if args[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        raise AssertionError(args)

    monkeypatch.setattr(workers, "run", clean_run)
    assert workers._require_seed_vc_vendor_integrity(tmp_path) == vendor

    def wrong_head(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="0" * 40 + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(workers, "run", wrong_head)
    with pytest.raises(RuntimeError, match="vendor HEAD mismatch"):
        workers._require_seed_vc_vendor_integrity(tmp_path)

    def untracked_file(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=SEED_VC_SOURCE_REVISION + "\n")
        if args[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(stdout="?? modules/injected.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(workers, "run", untracked_file)
    with pytest.raises(RuntimeError, match="untracked files"):
        workers._require_seed_vc_vendor_integrity(tmp_path)


def test_seed_vc_worker_call_runs_vendor_preflight_before_subprocess(tmp_path: Path, monkeypatch):
    project = tmp_path / "workers" / "seed_vc"
    project.mkdir(parents=True)
    instance = workers.Worker(name="seed_vc", project_dir=project)

    monkeypatch.setattr(
        workers,
        "require_current_environment",
        lambda _root, **_kwargs: {"irodori_backend": "cpu"},
    )
    monkeypatch.setattr(
        workers,
        "_require_seed_vc_vendor_integrity",
        lambda _root: (_ for _ in ()).throw(RuntimeError("vendor preflight blocked")),
    )
    monkeypatch.setattr(
        workers,
        "run_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess must not run")),
    )

    with pytest.raises(RuntimeError, match="vendor preflight blocked"):
        instance.call(tmp_path, "health", {})

    assert not (tmp_path / ".runtime" / "requests").exists()
