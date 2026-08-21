from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import workers
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

    assert len(removed) == 8
    for name in valid_values:
        directory = cache / name
        assert (directory / "valid.json").is_file()
        assert not (directory / "empty.json").exists()
        assert not (directory / "truncated.json").exists()


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

    monkeypatch.setattr(workers, "require_current_environment", lambda _root: None)
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
