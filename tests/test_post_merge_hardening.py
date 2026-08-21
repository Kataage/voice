from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import irodori
from personavoice.dataset import replace_utterances
from personavoice.model_assets import LFM_MODEL_REVISION
from personavoice.state import PREPARE_CACHE_POLICY_VERSION, StateStore
from personavoice.workers import Worker


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _state(path: Path, *, stage: str, fingerprint: str, result: dict) -> StateStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_value = {
        "status": "complete",
        "fingerprint": fingerprint,
        "result": result,
    }
    if stage == "prepare":
        stage_value["cache_policy_version"] = PREPARE_CACHE_POLICY_VERSION
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "persona": path.parent.name,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "stages": {stage: stage_value},
            }
        ),
        encoding="utf-8",
    )
    return StateStore(path)


def _write_master(path: Path, clip: Path) -> None:
    replace_utterances(
        path,
        [
            {
                "id": "source_000001",
                "source_id": "source",
                "source_path": "source.wav",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "target": True,
                "speaker_similarity": 0.9,
                "speaker_coverage": 1.0,
                "overlap_ratio": 0.0,
                "text": "a",
                "text_annotated": "a",
                "emotion": "NEUTRAL",
                "events": [],
                "caption": "",
                "audio_path": str(clip.resolve()),
                "quality": 1.0,
            }
        ],
    )


def test_worker_sync_refuses_missing_lockfile(tmp_path: Path):
    project = tmp_path / "workers" / "lfm"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n", encoding="utf-8")
    instance = Worker(name="lfm", project_dir=project)

    with pytest.raises(FileNotFoundError, match="Audited worker lockfile is missing"):
        instance.sync(tmp_path)


def test_irodori_backend_requires_explicit_setup_state(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="setup state is missing"):
        irodori.configured_backend(tmp_path)

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "setup.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="setup state is unreadable"):
        irodori.configured_backend(tmp_path)


def test_irodori_lora_completion_requires_config_and_weight(tmp_path: Path):
    adapter = tmp_path / "checkpoint_final"
    adapter.mkdir()
    assert not irodori.lora_adapter_complete(adapter)

    _write(adapter / "adapter_config.json", b"{}")
    assert not irodori.lora_adapter_complete(adapter)

    _write(adapter / "adapter_model.safetensors", b"weights")
    assert irodori.lora_adapter_complete(adapter)
    assert not irodori.lora_resume_checkpoint_complete(adapter)

    _write(adapter / "trainer_state.pt", b"state")
    assert irodori.lora_resume_checkpoint_complete(adapter)


def test_prepare_cache_hit_requires_exported_artifacts(tmp_path: Path):
    persona = tmp_path / "personas" / "alice"
    dataset = persona / "dataset"
    references = persona / "references"
    fingerprint = "prepare-fingerprint"

    _write(dataset / "source_inventory.json", b"[]")
    _write(dataset / "master.json", b"[]")
    clip = _write(dataset / "clips" / "u.flac", b"audio")
    _write_master(dataset / "master.sqlite3", clip)
    seed_clip = _write(dataset / "seed_vc" / "audio" / "u.flac", b"audio")
    (dataset / "irodori_source.jsonl").write_text(
        json.dumps({"audio": str(clip.resolve()), "text": "a"}) + "\n",
        encoding="utf-8",
    )
    (dataset / "lfm_train.jsonl").write_text(
        json.dumps({"prompt": [], "completion": []}) + "\n",
        encoding="utf-8",
    )
    (dataset / "seed_vc" / "manifest.jsonl").write_text(
        json.dumps({"audio": str(seed_clip.resolve())}) + "\n",
        encoding="utf-8",
    )
    ref = _write(references / "ref.flac", b"audio")
    (references / "bank.json").write_text(
        json.dumps({"files": [str(ref.resolve())], "seconds": 1.0}),
        encoding="utf-8",
    )
    result = {
        "prepare_schema": 4,
        "sources": 1,
        "skipped_sources": 0,
        "utterances": 1,
        "target_utterances": 1,
        "usable_tts_utterances": 1,
        "usable_seconds": 1.0,
        "master_db": str((dataset / "master.sqlite3").resolve()),
        "irodori_examples": 1,
        "lfm_examples": 1,
        "seed_vc_examples": 1,
        "references": 1,
    }
    store = _state(persona / "state.json", stage="prepare", fingerprint=fingerprint, result=result)
    assert store.is_complete("prepare", fingerprint)

    clip.unlink()
    assert not store.is_complete("prepare", fingerprint)


def test_train_cache_hit_requires_complete_adapters(tmp_path: Path):
    persona = tmp_path / "personas" / "alice"
    models = persona / "models"
    fingerprint = "train-fingerprint"
    base = _write(tmp_path / "models" / "irodori" / "base.safetensors", b"base")
    speaker = _write(models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors", b"s")
    irodori_lora = models / "irodori" / "lora" / "checkpoint_final"
    _write(irodori_lora / "adapter_config.json", b"{}")
    irodori_weight = _write(irodori_lora / "adapter_model.safetensors", b"w")
    lfm = models / "lfm" / "adapter"
    _write(lfm / "adapter_config.json", b"{}")
    _write(lfm / "adapter_model.safetensors", b"w")
    (lfm / ".personavoice-base-revision").write_text(LFM_MODEL_REVISION + "\n", encoding="utf-8")
    result = {
        "train_schema": 8,
        "fingerprint": fingerprint,
        "irodori": {
            "base": str(base.resolve()),
            "speaker_embedding": str(speaker.resolve()),
            "lora_adapter": str(irodori_lora.resolve()),
        },
        "lfm_adapter": str(lfm.resolve()),
        "seed_vc_cfm": None,
    }
    store = _state(persona / "state.json", stage="train", fingerprint=fingerprint, result=result)
    assert store.is_complete("train", fingerprint)

    irodori_weight.unlink()
    assert not store.is_complete("train", fingerprint)


def test_train_cache_hit_rejects_logically_incomplete_result(tmp_path: Path):
    persona = tmp_path / "personas" / "alice"
    fingerprint = "train-fingerprint"
    base = _write(tmp_path / "base.safetensors", b"base")
    result = {
        "train_schema": 8,
        "fingerprint": fingerprint,
        "irodori": {"base": str(base.resolve())},
        "lfm_adapter": None,
        # seed_vc_cfm intentionally missing: a truncated logical result must not cache-hit.
    }
    store = _state(persona / "state.json", stage="train", fingerprint=fingerprint, result=result)
    assert not store.is_complete("train", fingerprint)
