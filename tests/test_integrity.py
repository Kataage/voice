from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from personavoice import media
from personavoice.config import PersonaConfig
from personavoice.dataset import replace_utterances
from personavoice.media import inventory_fingerprint
from personavoice.pipeline import _prepare_fingerprint
from personavoice.project import init_persona
from personavoice.state import StateStore
from personavoice.training import train_persona


def test_inventory_fingerprint_detects_same_size_same_mtime_replacement(tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"AAAA")
    stat = source.stat()
    first = inventory_fingerprint(tmp_path)

    source.write_bytes(b"BBBB")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = inventory_fingerprint(tmp_path)

    assert first != second


def test_extract_audio_does_not_publish_partial_cache_on_failure(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    destination = tmp_path / "cache" / "audio.flac"

    def fake_run(args, **_kwargs):
        Path(args[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    monkeypatch.setattr(media, "ffmpeg_command", lambda name: name)
    with pytest.raises(subprocess.CalledProcessError):
        media.extract_lossless_audio(source, destination)

    assert not destination.exists()
    assert list(destination.parent.glob(".*.tmp.flac")) == []


def test_extract_audio_atomically_publishes_success(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    destination = tmp_path / "cache" / "audio.flac"

    def fake_run(args, **_kwargs):
        Path(args[-1]).write_bytes(b"complete")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    monkeypatch.setattr(media, "ffmpeg_command", lambda name: name)
    media.extract_lossless_audio(source, destination)

    assert destination.read_bytes() == b"complete"
    assert list(destination.parent.glob(".*.tmp.flac")) == []


def _write_minimal_complete_prepare_artifacts(paths) -> dict:
    dataset = paths.dataset
    clip = dataset / "clips" / "u.flac"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"audio")
    digest = "a" * 64
    source_id = digest[:16]
    row = {
        "id": f"{source_id}_000001",
        "source_id": source_id,
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
    replace_utterances(dataset / "master.sqlite3", [row])
    (dataset / "source_inventory.json").write_text(
        json.dumps([{"sha256": digest, "path": "source.wav"}]) + "\n",
        encoding="utf-8",
    )
    (dataset / "master.json").write_text(json.dumps([row]) + "\n", encoding="utf-8")
    (dataset / "irodori_source.jsonl").write_text("", encoding="utf-8")
    (dataset / "lfm_train.jsonl").write_text("", encoding="utf-8")
    seed_manifest = dataset / "seed_vc" / "manifest.jsonl"
    seed_manifest.parent.mkdir(parents=True, exist_ok=True)
    seed_manifest.write_text("", encoding="utf-8")
    bank = paths.references / "bank.json"
    bank.write_text(json.dumps({"files": [], "seconds": 0.0}), encoding="utf-8")
    return {
        "prepare_schema": 4,
        "sources": 1,
        "skipped_sources": 0,
        "utterances": 1,
        "target_utterances": 1,
        "usable_tts_utterances": 1,
        "usable_seconds": 1.0,
        "master_db": str((dataset / "master.sqlite3").resolve()),
        "irodori_examples": 0,
        "lfm_examples": 0,
        "seed_vc_examples": 0,
        "references": 0,
    }


def test_train_refuses_dataset_when_prepare_inputs_changed(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    source = paths.raw / "source.wav"
    source.write_bytes(b"before")

    store = StateStore(paths.state)
    prepared = _prepare_fingerprint(paths, cfg)
    with store.running("prepare", prepared):
        store.set_result("prepare", _write_minimal_complete_prepare_artifacts(paths))

    # Prove this fixture is a genuinely complete prepare cache before mutating inputs.
    assert store.is_complete("prepare", prepared)

    source.write_bytes(b"after!")
    assert _prepare_fingerprint(paths, cfg) != prepared
    with pytest.raises(RuntimeError, match="missing, stale, or incomplete"):
        train_persona(tmp_path, paths, cfg)
