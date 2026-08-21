from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from personavoice import media
from personavoice.config import PersonaConfig
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
    media.extract_lossless_audio(source, destination)

    assert destination.read_bytes() == b"complete"
    assert list(destination.parent.glob(".*.tmp.flac")) == []


def test_train_refuses_dataset_when_prepare_inputs_changed(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    source = paths.raw / "source.wav"
    source.write_bytes(b"before")

    store = StateStore(paths.state)
    prepared = _prepare_fingerprint(paths, cfg)
    with store.running("prepare", prepared):
        store.set_result("prepare", {"ok": True})

    source.write_bytes(b"after!")
    with pytest.raises(RuntimeError, match="missing or stale"):
        train_persona(tmp_path, paths, cfg)
