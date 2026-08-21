from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from personavoice.doctor import report as doctor_report


def _load_asr_worker(monkeypatch):
    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    worker_path = Path(__file__).parents[1] / "workers" / "asr" / "worker.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_asr_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_diarization_worker(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda value: value
    monkeypatch.setitem(sys.modules, "torch", torch)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    pyannote = types.ModuleType("pyannote")
    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = object
    pyannote.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)

    worker_path = Path(__file__).parents[1] / "workers" / "diarization" / "worker.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_diarization_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_complete_asr_files(local: Path, worker) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name in worker.REQUIRED_MODEL_FILES:
        if name.endswith(".json"):
            (local / name).write_text("{}\n", encoding="utf-8")
        else:
            (local / name).write_bytes(b"weights")


def test_asr_model_path_requires_nonempty_required_files_and_revision(tmp_path: Path, monkeypatch):
    worker = _load_asr_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    local = tmp_path / "models" / "asr" / worker.PINNED_MODEL_NAME
    _write_complete_asr_files(local, worker)
    (local / "model.bin").write_bytes(b"")
    (local / worker.REVISION_MARKER).write_text(worker.PINNED_MODEL_REVISION + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        worker.model_path(worker.PINNED_MODEL_NAME)

    (local / "model.bin").write_bytes(b"weights")
    (local / worker.REVISION_MARKER).write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited revision"):
        worker.model_path(worker.PINNED_MODEL_NAME)

    (local / worker.REVISION_MARKER).write_text(worker.PINNED_MODEL_REVISION + "\n", encoding="utf-8")
    assert Path(worker.model_path(worker.PINNED_MODEL_NAME)) == local


def test_asr_download_does_not_finalize_incomplete_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_asr_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    with pytest.raises(FileNotFoundError, match="required model files"):
        worker.download({})

    local = tmp_path / "models" / "asr" / worker.PINNED_MODEL_NAME
    _write_complete_asr_files(local, worker)
    result = worker.download({})
    assert result["revision"] == worker.PINNED_MODEL_REVISION
    assert (local / worker.REVISION_MARKER).read_text(encoding="utf-8").strip() == worker.PINNED_MODEL_REVISION


def test_doctor_asr_static_check_requires_full_offline_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_asr_worker(monkeypatch)
    local = tmp_path / "models" / "asr" / worker.PINNED_MODEL_NAME
    local.mkdir(parents=True)
    (local / "model.bin").write_bytes(b"weights")
    (local / "config.json").write_text("{}\n", encoding="utf-8")

    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["asr"] is False

    _write_complete_asr_files(local, worker)
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["asr"] is True

    (local / "tokenizer.json").unlink()
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["asr"] is False


def test_pyannote_local_source_requires_nonempty_config_and_revision(tmp_path: Path, monkeypatch):
    worker = _load_diarization_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    local = tmp_path / "models" / "pyannote" / "community-1"
    local.mkdir(parents=True)
    config = local / "config.yaml"
    marker = local / worker.REVISION_MARKER

    config.write_bytes(b"")
    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        worker.local_source()

    config.write_text("pipeline: {}\n", encoding="utf-8")
    marker.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited revision"):
        worker.local_source()

    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    assert Path(worker.local_source()) == local


def test_pyannote_download_does_not_finalize_empty_config(tmp_path: Path, monkeypatch):
    worker = _load_diarization_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    with pytest.raises(FileNotFoundError, match="without a valid config"):
        worker.download({})

    local = tmp_path / "models" / "pyannote" / "community-1"
    local.mkdir(parents=True, exist_ok=True)
    (local / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
    result = worker.download({})
    assert result["revision"] == worker.MODEL_REVISION
    assert (local / worker.REVISION_MARKER).read_text(encoding="utf-8").strip() == worker.MODEL_REVISION
