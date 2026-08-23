from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from personavoice import pipeline
from personavoice.atomic import atomic_write_json
from personavoice.config import PersonaConfig
from personavoice.prepare_checkpoints import (
    checkpoint_dir,
    prepare_batch_progress,
    recover_checkpoint,
)
from personavoice.project import init_persona
from personavoice.state import (
    PREPARE_CACHE_POLICY_COMPATIBILITY,
    PREPARE_CACHE_POLICY_VERSION,
    _prepare_policy_compatible,
)
from personavoice.status import persona_status


def _asr_result() -> dict:
    return {
        "language": "ja",
        "language_probability": 1.0,
        "duration": 1.0,
        "segments": [],
    }


def _checkpoint(directory: Path, item_id: str, result: dict) -> Path:
    path = directory / f"{item_id}.json"
    atomic_write_json(path, {"schema": 1, "id": item_id, "result": result})
    return path


def test_checkpoint_requires_matching_id_and_semantic_result(tmp_path: Path):
    directory = tmp_path / ".checkpoints"
    item_id = "abc123"
    path = _checkpoint(directory, item_id, _asr_result())
    assert recover_checkpoint(directory, item_id, "asr") == _asr_result()
    assert path.exists()

    atomic_write_json(path, {"schema": 1, "id": "other", "result": _asr_result()})
    assert recover_checkpoint(directory, item_id, "asr") is None
    assert not path.exists()

    path = _checkpoint(directory, item_id, {"language": "ja", "segments": []})
    assert recover_checkpoint(directory, item_id, "asr") is None
    assert not path.exists()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{truncated", encoding="utf-8")
    assert recover_checkpoint(directory, item_id, "asr") is None
    assert not path.exists()


def test_asr_batch_recovers_successful_item_after_worker_crash(tmp_path: Path, monkeypatch):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    source_id = "a" * 16
    audio = paths.cache / "audio" / f"{source_id}.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio")
    sources = [{"source_id": source_id, "audio": audio}]

    class CrashingWorker:
        def call(self, _root, command, payload):
            assert command == "batch_transcribe"
            directory = Path(payload["checkpoint_dir"])
            _checkpoint(directory, source_id, _asr_result())
            atomic_write_json(
                directory / "progress.json",
                {
                    "schema": 1,
                    "worker": "asr",
                    "command": "batch_transcribe",
                    "phase": "transcribe",
                    "completed": 1,
                    "total": 1,
                    "failed": 0,
                    "current_id": None,
                    "state": "running",
                },
            )
            raise RuntimeError("simulated process crash")

    monkeypatch.setattr(pipeline, "worker", lambda _root, _name: CrashingWorker())
    with pytest.raises(RuntimeError, match="simulated process crash"):
        pipeline._batch_asr(tmp_path, paths, cfg, sources)

    partial = checkpoint_dir(paths.cache / "asr") / f"{source_id}.json"
    assert partial.is_file()

    class MustNotRun:
        def call(self, *_args, **_kwargs):
            raise AssertionError("recovered ASR item must not be recomputed")

    monkeypatch.setattr(pipeline, "worker", lambda _root, _name: MustNotRun())
    result = pipeline._batch_asr(tmp_path, paths, cfg, sources)
    assert result[source_id] == _asr_result()
    assert json.loads((paths.cache / "asr" / f"{source_id}.json").read_text(encoding="utf-8")) == _asr_result()
    assert not checkpoint_dir(paths.cache / "asr").exists()


def test_status_exposes_advisory_batch_progress(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    directory = checkpoint_dir(paths.cache / "asr")
    _checkpoint(directory, "a" * 16, _asr_result())
    atomic_write_json(
        directory / "progress.json",
        {
            "schema": 1,
            "worker": "asr",
            "command": "batch_transcribe",
            "phase": "transcribe",
            "completed": 3,
            "total": 10,
            "failed": 1,
            "current_id": "b" * 16,
            "state": "running",
        },
    )
    progress = prepare_batch_progress(paths.root)
    assert progress["asr"]["completed"] == 3
    assert progress["asr"]["checkpointed_successes"] == 1
    status = persona_status(tmp_path, paths, cfg)
    assert status["audit"]["prepare"]["batch_progress"]["asr"]["total"] == 10


def _load_worker(monkeypatch, name: str):
    if name == "asr":
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 0
        ctranslate2.get_supported_compute_types = lambda *_args, **_kwargs: {"float32"}
        monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
        faster = types.ModuleType("faster_whisper")
        faster.WhisperModel = object
        monkeypatch.setitem(sys.modules, "faster_whisper", faster)
        runtime_policy = types.ModuleType("runtime_policy")
        runtime_policy.choose_compute_type = lambda _device, _supported, _requested: "float32"
        monkeypatch.setitem(sys.modules, "runtime_policy", runtime_policy)
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    elif name == "diarization":
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.device = lambda value: value
        torch.float32 = object()
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
    else:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch)
        funasr = types.ModuleType("funasr")
        funasr.AutoModel = object
        monkeypatch.setitem(sys.modules, "funasr", funasr)
        modelscope = types.ModuleType("modelscope")
        modelscope.snapshot_download = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "modelscope", modelscope)

    worker_path = Path(__file__).parents[1] / "workers" / name / "worker.py"
    spec = importlib.util.spec_from_file_location(f"checkpoint_test_{name}", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["asr", "diarization", "sense"])
def test_worker_checkpoint_path_is_confined_to_persona_cache(tmp_path: Path, monkeypatch, name: str):
    module = _load_worker(monkeypatch, name)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    valid = tmp_path / "personas" / "alice" / "cache" / name / ".checkpoints"
    resolved = module._checkpoint_directory({"checkpoint_dir": str(valid)})
    assert resolved == valid.resolve()

    with pytest.raises(ValueError, match="escapes PERSONAVOICE_ROOT"):
        module._checkpoint_directory({"checkpoint_dir": str(tmp_path.parent / "outside")})
    with pytest.raises(ValueError, match="Unsafe prepare checkpoint item id"):
        module._write_item_checkpoint(resolved, "../escape", {"ok": True})


def test_prepare_policy_migration_is_scoped_to_exact_new_generation():
    assert set(PREPARE_CACHE_POLICY_COMPATIBILITY) == {PREPARE_CACHE_POLICY_VERSION}
    previous = PREPARE_CACHE_POLICY_COMPATIBILITY[PREPARE_CACHE_POLICY_VERSION]
    assert previous
    assert PREPARE_CACHE_POLICY_VERSION not in previous
    assert all(_prepare_policy_compatible(value) for value in previous)
    assert not _prepare_policy_compatible("12-unrelated-old-policy")
