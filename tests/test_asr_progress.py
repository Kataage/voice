from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _worker(monkeypatch: pytest.MonkeyPatch):
    ctranslate2 = types.ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 0
    ctranslate2.get_supported_compute_types = lambda *_args, **_kwargs: {"float32"}
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    faster = types.ModuleType("faster_whisper")
    faster.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", faster)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    runtime_policy = types.ModuleType("runtime_policy")
    runtime_policy.choose_compute_type = lambda _device, _supported, _requested: "float32"
    monkeypatch.setitem(sys.modules, "runtime_policy", runtime_policy)

    path = Path(__file__).parents[1] / "workers" / "asr" / "worker.py"
    spec = importlib.util.spec_from_file_location("asr_progress_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_model():
    word = types.SimpleNamespace(start=0.1, end=0.8, word="テスト", probability=0.99)
    segments = [
        types.SimpleNamespace(
            start=0.0,
            end=1.0,
            text="テスト",
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            words=[word],
        ),
        types.SimpleNamespace(
            start=1.0,
            end=2.5,
            text="です",
            avg_logprob=-0.2,
            no_speech_prob=0.02,
            words=[],
        ),
    ]
    info = types.SimpleNamespace(language="ja", language_probability=0.98, duration=2.5)

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return iter(segments), info

    return FakeModel()


def test_transcribe_with_model_reports_each_yielded_segment(monkeypatch: pytest.MonkeyPatch):
    worker = _worker(monkeypatch)
    updates: list[tuple[float, int]] = []

    result = worker.transcribe_with_model(
        _fake_model(),
        "unused.flac",
        language="ja",
        progress=lambda seconds, count: updates.append((seconds, count)),
    )

    assert updates == [(1.0, 1), (2.5, 2)]
    assert result["duration"] == 2.5
    assert [row["text"] for row in result["segments"]] == ["テスト", "です"]


def test_batch_publishes_runtime_and_model_load_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(
        worker,
        "runtime_config",
        lambda _requested: ("cuda", "int8_float32", {"float32", "int8_float32"}),
    )
    monkeypatch.setattr(worker, "ASR_HEARTBEAT_SECONDS", 0.0)
    checkpoint = tmp_path / "personas" / "alice" / "cache" / "asr" / ".checkpoints"

    def make_model(_name: str, *, device: str, compute_type: str):
        progress = json.loads((checkpoint / "progress.json").read_text(encoding="utf-8"))
        assert progress["phase"] == "model_load"
        assert progress["device"] == "cuda"
        assert progress["compute_type"] == "int8_float32"
        assert device == "cuda"
        assert compute_type == "int8_float32"
        return _fake_model()

    monkeypatch.setattr(worker, "_make_model_with_runtime", make_model)
    result = worker.batch_transcribe(
        {
            "items": [{"id": "source123", "audio": str(tmp_path / "unused.flac")}],
            "model": "large-v3",
            "compute_type": "auto",
            "language": "ja",
            "checkpoint_dir": str(checkpoint.resolve()),
        }
    )

    assert result["results"][0]["ok"] is True
    item_checkpoint = json.loads((checkpoint / "source123.json").read_text(encoding="utf-8"))
    assert item_checkpoint["result"]["duration"] == 2.5
    progress = json.loads((checkpoint / "progress.json").read_text(encoding="utf-8"))
    assert progress["state"] == "finished"
    assert progress["completed"] == 1
    assert progress["device"] == "cuda"
    assert progress["compute_type"] == "int8_float32"
