from __future__ import annotations

import importlib.util
import struct
import sys
import types
from pathlib import Path

import pytest


class _FakeTensor:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.cloned = False
        self.unsqueeze_dim: int | None = None

    def clone(self):
        self.cloned = True
        return self

    def unsqueeze(self, dim: int):
        self.unsqueeze_dim = dim
        return self


def _load_worker(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda value: value
    torch.float32 = object()
    torch.frombuffer = lambda buffer, *, dtype: _FakeTensor(bytes(buffer))
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
    spec = importlib.util.spec_from_file_location("personavoice_test_diarization_waveform", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preload_audio_decodes_float32_pcm_with_audited_ffmpeg(tmp_path: Path, monkeypatch):
    worker = _load_worker(monkeypatch)
    source = tmp_path / "source.flac"
    source.write_bytes(b"local-audio")
    pcm = struct.pack("<ff", 0.25, -0.5)
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append([str(value) for value in args])
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return types.SimpleNamespace(returncode=0, stdout=pcm, stderr=b"")

    monkeypatch.setattr(worker, "_ffmpeg_executable", lambda: "audited-ffmpeg")
    monkeypatch.setattr(worker.subprocess, "run", fake_run)

    result = worker._preload_audio(str(source))

    assert result["sample_rate"] == 16000
    waveform = result["waveform"]
    assert waveform.raw == pcm
    assert waveform.cloned is True
    assert waveform.unsqueeze_dim == 0
    assert calls and calls[0][0] == "audited-ffmpeg"
    assert "-ac" in calls[0] and "1" in calls[0]
    assert "-ar" in calls[0] and "16000" in calls[0]
    assert "f32le" in calls[0]
    assert calls[0][-1] == "pipe:1"


def test_diarization_pipeline_receives_waveform_dict_not_path(monkeypatch):
    worker = _load_worker(monkeypatch)
    preloaded = {"waveform": object(), "sample_rate": 16000}
    monkeypatch.setattr(worker, "_preload_audio", lambda _audio: preloaded)

    class Annotation:
        def labels(self):
            return []

        def itertracks(self, *, yield_label):
            assert yield_label is True
            return iter(())

    class Output:
        speaker_diarization = Annotation()
        exclusive_speaker_diarization = None
        speaker_embeddings = None

    calls = []

    def pipeline(value, *, num_speakers):
        calls.append((value, num_speakers))
        return Output()

    result = worker.diarize_with_pipeline(pipeline, "never-passed-to-pyannote.flac", force_one=True)

    assert calls == [(preloaded, 1)]
    assert result == {"turns": [], "exclusive_turns": [], "speaker_embeddings": {}}


def test_preload_audio_rejects_nonlocal_input(monkeypatch):
    worker = _load_worker(monkeypatch)
    with pytest.raises(FileNotFoundError, match="not a local file"):
        worker._preload_audio("https://example.invalid/audio.wav")


def test_preload_audio_surfaces_ffmpeg_decode_failure(tmp_path: Path, monkeypatch):
    worker = _load_worker(monkeypatch)
    source = tmp_path / "source.wav"
    source.write_bytes(b"local-audio")
    monkeypatch.setattr(worker, "_ffmpeg_executable", lambda: "audited-ffmpeg")
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"bad input",
        ),
    )
    with pytest.raises(RuntimeError, match="bad input"):
        worker._preload_audio(str(source))


def test_preload_audio_rejects_empty_or_malformed_pcm(tmp_path: Path, monkeypatch):
    worker = _load_worker(monkeypatch)
    source = tmp_path / "source.wav"
    source.write_bytes(b"local-audio")
    monkeypatch.setattr(worker, "_ffmpeg_executable", lambda: "audited-ffmpeg")

    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    with pytest.raises(RuntimeError, match="no PCM samples"):
        worker._preload_audio(str(source))

    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=b"abc", stderr=b""),
    )
    with pytest.raises(RuntimeError, match="malformed float32 PCM"):
        worker._preload_audio(str(source))
