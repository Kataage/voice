from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import ffmpeg_materializer, runtime_dependencies
from personavoice.atomic import atomic_write_json


def _runtime(root: Path, *, source: str, name: str) -> runtime_dependencies.FfmpegRuntime:
    bin_dir = root / name
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "ffmpeg.exe").write_bytes(b"ffmpeg-generation-1")
    (bin_dir / "ffprobe.exe").write_bytes(b"ffprobe-generation-1")
    for dll in (
        "avutil-60.dll",
        "avcodec-62.dll",
        "avformat-62.dll",
        "swresample-6.dll",
        "swscale-9.dll",
    ):
        (bin_dir / dll).write_bytes(f"{dll}-generation-1".encode())
    return runtime_dependencies.FfmpegRuntime(
        ffmpeg=str(bin_dir / "ffmpeg.exe"),
        ffprobe=str(bin_dir / "ffprobe.exe"),
        bin_dir=str(bin_dir),
        version_major=8,
        shared_libraries=True,
        torchcodec_compatible=True,
        source=source,
        error=None,
    )


def test_setup_records_exact_ffmpeg_runtime_provenance(tmp_path: Path, monkeypatch) -> None:
    expected = _runtime(tmp_path, source="PATH", name="system-bin")
    monkeypatch.setattr(ffmpeg_materializer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ffmpeg_materializer, "require_ffmpeg_runtime", lambda: expected)

    assert ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path) is expected
    recorded = runtime_dependencies.recorded_ffmpeg_provenance(tmp_path)
    assert recorded == runtime_dependencies.ffmpeg_provenance(expected)
    assert recorded is not None
    assert recorded["critical_sha256"]["ffmpeg"]
    assert recorded["critical_sha256"]["ffprobe"]


def test_missing_pinned_runtime_cannot_silently_fall_back_to_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pinned = _runtime(tmp_path, source="PersonaVoice:pinned", name="pinned-bin")
    fallback = _runtime(tmp_path, source="PATH", name="path-bin")
    atomic_write_json(
        runtime_dependencies.ffmpeg_provenance_path(tmp_path),
        runtime_dependencies.ffmpeg_provenance(pinned),
    )
    monkeypatch.setattr(runtime_dependencies, "_discover_ffmpeg_runtime", lambda: fallback)

    status = runtime_dependencies.ffmpeg_provenance_status(tmp_path)
    assert status["ok"] is False
    assert "changed after PersonaVoice setup" in str(status["error"])


def test_in_place_ffmpeg_binary_change_is_detected(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, source="PATH", name="system-bin")
    atomic_write_json(
        runtime_dependencies.ffmpeg_provenance_path(tmp_path),
        runtime_dependencies.ffmpeg_provenance(runtime),
    )
    monkeypatch.setattr(runtime_dependencies, "_discover_ffmpeg_runtime", lambda: runtime)
    Path(runtime.ffmpeg).write_bytes(b"ffmpeg-generation-2")

    status = runtime_dependencies.ffmpeg_provenance_status(tmp_path)
    assert status["ok"] is False
    assert "changed after PersonaVoice setup" in str(status["error"])


def test_ffmpeg_command_refuses_runtime_switch_after_setup(tmp_path: Path, monkeypatch) -> None:
    pinned = _runtime(tmp_path, source="PersonaVoice:pinned", name="pinned-bin")
    fallback = _runtime(tmp_path, source="PATH", name="path-bin")
    runtime_dir = tmp_path / ".runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "setup.json").write_text(json.dumps({}), encoding="utf-8")
    atomic_write_json(
        runtime_dependencies.ffmpeg_provenance_path(tmp_path),
        runtime_dependencies.ffmpeg_provenance(pinned),
    )
    monkeypatch.setattr(runtime_dependencies, "_repo_root_if_available", lambda: tmp_path)
    monkeypatch.setattr(runtime_dependencies, "_discover_ffmpeg_runtime", lambda: fallback)

    with pytest.raises(RuntimeError, match="changed after PersonaVoice setup"):
        runtime_dependencies.command("ffmpeg")


def test_ffmpeg_command_requires_provenance_after_setup(tmp_path: Path, monkeypatch) -> None:
    current = _runtime(tmp_path, source="PATH", name="path-bin")
    runtime_dir = tmp_path / ".runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "setup.json").write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(runtime_dependencies, "_repo_root_if_available", lambda: tmp_path)
    monkeypatch.setattr(runtime_dependencies, "_discover_ffmpeg_runtime", lambda: current)

    with pytest.raises(RuntimeError, match="provenance is missing"):
        runtime_dependencies.command("ffprobe")
