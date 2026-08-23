from __future__ import annotations

from personavoice import ffmpeg_materializer, runtime_dependencies


def test_non_x64_windows_accepts_valid_explicit_ffmpeg_override(tmp_path, monkeypatch):
    monkeypatch.setattr(ffmpeg_materializer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ffmpeg_materializer.platform, "machine", lambda: "ARM64")
    monkeypatch.setenv("PERSONAVOICE_FFMPEG_BIN", str(tmp_path / "ffmpeg-bin"))
    expected = runtime_dependencies.FfmpegRuntime(
        ffmpeg=str(tmp_path / "ffmpeg-bin" / "ffmpeg.exe"),
        ffprobe=str(tmp_path / "ffmpeg-bin" / "ffprobe.exe"),
        bin_dir=str(tmp_path / "ffmpeg-bin"),
        version_major=8,
        shared_libraries=True,
        torchcodec_compatible=True,
        source="PERSONAVOICE_FFMPEG_BIN",
        error=None,
    )
    monkeypatch.setattr(ffmpeg_materializer, "_candidate_runtime", lambda *_args: expected)

    assert ffmpeg_materializer.materialize_windows_ffmpeg(tmp_path) is expected


def test_non_x64_windows_auto_materializer_fails_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(ffmpeg_materializer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ffmpeg_materializer.platform, "machine", lambda: "ARM64")
    monkeypatch.delenv("PERSONAVOICE_FFMPEG_BIN", raising=False)
    called = False

    def should_not_download(*_args):
        nonlocal called
        called = True
        raise AssertionError("download must not start on an unsupported auto-materializer host")

    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", should_not_download)

    try:
        ffmpeg_materializer.materialize_windows_ffmpeg(tmp_path)
    except RuntimeError as exc:
        assert "x86_64-only" in str(exc)
    else:
        raise AssertionError("non-x64 automatic materialization must fail closed")
    assert called is False
