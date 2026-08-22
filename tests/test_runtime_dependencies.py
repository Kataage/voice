from __future__ import annotations

from pathlib import Path

from personavoice import runtime_dependencies


def _windows_runtime_tree(root: Path, *, shared: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ffmpeg.exe").write_bytes(b"x")
    (root / "ffprobe.exe").write_bytes(b"x")
    if shared:
        for name in (
            "avutil-60.dll",
            "avcodec-62.dll",
            "avformat-62.dll",
            "swresample-6.dll",
            "swscale-9.dll",
        ):
            (root / name).write_bytes(b"x")
    return root


def test_windows_shared_ffmpeg_8_is_torchcodec_compatible(tmp_path: Path, monkeypatch):
    directory = _windows_runtime_tree(tmp_path / "bin")
    monkeypatch.setattr(runtime_dependencies.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_dependencies, "_version_major", lambda _path: 8)

    runtime = runtime_dependencies._candidate_runtime(directory, "test")

    assert runtime is not None
    assert runtime.version_major == 8
    assert runtime.shared_libraries is True
    assert runtime.torchcodec_compatible is True


def test_windows_static_ffmpeg_is_not_accepted_for_torchcodec(tmp_path: Path, monkeypatch):
    directory = _windows_runtime_tree(tmp_path / "bin", shared=False)
    monkeypatch.setattr(runtime_dependencies.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_dependencies, "_version_major", lambda _path: 8)

    runtime = runtime_dependencies._candidate_runtime(directory, "test")

    assert runtime is not None
    assert runtime.shared_libraries is False
    assert runtime.torchcodec_compatible is False


def test_ffmpeg_9_is_not_accepted_by_torchcodec_010_contract(tmp_path: Path, monkeypatch):
    directory = _windows_runtime_tree(tmp_path / "bin")
    monkeypatch.setattr(runtime_dependencies.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_dependencies, "_version_major", lambda _path: 9)

    runtime = runtime_dependencies._candidate_runtime(directory, "test")

    assert runtime is not None
    assert runtime.shared_libraries is True
    assert runtime.torchcodec_compatible is False


def test_explicit_ffmpeg_directory_is_exported_to_worker_environment(
    tmp_path: Path,
    monkeypatch,
):
    runtime = runtime_dependencies.FfmpegRuntime(
        ffmpeg=str(tmp_path / "ffmpeg.exe"),
        ffprobe=str(tmp_path / "ffprobe.exe"),
        bin_dir=str(tmp_path),
        version_major=8,
        shared_libraries=True,
        torchcodec_compatible=True,
        source="test",
        error=None,
    )
    monkeypatch.setattr(runtime_dependencies, "ffmpeg_runtime", lambda: runtime)
    monkeypatch.setenv("PATH", "existing-path")

    env = runtime_dependencies.ffmpeg_environment()

    assert env["PERSONAVOICE_FFMPEG_BIN"] == str(tmp_path)
    assert env["PATH"].split(runtime_dependencies.os.pathsep)[0] == str(tmp_path)
