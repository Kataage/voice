from __future__ import annotations

import json
from pathlib import Path

from personavoice import ffmpeg_contract


def _valid_runtime_tree(repo_root: Path) -> tuple[Path, Path]:
    root = ffmpeg_contract.runtime_root(repo_root)
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    names = (
        "ffmpeg.exe",
        "ffprobe.exe",
        "avutil-60.dll",
        "avcodec-62.dll",
        "avformat-62.dll",
        "avdevice-62.dll",
        "avfilter-11.dll",
        "swresample-6.dll",
        "swscale-9.dll",
    )
    files: dict[str, str] = {}
    for name in names:
        path = bin_dir / name
        path.write_bytes(name.encode("ascii"))
        files[name] = ffmpeg_contract.sha256_file(path)
    marker = {
        "schema_version": ffmpeg_contract.MARKER_SCHEMA_VERSION,
        "version": ffmpeg_contract.WINDOWS_FFMPEG_VERSION,
        "source_url": ffmpeg_contract.WINDOWS_FFMPEG_URL,
        "archive_sha256": ffmpeg_contract.WINDOWS_FFMPEG_ARCHIVE_SHA256,
        "files": files,
    }
    (root / ffmpeg_contract.PINNED_RUNTIME_MARKER).write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    assert ffmpeg_contract.validate_runtime_root(root)["ok"] is True
    return root, bin_dir


def test_pinned_runtime_rejects_runtime_root_indirection(tmp_path: Path, monkeypatch) -> None:
    root, _bin_dir = _valid_runtime_tree(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == root or original(self),
    )

    status = ffmpeg_contract.validate_runtime_root(root)

    assert status["ok"] is False
    assert any("runtime root is a symlink/junction" in str(error) for error in status["errors"])


def test_pinned_runtime_rejects_bin_root_indirection(tmp_path: Path, monkeypatch) -> None:
    root, bin_dir = _valid_runtime_tree(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == bin_dir or original(self),
    )

    status = ffmpeg_contract.validate_runtime_root(root)

    assert status["ok"] is False
    assert any("bin directory is a symlink/junction" in str(error) for error in status["errors"])


def test_marker_paths_reject_windows_drive_and_backslash_forms() -> None:
    assert ffmpeg_contract._safe_relative_file("C:/escape.dll") is None
    assert ffmpeg_contract._safe_relative_file("folder\\escape.dll") is None
    assert ffmpeg_contract._safe_relative_file("../escape.dll") is None
