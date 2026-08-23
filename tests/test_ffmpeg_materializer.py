from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from personavoice import ffmpeg_contract, ffmpeg_materializer, runtime_dependencies


def _fake_shared_archive(
    path: Path,
    *,
    traversal: bool = False,
    duplicate: bool = False,
    symlink: bool = False,
) -> str:
    root = ffmpeg_contract.WINDOWS_FFMPEG_ARCHIVE_ROOT
    files = {
        f"{root}/bin/ffmpeg.exe": b"ffmpeg",
        f"{root}/bin/ffprobe.exe": b"ffprobe",
        f"{root}/bin/avutil-60.dll": b"avutil",
        f"{root}/bin/avcodec-62.dll": b"avcodec",
        f"{root}/bin/avformat-62.dll": b"avformat",
        f"{root}/bin/avdevice-62.dll": b"avdevice",
        f"{root}/bin/avfilter-11.dll": b"avfilter",
        f"{root}/bin/swresample-6.dll": b"swresample",
        f"{root}/bin/swscale-9.dll": b"swscale",
        f"{root}/bin/helper-runtime.dll": b"helper",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
        if traversal:
            bundle.writestr(f"{root}/bin/../../escape.dll", b"escape")
        if duplicate:
            bundle.writestr(f"{root}/bin/FFMPEG.EXE", b"duplicate")
        if symlink:
            info = zipfile.ZipInfo(f"{root}/bin/link.dll")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            bundle.writestr(info, b"target")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _windows_test_runtime(monkeypatch, repo_root: Path, expected_sha: str) -> None:
    monkeypatch.setattr(ffmpeg_materializer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ffmpeg_materializer.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(runtime_dependencies.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_dependencies, "_version_major", lambda _path: 8)
    monkeypatch.setattr(ffmpeg_materializer, "WINDOWS_FFMPEG_ARCHIVE_SHA256", expected_sha)
    monkeypatch.setattr(ffmpeg_contract, "WINDOWS_FFMPEG_ARCHIVE_SHA256", expected_sha)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(repo_root))
    monkeypatch.delenv("PERSONAVOICE_FFMPEG_BIN", raising=False)


def _download_from(source: Path):
    def fake_download(_url: str, destination: Path) -> None:
        shutil.copyfile(source, destination)

    return fake_download


def _materialize_fake_runtime(tmp_path: Path, monkeypatch) -> Path:
    source = tmp_path / "source.zip"
    expected_sha = _fake_shared_archive(source)
    _windows_test_runtime(monkeypatch, tmp_path, expected_sha)
    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", _download_from(source))
    ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)
    return ffmpeg_contract.pinned_bin_dir(tmp_path)


def test_windows_setup_materializes_verified_repo_local_shared_ffmpeg(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "source.zip"
    expected_sha = _fake_shared_archive(source)
    _windows_test_runtime(monkeypatch, tmp_path, expected_sha)
    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", _download_from(source))

    runtime = ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)

    assert runtime.source == "PersonaVoice:pinned"
    assert runtime.version_major == 8
    assert runtime.torchcodec_compatible is True
    status = ffmpeg_contract.validate_pinned_runtime(tmp_path)
    assert status["ok"] is True
    marker = json.loads(
        (ffmpeg_contract.runtime_root(tmp_path) / ".personavoice-ffmpeg.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["schema_version"] == ffmpeg_contract.MARKER_SCHEMA_VERSION
    assert marker["source_url"] == ffmpeg_contract.WINDOWS_FFMPEG_URL
    assert marker["archive_sha256"] == expected_sha
    assert "helper-runtime.dll" in marker["files"]
    assert set(marker["files"]) == {
        path.relative_to(ffmpeg_contract.pinned_bin_dir(tmp_path)).as_posix()
        for path in ffmpeg_contract.pinned_bin_dir(tmp_path).rglob("*")
        if path.is_file()
    }


def test_windows_ffmpeg_archive_checksum_mismatch_is_rejected(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.zip"
    _fake_shared_archive(source)
    _windows_test_runtime(monkeypatch, tmp_path, "0" * 64)
    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", _download_from(source))

    with pytest.raises(RuntimeError, match="archive checksum mismatch"):
        ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)
    assert not ffmpeg_contract.runtime_root(tmp_path).exists()


def test_windows_ffmpeg_download_size_limit_removes_partial_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ffmpeg_materializer, "_MAX_ARCHIVE_DOWNLOAD_BYTES", 8)
    monkeypatch.setattr(ffmpeg_materializer, "_COPY_CHUNK_BYTES", 4)
    monkeypatch.setattr(
        ffmpeg_materializer.urllib.request,
        "urlopen",
        lambda _request, timeout: io.BytesIO(b"123456789"),
    )
    destination = tmp_path / "oversize.zip"

    with pytest.raises(RuntimeError, match="download size limit"):
        ffmpeg_materializer._download_archive("https://example.invalid/archive.zip", destination)
    assert not destination.exists()


def test_windows_ffmpeg_zip_traversal_is_rejected(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.zip"
    expected_sha = _fake_shared_archive(source, traversal=True)
    _windows_test_runtime(monkeypatch, tmp_path, expected_sha)
    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", _download_from(source))

    with pytest.raises(RuntimeError, match="Unsafe path"):
        ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)
    assert not (tmp_path / ".runtime" / "tools" / "ffmpeg" / "escape.dll").exists()


def test_windows_ffmpeg_duplicate_casefolded_path_is_rejected(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.zip"
    expected_sha = _fake_shared_archive(source, duplicate=True)
    _windows_test_runtime(monkeypatch, tmp_path, expected_sha)
    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", _download_from(source))

    with pytest.raises(RuntimeError, match="Duplicate path"):
        ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)


def test_windows_ffmpeg_symlink_is_rejected(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.zip"
    expected_sha = _fake_shared_archive(source, symlink=True)
    _windows_test_runtime(monkeypatch, tmp_path, expected_sha)
    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", _download_from(source))

    with pytest.raises(RuntimeError, match="Symlinks are not allowed"):
        ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)


def test_pinned_runtime_detects_post_setup_critical_file_corruption(tmp_path: Path, monkeypatch):
    bin_dir = _materialize_fake_runtime(tmp_path, monkeypatch)
    (bin_dir / "avcodec-62.dll").write_bytes(b"corrupted")

    status = ffmpeg_contract.validate_pinned_runtime(tmp_path)
    assert status["ok"] is False
    assert any("checksum mismatch" in str(error) for error in status["errors"])


def test_pinned_runtime_detects_noncritical_dependency_corruption(tmp_path: Path, monkeypatch):
    bin_dir = _materialize_fake_runtime(tmp_path, monkeypatch)
    (bin_dir / "helper-runtime.dll").write_bytes(b"corrupted-helper")

    status = ffmpeg_contract.validate_pinned_runtime(tmp_path)
    assert status["ok"] is False
    assert any(
        "helper-runtime.dll" in str(error) and "checksum mismatch" in str(error)
        for error in status["errors"]
    )


def test_pinned_runtime_rejects_post_setup_untracked_file(tmp_path: Path, monkeypatch):
    bin_dir = _materialize_fake_runtime(tmp_path, monkeypatch)
    (bin_dir / "injected.dll").write_bytes(b"injected")

    status = ffmpeg_contract.validate_pinned_runtime(tmp_path)
    assert status["ok"] is False
    assert any("untracked file: injected.dll" in str(error) for error in status["errors"])


def test_pinned_runtime_rejects_filesystem_symlink_or_junction(tmp_path: Path, monkeypatch):
    bin_dir = _materialize_fake_runtime(tmp_path, monkeypatch)
    injected = bin_dir / "injected-link.dll"
    injected.write_bytes(b"placeholder")
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == injected or original(self),
    )

    status = ffmpeg_contract.validate_pinned_runtime(tmp_path)
    assert status["ok"] is False
    assert any("symlink/junction" in str(error) for error in status["errors"])


def test_pinned_runtime_rejects_unsafe_marker_path_without_reading_outside_bin(
    tmp_path: Path,
):
    root = ffmpeg_contract.runtime_root(tmp_path)
    root.mkdir(parents=True)
    outside = root / "escape.dll"
    outside.write_bytes(b"outside")
    marker = {
        "schema_version": ffmpeg_contract.MARKER_SCHEMA_VERSION,
        "version": ffmpeg_contract.WINDOWS_FFMPEG_VERSION,
        "source_url": ffmpeg_contract.WINDOWS_FFMPEG_URL,
        "archive_sha256": ffmpeg_contract.WINDOWS_FFMPEG_ARCHIVE_SHA256,
        "files": {"../escape.dll": hashlib.sha256(b"outside").hexdigest()},
    }
    (root / ".personavoice-ffmpeg.json").write_text(json.dumps(marker), encoding="utf-8")

    status = ffmpeg_contract.validate_pinned_runtime(tmp_path)
    assert status["ok"] is False
    assert any("unsafe path" in str(error) for error in status["errors"])


def test_invalid_explicit_ffmpeg_override_fails_closed_without_download(
    tmp_path: Path,
    monkeypatch,
):
    _windows_test_runtime(monkeypatch, tmp_path, "0" * 64)
    monkeypatch.setenv("PERSONAVOICE_FFMPEG_BIN", str(tmp_path / "missing"))
    called = False

    def should_not_download(_url: str, _destination: Path) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", should_not_download)
    with pytest.raises(RuntimeError, match="PERSONAVOICE_FFMPEG_BIN"):
        ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)
    assert called is False


def test_normal_runtime_prefers_verified_repo_local_ffmpeg(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.zip"
    expected_sha = _fake_shared_archive(source)
    _windows_test_runtime(monkeypatch, tmp_path, expected_sha)
    monkeypatch.setattr(ffmpeg_materializer, "_download_archive", _download_from(source))
    ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)

    runtime = runtime_dependencies.ffmpeg_runtime()
    assert runtime.source == "PersonaVoice:pinned"
    assert runtime.torchcodec_compatible is True
