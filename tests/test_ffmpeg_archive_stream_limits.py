from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from personavoice import ffmpeg_contract, ffmpeg_materializer


def test_ffmpeg_archive_member_rejects_nul_path() -> None:
    info = zipfile.ZipInfo("placeholder")
    info.filename = f"{ffmpeg_contract.WINDOWS_FFMPEG_ARCHIVE_ROOT}/bin/bad\x00name.dll"

    with pytest.raises(RuntimeError, match="NUL bytes"):
        ffmpeg_materializer._safe_bin_member(info)


def test_ffmpeg_extraction_bounds_actual_streamed_bytes(tmp_path: Path, monkeypatch) -> None:
    info = zipfile.ZipInfo(
        f"{ffmpeg_contract.WINDOWS_FFMPEG_ARCHIVE_ROOT}/bin/ffmpeg.exe"
    )
    info.file_size = 1

    class FakeBundle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def infolist(self):
            return [info]

        def open(self, _info):
            return io.BytesIO(b"two-bytes")

    monkeypatch.setattr(ffmpeg_materializer.zipfile, "ZipFile", lambda _path: FakeBundle())
    destination = tmp_path / "runtime"

    with pytest.raises(RuntimeError, match="expanded beyond its declared size"):
        ffmpeg_materializer._extract_archive(tmp_path / "fake.zip", destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".runtime.*.extracting"))
