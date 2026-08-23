from __future__ import annotations

from pathlib import Path

import pytest

from personavoice import (
    ffmpeg_contract,
    ffmpeg_materializer,
    hardware,
    runtime_dependencies,
    setup_env,
)


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

    with pytest.raises(RuntimeError, match="x86_64-only"):
        ffmpeg_materializer.materialize_windows_ffmpeg(tmp_path)
    assert called is False


def test_unsafe_explicit_cuda_backend_fails_before_ffmpeg_materialization(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: "/tool")
    monkeypatch.setattr(
        setup_env,
        "selected_nvidia_gpu",
        lambda: hardware.GpuInfo(
            index=0,
            name="Pascal GPU",
            total_mib=11264,
            free_mib=10000,
            compute_capability="6.1",
            uuid="GPU-pascal",
            pci_bus_id="00000000:01:00.0",
            driver_version="999.1",
        ),
    )
    materialized = False

    def should_not_materialize(_repo_root):
        nonlocal materialized
        materialized = True
        raise AssertionError("FFmpeg materialization must happen after CUDA backend validation")

    monkeypatch.setattr(setup_env, "ensure_ffmpeg_runtime", should_not_materialize)

    with pytest.raises(ValueError, match="--backend cu126"):
        setup_env.install_environments(tmp_path, backend="cu128")
    assert materialized is False


def test_windows_ffmpeg_contract_is_fixed_to_audited_winget_release():
    assert ffmpeg_contract.WINDOWS_FFMPEG_VERSION == "8.1.1"
    assert ffmpeg_contract.WINDOWS_FFMPEG_URL == (
        "https://github.com/GyanD/codexffmpeg/releases/download/8.1.1/"
        "ffmpeg-8.1.1-full_build-shared.zip"
    )
    assert ffmpeg_contract.WINDOWS_FFMPEG_ARCHIVE_SHA256 == (
        "4296b396bdfd5fbc3dfc75ab4c8703354a56963232d65c4182993543df2d2f45"
    )



def test_non_windows_setup_can_reauthorize_changed_ffmpeg(tmp_path, monkeypatch):
    bin_dir = tmp_path / "ffmpeg-bin"
    bin_dir.mkdir()
    ffmpeg = bin_dir / "ffmpeg"
    ffprobe = bin_dir / "ffprobe"
    ffmpeg.write_bytes(b"new-ffmpeg-generation")
    ffprobe.write_bytes(b"new-ffprobe-generation")
    discovered = runtime_dependencies.FfmpegRuntime(
        ffmpeg=str(ffmpeg),
        ffprobe=str(ffprobe),
        bin_dir=str(bin_dir),
        version_major=8,
        shared_libraries=True,
        torchcodec_compatible=True,
        source="PATH",
        error=None,
    )
    monkeypatch.setattr(ffmpeg_materializer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        ffmpeg_materializer,
        "_discover_ffmpeg_runtime",
        lambda: discovered,
    )

    result = ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)
    assert result is discovered
    assert runtime_dependencies.recorded_ffmpeg_provenance(tmp_path) == (
        runtime_dependencies.ffmpeg_provenance(discovered)
    )


def test_windows_bootstrap_never_installs_or_requires_ffmpeg_before_setup():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")
    lowered = text.casefold()
    assert "winget install" not in lowered
    assert "gyan.ffmpeg.shared" not in lowered
    assert "persona doctor" not in lowered
    assert "persona --help" in lowered
    assert "persona setup --backend auto" in lowered
