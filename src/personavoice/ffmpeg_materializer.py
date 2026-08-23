from __future__ import annotations

import os
import platform
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from personavoice.atomic import atomic_write_json
from personavoice.ffmpeg_contract import (
    MARKER_SCHEMA_VERSION,
    REQUIRED_DLL_PATTERNS,
    REQUIRED_EXECUTABLES,
    WINDOWS_FFMPEG_ARCHIVE_ROOT,
    WINDOWS_FFMPEG_ARCHIVE_SHA256,
    WINDOWS_FFMPEG_URL,
    WINDOWS_FFMPEG_VERSION,
    archive_path,
    pinned_bin_dir,
    runtime_root,
    sha256_file,
    validate_pinned_runtime,
    validate_runtime_root,
)
from personavoice.runtime_dependencies import FfmpegRuntime, _candidate_runtime, require_ffmpeg_runtime

_MAX_ARCHIVE_FILES = 20_000
_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024


def _x64_windows() -> bool:
    return platform.machine().lower() in {"amd64", "x86_64", "x64"}


def _download_archive(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PersonaVoice/FFmpeg-materializer"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
        handle.flush()
        os.fsync(handle.fileno())


def _verified_archive(repo_root: Path) -> Path:
    archive = archive_path(repo_root)
    if archive.is_file():
        try:
            if sha256_file(archive).lower() == WINDOWS_FFMPEG_ARCHIVE_SHA256:
                return archive
        except OSError:
            pass
        archive.unlink(missing_ok=True)

    temp = archive.with_name(f".{archive.name}.{uuid4().hex}.tmp")
    try:
        _download_archive(WINDOWS_FFMPEG_URL, temp)
        actual = sha256_file(temp)
        if actual.lower() != WINDOWS_FFMPEG_ARCHIVE_SHA256:
            raise RuntimeError(
                "Downloaded Windows FFmpeg archive checksum mismatch: "
                f"expected {WINDOWS_FFMPEG_ARCHIVE_SHA256}, got {actual}. "
                "Refusing to extract an unaudited runtime."
            )
        archive.parent.mkdir(parents=True, exist_ok=True)
        temp.replace(archive)
        return archive
    finally:
        temp.unlink(missing_ok=True)


def _safe_bin_member(info: zipfile.ZipInfo) -> PurePosixPath | None:
    raw = info.filename.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe path in FFmpeg archive: {info.filename!r}")
    if any(":" in part for part in path.parts):
        raise RuntimeError(f"Unsafe drive-like path in FFmpeg archive: {info.filename!r}")

    prefix = PurePosixPath(WINDOWS_FFMPEG_ARCHIVE_ROOT, "bin")
    if path.parts[: len(prefix.parts)] != prefix.parts:
        return None
    relative = PurePosixPath(*path.parts[len(prefix.parts) :])
    if not relative.parts or info.is_dir():
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"Unsafe path in FFmpeg archive: {info.filename!r}")

    file_type = (info.external_attr >> 16) & 0o170000
    if file_type == 0o120000:
        raise RuntimeError(f"Symlinks are not allowed in the pinned FFmpeg archive: {info.filename!r}")
    return relative


def _critical_file_hashes(bin_dir: Path) -> dict[str, str]:
    selected: set[Path] = set()
    for name in REQUIRED_EXECUTABLES:
        path = bin_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Pinned FFmpeg archive is missing required executable {name}")
        selected.add(path)
    for pattern in REQUIRED_DLL_PATTERNS:
        matches = sorted(path for path in bin_dir.glob(pattern) if path.is_file())
        if not matches:
            raise RuntimeError(f"Pinned FFmpeg archive is missing required DLL family {pattern}")
        selected.update(matches)
    return {
        path.relative_to(bin_dir).as_posix(): sha256_file(path)
        for path in sorted(selected, key=lambda value: value.as_posix().casefold())
    }


def _publish_directory(temp_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.{uuid4().hex}.backup")
    shutil.rmtree(backup, ignore_errors=True)
    moved_old = False
    try:
        if destination.exists():
            destination.replace(backup)
            moved_old = True
        temp_root.replace(destination)
    except Exception:
        if not destination.exists() and moved_old and backup.exists():
            backup.replace(destination)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def _extract_archive(archive: Path, destination: Path) -> None:
    temp_root = destination.with_name(f".{destination.name}.{uuid4().hex}.extracting")
    shutil.rmtree(temp_root, ignore_errors=True)
    try:
        bin_dir = temp_root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > _MAX_ARCHIVE_FILES:
                raise RuntimeError("Pinned FFmpeg archive contains an unreasonable number of files")
            total_size = sum(max(0, int(info.file_size)) for info in infos)
            if total_size > _MAX_UNCOMPRESSED_BYTES:
                raise RuntimeError("Pinned FFmpeg archive exceeds the audited extraction size limit")

            extracted = 0
            seen: set[str] = set()
            for info in infos:
                relative = _safe_bin_member(info)
                if relative is None:
                    continue
                key = relative.as_posix().casefold()
                if key in seen:
                    raise RuntimeError(f"Duplicate path in FFmpeg archive: {relative.as_posix()!r}")
                seen.add(key)
                target = bin_dir.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                extracted += 1
        if extracted == 0:
            raise RuntimeError("Pinned FFmpeg archive contained no bin/ runtime files")

        files = _critical_file_hashes(bin_dir)
        atomic_write_json(
            temp_root / ".personavoice-ffmpeg.json",
            {
                "schema_version": MARKER_SCHEMA_VERSION,
                "version": WINDOWS_FFMPEG_VERSION,
                "source_url": WINDOWS_FFMPEG_URL,
                "archive_sha256": WINDOWS_FFMPEG_ARCHIVE_SHA256,
                "files": files,
            },
        )
        status = validate_runtime_root(temp_root)
        if not status["ok"]:
            raise RuntimeError(
                "Extracted pinned FFmpeg runtime failed integrity validation: "
                + "; ".join(str(value) for value in status["errors"])
            )
        _publish_directory(temp_root, destination)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def materialize_windows_ffmpeg(repo_root: Path) -> FfmpegRuntime:
    if platform.system() != "Windows":
        return require_ffmpeg_runtime()
    if not _x64_windows():
        raise RuntimeError(
            "PersonaVoice's audited local Windows FFmpeg runtime is currently x86_64-only. "
            "Provide a compatible shared FFmpeg runtime with PERSONAVOICE_FFMPEG_BIN or use "
            "a supported x86_64 Windows host."
        )

    explicit = os.getenv("PERSONAVOICE_FFMPEG_BIN")
    if explicit:
        candidate = _candidate_runtime(Path(explicit).expanduser(), "PERSONAVOICE_FFMPEG_BIN")
        if candidate is None or not candidate.torchcodec_compatible:
            raise RuntimeError(
                "PERSONAVOICE_FFMPEG_BIN is set but does not point to a compatible shared "
                "FFmpeg 4-8 runtime. Fix or unset the explicit override before setup."
            )
        return candidate

    status = validate_pinned_runtime(repo_root)
    if not status["ok"]:
        archive = _verified_archive(repo_root)
        _extract_archive(archive, runtime_root(repo_root))
        status = validate_pinned_runtime(repo_root)
    if not status["ok"]:
        raise RuntimeError(
            "Pinned Windows FFmpeg runtime could not be materialized safely: "
            + "; ".join(str(value) for value in status["errors"])
        )

    candidate = _candidate_runtime(pinned_bin_dir(repo_root), "PersonaVoice:pinned")
    if candidate is None or not candidate.torchcodec_compatible:
        raise RuntimeError(
            "Pinned Windows FFmpeg files passed checksum validation but did not pass the "
            "TorchCodec runtime/version contract."
        )
    return candidate


def ensure_ffmpeg_runtime(repo_root: Path) -> FfmpegRuntime:
    """Ensure setup has FFmpeg while keeping normal runtime network-free."""

    if platform.system() == "Windows":
        return materialize_windows_ffmpeg(repo_root)
    return require_ffmpeg_runtime()
