from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

WINDOWS_FFMPEG_VERSION = "8.1.1"
WINDOWS_FFMPEG_ARCHIVE = f"ffmpeg-{WINDOWS_FFMPEG_VERSION}-full_build-shared.zip"
WINDOWS_FFMPEG_ARCHIVE_ROOT = f"ffmpeg-{WINDOWS_FFMPEG_VERSION}-full_build-shared"
WINDOWS_FFMPEG_URL = (
    "https://github.com/GyanD/codexffmpeg/releases/download/"
    f"{WINDOWS_FFMPEG_VERSION}/{WINDOWS_FFMPEG_ARCHIVE}"
)
# Independently audited against the Microsoft winget-pkgs manifest for
# Gyan.FFmpeg.Shared 8.1.1.
WINDOWS_FFMPEG_ARCHIVE_SHA256 = (
    "4296b396bdfd5fbc3dfc75ab4c8703354a56963232d65c4182993543df2d2f45"
)
PINNED_RUNTIME_MARKER = ".personavoice-ffmpeg.json"
MARKER_SCHEMA_VERSION = 2
REQUIRED_EXECUTABLES = ("ffmpeg.exe", "ffprobe.exe")
REQUIRED_DLL_PATTERNS = (
    "avutil-*.dll",
    "avcodec-*.dll",
    "avformat-*.dll",
    "avdevice-*.dll",
    "avfilter-*.dll",
    "swresample-*.dll",
    "swscale-*.dll",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_root(repo_root: Path) -> Path:
    return repo_root / ".runtime" / "tools" / "ffmpeg" / WINDOWS_FFMPEG_VERSION


def pinned_bin_dir(repo_root: Path) -> Path:
    return runtime_root(repo_root) / "bin"


def archive_path(repo_root: Path) -> Path:
    return repo_root / ".runtime" / "downloads" / WINDOWS_FFMPEG_ARCHIVE


def marker_path(root: Path) -> Path:
    return root / PINNED_RUNTIME_MARKER


def ffmpeg_contract() -> dict[str, object]:
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "version": WINDOWS_FFMPEG_VERSION,
        "url": WINDOWS_FFMPEG_URL,
        "archive": WINDOWS_FFMPEG_ARCHIVE,
        "archive_sha256": WINDOWS_FFMPEG_ARCHIVE_SHA256,
        "archive_root": WINDOWS_FFMPEG_ARCHIVE_ROOT,
        "required_executables": list(REQUIRED_EXECUTABLES),
        "required_dll_patterns": list(REQUIRED_DLL_PATTERNS),
    }


def _marker(root: Path) -> dict:
    path = marker_path(root)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_relative_file(value: str) -> PurePosixPath | None:
    if not value or "\\" in value or "\x00" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    if any(":" in part for part in path.parts):
        return None
    return path


def validate_runtime_root(root: Path) -> dict[str, object]:
    """Validate a materialized pinned runtime without trusting its marker paths."""

    errors: list[str] = []
    value = _marker(root)
    bin_dir = root / "bin"
    if value.get("schema_version") != MARKER_SCHEMA_VERSION:
        errors.append("pinned FFmpeg marker has the wrong schema version")
    if value.get("version") != WINDOWS_FFMPEG_VERSION:
        errors.append("pinned FFmpeg marker has the wrong version")
    if value.get("source_url") != WINDOWS_FFMPEG_URL:
        errors.append("pinned FFmpeg marker has the wrong source URL")
    if value.get("archive_sha256") != WINDOWS_FFMPEG_ARCHIVE_SHA256:
        errors.append("pinned FFmpeg marker has the wrong archive SHA256")

    raw_files = value.get("files")
    files = raw_files if isinstance(raw_files, dict) else {}
    safe_files: dict[str, str] = {}
    if not files:
        errors.append("pinned FFmpeg marker has no extracted-file hashes")

    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append("pinned FFmpeg marker contains an invalid file hash entry")
            continue
        safe = _safe_relative_file(relative)
        if safe is None:
            errors.append(f"pinned FFmpeg marker contains an unsafe path: {relative!r}")
            continue
        if not _SHA256_RE.fullmatch(expected):
            errors.append(f"pinned FFmpeg marker contains an invalid SHA256 for {relative!r}")
            continue
        normalized = safe.as_posix()
        if normalized in safe_files:
            errors.append(f"pinned FFmpeg marker contains a duplicate path: {normalized}")
            continue
        safe_files[normalized] = expected.lower()

    for name in REQUIRED_EXECUTABLES:
        if name not in safe_files:
            errors.append(f"pinned FFmpeg marker is missing {name}")
    for pattern in REQUIRED_DLL_PATTERNS:
        matches = [
            relative
            for relative in safe_files
            if "/" not in relative and PurePosixPath(relative).match(pattern)
        ]
        if not matches:
            errors.append(f"pinned FFmpeg marker is missing DLL family {pattern}")

    for relative, expected in safe_files.items():
        safe = PurePosixPath(relative)
        path = bin_dir.joinpath(*safe.parts)
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"pinned FFmpeg runtime file is missing/empty: {relative}")
                continue
            actual = sha256_file(path)
        except OSError as exc:
            errors.append(f"pinned FFmpeg runtime file could not be read: {relative}: {exc}")
            continue
        if actual.lower() != expected:
            errors.append(
                f"pinned FFmpeg runtime checksum mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )

    return {
        "ok": not errors,
        "root": str(root),
        "bin_dir": str(bin_dir),
        "version": WINDOWS_FFMPEG_VERSION,
        "archive_sha256": WINDOWS_FFMPEG_ARCHIVE_SHA256,
        "errors": errors,
    }


def validate_pinned_runtime(repo_root: Path) -> dict[str, object]:
    return validate_runtime_root(runtime_root(repo_root))
