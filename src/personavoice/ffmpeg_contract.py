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
MARKER_SCHEMA_VERSION = 3
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


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except OSError:
        return True


def _disk_inventory(bin_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Return every regular runtime file while rejecting filesystem indirection."""

    errors: list[str] = []
    files: dict[str, Path] = {}
    seen_casefold: dict[str, str] = {}
    try:
        root = bin_dir.resolve(strict=True)
    except OSError as exc:
        return {}, [f"pinned FFmpeg bin directory is missing/unreadable: {exc}"]
    if not root.is_dir():
        return {}, ["pinned FFmpeg bin path is not a directory"]

    stack = [bin_dir]
    while stack:
        directory = stack.pop()
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            errors.append(f"pinned FFmpeg runtime directory could not be read: {directory}: {exc}")
            continue
        for path in entries:
            relative = path.relative_to(bin_dir).as_posix()
            try:
                if path.is_symlink() or _is_junction(path):
                    errors.append(
                        f"pinned FFmpeg runtime contains a symlink/junction: {relative}"
                    )
                    continue
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    errors.append(
                        f"pinned FFmpeg runtime path escapes its bin directory: {relative}"
                    )
                    continue
                if path.is_dir():
                    stack.append(path)
                    continue
                if not path.is_file():
                    errors.append(
                        f"pinned FFmpeg runtime contains a non-regular file: {relative}"
                    )
                    continue
                if path.stat().st_size <= 0:
                    errors.append(f"pinned FFmpeg runtime file is empty: {relative}")
                    continue
            except OSError as exc:
                errors.append(
                    f"pinned FFmpeg runtime file could not be inspected: {relative}: {exc}"
                )
                continue

            key = relative.casefold()
            previous = seen_casefold.get(key)
            if previous is not None:
                errors.append(
                    "pinned FFmpeg runtime contains case-insensitive duplicate paths: "
                    f"{previous!r}, {relative!r}"
                )
                continue
            seen_casefold[key] = relative
            files[relative] = path
    return files, errors


def validate_runtime_root(root: Path) -> dict[str, object]:
    """Validate the exact materialized runtime without trusting marker paths."""

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
    marker_casefold: dict[str, str] = {}
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
        key = normalized.casefold()
        previous = marker_casefold.get(key)
        if previous is not None:
            errors.append(
                "pinned FFmpeg marker contains a case-insensitive duplicate path: "
                f"{previous!r}, {normalized!r}"
            )
            continue
        marker_casefold[key] = normalized
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

    disk_files, disk_errors = _disk_inventory(bin_dir)
    errors.extend(disk_errors)
    disk_by_casefold = {relative.casefold(): relative for relative in disk_files}
    marker_keys = set(marker_casefold)
    disk_keys = set(disk_by_casefold)
    for key in sorted(marker_keys - disk_keys):
        errors.append(f"pinned FFmpeg runtime file is missing: {marker_casefold[key]}")
    for key in sorted(disk_keys - marker_keys):
        errors.append(
            f"pinned FFmpeg runtime contains an untracked file: {disk_by_casefold[key]}"
        )

    for key in sorted(marker_keys & disk_keys):
        marker_relative = marker_casefold[key]
        disk_relative = disk_by_casefold[key]
        expected = safe_files[marker_relative]
        path = disk_files[disk_relative]
        try:
            actual = sha256_file(path)
        except OSError as exc:
            errors.append(
                f"pinned FFmpeg runtime file could not be read: {disk_relative}: {exc}"
            )
            continue
        if actual.lower() != expected:
            errors.append(
                f"pinned FFmpeg runtime checksum mismatch for {disk_relative}: "
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
