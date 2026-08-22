from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from personavoice.runtime_dependencies import command as ffmpeg_command

MEDIA_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffmpeg_command("ffprobe"),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(result.stdout)


def media_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != ".gitkeep" and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def inventory(raw_dir: Path) -> list[dict[str, Any]]:
    """Return one canonical source row per unique media content hash.

    Copying the same recording into `raw/` under multiple filenames must not
    duplicate the conversation in training or create colliding utterance IDs.
    Duplicate paths are retained as provenance on the canonical row. The current
    pipeline uses the first 16 SHA256 hex characters as its compact source ID, so
    a different full digest with the same prefix is rejected instead of silently
    colliding in downstream utterance IDs.
    """

    rows: list[dict[str, Any]] = []
    by_sha: dict[str, dict[str, Any]] = {}
    by_source_prefix: dict[str, str] = {}
    for path in media_files(raw_dir):
        relative = path.relative_to(raw_dir).as_posix()
        digest = sha256_file(path)
        existing = by_sha.get(digest)
        if existing is not None:
            existing["duplicate_paths"].append(relative)
            continue
        prefix = digest[:16]
        previous_digest = by_source_prefix.get(prefix)
        if previous_digest is not None and previous_digest != digest:
            raise RuntimeError(
                "Two different raw media files share the same truncated source ID "
                f"{prefix}. Remove one file or update PersonaVoice to a wider source-ID scheme."
            )
        row = {
            "path": relative,
            "duplicate_paths": [],
            "absolute_path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "probe": ffprobe(path),
        }
        by_sha[digest] = row
        by_source_prefix[prefix] = digest
        rows.append(row)
    return rows


def inventory_fingerprint(raw_dir: Path) -> str:
    """Hash media contents, logical paths, and the materialization location.

    The exported canonical datasets intentionally contain absolute local audio
    paths for downstream tools. Moving a persona/repository must therefore make
    a completed prepare stage stale so those paths are rebuilt at the new root.
    Including every logical path also means adding/removing a duplicate source
    is observable even though `inventory` suppresses duplicate training input.
    """

    digest = hashlib.sha256()
    digest.update(str(raw_dir.resolve()).encode("utf-8"))
    for path in media_files(raw_dir):
        digest.update(path.relative_to(raw_dir).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _atomic_ffmpeg_output(destination: Path) -> tuple[Path, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}")
    temp.unlink(missing_ok=True)
    return temp, destination


def extract_lossless_audio(source: Path, destination: Path, *, sample_rate: int = 48000) -> None:
    temp, final = _atomic_ffmpeg_output(destination)
    try:
        subprocess.run(
            [
                ffmpeg_command("ffmpeg"),
                "-nostdin", "-y", "-v", "error", "-i", str(source),
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-c:a", "flac", "-compression_level", "8", str(temp),
            ],
            check=True,
        )
        if not temp.is_file() or temp.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg completed without a valid audio file: {temp}")
        temp.replace(final)
    finally:
        temp.unlink(missing_ok=True)


def cut_audio(source: Path, destination: Path, start: float, end: float) -> None:
    duration = max(0.01, end - start)
    temp, final = _atomic_ffmpeg_output(destination)
    try:
        subprocess.run(
            [
                ffmpeg_command("ffmpeg"),
                "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}",
                "-i", str(source), "-t", f"{duration:.3f}", "-ac", "1", "-ar", "48000",
                "-c:a", "flac", "-compression_level", "8", str(temp),
            ],
            check=True,
        )
        if not temp.is_file() or temp.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg completed without a valid audio clip: {temp}")
        temp.replace(final)
    finally:
        temp.unlink(missing_ok=True)
