from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
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
    rows: list[dict[str, Any]] = []
    for path in media_files(raw_dir):
        rows.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "probe": ffprobe(path),
            }
        )
    return rows


def inventory_fingerprint(raw_dir: Path) -> str:
    """Hash the actual media contents, not mutable filesystem metadata.

    mtime/size fingerprints can miss a same-size replacement whose timestamp was
    preserved by a copy/restore tool. Preparation correctness is more important
    than avoiding a sequential read of the source files at command startup.
    """

    digest = hashlib.sha256()
    for path in media_files(raw_dir):
        digest.update(path.relative_to(raw_dir).as_posix().encode())
        digest.update(sha256_file(path).encode())
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
                "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
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
                "ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}",
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
