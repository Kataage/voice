from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MEDIA_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".mp4", ".mkv", ".mov", ".webm"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
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
    )
    return json.loads(result.stdout)


def inventory(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS):
        rows.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "probe": ffprobe(path),
            }
        )
    return rows


def inventory_fingerprint(raw_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS):
        stat = path.stat()
        digest.update(path.relative_to(raw_dir).as_posix().encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()
