from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a UTF-8 text file atomically from a sibling temporary file.

    Persona configuration and setup state are recovery-critical. A process kill,
    power loss, or exception while writing them must leave either the previous
    complete file or the new complete file, never a truncated destination.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )
