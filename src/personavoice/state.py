from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def stage(self, name: str) -> dict[str, Any]:
        return self.load().setdefault("stages", {}).get(name, {})

    def is_complete(self, name: str, fingerprint: str) -> bool:
        stage = self.stage(name)
        return stage.get("status") == "complete" and stage.get("fingerprint") == fingerprint

    @contextmanager
    def running(self, name: str, fingerprint: str) -> Iterator[dict[str, Any]]:
        state = self.load()
        stage = state.setdefault("stages", {}).setdefault(name, {})
        stage.update(
            {
                "status": "running",
                "fingerprint": fingerprint,
                "started_at": _now(),
                "finished_at": None,
                "error": None,
            }
        )
        self.save(state)
        try:
            yield stage
        except Exception as exc:
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            stage.update({"status": "failed", "finished_at": _now(), "error": str(exc)})
            self.save(state)
            raise
        else:
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            stage.update({"status": "complete", "finished_at": _now(), "error": None})
            self.save(state)
