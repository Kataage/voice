from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Increment this whenever the semantics of cached prepare artifacts change.
# Keeping it here lets us invalidate old caches even when the user-facing
# prepare configuration itself did not change.
PREPARE_CACHE_POLICY_VERSION = 2


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
        temp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)

    def stage(self, name: str) -> dict[str, Any]:
        return self.load().setdefault("stages", {}).get(name, {})

    def is_complete(self, name: str, fingerprint: str) -> bool:
        stage = self.stage(name)
        if name == "prepare" and stage.get("cache_policy_version") != PREPARE_CACHE_POLICY_VERSION:
            return False
        return stage.get("status") == "complete" and stage.get("fingerprint") == fingerprint

    def set_result(self, name: str, result: dict[str, Any]) -> None:
        state = self.load()
        state.setdefault("stages", {}).setdefault(name, {})["result"] = result
        self.save(state)

    def _invalidate_prepare_derived(self) -> None:
        """Remove only artifacts whose identity depends on prepare semantics.

        The lossless source-audio cache is intentionally retained because it is
        content-addressed by source SHA. ASR/diarization/Sense caches and cut
        clips are not safe to reuse across arbitrary prepare-setting or code
        changes, so they are rebuilt when the prepare fingerprint/policy changes
        or the caller explicitly forces a rebuild.
        """

        persona_root = self.path.parent
        for relative in (
            Path("cache/asr"),
            Path("cache/diarization"),
            Path("cache/identity"),
            Path("cache/sense"),
            Path("dataset/clips"),
        ):
            shutil.rmtree(persona_root / relative, ignore_errors=True)

    @contextmanager
    def running(
        self,
        name: str,
        fingerprint: str,
        *,
        force: bool = False,
    ) -> Iterator[dict[str, Any]]:
        state = self.load()
        stage = state.setdefault("stages", {}).setdefault(name, {})

        if name == "prepare":
            old_fingerprint = stage.get("fingerprint")
            old_policy = stage.get("cache_policy_version")
            # Same-fingerprint failed/running stages keep expensive caches for
            # normal resumability. Explicit --force always starts from fresh
            # prepare-derived caches regardless of the prior stage status.
            must_invalidate = (
                force
                or old_policy != PREPARE_CACHE_POLICY_VERSION
                or (old_fingerprint is not None and old_fingerprint != fingerprint)
            )
            if must_invalidate:
                self._invalidate_prepare_derived()

        stage.update(
            {
                "status": "running",
                "fingerprint": fingerprint,
                "started_at": _now(),
                "finished_at": None,
                "error": None,
            }
        )
        if name == "prepare":
            stage["cache_policy_version"] = PREPARE_CACHE_POLICY_VERSION
        self.save(state)
        try:
            yield stage
        except Exception as exc:
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            stage.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "error": str(exc),
                }
            )
            self.save(state)
            raise
        else:
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            stage.update(
                {
                    "status": "complete",
                    "finished_at": _now(),
                    "error": None,
                }
            )
            self.save(state)
