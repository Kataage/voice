from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from personavoice.model_assets import (
    ASR_MODEL_REVISION,
    PYANNOTE_MODEL_REVISION,
    SENSE_MODEL_CMVN_SHA256,
    SENSE_MODEL_TOKENIZER_SHA256,
    SENSE_MODEL_WEIGHT_SHA256,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_contract(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_cache_policy() -> str:
    """Return the exact preprocessing/model/environment implementation contract.

    Prepared artifacts are reproducible only when model assets, dependency
    graphs, and the code that interprets their outputs are unchanged. Including
    implementation hashes prevents a future code fix from accidentally reusing
    stale ASR/diarization/Sense/segmentation results when a manual schema bump is
    forgotten.
    """

    repo = _repo_root()
    contract = {
        "schema": 6,
        "asr_revision": ASR_MODEL_REVISION,
        "pyannote_revision": PYANNOTE_MODEL_REVISION,
        "sense_weight_sha256": SENSE_MODEL_WEIGHT_SHA256,
        "sense_cmvn_sha256": SENSE_MODEL_CMVN_SHA256,
        "sense_tokenizer_sha256": SENSE_MODEL_TOKENIZER_SHA256,
        "asr_lock_sha256": _file_contract(repo / "workers" / "asr" / "uv.lock"),
        "diarization_lock_sha256": _file_contract(
            repo / "workers" / "diarization" / "uv.lock"
        ),
        "sense_lock_sha256": _file_contract(repo / "workers" / "sense" / "uv.lock"),
        "pipeline_code_sha256": _file_contract(repo / "src" / "personavoice" / "pipeline.py"),
        "media_code_sha256": _file_contract(repo / "src" / "personavoice" / "media.py"),
        "speaker_code_sha256": _file_contract(repo / "src" / "personavoice" / "speaker.py"),
        "captions_code_sha256": _file_contract(repo / "src" / "personavoice" / "captions.py"),
        "dataset_code_sha256": _file_contract(repo / "src" / "personavoice" / "dataset.py"),
        "asr_worker_code_sha256": _file_contract(repo / "workers" / "asr" / "worker.py"),
        "diarization_worker_code_sha256": _file_contract(
            repo / "workers" / "diarization" / "worker.py"
        ),
        "sense_worker_code_sha256": _file_contract(repo / "workers" / "sense" / "worker.py"),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"6-{hashlib.sha256(encoded).hexdigest()[:20]}"


# Cached prepare artifacts are valid only for this exact preprocessing contract.
PREPARE_CACHE_POLICY_VERSION = _prepare_cache_policy()


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
        if (
            name == "prepare"
            and "usable_tts_utterances" in result
            and int(result["usable_tts_utterances"]) <= 0
        ):
            raise RuntimeError(
                "Preparation found no usable authorized-speaker utterances. Sources where the "
                "authorized speaker was not selected are listed in dataset/skipped_sources.json. "
                "Add/clean identity reference audio, add source recordings containing the target "
                "speaker, or deliberately review prepare.min_identity_similarity."
            )
        state = self.load()
        state.setdefault("stages", {}).setdefault(name, {})["result"] = result
        self.save(state)

    def _invalidate_prepare_derived(self) -> None:
        """Remove only artifacts whose identity depends on prepare semantics.

        The lossless source-audio cache is intentionally retained because it is
        content-addressed by source SHA. ASR/diarization/Sense caches and cut
        clips are not safe to reuse across arbitrary prepare-setting, model,
        dependency, or code changes, so they are rebuilt when the prepare
        fingerprint/policy changes or the caller explicitly forces a rebuild.
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
