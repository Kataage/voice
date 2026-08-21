from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

ResultValidator = Callable[[Any], bool]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _timed_row(value: Any, *, require_speaker: bool) -> bool:
    if not isinstance(value, dict):
        return False
    start = value.get("start")
    end = value.get("end")
    if not _finite_number(start) or not _finite_number(end):
        return False
    if float(start) < 0 or float(end) < float(start):
        return False
    return not (require_speaker and not isinstance(value.get("speaker"), str))


def valid_embedding_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    embedding = value.get("embedding")
    return isinstance(embedding, list) and bool(embedding) and all(
        _finite_number(item) for item in embedding
    )


def valid_asr_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("language"), str) or not value["language"]:
        return False
    duration = value.get("duration")
    if not _finite_number(duration) or float(duration) < 0:
        return False
    language_probability = value.get("language_probability")
    if language_probability is not None and not _finite_number(language_probability):
        return False
    segments = value.get("segments")
    if not isinstance(segments, list):
        return False
    for segment in segments:
        if not _timed_row(segment, require_speaker=False):
            return False
        if not isinstance(segment.get("text"), str):
            return False
        avg_logprob = segment.get("avg_logprob")
        if avg_logprob is not None and not _finite_number(avg_logprob):
            return False
        no_speech_prob = segment.get("no_speech_prob")
        if no_speech_prob is not None and not _finite_number(no_speech_prob):
            return False
        words = segment.get("words")
        if not isinstance(words, list):
            return False
        for word in words:
            if not _timed_row(word, require_speaker=False):
                return False
            if not isinstance(word.get("word"), str):
                return False
            probability = word.get("probability")
            if probability is not None and not _finite_number(probability):
                return False
    return True


def valid_diarization_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    turns = value.get("turns")
    exclusive_turns = value.get("exclusive_turns")
    embeddings = value.get("speaker_embeddings")
    if not isinstance(turns, list) or not isinstance(exclusive_turns, list):
        return False
    if not all(_timed_row(row, require_speaker=True) for row in turns):
        return False
    if not all(_timed_row(row, require_speaker=True) for row in exclusive_turns):
        return False
    if not isinstance(embeddings, dict):
        return False
    for label, embedding in embeddings.items():
        if not isinstance(label, str):
            return False
        if not isinstance(embedding, list) or not embedding:
            return False
        if not all(_finite_number(item) for item in embedding):
            return False
    return True


def valid_sense_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("raw"), str):
        return False
    if not isinstance(value.get("emotion"), str):
        return False
    events = value.get("events")
    tags = value.get("tags")
    return (
        isinstance(events, list)
        and all(isinstance(item, str) for item in events)
        and isinstance(tags, list)
        and all(isinstance(item, str) for item in tags)
    )


def valid_lfm_infer_result(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("text"), str)


def _valid_batch_rows(rows: Any, validator: ResultValidator) -> bool:
    if not isinstance(rows, list):
        return False
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("ok"), bool):
            return False
        raw_id = row.get("id")
        if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool):
            return False
        item_id = str(raw_id)
        if not item_id or item_id in seen:
            return False
        seen.add(item_id)
        if row["ok"]:
            if "result" not in row or not validator(row["result"]):
                return False
        elif not isinstance(row.get("error"), str) or not row["error"]:
            return False
    return True


def validate_worker_response(worker_name: str, command: str, value: Any) -> None:
    """Reject structurally invalid worker output before it can enter a prepare cache."""

    valid = True
    if worker_name == "asr" and command == "transcribe":
        valid = valid_asr_result(value)
    elif worker_name == "asr" and command == "batch_transcribe":
        valid = isinstance(value, dict) and _valid_batch_rows(
            value.get("results"), valid_asr_result
        )
    elif worker_name == "diarization" and command == "diarize":
        valid = valid_diarization_result(value)
    elif worker_name == "diarization" and command == "embed":
        valid = valid_embedding_result(value)
    elif worker_name == "diarization" and command == "batch":
        valid = (
            isinstance(value, dict)
            and _valid_batch_rows(value.get("embeddings"), valid_embedding_result)
            and _valid_batch_rows(value.get("diarizations"), valid_diarization_result)
        )
    elif worker_name == "sense" and command == "analyze":
        valid = valid_sense_result(value)
    elif worker_name == "sense" and command == "batch_analyze":
        valid = isinstance(value, dict) and _valid_batch_rows(
            value.get("results"), valid_sense_result
        )
    elif worker_name == "lfm" and command == "infer":
        valid = valid_lfm_infer_result(value)
    elif worker_name in {"asr", "diarization", "sense", "lfm", "seed_vc"}:
        valid = isinstance(value, dict)

    if not valid:
        raise RuntimeError(
            f"{worker_name} worker returned an invalid response schema for {command!r}"
        )


PREPARE_CACHE_VALIDATORS: dict[str, ResultValidator] = {
    "asr": valid_asr_result,
    "diarization": valid_diarization_result,
    "identity": valid_embedding_result,
    "sense": valid_sense_result,
}


def purge_invalid_prepare_caches(persona_root: Path) -> list[str]:
    """Delete parseable prepare caches that fail semantic worker contracts.

    Syntax-corrupt/truncated JSON remains the responsibility of the pipeline's
    per-cache reader, which removes it on access. This preserves expensive
    same-fingerprint resume state until a cache is actually needed while still
    preventing parseable-but-logically-invalid values from being reused.
    """

    removed: list[str] = []
    cache_root = persona_root / "cache"
    for directory, validator in PREPARE_CACHE_VALIDATORS.items():
        target = cache_root / directory
        if not target.is_dir():
            continue
        for path in target.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            try:
                valid = validator(value)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if valid:
                continue
            path.unlink(missing_ok=True)
            removed.append(str(path))
    return removed
