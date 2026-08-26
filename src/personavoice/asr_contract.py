"""Normalized ASR and alignment representations.

Worker-specific confidence fields stay in ``backend_metadata``.  In
particular, Qwen output is not converted into a fabricated Whisper
``avg_logprob`` or probability score.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from personavoice.lineage import (
    ALIGNMENT_CONTRACT_VERSION,
    ASR_CONTRACT_VERSION,
    resolve_alignment,
    resolve_backend,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _safe_time(value: Any, default: float = 0.0) -> float:
    parsed = _number(value)
    return max(0.0, parsed if parsed is not None else default)


def _word(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    text = value.get("word", value.get("text", ""))
    if not isinstance(text, str) or not text.strip():
        return None
    start = _safe_time(value.get("start"))
    end = max(start, _safe_time(value.get("end"), start))
    probability = _number(value.get("probability"))
    return {
        "start": start,
        "end": end,
        "word": text,
        "probability": probability,
    }


def _segment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ASR segment must be an object")
    start = _safe_time(value.get("start"))
    end = max(start, _safe_time(value.get("end"), start))
    text = value.get("text", "")
    if not isinstance(text, str):
        raise ValueError("ASR segment text must be a string")
    words = []
    for raw_word in value.get("words") or []:
        normalized = _word(raw_word)
        if normalized is not None:
            words.append(normalized)
    return {
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": _number(value.get("avg_logprob")),
        "no_speech_prob": _number(value.get("no_speech_prob")),
        # This is only a worker-provided value.  No Qwen score is inferred from
        # Whisper fields when the backend does not emit one.
        "confidence": _number(value.get("confidence")),
        "words": words,
    }


def transcript_hash(segments: list[dict[str, Any]]) -> str:
    return _hash(
        [
            {
                "start": row["start"],
                "end": row["end"],
                "text": row["text"],
                "words": row["words"],
            }
            for row in segments
        ]
    )


def alignment_hash(units: list[dict[str, Any]]) -> str:
    return _hash(units)


def words_alignment(segments: list[dict[str, Any]]) -> dict[str, Any]:
    units = []
    for segment in segments:
        for word in segment.get("words") or []:
            units.append(
                {
                    "unit": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "confidence": word.get("probability"),
                }
            )
    return {
        "contract_version": ALIGNMENT_CONTRACT_VERSION,
        "units": units,
        "units_kind": "word",
        "hash": alignment_hash(units),
    }


def normalize_asr_result(
    value: dict[str, Any],
    *,
    backend: str,
    source_audio: Path | str,
    analysis_audio: Path | str | None = None,
    alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a worker response while preserving backend evidence."""

    spec = resolve_backend(backend)
    if not isinstance(value, dict):
        raise ValueError("ASR result must be an object")
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("ASR result segments must be a list")
    segments = [_segment(item) for item in raw_segments]
    language = value.get("language") or "und"
    if not isinstance(language, str) or not language:
        raise ValueError("ASR result language must be a non-empty string")
    duration = _number(value.get("duration"))
    if duration is None or duration < 0:
        raise ValueError("ASR result duration must be finite and non-negative")
    normalized_alignment = alignment
    if normalized_alignment is None and spec.key == "whisper-large-v3":
        normalized_alignment = words_alignment(segments)
    if normalized_alignment is not None:
        if not isinstance(normalized_alignment, dict):
            raise ValueError("alignment must be an object")
        units = normalized_alignment.get("units")
        if not isinstance(units, list):
            raise ValueError("alignment units must be a list")
        requested_alignment = normalized_alignment.get("key") or "auto"
        alignment_spec = resolve_alignment(spec.key, str(requested_alignment))
        normalized_alignment = {
            **alignment_spec.as_dict(),
            **normalized_alignment,
            "contract_version": ALIGNMENT_CONTRACT_VERSION,
            "hash": alignment_hash(units),
        }
        if (
            normalized_alignment.get("key") != alignment_spec.key
            or normalized_alignment.get("model_id") != alignment_spec.model_id
            or normalized_alignment.get("revision") != alignment_spec.revision
        ):
            raise ValueError("ASR alignment does not match its versioned backend contract")
        validate_alignment_coupling(
            normalized_alignment,
            asr_model_id=spec.model_id,
            asr_model_revision=spec.revision,
        )
    transcript = transcript_hash(segments)
    provenance = {
        "contract_version": ASR_CONTRACT_VERSION,
        "backend": spec.key,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "source_audio": str(source_audio),
        "analysis_audio": str(analysis_audio or source_audio),
        "transcript_hash": transcript,
        "alignment": (
            {
                "backend": normalized_alignment.get("backend"),
                "model_id": normalized_alignment.get("model_id"),
                "model_revision": normalized_alignment.get("model_revision"),
                "hash": normalized_alignment.get("hash"),
            }
            if normalized_alignment is not None
            else None
        ),
    }
    output = {
        "language": language,
        "language_probability": _number(value.get("language_probability")),
        "duration": duration,
        "segments": segments,
        "backend": spec.key,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "backend_metadata": value.get("backend_metadata") or {},
        "provenance": provenance,
    }
    if normalized_alignment is not None:
        output["alignment"] = normalized_alignment
    return output


def alignment_contract_for_result(
    asr_result: dict[str, Any],
    *,
    asr_backend: str,
    requested: str = "auto",
) -> dict[str, Any]:
    """Build a versioned alignment request without crossing encoder lineages."""

    spec = resolve_backend(asr_backend)
    alignment_spec = resolve_alignment(spec.key, requested)
    return {
        **alignment_spec.as_dict(),
        "asr_backend": spec.key,
        "asr_model_id": spec.model_id,
        "asr_model_revision": spec.revision,
        "transcript_hash": (
            asr_result.get("provenance", {}).get("transcript_hash")
            if isinstance(asr_result.get("provenance"), dict)
            else transcript_hash(asr_result.get("segments") or [])
        ),
    }


def validate_alignment_coupling(
    alignment: dict[str, Any],
    *,
    asr_model_id: str,
    asr_model_revision: str,
) -> None:
    """Reject a domain CTC head attached to any other audio encoder."""

    if alignment.get("key") != "domain-ctc-aligner":
        return
    if (
        alignment.get("coupled_encoder_model_id") != asr_model_id
        or alignment.get("coupled_encoder_revision") != asr_model_revision
    ):
        raise ValueError("Domain CTC alignment head is coupled to a different ASR encoder")
