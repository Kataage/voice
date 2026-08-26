from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from personavoice.lfm_contract import (
    LFM_CONTRACT_FINGERPRINT,
    LFM_CONTRACT_SCHEMA_VERSION,
    build_lfm_system_prompt,
    normalize_voice_fields,
)
from personavoice.profile import PROFILE_FILENAME, CoreProfile, load_core_profile

SCHEMA_VERSION = 1
LFM_QUALITY_REPORT_SCHEMA = 1
IRODORI_QUALITY_REPORT_SCHEMA = 1


def load_lfm_tokenizer(model_dir: Path):
    """Load the pinned LFM tokenizer for exact chat-template accounting."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The LFM tokenizer contract requires the pinned Transformers runtime; run `uv sync --locked`."
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"The pinned LFM tokenizer is unavailable at {model_dir}; run `persona setup` first."
        ) from exc


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def initialize(path: Path) -> None:
    with connect(path) as con:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS utterances (
              id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              start REAL NOT NULL,
              end REAL NOT NULL,
              speaker TEXT,
              target INTEGER NOT NULL,
              speaker_similarity REAL,
              speaker_coverage REAL,
              overlap_ratio REAL NOT NULL,
              text TEXT NOT NULL,
              text_annotated TEXT NOT NULL,
              emotion TEXT,
              events_json TEXT NOT NULL,
              caption TEXT NOT NULL,
              audio_path TEXT,
              quality REAL NOT NULL,
              raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_utterances_source_start ON utterances(source_id, start);
            CREATE INDEX IF NOT EXISTS idx_utterances_target ON utterances(target, quality DESC);
            """
        )
        con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _source_inventory(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return []
    return value


def _publish_skipped_sources(master_db: Path, rows: list[dict[str, Any]]) -> None:
    """Record every source recording that yielded no authorized target segment.

    The canonical source inventory is included so recordings that yield zero
    utterance rows (for example silence or an empty ASR/diarization result) are
    still visible. The pipeline later enriches entries rejected by speaker
    identity with the measured similarity and configured threshold.
    """

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)

    source_paths: dict[str, Any] = {}
    for source in _source_inventory(master_db.parent / "source_inventory.json"):
        digest = source.get("sha256")
        if not isinstance(digest, str) or len(digest) < 16:
            continue
        source_paths[digest[:16]] = source.get("path")

    source_ids = sorted(set(by_source) | set(source_paths))
    skipped = []
    for source_id in source_ids:
        source_rows = by_source.get(source_id, [])
        if any(bool(row.get("target")) for row in source_rows):
            continue
        speakers = sorted(
            {str(row.get("speaker")) for row in source_rows if row.get("speaker") is not None}
        )
        skipped.append(
            {
                "source_id": source_id,
                "source_path": (
                    source_rows[0].get("source_path")
                    if source_rows
                    else source_paths.get(source_id)
                ),
                "reason": (
                    "authorized_speaker_not_selected"
                    if source_rows
                    else "no_utterances"
                ),
                "detected_speakers": speakers,
                "utterances": len(source_rows),
            }
        )
    _write_json_atomic(master_db.parent / "skipped_sources.json", skipped)


def replace_utterances(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    initialize(path)
    with connect(path) as con:
        con.execute("DELETE FROM utterances")
        for row in row_list:
            con.execute(
                """
                INSERT INTO utterances(
                  id, source_id, start, end, speaker, target, speaker_similarity,
                  speaker_coverage, overlap_ratio, text, text_annotated, emotion,
                  events_json, caption, audio_path, quality, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["source_id"],
                    row["start"],
                    row["end"],
                    row.get("speaker"),
                    1 if row.get("target") else 0,
                    row.get("speaker_similarity"),
                    row.get("speaker_coverage"),
                    row.get("overlap_ratio", 0.0),
                    row.get("text", ""),
                    row.get("text_annotated", row.get("text", "")),
                    row.get("emotion"),
                    json.dumps(row.get("events", []), ensure_ascii=False),
                    row.get("caption", ""),
                    row.get("audio_path"),
                    row.get("quality", 0.0),
                    json.dumps(row, ensure_ascii=False),
                ),
            )
    _publish_skipped_sources(path, row_list)


def load_utterances(path: Path, *, target_only: bool = False) -> list[dict[str, Any]]:
    query = "SELECT raw_json FROM utterances"
    if target_only:
        query += " WHERE target = 1"
    query += " ORDER BY source_id, start"
    with connect(path) as con:
        return [json.loads(row[0]) for row in con.execute(query)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Publish complete JSONL exports atomically.

    A killed prepare process must not leave a half-written training dataset at
    the canonical output path. The temporary file lives beside the destination
    so os.replace/Path.replace remains atomic on the same filesystem.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    count = 0
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
        return count
    finally:
        temp.unlink(missing_ok=True)


def export_irodori(
    master_db: Path,
    output: Path,
    persona_name: str,
    *,
    report_path: Path | None = None,
    lineage_metadata: dict[str, Any] | None = None,
    require_provenance: bool = False,
) -> int:
    return _export_irodori(
        master_db,
        output,
        persona_name,
        report_path=report_path,
        lineage_metadata=lineage_metadata,
        require_provenance=require_provenance,
    )


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _row_provenance(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "asr_backend": item.get("asr_backend"),
        "asr_model_id": item.get("asr_model_id"),
        "asr_model_revision": item.get("asr_model_revision"),
        "asr_language_probability": item.get("asr_language_probability"),
        "asr_segment_confidence": item.get("asr_segment_confidence"),
        "transcript_hash": item.get("transcript_hash"),
        "alignment_backend": item.get("alignment_backend"),
        "alignment_model_id": item.get("alignment_model_id"),
        "alignment_model_revision": item.get("alignment_model_revision"),
        "alignment_hash": item.get("alignment_hash"),
        "boundary_evidence": item.get("boundary_evidence"),
        "canonical_audio_path": item.get("canonical_audio_path"),
        "analysis_audio_path": item.get("analysis_audio_path"),
        "separation": item.get("separation"),
    }


def _bounded_number(value: Any, *, lower: float, upper: float) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and lower <= parsed <= upper else None


def _irodori_row_reason(
    item: dict[str, Any],
    *,
    require_provenance: bool,
) -> tuple[str | None, str | None]:
    if not item.get("target"):
        return "not_target_speaker", None
    if item.get("excluded_reason"):
        return str(item["excluded_reason"]), None
    start = _finite(item.get("start"))
    end = _finite(item.get("end"))
    duration = None if start is None or end is None else end - start
    if duration is None or duration <= 0:
        return "invalid_duration_or_boundary", "invalid_boundary"
    if not item.get("audio_path"):
        return "missing_audio", "missing_audio"
    audio = Path(str(item["audio_path"]))
    if not audio.is_file() or audio.stat().st_size <= 0:
        return "missing_audio", "missing_audio"
    text = str(item.get("text_annotated") or "").strip()
    events = item.get("events") or []
    if not text and not events:
        return "empty_text_without_event", "empty_text"
    if "\ufffd" in text:
        return "replacement_character", "replacement_character"
    overlap = _bounded_number(item.get("overlap_ratio"), lower=0.0, upper=1.0)
    if overlap is None or overlap > 0.35:
        return "overlap_or_missing_signal", "overlap"
    coverage = _bounded_number(item.get("speaker_coverage"), lower=0.0, upper=1.0)
    if coverage is None or coverage < 0.5:
        return "low_speaker_coverage", "low_speaker_coverage"
    quality = _bounded_number(item.get("quality"), lower=0.0, upper=1.0)
    if quality is None or quality < 0.45:
        return "quality_below_threshold", "low_quality"
    if require_provenance:
        required = (
            "asr_backend",
            "asr_model_revision",
            "transcript_hash",
            "alignment_backend",
            "alignment_model_revision",
            "alignment_hash",
            "boundary_evidence",
        )
        if any(item.get(key) in (None, "", {}) for key in required):
            return "missing_transcript_alignment_provenance", "missing_provenance"
        similarity = _bounded_number(item.get("speaker_similarity"), lower=-1.0, upper=1.0)
        if similarity is None:
            return "missing_target_speaker_evidence", "speaker_evidence"
        if text and duration > 0 and len(text) / duration > 80.0:
            return "fragmented_text_audio_pair", "fragmented_text"
    return None, None


def _export_irodori(
    master_db: Path,
    output: Path,
    persona_name: str,
    *,
    report_path: Path | None,
    lineage_metadata: dict[str, Any] | None,
    require_provenance: bool,
) -> int:
    rows = []
    reason_counts: Counter[str] = Counter()
    pathology_counts: Counter[str] = Counter()
    candidates = load_utterances(master_db, target_only=False)
    for item in candidates:
        reason, pathology = _irodori_row_reason(item, require_provenance=require_provenance)
        if reason is not None:
            reason_counts[reason] += 1
            if pathology:
                pathology_counts[pathology] += 1
            continue
        text = str(item.get("text_annotated") or "")
        rows.append(
            {
                "audio": str(Path(item["audio_path"]).resolve()),
                "text": text,
                "text_hash": _text_hash(text),
                "caption": item.get("caption") or "自然に話している。",
                "speaker": persona_name,
                "utterance_id": item["id"],
                "duration": round(float(item["end"]) - float(item["start"]), 3),
                "overlap_ratio": item.get("overlap_ratio"),
                "speaker_coverage": item.get("speaker_coverage"),
                "speaker_similarity": item.get("speaker_similarity"),
                "provenance": _row_provenance(item),
                "boundary_evidence": item.get("boundary_evidence"),
            }
        )
    count = write_jsonl(output, rows)
    if report_path is not None:
        _write_json_atomic(
            report_path,
            {
                "schema_version": IRODORI_QUALITY_REPORT_SCHEMA,
                "lineage": lineage_metadata,
                "candidate_count": len(candidates),
                "accepted_count": count,
                "rejected_count": sum(reason_counts.values()),
                "rejection_reasons": dict(sorted(reason_counts.items())),
                "pathology_counters": dict(sorted(pathology_counts.items())),
                "text_hashes": sorted(row["text_hash"] for row in rows),
                "candidate_text_hashes": sorted(
                    _text_hash(str(item.get("text_annotated") or "")) for item in candidates
                ),
                "quality_gate": {
                    "passed": count > 0,
                    "reason": None if count > 0 else "no_accepted_target_pairs",
                },
            },
        )
    return count


def export_seed_vc(master_db: Path, output_dir: Path) -> int:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for existing in audio_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    manifest = []
    for item in load_utterances(master_db, target_only=True):
        if not item.get("audio_path") or item.get("quality", 0.0) < 0.55:
            continue
        source = Path(item["audio_path"])
        destination = audio_dir / f"{item['id']}{source.suffix}"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        manifest.append({"audio": str(destination.resolve()), "speaker": "target"})
    return write_jsonl(output_dir / "manifest.jsonl", manifest)


def _conversation_blocks(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive segments from the same conversational side.

    Prepare may split one long turn into several clips. Treating each target
    clip as a new assistant response teaches the persona model to answer its own
    immediately preceding speech. Collapsing same-side segments reconstructs a
    cleaner dialogue-turn view while retaining the original acoustic rows for
    TTS training.
    """

    blocks: list[dict[str, Any]] = []
    for row in source_rows:
        text = str(row.get("text") or "").strip()
        events = list(row.get("events") or [])
        # A supported event-only target turn is a valid conversational answer;
        # dropping it makes the model learn that laughter/breaths never occur.
        if not text and not events:
            continue
        target = bool(row.get("target"))
        speaker = str(row.get("speaker") or "")
        same_side = bool(
            blocks
            and blocks[-1]["target"] == target
            and (target or blocks[-1]["speaker"] == speaker)
        )
        if same_side:
            block = blocks[-1]
            block["text"] += text
            block["end"] = row.get("end", block["end"])
            block["rows"].append(row)
            block["events"] = sorted(set(block.get("events", [])) | set(events))
            if target and float(row.get("quality", 0.0)) > float(block["voice_row"].get("quality", 0.0)):
                block["voice_row"] = row
            continue
        blocks.append(
            {
                "target": target,
                "speaker": speaker,
                "text": text,
                "start": row.get("start"),
                "end": row.get("end"),
                "rows": [row],
                "voice_row": row,
                "events": events,
            }
        )
    return blocks


def export_lfm(
    master_db: Path,
    output: Path,
    persona_name: str,
    *,
    profile: CoreProfile | None = None,
    report_path: Path | None = None,
    lineage_metadata: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
    max_tokens: int = 2048,
) -> int:
    """Export conversational prompt/completion examples for persona SFT.

    TRL applies the model chat template to conversational prompt/completion data
    and, with completion-only loss, trains only on the authorized persona reply.
    This avoids teaching the model to imitate system instructions or the other
    speaker while preserving those turns as conditioning context.
    """

    if isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    all_rows = load_utterances(master_db)
    examples: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    pathology_counts: Counter[str] = Counter()
    text_lengths: list[int] = []
    token_lengths: list[int] = []
    candidate_text_lengths: list[int] = []
    candidate_token_lengths: list[int] = []
    token_sources: list[str] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_source.setdefault(row["source_id"], []).append(row)
    profile = profile or load_core_profile(
        (
            master_db.parent.parent / PROFILE_FILENAME
            if (master_db.parent.parent / PROFILE_FILENAME).is_file()
            else master_db.parent.parents[3] / PROFILE_FILENAME
        ),
        persona_name=persona_name,
    )
    system = build_lfm_system_prompt(profile)
    candidate_count = 0
    for source_rows in by_source.values():
        blocks = _conversation_blocks(source_rows)
        for index, block in enumerate(blocks):
            if not block["target"]:
                continue
            candidate_count += 1
            if index == 0:
                reason_counts["no_context_before_target"] += 1
                pathology_counts["missing_context"] += 1
                continue
            if blocks[index - 1]["target"]:
                reason_counts["consecutive_target_without_context"] += 1
                pathology_counts["missing_context"] += 1
                continue
            context = blocks[max(0, index - 4) : index]
            lines = []
            for context_block in context:
                speaker = persona_name if context_block["target"] else "相手"
                context_text = context_block["text"] or (
                    "（非言語: " + ",".join(context_block.get("events") or []) + "）"
                )
                lines.append(f"{speaker}: {context_text}")
            if not lines:
                continue
            user = "直前の会話:\n" + "\n".join(lines) + "\nこの続きとして自然に返答してください。"
            voice_row = block["voice_row"]
            strict_lineage = lineage_metadata is not None
            duration = sum(
                max(0.0, float(row.get("end", 0.0)) - float(row.get("start", 0.0)))
                for row in block["rows"]
            )
            block_text = str(block["text"] or "").strip()
            events = list(block.get("events") or voice_row.get("events") or [])
            target_evidence = {
                "target": bool(voice_row.get("target")),
                "speaker_similarity": voice_row.get("speaker_similarity"),
                "speaker_coverage": voice_row.get("speaker_coverage"),
            }
            row_overlaps = [
                _bounded_number(row.get("overlap_ratio"), lower=0.0, upper=1.0)
                for row in block["rows"]
            ]
            row_coverages = [
                _bounded_number(row.get("speaker_coverage"), lower=0.0, upper=1.0)
                for row in block["rows"]
            ]
            evidence_similarity = _bounded_number(
                target_evidence["speaker_similarity"], lower=-1.0, upper=1.0
            )
            evidence_coverage = _bounded_number(
                target_evidence["speaker_coverage"], lower=0.0, upper=1.0
            )
            reason = None
            pathology = None
            if not target_evidence["target"]:
                reason = "not_target_speaker"
            elif voice_row.get("excluded_reason"):
                reason = str(voice_row["excluded_reason"])
            elif duration <= 0:
                reason, pathology = "invalid_duration", "invalid_duration"
            elif "\ufffd" in block_text:
                reason, pathology = "replacement_character", "replacement_character"
            elif not block_text and not events:
                reason, pathology = "empty_text_without_event", "empty_text"
            elif strict_lineage and (
                evidence_similarity is None
                or evidence_coverage is None
                or evidence_coverage < 0.5
            ):
                reason, pathology = "insufficient_target_speaker_evidence", "speaker_evidence"
            elif strict_lineage and (
                not voice_row.get("audio_path")
                or any(not row.get("audio_path") for row in block["rows"] if row.get("text"))
            ):
                reason, pathology = "missing_target_audio", "missing_audio"
            elif any(value is None or value > 0.35 for value in row_overlaps):
                reason, pathology = "overlap_too_high", "overlap"
            elif strict_lineage and any(
                key not in voice_row or voice_row.get(key) in (None, "", {})
                for key in (
                    "asr_backend",
                    "asr_model_revision",
                    "transcript_hash",
                    "alignment_backend",
                    "alignment_model_revision",
                    "alignment_hash",
                )
            ):
                reason, pathology = "missing_transcript_alignment_provenance", "provenance"
            caption, emotion, events, _ = normalize_voice_fields(
                {
                    "caption": voice_row.get("caption") or "自然に話している。",
                    "emotion": voice_row.get("emotion") or "NEUTRAL",
                    "events": events,
                }
            )
            answer = {
                "text": block_text,
                "voice": {
                    "caption": caption,
                    "emotion": emotion,
                    "events": list(events),
                },
            }
            example = {
                "lfm_contract": {
                    "schema_version": LFM_CONTRACT_SCHEMA_VERSION,
                    "fingerprint": LFM_CONTRACT_FINGERPRINT,
                },
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "completion": [
                    {
                        "role": "assistant",
                        "content": json.dumps(answer, ensure_ascii=False),
                    }
                ],
            }
            estimated = _estimate_lfm_tokens(example, tokenizer, source=token_sources)
            candidate_text_lengths.append(len(block_text))
            candidate_token_lengths.append(estimated)
            if estimated > max_tokens:
                reason, pathology = "token_budget_exceeded", "token_budget"
            if reason is not None:
                reason_counts[reason] += 1
                if pathology:
                    pathology_counts[pathology] += 1
                continue
            examples.append(
                {
                    **example,
                    "quality": {
                        "duration": round(duration, 3),
                        "target_speaker_evidence": target_evidence,
                        "overlap_max": max(
                            value if value is not None else 1.0 for value in row_overlaps
                        ),
                        "speaker_coverage_min": min(
                            value if value is not None else 0.0 for value in row_coverages
                        ),
                        "asr_confidence": {
                            "word_probability": voice_row.get("word_probability"),
                            "language_probability": voice_row.get("asr_language_probability"),
                            "segment_confidence": voice_row.get("asr_segment_confidence"),
                        },
                        "token_count": estimated,
                        "text_hash": _text_hash(block_text),
                        "transcript_hash": voice_row.get("transcript_hash"),
                        "alignment_backend": voice_row.get("alignment_backend"),
                        "alignment_revision": voice_row.get("alignment_model_revision"),
                        "provenance": {
                            "asr_backend": voice_row.get("asr_backend"),
                            "asr_model_revision": voice_row.get("asr_model_revision"),
                            "transcript_hash": voice_row.get("transcript_hash"),
                            "alignment_backend": voice_row.get("alignment_backend"),
                            "alignment_revision": voice_row.get("alignment_model_revision"),
                            "alignment_hash": voice_row.get("alignment_hash"),
                        },
                    },
                }
            )
            text_lengths.append(len(block_text))
            token_lengths.append(estimated)
    count = write_jsonl(output, examples)
    if report_path is not None:
        _write_json_atomic(
            report_path,
            {
                "schema_version": LFM_QUALITY_REPORT_SCHEMA,
                "lineage": lineage_metadata,
                "candidate_count": candidate_count,
                "accepted_count": count,
                "rejected_count": sum(reason_counts.values()),
                "rejection_reasons": dict(sorted(reason_counts.items())),
                "pathology_counters": dict(sorted(pathology_counts.items())),
                "text_distribution": _distribution(text_lengths),
                "token_distribution": _distribution(token_lengths),
                "candidate_text_distribution": _distribution(candidate_text_lengths),
                "candidate_token_distribution": _distribution(candidate_token_lengths),
                "max_tokens": max_tokens,
                "token_count_source": (
                    token_sources[0]
                    if token_sources and len(set(token_sources)) == 1
                    else "mixed_token_count_sources"
                    if token_sources
                    else "not_applicable"
                ),
                "valid_short_or_nonverbal_retention": True,
                "quality_gate": {
                    "passed": count > 0,
                    "reason": None if count > 0 else "no_accepted_conversation_examples",
                },
            },
        )
    return count


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
    }


def _estimate_lfm_tokens(
    example: dict[str, Any],
    tokenizer: Any | None,
    *,
    source: list[str] | None = None,
) -> int:
    """Count the exact pinned chat template, with a conservative test fallback."""

    prompt = example.get("prompt") or []
    completion = example.get("completion") or []
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            processed = tokenizer.apply_chat_template(
                prompt + completion,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=False,
            )
            if isinstance(processed, dict) and "input_ids" in processed:
                encoded = processed["input_ids"]
            else:
                encoded = processed
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
                encoded = encoded[0]
            if source is not None:
                source.append("pinned_lfm_chat_template")
            return len(encoded)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Training-time worker validation remains authoritative; the export
            # report records the deterministic fallback rather than guessing a
            # model-specific score from an ASR confidence field.
            pass
    characters = sum(
        len(str(message.get("content") or ""))
        for message in (*prompt, *completion)
        if isinstance(message, dict)
    )
    if source is not None:
        source.append("conservative_character_estimate")
    return max(1, characters + 8)
