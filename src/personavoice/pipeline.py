from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from personavoice.asr_contract import (
    alignment_contract_for_result,
    alignment_hash,
    normalize_asr_result,
    words_alignment,
)
from personavoice.atomic import atomic_write_json
from personavoice.captions import annotate_text, build_caption, normalize_events
from personavoice.config import PersonaConfig
from personavoice.dataset import (
    export_irodori,
    export_lfm,
    export_seed_vc,
    load_lfm_tokenizer,
    replace_utterances,
)
from personavoice.lineage import (
    ALIGNMENT_CONTRACT_VERSION,
    ASR_CONTRACT_VERSION,
    build_lineage_record,
    lineage_identity,
    master_fingerprint,
    prepare_lineage_seed,
    resolve_alignment,
    resolve_backend,
)
from personavoice.media import (
    cut_audio,
    extract_lossless_audio,
    inventory,
    inventory_fingerprint,
    media_files,
    sha256_file,
)
from personavoice.prepare_checkpoints import (
    checkpoint_dir,
    cleanup_checkpoint_dir,
    discard_checkpoint,
    recover_checkpoint,
)
from personavoice.project import PersonaPaths
from personavoice.separation import materialize_analysis_audio
from personavoice.speaker import (
    TARGET_NOT_FOUND,
    dominant_speaker,
    overlap_ratio,
    select_target_speaker,
)
from personavoice.state import StateStore, _prepare_cache_policy
from personavoice.worker_contracts import valid_alignment_result
from personavoice.workers import worker

PREPARE_SCHEMA_VERSION = 4


def _dump(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_cache_json(path: Path) -> dict[str, Any] | None:
    """Read a disposable prepare cache, self-healing corrupt/truncated files."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(value, dict):
        path.unlink(missing_ok=True)
        return None
    return value


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _batch_results(rows: list[dict[str, Any]], *, operation: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    errors = []
    for row in rows:
        item_id = str(row.get("id"))
        if row.get("ok"):
            output[item_id] = row.get("result")
        else:
            errors.append(f"{item_id}: {row.get('error') or 'unknown error'}")
    if errors:
        joined = "\n".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n... and {len(errors) - 20} more"
        raise RuntimeError(f"{operation} failed for {len(errors)} item(s):\n{joined}{suffix}")
    return output


def _words(
    asr: dict[str, Any],
    alignment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for segment in asr.get("segments", []):
        for word in segment.get("words") or []:
            start = word.get("start")
            end = word.get("end")
            text = word.get("word") or ""
            if start is None or end is None or not text.strip():
                continue
            out.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": str(text),
                    "probability": word.get("probability"),
                }
            )
    if out or not isinstance(alignment, dict):
        return out
    for unit in alignment.get("units") or []:
        if not isinstance(unit, dict):
            continue
        start = unit.get("start")
        end = unit.get("end")
        text = unit.get("unit") or unit.get("text") or ""
        if start is None or end is None or not isinstance(text, str) or not text.strip():
            continue
        out.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "probability": unit.get("confidence"),
            }
        )
    return out


def _words_for_turn(turn: dict[str, Any], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = float(turn["start"])
    end = float(turn["end"])
    selected = []
    for word in words:
        midpoint = (float(word["start"]) + float(word["end"])) / 2.0
        if start - 0.08 <= midpoint <= end + 0.08:
            selected.append(word)
    return selected


def _empty_turn_rows(turn: dict[str, Any], *, max_seconds: float) -> list[dict[str, Any]]:
    start = float(turn["start"])
    end = float(turn["end"])
    if end <= start:
        return []
    count = max(1, math.ceil((end - start) / max_seconds))
    width = (end - start) / count
    rows = []
    for index in range(count):
        chunk_start = start + index * width
        chunk_end = min(end, start + (index + 1) * width)
        rows.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "speaker": str(turn["speaker"]),
                "speaker_coverage": 1.0,
                "text": "",
                "word_probability": None,
                "avg_logprob": None,
            }
        )
    return rows


def _word_turn_rows(
    turn: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    max_seconds: float,
) -> list[dict[str, Any]]:
    if not selected:
        return _empty_turn_rows(turn, max_seconds=max_seconds)

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in selected:
        if current and float(word["end"]) - float(current[0]["start"]) > max_seconds:
            chunks.append(current)
            current = []
        current.append(word)
    if current:
        chunks.append(current)

    turn_start = float(turn["start"])
    turn_end = float(turn["end"])
    rows = []
    for chunk in chunks:
        probabilities = [
            float(word["probability"])
            for word in chunk
            if word.get("probability") is not None
        ]
        rows.append(
            {
                "start": max(turn_start, float(chunk[0]["start"]) - 0.12),
                "end": min(turn_end, float(chunk[-1]["end"]) + 0.18),
                "speaker": str(turn["speaker"]),
                "speaker_coverage": 1.0,
                "text": "".join(str(word["text"]) for word in chunk).strip(),
                "word_probability": (
                    sum(probabilities) / len(probabilities) if probabilities else None
                ),
                "avg_logprob": None,
            }
        )
    return rows


def _fallback_segments(asr: dict[str, Any], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for segment in asr.get("segments", []):
        start = float(segment["start"])
        end = float(segment["end"])
        speaker, coverage = dominant_speaker(start, end, turns)
        rows.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "speaker_coverage": coverage,
                "text": (segment.get("text") or "").strip(),
                "word_probability": None,
                "avg_logprob": segment.get("avg_logprob"),
            }
        )
    return rows


def _turn_rows(
    asr: dict[str, Any],
    exclusive_turns: list[dict[str, Any]],
    *,
    max_seconds: float,
    alignment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    words = _words(asr, alignment)
    if not words:
        return _fallback_segments(asr, exclusive_turns)
    rows: list[dict[str, Any]] = []
    for turn in exclusive_turns:
        rows.extend(
            _word_turn_rows(
                turn,
                _words_for_turn(turn, words),
                max_seconds=max_seconds,
            )
        )
    return rows


def _merge_rows(
    rows: list[dict[str, Any]],
    *,
    gap: float,
    max_seconds: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not out:
            out.append(dict(row))
            continue
        previous = out[-1]
        can_merge = (
            row.get("speaker") == previous.get("speaker")
            and float(row["start"]) - float(previous["end"]) <= gap
            and float(row["end"]) - float(previous["start"]) <= max_seconds
        )
        if not can_merge:
            out.append(dict(row))
            continue
        previous["end"] = row["end"]
        previous["text"] = (
            str(previous.get("text") or "") + str(row.get("text") or "")
        ).strip()
        previous["speaker_coverage"] = min(
            float(previous.get("speaker_coverage") or 0.0),
            float(row.get("speaker_coverage") or 0.0),
        )
        probabilities = [
            value
            for value in (
                previous.get("word_probability"),
                row.get("word_probability"),
            )
            if value is not None
        ]
        previous["word_probability"] = (
            sum(float(value) for value in probabilities) / len(probabilities)
            if probabilities
            else None
        )
    return out


def _quality(row: dict[str, Any]) -> float:
    quality = 1.0
    quality -= min(0.7, float(row.get("overlap_ratio") or 0.0) * 3.0)
    quality *= max(0.4, min(1.0, float(row.get("speaker_coverage") or 0.0)))
    similarity = row.get("speaker_similarity")
    if similarity is not None:
        quality *= max(0.5, min(1.0, (float(similarity) + 1.0) / 2.0))
    probability = row.get("word_probability")
    if probability is not None:
        quality *= max(0.55, min(1.0, float(probability)))
    duration = float(row["end"]) - float(row["start"])
    if duration < 1.0:
        quality *= 0.55
    elif duration > 18.0:
        quality *= 0.75
    events = set(row.get("events") or [])
    if "BGM" in events:
        quality *= 0.7
    if row.get("excluded_reason"):
        quality *= 0.35
    return round(max(0.0, min(1.0, quality)), 4)


def _identity_embeddings(repo_root: Path, paths: PersonaPaths) -> list[list[float]]:
    identity = media_files(paths.identity)
    if not identity:
        return []
    diarization = worker(repo_root, "diarization")
    values_by_key: dict[str, list[float]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "identity")
    for source in identity:
        key = sha256_file(source)[:20]
        cache = paths.cache / "identity" / f"{key}.json"
        cache_paths[key] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        embedding = cached.get("embedding") if cached is not None else None
        if isinstance(embedding, list) and embedding:
            discard_checkpoint(checkpoints, key)
            values_by_key[key] = [float(value) for value in embedding]
            continue
        if cache.exists():
            cache.unlink(missing_ok=True)
        recovered = recover_checkpoint(checkpoints, key, "identity")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, key)
            embedding = recovered.get("embedding")
            if isinstance(embedding, list) and embedding:
                values_by_key[key] = [float(value) for value in embedding]
                continue
        pending.append({"id": key, "audio": str(source.resolve())})
    if pending:
        response = diarization.call(
            repo_root,
            "batch",
            {
                "embeddings": pending,
                "diarizations": [],
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(response.get("embeddings") or [], operation="identity embedding")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            discard_checkpoint(checkpoints, key)
            embedding = result.get("embedding")
            if embedding:
                values_by_key[key] = [float(value) for value in embedding]
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return [values_by_key[key] for key in sorted(values_by_key)]


def _batch_asr(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "asr")

    def normalize(source: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
        # A pre-v0.4 legacy cache may not carry backend provenance.  It remains
        # readable for compatibility, while every worker result written by the
        # current contract is normalized and therefore lineage-bound.
        if not value.get("backend"):
            return value
        return normalize_asr_result(
            value,
            backend=cfg.prepare.asr_model,
            source_audio=source["audio"],
            analysis_audio=source.get("analysis_audio") or source["audio"],
        )

    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "asr" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = normalize(source, cached)
            continue
        recovered = recover_checkpoint(checkpoints, source_id, "asr")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = normalize(source, recovered)
            continue
        analysis_audio = source.get("analysis_audio") or source["audio"]
        pending.append({"id": source_id, "audio": str(Path(analysis_audio).resolve())})
    if pending:
        response = worker(repo_root, "asr").call(
            repo_root,
            "batch_transcribe",
            {
                "items": pending,
                "model": cfg.prepare.asr_model,
                "compute_type": cfg.prepare.asr_compute_type,
                "device": cfg.prepare.asr_device,
                "dtype": cfg.prepare.asr_dtype,
                "language": cfg.prepare.language,
                "contract_version": ASR_CONTRACT_VERSION,
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(response.get("results") or [], operation="ASR")
        for source_id, result in results.items():
            source = next(item for item in sources if str(item["source_id"]) == source_id)
            result = normalize(source, result)
            _dump(cache_paths[source_id], result)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = result
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return values


def _batch_alignment(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    sources: list[dict[str, Any]],
    asr_by_source: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach a separately versioned alignment contract to every ASR result."""

    backend = resolve_backend(cfg.prepare.asr_model)
    alignment_spec = resolve_alignment(backend.key, cfg.prepare.alignment_backend)
    cache_root = paths.cache / "alignment"
    cache_root.mkdir(parents=True, exist_ok=True)
    values: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    cache_paths: dict[str, Path] = {}
    for source in sources:
        source_id = str(source["source_id"])
        asr = asr_by_source[source_id]
        request = alignment_contract_for_result(
            asr,
            asr_backend=backend.key,
            requested=cfg.prepare.alignment_backend,
        )
        transcript_hash = str(request["transcript_hash"])
        cache = cache_root / f"{source_id}-{alignment_spec.revision[:16]}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if (
            isinstance(cached, dict)
            and cached.get("contract_version") == ALIGNMENT_CONTRACT_VERSION
            and cached.get("key") == alignment_spec.key
            and cached.get("backend") == alignment_spec.key
            and cached.get("model_id") == alignment_spec.model_id
            and cached.get("model_revision") == alignment_spec.revision
            and cached.get("transcript_hash") == transcript_hash
            and cached.get("asr_backend") == backend.key
            and cached.get("asr_model_id") == backend.model_id
            and cached.get("asr_model_revision") == backend.revision
            and cached.get("revision") == alignment_spec.revision
            and isinstance(cached.get("hash"), str)
            and cached.get("hash") == alignment_hash(cached.get("units") or [])
            and valid_alignment_result(cached)
        ):
            values[source_id] = cached
            continue
        if cache.exists():
            cache.unlink(missing_ok=True)

        embedded = asr.get("alignment") if isinstance(asr, dict) else None
        if (
            isinstance(embedded, dict)
            and embedded.get("model_revision") == alignment_spec.revision
        ):
            embedded_value = {
                **embedded,
                **request,
                "transcript_hash": transcript_hash,
                "hash": alignment_hash(embedded.get("units") or []),
            }
            if (
                embedded_value.get("key") == alignment_spec.key
                and embedded_value.get("backend") == alignment_spec.key
                and embedded_value.get("model_id") == alignment_spec.model_id
                and embedded_value.get("asr_backend") == backend.key
                and embedded_value.get("asr_model_id") == backend.model_id
                and embedded_value.get("asr_model_revision") == backend.revision
                and valid_alignment_result(embedded_value)
            ):
                values[source_id] = embedded_value
                _dump(cache, values[source_id])
                continue
        if alignment_spec.key == "whisper-native-words":
            alignment = words_alignment(asr.get("segments") or [])
            values[source_id] = {
                **alignment,
                **request,
                "backend": alignment_spec.key,
                "model_id": alignment_spec.model_id,
                "model_revision": alignment_spec.revision,
                "transcript_hash": transcript_hash,
            }
            _dump(cache, values[source_id])
            continue
        analysis_audio = source.get("analysis_audio") or source["audio"]
        pending.append(
            {
                "id": source_id,
                "audio": str(Path(analysis_audio).resolve()),
                "transcript": "".join(
                    str(segment.get("text") or "")
                    for segment in asr.get("segments", [])
                ).strip(),
                "segments": asr.get("segments") or [],
                "contract": request,
            }
        )
    if pending:
        response = worker(repo_root, "asr").call(
            repo_root,
            "batch_align",
            {
                "items": pending,
                "contract_version": ALIGNMENT_CONTRACT_VERSION,
                "backend": alignment_spec.key,
                "model": alignment_spec.model_id,
                "revision": alignment_spec.revision,
                "asr_backend": backend.key,
                "device": cfg.prepare.asr_device,
                "dtype": cfg.prepare.asr_dtype,
                "checkpoint_dir": str(checkpoint_dir(cache_root)),
            },
        )
        results = _batch_results(response.get("results") or [], operation="ASR alignment")
        for source_id, result in results.items():
            if not isinstance(result, dict):
                raise RuntimeError(f"ASR alignment returned an invalid result for {source_id}")
            expected = alignment_contract_for_result(
                asr_by_source[source_id],
                asr_backend=backend.key,
                requested=cfg.prepare.alignment_backend,
            )
            if result.get("revision") != alignment_spec.revision:
                raise RuntimeError("ASR alignment revision does not match the requested contract")
            result = {
                **result,
                **expected,
                "hash": alignment_hash(result.get("units") or []),
            }
            if not valid_alignment_result(result):
                raise RuntimeError(f"ASR alignment returned an invalid result for {source_id}")
            _dump(cache_paths[source_id], result)
            values[source_id] = result
    return values


def _batch_diarization(
    repo_root: Path,
    paths: PersonaPaths,
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "diarization")
    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "diarization" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = cached
            continue
        recovered = recover_checkpoint(checkpoints, source_id, "diarization")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = recovered
            continue
        pending.append({"id": source_id, "audio": str(source["audio"].resolve())})
    if pending:
        response = worker(repo_root, "diarization").call(
            repo_root,
            "batch",
            {
                "embeddings": [],
                "diarizations": pending,
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(
            response.get("diarizations") or [],
            operation="speaker diarization",
        )
        for source_id, result in results.items():
            _dump(cache_paths[source_id], result)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = result
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return values


def _batch_sense(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not cfg.prepare.use_sensevoice:
        return {}
    values: dict[str, dict[str, Any]] = {}
    pending = []
    pending_keys: set[str] = set()
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "sense")
    for row in rows:
        audio_path = row.get("audio_path")
        if not audio_path:
            continue
        audio = Path(audio_path)
        key = sha256_file(audio)[:20]
        row["sense_key"] = key
        cache = paths.cache / "sense" / f"{key}.json"
        cache_paths[key] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            discard_checkpoint(checkpoints, key)
            values[key] = cached
            continue
        recovered = recover_checkpoint(checkpoints, key, "sense")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, key)
            values[key] = recovered
            continue
        if key not in pending_keys:
            pending.append({"id": key, "audio": str(audio.resolve())})
            pending_keys.add(key)
    if pending:
        response = worker(repo_root, "sense").call(
            repo_root,
            "batch_analyze",
            {
                "items": pending,
                "language": cfg.prepare.language,
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(response.get("results") or [], operation="SenseVoice analysis")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            discard_checkpoint(checkpoints, key)
            values[key] = result
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return values


def _select_references(
    paths: PersonaPaths,
    rows: list[dict[str, Any]],
    cfg: PersonaConfig,
) -> list[str]:
    shutil.rmtree(paths.references, ignore_errors=True)
    paths.references.mkdir(parents=True, exist_ok=True)
    candidates = [
        row
        for row in rows
        if row.get("target")
        and row.get("audio_path")
        and row.get("text")
        and row.get("quality", 0) >= 0.6
        and float(row["end"]) - float(row["start"])
        <= cfg.prepare.reference_clip_max_seconds
    ]
    by_emotion: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(candidates, key=lambda item: float(item.get("quality", 0)), reverse=True):
        emotion = str(row.get("emotion") or "NEUTRAL").lower()
        by_emotion.setdefault(emotion, []).append(row)
    for emotion, items in by_emotion.items():
        directory = paths.references / "by_emotion" / emotion
        directory.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(items[:3], start=1):
            shutil.copy2(Path(row["audio_path"]), directory / f"ref_{index:02d}.flac")

    candidates.sort(
        key=lambda row: (
            str(row.get("emotion") or "NEUTRAL").upper() != "NEUTRAL",
            -float(row.get("quality", 0)),
        )
    )
    selected: list[dict[str, Any]] = []
    total = 0.0
    for row in candidates:
        duration = float(row["end"]) - float(row["start"])
        if total + duration > cfg.prepare.reference_seconds and selected:
            continue
        selected.append(row)
        total += duration
        if total >= cfg.prepare.reference_seconds:
            break
    outputs = []
    for index, row in enumerate(selected, start=1):
        source = Path(row["audio_path"])
        destination = (
            paths.references
            / f"ref_{index:03d}_{str(row.get('emotion') or 'neutral').lower()}.flac"
        )
        shutil.copy2(source, destination)
        outputs.append(str(destination.resolve()))
    _dump(paths.references / "bank.json", {"files": outputs, "seconds": round(total, 2)})
    return outputs


def _prepare_fingerprint(paths: PersonaPaths, cfg: PersonaConfig) -> str:
    payload = {
        "schema": PREPARE_SCHEMA_VERSION,
        "raw": inventory_fingerprint(paths.raw),
        "identity": inventory_fingerprint(paths.identity),
        "config": cfg.prepare.model_dump(mode="json"),
        # State-level completion must notice a changed ASR/alignment/
        # separation implementation before it can return an older lineage.
        "implementation_policy": _prepare_cache_policy(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _skipped_source_report(
    path: Path,
    identity_rejections: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RuntimeError(f"Invalid skipped-source report: {path}")
    for item in value:
        extra = identity_rejections.get(str(item.get("source_id")))
        if extra is not None:
            item.update(extra)
    _dump(path, value)
    return value


def prepare_persona(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not cfg.consent.authorized:
        raise PermissionError(
            "persona.yaml does not mark this voice as authorized. "
            "Confirm permission and set consent.authorized: true."
        )
    raw_files = media_files(paths.raw)
    if not raw_files:
        raise FileNotFoundError(f"No supported media was found in {paths.raw}")

    store = StateStore(paths.state)
    fingerprint = _prepare_fingerprint(paths, cfg)
    if not force and store.is_complete("prepare", fingerprint):
        return store.stage("prepare").get("result", {})

    seed = prepare_lineage_seed(
        paths,
        cfg,
        prepare_schema=PREPARE_SCHEMA_VERSION,
        # The dynamic implementation contract is recorded in the lineage;
        # the stable state marker remains reserved for legacy root caches.
        cache_policy_version=_prepare_cache_policy(),
    )
    lineage_id, lineage_fingerprint = lineage_identity(seed)
    candidate_paths = paths.for_lineage(lineage_id)
    candidate_paths.ensure_lineage()
    # All generated data below is written through this immutable candidate view.
    # raw/identity/config/state still resolve to the persona root.
    paths = candidate_paths

    with store.running(
        "prepare",
        fingerprint,
        force=force,
        lineage=True,
        lineage_cache_root=paths.cache,
    ):
        source_inventory = inventory(paths.raw)
        _dump(paths.dataset / "source_inventory.json", source_inventory)

        prepared_sources: list[dict[str, Any]] = []
        separation_rows: list[dict[str, Any]] = []
        for source in source_inventory:
            source_id = source["sha256"][:16]
            source_audio = paths.cache / "audio" / f"{source_id}.flac"
            if not _nonempty_file(source_audio):
                source_audio.unlink(missing_ok=True)
                extract_lossless_audio(Path(source["absolute_path"]), source_audio)
            analysis_audio, separation = materialize_analysis_audio(
                source_audio,
                paths,
                source_id,
                policy=cfg.prepare.separation_policy,
                metadata=source,
            )
            separation_rows.append(
                {
                    "source_id": source_id,
                    "canonical_audio": str(source_audio.resolve()),
                    "analysis_audio": str(analysis_audio.resolve()),
                    **separation,
                }
            )
            prepared_sources.append(
                {
                    "source_id": source_id,
                    "source": source,
                    "audio": source_audio,
                    "canonical_audio": source_audio,
                    "analysis_audio": analysis_audio,
                    "separation": separation,
                }
            )
        _dump(paths.dataset / "analysis_audio.json", separation_rows)

        identity_embeddings = _identity_embeddings(repo_root, paths)
        asr_by_source = _batch_asr(repo_root, paths, cfg, prepared_sources)
        alignment_by_source = _batch_alignment(
            repo_root,
            paths,
            cfg,
            prepared_sources,
            asr_by_source,
        )
        diar_by_source = _batch_diarization(repo_root, paths, prepared_sources)
        all_rows: list[dict[str, Any]] = []
        identity_rejections: dict[str, dict[str, Any]] = {}

        for prepared in prepared_sources:
            source_id = str(prepared["source_id"])
            source = prepared["source"]
            source_audio: Path = prepared["audio"]
            analysis_audio: Path = prepared["analysis_audio"]
            asr = asr_by_source[source_id]
            alignment = alignment_by_source[source_id]
            asr_spec = resolve_backend(cfg.prepare.asr_model)
            diarization = diar_by_source[source_id]
            embeddings = {
                str(key): value
                for key, value in (diarization.get("speaker_embeddings") or {}).items()
            }
            target_label, target_similarity = select_target_speaker(
                embeddings,
                identity_embeddings,
                threshold=cfg.prepare.min_identity_similarity,
            )
            if target_label == TARGET_NOT_FOUND:
                identity_rejections[source_id] = {
                    "reason": "authorized_speaker_below_identity_threshold",
                    "best_similarity": round(float(target_similarity), 6),
                    "threshold": cfg.prepare.min_identity_similarity,
                }
            exclusive = diarization.get("exclusive_turns") or diarization.get("turns") or []
            regular = diarization.get("turns") or exclusive
            source_rows = _turn_rows(
                asr,
                exclusive,
                max_seconds=cfg.prepare.max_clip_seconds,
                alignment=alignment,
            )
            source_rows = _merge_rows(
                source_rows,
                gap=cfg.prepare.merge_gap_seconds,
                max_seconds=cfg.prepare.max_clip_seconds,
            )

            for index, row in enumerate(source_rows):
                start = max(0.0, float(row["start"]))
                end = max(start + 0.05, float(row["end"]))
                target = row.get("speaker") == target_label
                overlap = overlap_ratio(start, end, regular)
                item: dict[str, Any] = {
                    "id": f"{source_id}_{index:06d}",
                    "source_id": source_id,
                    "source_path": source["path"],
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": row.get("speaker"),
                    "target": target,
                    "speaker_similarity": target_similarity if target else None,
                    "speaker_coverage": float(row.get("speaker_coverage") or 0.0),
                    "overlap_ratio": overlap,
                    "word_probability": row.get("word_probability"),
                    "text": str(row.get("text") or "").strip(),
                    "emotion": None,
                    "events": [],
                    "caption": "",
                    "audio_path": None,
                    "excluded_reason": None,
                    "canonical_audio_path": str(source_audio.resolve()),
                    "analysis_audio_path": str(analysis_audio.resolve()),
                    "source_audio_kind": "canonical",
                    "separation": prepared["separation"],
                    "asr_backend": asr.get("backend") or asr_spec.key,
                    "asr_model_id": asr.get("model_id") or asr_spec.model_id,
                    "asr_model_revision": asr.get("model_revision") or asr_spec.revision,
                    "asr_language_probability": asr.get("language_probability"),
                    "asr_segment_confidence": [
                        segment.get("confidence")
                        for segment in asr.get("segments") or []
                        if isinstance(segment, dict) and segment.get("confidence") is not None
                    ],
                    "transcript_hash": (
                        asr.get("provenance", {}).get("transcript_hash")
                        if isinstance(asr.get("provenance"), dict)
                        else hashlib.sha256(
                            json.dumps(
                                asr.get("segments") or [],
                                sort_keys=True,
                                ensure_ascii=False,
                            ).encode("utf-8")
                        ).hexdigest()
                    ),
                    "alignment_backend": alignment.get("key"),
                    "alignment_model_id": alignment.get("model_id"),
                    "alignment_model_revision": alignment.get("revision"),
                    "alignment_hash": alignment.get("hash"),
                    "boundary_evidence": {
                        "source": "diarization-turn-and-asr-word-boundary",
                        "clip_start": round(start, 3),
                        "clip_end": round(end, 3),
                        "canonical_audio": str(source_audio.resolve()),
                        "analysis_audio": str(analysis_audio.resolve()),
                        "asr_segment_count": len(asr.get("segments") or []),
                        "alignment_units": len(alignment.get("units") or []),
                    },
                }
                if target and overlap > cfg.prepare.max_overlap_ratio:
                    item["excluded_reason"] = "overlap"
                if (
                    target
                    and not item["excluded_reason"]
                    and (end - start) >= cfg.prepare.min_clip_seconds
                ):
                    clip = paths.dataset / "clips" / f"{item['id']}.flac"
                    if not _nonempty_file(clip):
                        clip.unlink(missing_ok=True)
                        cut_audio(source_audio, clip, start, end)
                    item["audio_path"] = str(clip.resolve())
                all_rows.append(item)

        sense_by_key = _batch_sense(repo_root, paths, cfg, all_rows)
        for item in all_rows:
            sense_key = item.pop("sense_key", None)
            if sense_key and sense_key in sense_by_key:
                sense = sense_by_key[sense_key]
                item["emotion"] = sense.get("emotion")
                item["events"] = normalize_events(sense.get("events") or [])
            if item.get("audio_path") and not item.get("text"):
                keep_nonverbal = cfg.prepare.keep_nonverbal_only and bool(item["events"])
                if not keep_nonverbal:
                    item["audio_path"] = None
                    item["excluded_reason"] = item.get("excluded_reason") or "empty_text"
            chars_per_second = (
                len(item["text"]) / max(0.1, float(item["end"]) - float(item["start"]))
                if item.get("text")
                else None
            )
            item["text_annotated"] = annotate_text(item.get("text", ""), item["events"])
            item["caption"] = build_caption(
                emotion=item["emotion"],
                events=item["events"],
                chars_per_second=chars_per_second,
            )
            item["quality"] = _quality(item)

        master_db = paths.dataset / "master.sqlite3"
        replace_utterances(master_db, all_rows)
        _dump(paths.dataset / "master.json", all_rows)
        skipped_report = _skipped_source_report(
            paths.dataset / "skipped_sources.json",
            identity_rejections,
        )
        references = _select_references(paths, all_rows, cfg)
        irodori_count = export_irodori(
            master_db,
            paths.dataset / "irodori_source.jsonl",
            cfg.name,
            report_path=paths.dataset / "irodori_quality_report.json",
            lineage_metadata={
                "lineage_id": lineage_id,
                "lineage_fingerprint": lineage_fingerprint,
                "master_fingerprint": master_fingerprint(all_rows),
            },
            require_provenance=True,
        )
        lfm_report_path = paths.dataset / "lfm_quality_report.json"
        lfm_tokenizer = (
            load_lfm_tokenizer(repo_root / "models" / "lfm" / "base")
            if cfg.training.lfm.enabled
            else None
        )
        lfm_count = export_lfm(
            master_db,
            paths.dataset / "lfm_train.jsonl",
            cfg.name,
            report_path=lfm_report_path,
            lineage_metadata={
                "lineage_id": lineage_id,
                "lineage_fingerprint": lineage_fingerprint,
                "master_fingerprint": master_fingerprint(all_rows),
            },
            tokenizer=lfm_tokenizer,
        )
        seed_count = export_seed_vc(master_db, paths.dataset / "seed_vc")
        target_rows = [row for row in all_rows if row.get("target")]
        usable = [
            row
            for row in target_rows
            if row.get("audio_path") and row.get("text_annotated")
        ]
        prepared_master_fingerprint = master_fingerprint(all_rows)
        lineage_record = build_lineage_record(
            seed,
            lineage_id=lineage_id,
            lineage_fingerprint=lineage_fingerprint,
            master_fingerprint=prepared_master_fingerprint,
            source_count=len(source_inventory),
            utterance_count=len(all_rows),
        )
        _dump(paths.lineage_record, lineage_record)

        def _relative(path: Path) -> str:
            return path.resolve().relative_to(paths.root.resolve()).as_posix()

        result = {
            "prepare_schema": PREPARE_SCHEMA_VERSION,
            "lineage_schema": 1,
            "lineage_id": lineage_id,
            "lineage_fingerprint": lineage_fingerprint,
            "master_fingerprint": prepared_master_fingerprint,
            "asr_backend": resolve_backend(cfg.prepare.asr_model).key,
            "alignment_backend": resolve_alignment(
                cfg.prepare.asr_model,
                cfg.prepare.alignment_backend,
            ).key,
            "separation_policy": cfg.prepare.separation_policy,
            "separation_report": _relative(paths.dataset / "analysis_audio.json"),
            "sources": len(source_inventory),
            "skipped_sources": len(skipped_report),
            "utterances": len(all_rows),
            "target_utterances": len(target_rows),
            "usable_tts_utterances": len(usable),
            "usable_seconds": round(
                sum(float(row["end"]) - float(row["start"]) for row in usable),
                2,
            ),
            "references": len(references),
            "irodori_examples": irodori_count,
            "lfm_examples": lfm_count,
            "seed_vc_examples": seed_count,
            "master_db": _relative(master_db),
            "dataset_root": _relative(paths.dataset),
            "references_root": _relative(paths.references),
            "cache_root": _relative(paths.cache),
            "models_root": _relative(paths.models),
            "outputs_root": _relative(paths.outputs),
            "lineage_record": _relative(paths.lineage_record),
            "lfm_quality_report": _relative(lfm_report_path),
            "irodori_quality_report": _relative(
                paths.dataset / "irodori_quality_report.json"
            ),
        }
        store.set_result("prepare", result)
        return result
