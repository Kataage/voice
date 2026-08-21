from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from personavoice.atomic import atomic_write_json
from personavoice.captions import annotate_text, build_caption, normalize_events
from personavoice.config import PersonaConfig
from personavoice.dataset import export_irodori, export_lfm, export_seed_vc, replace_utterances
from personavoice.media import (
    cut_audio,
    extract_lossless_audio,
    inventory,
    inventory_fingerprint,
    media_files,
    sha256_file,
)
from personavoice.project import PersonaPaths
from personavoice.speaker import (
    TARGET_NOT_FOUND,
    dominant_speaker,
    overlap_ratio,
    select_target_speaker,
)
from personavoice.state import StateStore
from personavoice.workers import worker

PREPARE_SCHEMA_VERSION = 3


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


def _words(asr: dict[str, Any]) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
    words = _words(asr)
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
    for source in identity:
        key = sha256_file(source)[:20]
        cache = paths.cache / "identity" / f"{key}.json"
        cache_paths[key] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        embedding = cached.get("embedding") if cached is not None else None
        if isinstance(embedding, list) and embedding:
            values_by_key[key] = [float(value) for value in embedding]
        else:
            if cache.exists():
                cache.unlink(missing_ok=True)
            pending.append({"id": key, "audio": str(source.resolve())})
    if pending:
        response = diarization.call(
            repo_root,
            "batch",
            {"embeddings": pending, "diarizations": []},
        )
        results = _batch_results(response.get("embeddings") or [], operation="identity embedding")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            embedding = result.get("embedding")
            if embedding:
                values_by_key[key] = [float(value) for value in embedding]
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
    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "asr" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            values[source_id] = cached
        else:
            pending.append({"id": source_id, "audio": str(source["audio"].resolve())})
    if pending:
        response = worker(repo_root, "asr").call(
            repo_root,
            "batch_transcribe",
            {
                "items": pending,
                "model": cfg.prepare.asr_model,
                "compute_type": cfg.prepare.asr_compute_type,
                "language": cfg.prepare.language,
            },
        )
        results = _batch_results(response.get("results") or [], operation="ASR")
        for source_id, result in results.items():
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
    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "diarization" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            values[source_id] = cached
        else:
            pending.append({"id": source_id, "audio": str(source["audio"].resolve())})
    if pending:
        response = worker(repo_root, "diarization").call(
            repo_root,
            "batch",
            {"embeddings": [], "diarizations": pending},
        )
        results = _batch_results(
            response.get("diarizations") or [],
            operation="speaker diarization",
        )
        for source_id, result in results.items():
            _dump(cache_paths[source_id], result)
            values[source_id] = result
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
            values[key] = cached
        elif key not in pending_keys:
            pending.append({"id": key, "audio": str(audio.resolve())})
            pending_keys.add(key)
    if pending:
        response = worker(repo_root, "sense").call(
            repo_root,
            "batch_analyze",
            {"items": pending, "language": cfg.prepare.language},
        )
        results = _batch_results(response.get("results") or [], operation="SenseVoice analysis")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            values[key] = result
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
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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

    with store.running("prepare", fingerprint, force=force):
        source_inventory = inventory(paths.raw)
        _dump(paths.dataset / "source_inventory.json", source_inventory)

        prepared_sources: list[dict[str, Any]] = []
        for source in source_inventory:
            source_id = source["sha256"][:16]
            source_audio = paths.cache / "audio" / f"{source_id}.flac"
            if not _nonempty_file(source_audio):
                source_audio.unlink(missing_ok=True)
                extract_lossless_audio(Path(source["absolute_path"]), source_audio)
            prepared_sources.append(
                {
                    "source_id": source_id,
                    "source": source,
                    "audio": source_audio,
                }
            )

        identity_embeddings = _identity_embeddings(repo_root, paths)
        asr_by_source = _batch_asr(repo_root, paths, cfg, prepared_sources)
        diar_by_source = _batch_diarization(repo_root, paths, prepared_sources)
        all_rows: list[dict[str, Any]] = []
        skipped_sources: list[dict[str, Any]] = []

        for prepared in prepared_sources:
            source_id = str(prepared["source_id"])
            source = prepared["source"]
            source_audio: Path = prepared["audio"]
            asr = asr_by_source[source_id]
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
                skipped_sources.append(
                    {
                        "source_id": source_id,
                        "source_path": source["path"],
                        "reason": "authorized_speaker_below_identity_threshold",
                        "best_similarity": round(float(target_similarity), 6),
                        "threshold": cfg.prepare.min_identity_similarity,
                    }
                )
            exclusive = diarization.get("exclusive_turns") or diarization.get("turns") or []
            regular = diarization.get("turns") or exclusive
            source_rows = _turn_rows(
                asr,
                exclusive,
                max_seconds=cfg.prepare.max_clip_seconds,
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
        _dump(paths.dataset / "skipped_sources.json", skipped_sources)
        references = _select_references(paths, all_rows, cfg)
        irodori_count = export_irodori(
            master_db,
            paths.dataset / "irodori_source.jsonl",
            cfg.name,
        )
        lfm_count = export_lfm(master_db, paths.dataset / "lfm_train.jsonl", cfg.name)
        seed_count = export_seed_vc(master_db, paths.dataset / "seed_vc")
        target_rows = [row for row in all_rows if row.get("target")]
        usable = [
            row
            for row in target_rows
            if row.get("audio_path") and row.get("text_annotated")
        ]
        result = {
            "prepare_schema": PREPARE_SCHEMA_VERSION,
            "sources": len(source_inventory),
            "skipped_sources": len(skipped_sources),
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
            "master_db": str(master_db.resolve()),
        }
        store.set_result("prepare", result)
        return result
