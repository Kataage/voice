from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from personavoice.captions import annotate_text, build_caption, normalize_events
from personavoice.config import PersonaConfig
from personavoice.dataset import export_irodori, export_lfm, export_seed_vc, replace_utterances
from personavoice.media import cut_audio, extract_lossless_audio, inventory, inventory_fingerprint, media_files
from personavoice.project import PersonaPaths
from personavoice.speaker import dominant_speaker, overlap_ratio, select_target_speaker
from personavoice.state import StateStore
from personavoice.workers import worker


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_or_call(cache: Path, fn: Any) -> Any:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    value = fn()
    _dump(cache, value)
    return value


def _words(asr: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for segment in asr.get("segments", []):
        for word in segment.get("words") or []:
            start = word.get("start")
            end = word.get("end")
            text = word.get("word") or ""
            if start is None or end is None or not text.strip():
                continue
            out.append({"start": float(start), "end": float(end), "text": text})
    return out


def _text_for_turn(turn: dict[str, Any], words: list[dict[str, Any]]) -> tuple[str, float | None, float | None]:
    start = float(turn["start"])
    end = float(turn["end"])
    selected = []
    for word in words:
        midpoint = (float(word["start"]) + float(word["end"])) / 2
        if start - 0.08 <= midpoint <= end + 0.08:
            selected.append(word)
    if not selected:
        return "", None, None
    text = "".join(word["text"] for word in selected).strip()
    return text, float(selected[0]["start"]), float(selected[-1]["end"])


def _fallback_segments(asr: dict[str, Any], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seg in asr.get("segments", []):
        start = float(seg["start"])
        end = float(seg["end"])
        speaker, coverage = dominant_speaker(start, end, turns)
        rows.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "speaker_coverage": coverage,
                "text": (seg.get("text") or "").strip(),
                "avg_logprob": seg.get("avg_logprob"),
            }
        )
    return rows


def _turn_rows(asr: dict[str, Any], exclusive_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words = _words(asr)
    if not words:
        return _fallback_segments(asr, exclusive_turns)
    rows: list[dict[str, Any]] = []
    for turn in exclusive_turns:
        text, first, last = _text_for_turn(turn, words)
        start = float(turn["start"])
        end = float(turn["end"])
        if first is not None and last is not None:
            start = max(start, first - 0.12)
            end = min(end, last + 0.18)
        rows.append(
            {
                "start": start,
                "end": end,
                "speaker": str(turn["speaker"]),
                "speaker_coverage": 1.0,
                "text": text,
                "avg_logprob": None,
            }
        )
    return rows


def _split_long(row: dict[str, Any], *, max_seconds: float) -> list[dict[str, Any]]:
    duration = float(row["end"]) - float(row["start"])
    if duration <= max_seconds:
        return [row]
    parts = max(2, int(duration // max_seconds) + (1 if duration % max_seconds else 0))
    width = duration / parts
    out = []
    for index in range(parts):
        item = dict(row)
        item["start"] = float(row["start"]) + index * width
        item["end"] = min(float(row["end"]), float(item["start"]) + width)
        if index:
            item["text"] = ""
        out.append(item)
    return out


def _merge_rows(rows: list[dict[str, Any]], *, gap: float, max_seconds: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not out:
            out.append(dict(row))
            continue
        prev = out[-1]
        can_merge = (
            row.get("speaker") == prev.get("speaker")
            and float(row["start"]) - float(prev["end"]) <= gap
            and float(row["end"]) - float(prev["start"]) <= max_seconds
        )
        if not can_merge:
            out.append(dict(row))
            continue
        prev["end"] = row["end"]
        prev["text"] = (str(prev.get("text") or "") + str(row.get("text") or "")).strip()
        prev["speaker_coverage"] = min(
            float(prev.get("speaker_coverage") or 0.0), float(row.get("speaker_coverage") or 0.0)
        )
    return out


def _quality(row: dict[str, Any]) -> float:
    quality = 1.0
    quality -= min(0.6, float(row.get("overlap_ratio") or 0.0) * 2.5)
    quality *= max(0.45, min(1.0, float(row.get("speaker_coverage") or 0.0)))
    sim = row.get("speaker_similarity")
    if sim is not None:
        quality *= max(0.55, min(1.0, (float(sim) + 1.0) / 2.0))
    duration = float(row["end"]) - float(row["start"])
    if duration < 1.0:
        quality *= 0.55
    elif duration > 18.0:
        quality *= 0.75
    if "BGM" in set(row.get("events") or []):
        quality *= 0.75
    return round(max(0.0, min(1.0, quality)), 4)


def _identity_embeddings(repo_root: Path, paths: PersonaPaths) -> list[list[float]]:
    identity = media_files(paths.identity)
    if not identity:
        return []
    diar = worker(repo_root, "diarization")
    values = []
    for source in identity:
        key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
        cache = paths.cache / "identity" / f"{key}.json"
        result = _load_or_call(cache, lambda source=source: diar.call(repo_root, "embed", {"audio": str(source.resolve())}))
        embedding = result.get("embedding")
        if embedding:
            values.append([float(v) for v in embedding])
    return values


def _sense_tags(repo_root: Path, paths: PersonaPaths, audio: Path) -> dict[str, Any]:
    key = hashlib.sha256(audio.read_bytes()).hexdigest()[:20]
    cache = paths.cache / "sense" / f"{key}.json"
    sense = worker(repo_root, "sense")
    return _load_or_call(cache, lambda: sense.call(repo_root, "analyze", {"audio": str(audio.resolve()), "language": "ja"}))


def _select_references(paths: PersonaPaths, rows: list[dict[str, Any]], cfg: PersonaConfig) -> list[str]:
    shutil.rmtree(paths.references, ignore_errors=True)
    paths.references.mkdir(parents=True, exist_ok=True)
    candidates = [
        row for row in rows
        if row.get("target") and row.get("audio_path") and row.get("text") and row.get("quality", 0) >= 0.6
        and float(row["end"]) - float(row["start"]) <= cfg.prepare.reference_clip_max_seconds
    ]
    by_emotion: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(candidates, key=lambda r: float(r.get("quality", 0)), reverse=True):
        emotion = str(row.get("emotion") or "NEUTRAL").lower()
        by_emotion.setdefault(emotion, []).append(row)
    for emotion, items in by_emotion.items():
        directory = paths.references / "by_emotion" / emotion
        directory.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(items[:3], start=1):
            shutil.copy2(Path(row["audio_path"]), directory / f"ref_{index:02d}.flac")
    candidates.sort(
        key=lambda row: (str(row.get("emotion") or "NEUTRAL").upper() != "NEUTRAL", -float(row.get("quality", 0)))
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
        destination = paths.references / f"ref_{index:03d}_{row.get('emotion','neutral').lower()}.flac"
        shutil.copy2(source, destination)
        outputs.append(str(destination.resolve()))
    _dump(paths.references / "bank.json", {"files": outputs, "seconds": round(total, 2)})
    return outputs


def prepare_persona(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig, *, force: bool = False) -> dict[str, Any]:
    if not cfg.consent.authorized:
        raise PermissionError(
            "persona.yaml does not mark this voice as authorized. Confirm permission and set consent.authorized: true."
        )
    raw_files = media_files(paths.raw)
    if not raw_files:
        raise FileNotFoundError(f"No supported media was found in {paths.raw}")

    store = StateStore(paths.state)
    fingerprint = inventory_fingerprint(paths.raw) + ":" + json.dumps(cfg.prepare.model_dump(), sort_keys=True)
    fingerprint = hashlib.sha256(fingerprint.encode()).hexdigest()
    if not force and store.is_complete("prepare", fingerprint):
        return store.stage("prepare").get("result", {})

    with store.running("prepare", fingerprint):
        inv = inventory(paths.raw)
        _dump(paths.dataset / "source_inventory.json", inv)
        identity_embeddings = _identity_embeddings(repo_root, paths)
        asr_worker = worker(repo_root, "asr")
        diar_worker = worker(repo_root, "diarization")
        all_rows: list[dict[str, Any]] = []

        for source in inv:
            source_id = source["sha256"][:16]
            source_audio = paths.cache / "audio" / f"{source_id}.flac"
            if not source_audio.exists():
                extract_lossless_audio(Path(source["absolute_path"]), source_audio)

            asr_cache = paths.cache / "asr" / f"{source_id}.json"
            asr = _load_or_call(
                asr_cache,
                lambda: asr_worker.call(
                    repo_root,
                    "transcribe",
                    {"audio": str(source_audio.resolve()), "model": cfg.prepare.asr_model, "language": cfg.prepare.language},
                ),
            )
            diar_cache = paths.cache / "diarization" / f"{source_id}.json"
            diar = _load_or_call(
                diar_cache,
                lambda: diar_worker.call(repo_root, "diarize", {"audio": str(source_audio.resolve())}),
            )
            embeddings = {str(k): v for k, v in (diar.get("speaker_embeddings") or {}).items()}
            target_label, target_similarity = select_target_speaker(
                embeddings,
                identity_embeddings,
                threshold=cfg.prepare.min_identity_similarity,
            )
            exclusive = diar.get("exclusive_turns") or diar.get("turns") or []
            regular = diar.get("turns") or exclusive
            source_rows = _turn_rows(asr, exclusive)
            expanded: list[dict[str, Any]] = []
            for row in source_rows:
                expanded.extend(_split_long(row, max_seconds=cfg.prepare.max_clip_seconds))
            source_rows = _merge_rows(
                expanded,
                gap=cfg.prepare.merge_gap_seconds,
                max_seconds=cfg.prepare.max_clip_seconds,
            )

            for index, row in enumerate(source_rows):
                start = max(0.0, float(row["start"]))
                end = max(start + 0.05, float(row["end"]))
                target = row.get("speaker") == target_label
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
                    "overlap_ratio": overlap_ratio(start, end, regular),
                    "text": str(row.get("text") or "").strip(),
                    "emotion": None,
                    "events": [],
                    "caption": "",
                    "audio_path": None,
                }
                if target and (end - start) >= cfg.prepare.min_clip_seconds:
                    clip = paths.dataset / "clips" / f"{item['id']}.flac"
                    if not clip.exists():
                        cut_audio(source_audio, clip, start, end)
                    item["audio_path"] = str(clip.resolve())
                    if cfg.prepare.use_sensevoice:
                        sense = _sense_tags(repo_root, paths, clip)
                        item["emotion"] = sense.get("emotion")
                        item["events"] = normalize_events(sense.get("events") or [])
                    keep_nonverbal = cfg.prepare.keep_nonverbal_only and bool(item["events"])
                    if not item["text"] and not keep_nonverbal:
                        item["audio_path"] = None
                chars_per_second = len(item["text"]) / max(0.1, end - start) if item["text"] else None
                item["text_annotated"] = annotate_text(item["text"], item["events"])
                item["caption"] = build_caption(
                    emotion=item["emotion"], events=item["events"], chars_per_second=chars_per_second
                )
                item["quality"] = _quality(item)
                all_rows.append(item)

        master_db = paths.dataset / "master.sqlite3"
        replace_utterances(master_db, all_rows)
        _dump(paths.dataset / "master.json", all_rows)
        references = _select_references(paths, all_rows, cfg)
        irodori_count = export_irodori(master_db, paths.dataset / "irodori_source.jsonl", cfg.name)
        lfm_count = export_lfm(master_db, paths.dataset / "lfm_train.jsonl", cfg.name)
        seed_count = export_seed_vc(master_db, paths.dataset / "seed_vc")
        target_rows = [row for row in all_rows if row.get("target")]
        usable = [row for row in target_rows if row.get("audio_path") and row.get("text_annotated")]
        result = {
            "sources": len(inv),
            "utterances": len(all_rows),
            "target_utterances": len(target_rows),
            "usable_tts_utterances": len(usable),
            "usable_seconds": round(sum(float(r["end"]) - float(r["start"]) for r in usable), 2),
            "references": len(references),
            "irodori_examples": irodori_count,
            "lfm_examples": lfm_count,
            "seed_vc_examples": seed_count,
            "master_db": str(master_db.resolve()),
        }
        store.set_result("prepare", result)
        return result
