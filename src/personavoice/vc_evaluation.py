from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import wave
from array import array
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from personavoice.atomic import atomic_write_json, atomic_write_text
from personavoice.config import PersonaConfig, VCEvaluationConfig
from personavoice.dataset import write_jsonl
from personavoice.evaluation_metrics import character_error_rate, word_error_rate
from personavoice.inference import VC_BACKENDS, reenact
from personavoice.lineage import load_lineage, prepared_paths
from personavoice.project import PersonaPaths
from personavoice.speaker import cosine_similarity
from personavoice.workers import worker

VC_EVALUATION_SCHEMA_VERSION = 1
VC_MANIFEST_FILENAME = "vc_evaluation_manifest.jsonl"
VC_EVALUATION_OUTPUT_DIR = "vc-evaluation"
RECOMMENDED_SAMPLE_MINIMUM = 100
RECOMMENDED_SAMPLE_MAXIMUM = 300
VC_BUCKETS = ("normal_speech", "mixed_speech_event", "nonverbal_only")
VC_METRIC_KEYS = (
    "japanese_cer",
    "wer_secondary",
    "speaker_similarity",
    "duration_ratio",
    "duration_ratio_error",
    "f0_correlation",
    "voiced_unvoiced_f1",
    "pause_ratio_error",
    "speech_rate_ratio",
    "nonverbal_event_preservation_rate",
    "laughter_preservation_rate",
    "breath_preservation_rate",
    "mixed_speech_event_preservation_rate",
    "nonverbal_only_success_rate",
)

_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
_EVENT_ALIASES = {
    "laugh": "laughter",
    "laughter": "laughter",
    "laughing": "laughter",
    "笑い": "laughter",
    "笑": "laughter",
    "breath": "breath",
    "breathing": "breath",
    "息": "breath",
    "sigh": "sigh",
    "ため息": "sigh",
    "gasp": "gasp",
    "surprise": "gasp",
    "驚き": "gasp",
    "cry": "cry",
    "泣き": "cry",
    "cough": "cough",
    "咳": "cough",
    "throat": "throat",
    "喉": "throat",
}


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _persona_relative(paths: PersonaPaths, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(paths.root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"VC evaluation paths must stay inside the persona root: {path}"
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"VC evaluation path is not portable: {path}")
    return relative.as_posix()


def _resolve_persona_relative(paths: PersonaPaths, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty persona-relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in value:
        raise ValueError(f"{label} must be a portable persona-relative path: {value!r}")
    path = (paths.root / candidate).resolve()
    try:
        path.relative_to(paths.root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the persona root: {value!r}") from exc
    if not _nonempty_file(path):
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _master_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(
            "Prepared dataset is missing master.json. Run the authorized `persona prepare` "
            "stage before creating the VC evaluation manifest; no evaluation result is inferred."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Prepared master.json is unreadable: {path}") from exc
    if isinstance(value, dict):
        value = value.get("rows")
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError(f"Prepared master.json must contain a list of row objects: {path}")
    return value


def _event_name(value: object) -> str:
    raw = str(value).strip().casefold()
    if not raw:
        return ""
    compact = " ".join(raw.replace("_", " ").replace("-", " ").split())
    return _EVENT_ALIASES.get(compact, compact.replace(" ", "_"))


def _events(row: dict[str, Any]) -> list[str]:
    raw = row.get("events")
    if raw is None:
        raw = row.get("event")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    result = sorted({_event_name(value) for value in raw if _event_name(value)})
    return result


def _row_text(row: dict[str, Any]) -> str:
    for key in ("text", "text_annotated"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _bucket(text: str, events: list[str]) -> str:
    if events and text:
        return "mixed_speech_event"
    if events:
        return "nonverbal_only"
    return "normal_speech"


def _best_reference(paths: PersonaPaths) -> Path:
    references = sorted(
        path
        for path in paths.references.rglob("*")
        if path.is_file() and path.suffix.casefold() in _AUDIO_SUFFIXES and _nonempty_file(path)
    )
    if not references:
        raise RuntimeError(
            "A prepared target reference is missing. Put an authorized reference clip in "
            "references/ or pass --reference; VC A/B cannot use an inferred speaker."
        )
    return references[0]


def _canonical_sample(
    paths: PersonaPaths,
    row: dict[str, Any],
    *,
    reference: Path,
    seed: int,
) -> dict[str, Any] | None:
    if not bool(row.get("target", True)):
        return None
    raw_audio = row.get("audio_path") or row.get("audio")
    if not isinstance(raw_audio, str) or not raw_audio:
        return None
    source = Path(raw_audio).expanduser()
    if not source.is_absolute():
        source = paths.root / source
    source = source.resolve()
    if not _nonempty_file(source):
        return None
    if source.suffix.casefold() not in _AUDIO_SUFFIXES:
        return None
    text = _row_text(row)
    events = _events(row)
    source_id = str(row.get("source_id") or "unknown-source")
    raw_id = row.get("id")
    item_id = str(raw_id) if raw_id is not None else ""
    if not item_id:
        item_id = hashlib.sha256(
            f"{source_id}|{source}|{row.get('start')}|{row.get('end')}".encode()
        ).hexdigest()[:20]
    return {
        "schema_version": VC_EVALUATION_SCHEMA_VERSION,
        "id": item_id,
        "source_id": source_id,
        "source_audio": _persona_relative(paths, source),
        "reference_audio": _persona_relative(paths, reference),
        "source_sha256": _sha256(source),
        "reference_sha256": _sha256(reference),
        "text": text,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "language": str(row.get("language") or "ja"),
        "bucket": _bucket(text, events),
        "events": events,
        "selection_seed": seed,
        "source_start": row.get("start"),
        "source_end": row.get("end"),
        "duration": (
            float(row.get("end")) - float(row.get("start"))
            if isinstance(row.get("start"), (int, float))
            and isinstance(row.get("end"), (int, float))
            else None
        ),
        "transcript_hash": row.get("transcript_hash"),
        "alignment_provenance": {
            "asr_backend": row.get("asr_backend"),
            "asr_model_revision": row.get("asr_model_revision"),
            "alignment_backend": row.get("alignment_backend"),
            "alignment_model_revision": row.get("alignment_model_revision"),
            "alignment_hash": row.get("alignment_hash"),
        },
        "boundary_evidence": row.get("boundary_evidence"),
        "quality": float(row.get("quality", 1.0) or 0.0),
    }


def _select_samples(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in VC_BUCKETS}
    for row in rows:
        grouped[row["bucket"]].append(row)
    rng = random.Random(seed)
    for bucket in VC_BUCKETS:
        grouped[bucket].sort(key=lambda item: (item["source_id"], item["id"]))
        rng.shuffle(grouped[bucket])

    # Rare event buckets are consumed first. This is deterministic oversampling
    # of scarce categories, without duplicating a source clip or mutating data.
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(grouped.values()):
        for bucket in ("nonverbal_only", "mixed_speech_event", "normal_speech"):
            if grouped[bucket] and len(selected) < limit:
                selected.append(grouped[bucket].pop())
    selected.sort(key=lambda item: item["id"])
    return selected


def build_vc_evaluation_manifest(
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    output: Path | None = None,
    reference: str | Path | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build the immutable same-source/same-reference VC A/B input manifest."""

    # The manifest is a validation view over the newest candidate lineage.  It
    # must not accidentally read an old root-level master before activation.
    paths = prepared_paths(paths)

    language = cfg.language.strip().casefold()
    if language not in {"ja", "jpn", "japanese"}:
        raise RuntimeError(
            "The canonical VC gate is Japanese-only; configure persona language=ja. "
            "English/Chinese public benchmarks cannot authorize a Vevo2 default change."
        )
    evaluation = cfg.vc_evaluation
    sample_limit = evaluation.sample_count if limit is None else limit
    if sample_limit < 1 or sample_limit > RECOMMENDED_SAMPLE_MAXIMUM:
        raise ValueError("VC evaluation limit must be between 1 and 300")
    selection_seed = evaluation.seed if seed is None else seed
    if reference is None:
        reference_path = _best_reference(paths)
    else:
        candidate = Path(reference).expanduser()
        if not candidate.is_absolute():
            candidate = paths.root / candidate
        reference_path = candidate.resolve()
    if not _nonempty_file(reference_path):
        raise FileNotFoundError(f"VC evaluation reference is missing or empty: {reference_path}")
    master_path = paths.dataset / "master.json"
    candidates = []
    for row in _master_rows(master_path):
        sample = _canonical_sample(
            paths,
            row,
            reference=reference_path,
            seed=selection_seed,
        )
        if sample is None or str(sample["language"]).strip().casefold() not in {
            "ja",
            "jpn",
            "japanese",
        }:
            continue
        sample["language"] = "ja"
        if sample["quality"] < 0.55:
            continue
        candidates.append(sample)
    if not candidates:
        raise RuntimeError(
            "Prepared master.json contains no eligible target clips with audio. "
            "VC A/B status remains pending target-machine validation; no samples were fabricated."
        )
    selected = _select_samples(candidates, limit=sample_limit, seed=selection_seed)
    ids = [row["id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Prepared master.json contains duplicate evaluation clip IDs")
    lineage = load_lineage(paths)
    if lineage is not None:
        for row in selected:
            row.update(
                {
                    "lineage_id": lineage.get("lineage_id"),
                    "lineage_fingerprint": lineage.get("lineage_fingerprint"),
                    "master_fingerprint": lineage.get("master_fingerprint"),
                }
            )
    destination = output or paths.dataset / VC_MANIFEST_FILENAME
    destination = destination.expanduser().resolve()
    # Reject a non-portable destination before writing any manifest bytes.
    _persona_relative(paths, destination)
    write_jsonl(destination, selected)
    counts = Counter(row["bucket"] for row in selected)
    return {
        "schema_version": VC_EVALUATION_SCHEMA_VERSION,
        "manifest": str(destination),
        "manifest_relative": _persona_relative(paths, destination),
        "manifest_sha256": _sha256(destination),
        "sample_count": len(selected),
        "available_count": len(candidates),
        "selection_seed": selection_seed,
        "reference_audio": selected[0]["reference_audio"],
        "reference_sha256": selected[0]["reference_sha256"],
        "bucket_counts": dict(sorted(counts.items())),
        "recommended_range": [RECOMMENDED_SAMPLE_MINIMUM, RECOMMENDED_SAMPLE_MAXIMUM],
        "underpowered": len(selected) < RECOMMENDED_SAMPLE_MINIMUM,
        "status": (
            "pending target-machine validation"
            if len(selected) < RECOMMENDED_SAMPLE_MINIMUM
            else "ready_for_runtime_evaluation"
        ),
    }


def load_vc_manifest(
    paths: PersonaPaths,
    manifest: Path,
    *,
    verify_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Load and validate a canonical manifest before any worker is invoked."""

    path = manifest.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VC evaluation manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"VC evaluation manifest is unreadable: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"VC evaluation manifest line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict) or row.get("schema_version") != VC_EVALUATION_SCHEMA_VERSION:
            raise RuntimeError(f"VC evaluation manifest line {line_number} has an unsupported schema")
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise RuntimeError(f"VC evaluation manifest line {line_number} has a duplicate/invalid id")
        bucket = row.get("bucket")
        if bucket not in VC_BUCKETS:
            raise RuntimeError(f"VC evaluation manifest line {line_number} has an invalid bucket")
        events = row.get("events")
        if not isinstance(events, list) or not all(isinstance(item, str) for item in events):
            raise RuntimeError(f"VC evaluation manifest line {line_number} has invalid events")
        source = _resolve_persona_relative(paths, row.get("source_audio"), label="source_audio")
        reference = _resolve_persona_relative(
            paths,
            row.get("reference_audio"),
            label="reference_audio",
        )
        for key in ("source_sha256", "reference_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise RuntimeError(f"VC evaluation manifest line {line_number} has invalid {key}")
        if verify_hashes:
            if _sha256(source) != row["source_sha256"]:
                raise RuntimeError(f"VC evaluation source checksum mismatch on line {line_number}")
            if _sha256(reference) != row["reference_sha256"]:
                raise RuntimeError(
                    f"VC evaluation reference checksum mismatch on line {line_number}"
                )
        if not isinstance(row.get("text"), str) or not isinstance(row.get("language"), str):
            raise RuntimeError(f"VC evaluation manifest line {line_number} has invalid text/language")
        text_hash = row.get("text_hash")
        if text_hash is not None and (
            not isinstance(text_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", text_hash)
            or hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != text_hash
        ):
            raise RuntimeError(f"VC evaluation manifest line {line_number} has an invalid text hash")
        duration = row.get("duration")
        if duration is not None and (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise RuntimeError(f"VC evaluation manifest line {line_number} has an invalid duration")
        if not isinstance(row.get("source_id"), str) or not row["source_id"]:
            raise RuntimeError(f"VC evaluation manifest line {line_number} has no source_id")
        lineage_fields = tuple(
            row.get(key)
            for key in ("lineage_id", "lineage_fingerprint", "master_fingerprint")
        )
        if any(value is not None for value in lineage_fields) and (
            not isinstance(lineage_fields[0], str)
            or not re.fullmatch(r"pl-[0-9a-f]{32}", lineage_fields[0])
            or not isinstance(lineage_fields[1], str)
            or not re.fullmatch(r"[0-9a-f]{64}", lineage_fields[1])
            or not isinstance(lineage_fields[2], str)
            or not re.fullmatch(r"[0-9a-f]{64}", lineage_fields[2])
        ):
            raise RuntimeError(
                f"VC evaluation manifest line {line_number} has invalid Prepare lineage metadata"
            )
        seen.add(item_id)
        row["source_path"] = source
        row["reference_path"] = reference
        rows.append(row)
    if not rows:
        raise RuntimeError(
            "VC evaluation manifest is empty; status is pending target-machine validation."
        )
    return rows


def _decode_audio(path: Path, *, sample_rate: int = 16000) -> tuple[list[float], int]:
    """Decode through the pinned FFmpeg path, with a WAV-only stdlib fallback."""

    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.readframes(handle.getnframes())
        values = array("h")
        values.frombytes(frames)
        if sys.byteorder != "little":
            values.byteswap()
        if channels > 1:
            mono = [
                sum(values[index : index + channels]) / channels
                for index in range(0, len(values) - channels + 1, channels)
            ]
        else:
            mono = list(values)
        if rate == sample_rate:
            return [float(value) / 32768.0 for value in mono], rate
        # The stdlib fallback is intentionally nearest-neighbour and is used
        # only for simple WAV smoke tests; production decoding uses FFmpeg.
        length = max(1, round(len(mono) * sample_rate / rate))
        return [float(mono[min(len(mono) - 1, round(index * rate / sample_rate))]) / 32768.0 for index in range(length)], sample_rate
    except (OSError, EOFError, wave.Error, ValueError):
        pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Cannot decode {path}: it is not a readable WAV and FFmpeg is unavailable"
        )
    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg could not decode {path}: {detail or 'unknown error'}")
    values = array("h")
    values.frombytes(completed.stdout)
    if sys.byteorder != "little":
        values.byteswap()
    return [float(value) / 32768.0 for value in values], sample_rate


def _audio_features(path: Path) -> dict[str, Any]:
    values, rate = _decode_audio(path)
    if not values:
        raise RuntimeError(f"Audio has no samples: {path}")
    frame = max(1, round(rate * 0.025))
    hop = max(1, round(rate * 0.010))
    rms: list[float] = []
    f0: list[float | None] = []
    for start in range(0, max(1, len(values) - frame + 1), hop):
        window = values[start : start + frame]
        if not window:
            continue
        energy = math.sqrt(sum(value * value for value in window) / len(window))
        rms.append(energy)
        crossings = sum(
            1
            for left, right in zip(window, window[1:], strict=False)
            if (left <= 0 < right) or (right <= 0 < left)
        )
        estimate = crossings * rate / (2.0 * max(1, len(window)))
        f0.append(estimate if energy >= 0.008 and 50.0 <= estimate <= 1200.0 else None)
    peak = max(rms, default=0.0)
    threshold = max(0.008, peak * 0.12)
    voiced = [energy >= threshold for energy in rms]
    voiced_frames = sum(voiced)
    duration = len(values) / rate
    return {
        "duration_seconds": duration,
        "voiced": voiced,
        "f0": f0,
        "pause_ratio": 1.0 - voiced_frames / max(1, len(voiced)),
        "speech_rate": voiced_frames / max(0.001, duration),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 1e-12 or right_var <= 1e-12:
        return None
    value = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / math.sqrt(left_var * right_var)
    return value if math.isfinite(value) else None


def _voiced_f1(reference: list[bool], generated: list[bool]) -> float | None:
    size = min(len(reference), len(generated))
    if size == 0:
        return None
    reference = reference[:size]
    generated = generated[:size]
    true_positive = sum(a and b for a, b in zip(reference, generated, strict=True))
    false_positive = sum((not a) and b for a, b in zip(reference, generated, strict=True))
    false_negative = sum(a and (not b) for a, b in zip(reference, generated, strict=True))
    if true_positive == 0:
        return 1.0 if false_positive == 0 and false_negative == 0 else 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def _signal_metrics(reference: Path, generated: Path) -> dict[str, float | None]:
    source = _audio_features(reference)
    output = _audio_features(generated)
    f0_pairs = [
        (left, right)
        for left, right in zip(source["f0"], output["f0"], strict=False)
        if left is not None and right is not None
    ]
    duration_ratio = output["duration_seconds"] / max(1e-9, source["duration_seconds"])
    return {
        "duration_seconds": output["duration_seconds"],
        "duration_ratio": duration_ratio,
        "duration_ratio_error": abs(1.0 - duration_ratio),
        "f0_correlation": _pearson(
            [pair[0] for pair in f0_pairs],
            [pair[1] for pair in f0_pairs],
        ),
        "voiced_unvoiced_f1": _voiced_f1(source["voiced"], output["voiced"]),
        "pause_ratio_error": abs(output["pause_ratio"] - source["pause_ratio"]),
        "speech_rate_ratio": output["speech_rate"] / max(1e-9, source["speech_rate"]),
    }


def _transcript(value: dict[str, Any]) -> str:
    return "".join(
        str(segment.get("text") or "")
        for segment in value.get("segments", [])
        if isinstance(segment, dict)
    ).strip()


def _detected_events(value: dict[str, Any]) -> list[str]:
    raw = value.get("events") if isinstance(value, dict) else []
    if not isinstance(raw, list):
        return []
    return sorted({_event_name(item) for item in raw if _event_name(item)})


def _event_score(expected: list[str], detected: list[str]) -> float | None:
    if not expected:
        return None
    expected_set = set(expected)
    return len(expected_set & set(detected)) / len(expected_set)


def _finite_mean(values: list[object]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return sum(numbers) / len(numbers) if numbers else None


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[object]:
    return [row.get("metrics", {}).get(key) for row in rows]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("ok")]
    event_rows = [
        row
        for row in successful
        if row.get("events", {}).get("expected")
    ]
    laughter_rows = [
        row
        for row in event_rows
        if "laughter" in row["events"].get("expected", [])
    ]
    breath_rows = [
        row for row in event_rows if "breath" in row["events"].get("expected", [])
    ]
    mixed_rows = [row for row in event_rows if row.get("bucket") == "mixed_speech_event"]
    nonverbal_only = [row for row in rows if row.get("bucket") == "nonverbal_only"]
    event_metric = lambda selected: _finite_mean(  # noqa: E731
        [row.get("events", {}).get("preservation") for row in selected]
    )
    values: dict[str, float | None] = {
        "japanese_cer": _finite_mean(_metric_values(successful, "cer")),
        "wer_secondary": _finite_mean(_metric_values(successful, "wer")),
        "speaker_similarity": _finite_mean(_metric_values(successful, "speaker_similarity")),
        "duration_ratio": _finite_mean(_metric_values(successful, "duration_ratio")),
        "duration_ratio_error": _finite_mean(
            _metric_values(successful, "duration_ratio_error")
        ),
        "f0_correlation": _finite_mean(_metric_values(successful, "f0_correlation")),
        "voiced_unvoiced_f1": _finite_mean(
            _metric_values(successful, "voiced_unvoiced_f1")
        ),
        "pause_ratio_error": _finite_mean(_metric_values(successful, "pause_ratio_error")),
        "speech_rate_ratio": _finite_mean(_metric_values(successful, "speech_rate_ratio")),
        "nonverbal_event_preservation_rate": event_metric(event_rows),
        "laughter_preservation_rate": event_metric(laughter_rows),
        "breath_preservation_rate": event_metric(breath_rows),
        "mixed_speech_event_preservation_rate": event_metric(mixed_rows),
        "nonverbal_only_success_rate": (
            sum(bool(row.get("ok")) for row in nonverbal_only) / len(nonverbal_only)
            if nonverbal_only
            else None
        ),
    }
    counts = {
        "samples": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "metric_counts": {
            key: sum(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in _metric_values(successful, key)
            )
            for key in (
                "cer",
                "wer",
                "speaker_similarity",
                "duration_ratio_error",
                "f0_correlation",
                "voiced_unvoiced_f1",
                "pause_ratio_error",
            )
        },
        "bucket_counts": dict(sorted(Counter(row.get("bucket") for row in rows).items())),
    }
    return {"metrics": values, "counts": counts}


def _human_review_complete(path: Path | None) -> tuple[bool, dict[str, Any]]:
    if path is None or not path.is_file():
        return False, {"status": "pending target-machine validation", "path": None}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, {"status": "invalid", "path": str(path)}
    complete = isinstance(value, dict) and value.get("status") == "complete"
    return complete, {
        "status": "complete" if complete else "pending target-machine validation",
        "path": str(path),
        "reviewer": value.get("reviewer") if isinstance(value, dict) else None,
    }


def _gate(
    aggregates: dict[str, dict[str, Any]],
    *,
    sample_count: int,
    policy: VCEvaluationConfig,
    human_complete: bool,
) -> dict[str, Any]:
    seed = aggregates["seed-vc-v2"]["metrics"]
    vevo = aggregates["vevo2-fm"]["metrics"]
    checks: list[dict[str, Any]] = []

    def compare(
        name: str,
        *,
        higher_is_better: bool,
        tolerance: float,
    ) -> None:
        baseline = seed.get(name)
        candidate = vevo.get(name)
        if not isinstance(baseline, (int, float)) or not isinstance(candidate, (int, float)):
            checks.append(
                {"metric": name, "pass": False, "status": "missing", "baseline": baseline, "vevo2": candidate}
            )
            return
        if higher_is_better:
            passed = candidate >= baseline - tolerance
        else:
            passed = candidate <= baseline + tolerance
        checks.append(
            {
                "metric": name,
                "pass": passed,
                "status": "pass" if passed else "material_regression",
                "baseline": baseline,
                "vevo2": candidate,
                "allowed_regression": tolerance,
            }
        )

    compare("japanese_cer", higher_is_better=False, tolerance=policy.max_cer_regression)
    compare(
        "speaker_similarity",
        higher_is_better=True,
        tolerance=policy.max_speaker_similarity_regression,
    )
    compare(
        "duration_ratio_error",
        higher_is_better=False,
        tolerance=policy.max_duration_ratio_error_regression,
    )
    compare(
        "f0_correlation",
        higher_is_better=True,
        tolerance=policy.max_f0_correlation_regression,
    )
    compare(
        "voiced_unvoiced_f1",
        higher_is_better=True,
        tolerance=policy.max_voiced_unvoiced_f1_regression,
    )
    compare(
        "pause_ratio_error",
        higher_is_better=False,
        tolerance=policy.max_pause_ratio_error_regression,
    )
    for event_metric in (
        "nonverbal_event_preservation_rate",
        "laughter_preservation_rate",
        "breath_preservation_rate",
        "mixed_speech_event_preservation_rate",
        "nonverbal_only_success_rate",
    ):
        compare(
            event_metric,
            higher_is_better=True,
            tolerance=policy.max_nonverbal_event_regression,
        )
    reasons: list[str] = []
    for backend, value in aggregates.items():
        failed = int(value["counts"].get("failed", 0))
        if failed:
            reasons.append(f"{backend} has {failed} failed sample(s)")
        missing_buckets = [
            bucket
            for bucket in VC_BUCKETS
            if int(value["counts"].get("bucket_counts", {}).get(bucket, 0)) == 0
        ]
        if missing_buckets:
            reasons.append(f"{backend} has no samples in bucket(s): {', '.join(missing_buckets)}")
    improvements = [
        item
        for item in checks
        if item.get("pass")
        and isinstance(item.get("baseline"), (int, float))
        and isinstance(item.get("vevo2"), (int, float))
        and item["baseline"] != item["vevo2"]
        and (
            item["vevo2"] > item["baseline"]
            if item["metric"]
            in {
                "speaker_similarity",
                "f0_correlation",
                "voiced_unvoiced_f1",
                "nonverbal_event_preservation_rate",
                "laughter_preservation_rate",
                "breath_preservation_rate",
                "mixed_speech_event_preservation_rate",
                "nonverbal_only_success_rate",
            }
            else item["vevo2"] < item["baseline"]
        )
    ]
    if sample_count < RECOMMENDED_SAMPLE_MINIMUM:
        reasons.append(
            f"only {sample_count} samples; at least {RECOMMENDED_SAMPLE_MINIMUM} are recommended"
        )
    if policy.require_human_review and not human_complete:
        reasons.append("human listening review is incomplete")
    if not improvements:
        reasons.append("Vevo2 has no measured clear improvement over Seed-VC")
    passed = (
        sample_count >= RECOMMENDED_SAMPLE_MINIMUM
        and all(item.get("pass") is True for item in checks)
        and all(int(value["counts"].get("failed", 0)) == 0 for value in aggregates.values())
        and all(
            int(value["counts"].get("bucket_counts", {}).get(bucket, 0)) > 0
            for value in aggregates.values()
            for bucket in VC_BUCKETS
        )
        and (human_complete or not policy.require_human_review)
        and bool(improvements)
    )
    status = "passed" if passed else "pending target-machine validation"
    if sample_count >= RECOMMENDED_SAMPLE_MINIMUM and any(
        item.get("status") == "material_regression" for item in checks
    ):
        status = "failed"
    return {
        "status": status,
        "passed": passed,
        "recommended_default": "vevo2-fm" if passed else "seed-vc-v2",
        "default_changed": False,
        "checks": checks,
        "clear_improvements": improvements,
        "reasons": reasons,
        "human_review_required": policy.require_human_review,
        "human_review_complete": human_complete,
    }


def _report_markdown(report: dict[str, Any]) -> str:
    gate = report["decision_gate"]
    lines = [
        "# Vevo2 vs Seed-VC canonical VC A/B evaluation",
        "",
        f"- Status: **{report['status']}**",
        f"- Manifest: `{report['manifest']['path']}`",
        f"- Samples: `{report['manifest']['count']}`",
        f"- Same manifest: `{report['same_input_contract']['same_manifest']}`",
        f"- Recommended default: `{gate['recommended_default']}`",
        f"- Default changed automatically: `{gate['default_changed']}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Seed-VC | Vevo2 FM |",
        "|---|---:|---:|",
    ]
    seed = report["results"]["seed-vc-v2"]["aggregates"]["metrics"]
    vevo = report["results"]["vevo2-fm"]["aggregates"]["metrics"]
    for key in VC_METRIC_KEYS:
        lines.append(f"| `{key}` | {seed.get(key)!r} | {vevo.get(key)!r} |" )
    lines += ["", "## Decision gate", "", f"- Status: **{gate['status']}**"]
    for item in gate["checks"]:
        lines.append(
            f"- `{item['metric']}`: **{item['status']}** "
            f"(Seed={item.get('baseline')!r}, Vevo2={item.get('vevo2')!r})"
        )
    lines += ["", "## Human listening", "", f"- {report['human_listening']['status']}", ""]
    for item in report["human_listening"]["samples"]:
        lines.append(f"- `{item['id']}` ({item['bucket']}): {item['status']}")
        for backend, path in item.get("outputs", {}).items():
            lines.append(f"  - {backend}: `{path}`")
    lines += [
        "",
        "The default remains Seed-VC until a completed Japanese/non-verbal gate is reviewed.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_vc(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    manifest: Path,
    *,
    output_dir: Path | None = None,
    human_review: Path | None = None,
) -> dict[str, Any]:
    """Run both VC backends against exactly the same immutable inputs."""

    # Evaluation happens before explicit activation, so select the candidate
    # recorded by Prepare while preserving the base persona root for portable
    # manifest paths and rollback-safe runtime selection.
    paths = prepared_paths(paths)

    rows = load_vc_manifest(paths, manifest, verify_hashes=True)
    lineage_ids = {row.get("lineage_id") for row in rows if row.get("lineage_id") is not None}
    if len(lineage_ids) > 1:
        raise RuntimeError("VC evaluation manifest mixes multiple Prepare lineages")
    if lineage_ids:
        manifest_lineage = next(iter(lineage_ids))
        if paths.lineage_id is not None and paths.lineage_id != manifest_lineage:
            raise RuntimeError(
                "VC evaluation manifest belongs to a different Prepare lineage; regenerate it"
            )
        if paths.lineage_id is None:
            paths = paths.for_lineage(str(manifest_lineage))
            rows = load_vc_manifest(paths, manifest, verify_hashes=True)
    manifest_path = manifest.expanduser().resolve()
    manifest_relative = _persona_relative(paths, manifest_path)
    evaluation_root = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (paths.outputs / VC_EVALUATION_OUTPUT_DIR / _utc_stamp()).resolve()
    )
    _persona_relative(paths, evaluation_root)
    if human_review is not None:
        _persona_relative(paths, human_review.expanduser().resolve())
    evaluation_root.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, list[dict[str, Any]]] = {}
    reference_embedding_cache: dict[str, list[float] | None] = {}
    for backend in VC_BACKENDS:
        backend_dir = evaluation_root / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        backend_results: list[dict[str, Any]] = []
        for row in rows:
            result: dict[str, Any] = {
                "id": row["id"],
                "bucket": row["bucket"],
                "source_id": row["source_id"],
                "source_sha256": row["source_sha256"],
                "reference_sha256": row["reference_sha256"],
                "expected_events": row["events"],
                "ok": False,
                "output": None,
                "output_sha256": None,
                "metrics": {},
                "events": {"expected": row["events"], "detected": [], "preservation": None},
                "errors": [],
            }
            output_path: Path | None = None
            try:
                output_path = reenact(
                    repo_root,
                    paths,
                    cfg,
                    row["source_path"],
                    ref=row["reference_path"],
                    transfer_style=True,
                    backend=backend,
                    output_dir=backend_dir,
                    metadata={
                        "evaluation_id": row["id"],
                        "manifest_sha256": _sha256(manifest_path),
                        "source_sha256": row["source_sha256"],
                        "reference_sha256": row["reference_sha256"],
                    },
                )
                output_path = output_path.resolve()
                if not _nonempty_file(output_path):
                    raise RuntimeError("backend returned a missing/empty output")
                output_path.relative_to(evaluation_root)
                result["output"] = _persona_relative(paths, output_path)
                result["output_sha256"] = _sha256(output_path)
                result["metrics"].update(
                    _signal_metrics(row["source_path"], output_path)
                )
                if row["text"]:
                    asr_result = worker(repo_root, "asr").call(
                        repo_root,
                        "transcribe",
                        {
                            "audio": str(output_path),
                            "model": cfg.prepare.asr_model,
                            "compute_type": cfg.prepare.asr_compute_type,
                            "device": cfg.prepare.asr_device,
                            "dtype": cfg.prepare.asr_dtype,
                            "language": row["language"],
                        },
                    )
                    hypothesis = _transcript(asr_result)
                    result["asr"] = {"reference": row["text"], "hypothesis": hypothesis}
                    result["metrics"]["cer"] = character_error_rate(row["text"], hypothesis)
                    result["metrics"]["wer"] = word_error_rate(row["text"], hypothesis)
                reference_key = row["reference_sha256"]
                if reference_key not in reference_embedding_cache:
                    reference_embedding_cache[reference_key] = worker(
                        repo_root, "diarization"
                    ).call(
                        repo_root,
                        "embed",
                        {"audio": str(row["reference_path"])},
                    ).get("embedding")
                reference_embedding = reference_embedding_cache[reference_key]
                generated_embedding = worker(repo_root, "diarization").call(
                    repo_root,
                    "embed",
                    {"audio": str(output_path)},
                ).get("embedding")
                if isinstance(reference_embedding, list) and isinstance(generated_embedding, list):
                    result["metrics"]["speaker_similarity"] = cosine_similarity(
                        reference_embedding,
                        generated_embedding,
                    )
                if row["events"]:
                    sense_result = worker(repo_root, "sense").call(
                        repo_root,
                        "analyze",
                        {"audio": str(output_path), "language": row["language"]},
                    )
                    detected = _detected_events(sense_result)
                    result["events"] = {
                        "expected": row["events"],
                        "detected": detected,
                        "preservation": _event_score(row["events"], detected),
                    }
                result["ok"] = True
            except Exception as exc:
                result["errors"].append(f"{type(exc).__name__}: {exc}")
            backend_results.append(result)
        output_rows[backend] = backend_results

    aggregates = {
        backend: _aggregate(output_rows[backend]) for backend in VC_BACKENDS
    }
    human_complete, human_info = _human_review_complete(human_review)
    human_samples = []
    for row in rows:
        outputs = {
            backend: next(
                (
                    item["output"]
                    for item in output_rows[backend]
                    if item["id"] == row["id"] and item.get("output")
                ),
                None,
            )
            for backend in VC_BACKENDS
        }
        human_samples.append(
            {
                "id": row["id"],
                "bucket": row["bucket"],
                "outputs": outputs,
                "status": "reviewed" if human_complete else "pending target-machine validation",
            }
        )
    gate = _gate(
        aggregates,
        sample_count=len(rows),
        policy=cfg.vc_evaluation,
        human_complete=human_complete,
    )
    report_dir = evaluation_root
    report = {
        "schema_version": VC_EVALUATION_SCHEMA_VERSION,
        "status": gate["status"],
        "created_at": datetime.now(UTC).isoformat(),
        "persona": cfg.name,
        "language": cfg.language,
        "manifest": {
            "path": manifest_relative,
            "sha256": _sha256(manifest_path),
            "count": len(rows),
            "selection_seed": rows[0].get("selection_seed"),
            "recommended_range": [RECOMMENDED_SAMPLE_MINIMUM, RECOMMENDED_SAMPLE_MAXIMUM],
            "underpowered": len(rows) < RECOMMENDED_SAMPLE_MINIMUM,
        },
        "backends": list(VC_BACKENDS),
        "same_input_contract": {
            "same_manifest": True,
            "same_source_per_sample": True,
            "same_reference_per_sample": True,
            "source_reference_hashes_verified": True,
            "source_mutation": False,
        },
        "provenance": {
            "source_and_reference_hashes": [
                {
                    "id": row["id"],
                    "source_sha256": row["source_sha256"],
                    "reference_sha256": row["reference_sha256"],
                }
                for row in rows
            ],
            "output_root": _persona_relative(paths, report_dir),
        },
        "results": {
            backend: {"samples": output_rows[backend], "aggregates": aggregates[backend]}
            for backend in VC_BACKENDS
        },
        "human_listening": {
            **human_info,
            "samples": human_samples,
            "instructions": (
                "Review intelligibility, target-speaker identity, prosody/timing, pauses, "
                "and non-verbal event preservation for matched Seed-VC/Vevo2 pairs."
            ),
        },
        "decision_gate": gate,
        "commands": [
            "uv run --locked persona eval-vc-manifest <persona>",
            "uv run --locked persona eval-vc <persona> --manifest <manifest>",
        ],
    }
    json_path = report_dir / "report.json"
    markdown_path = report_dir / "report.md"
    human_review_path = report_dir / "human_review.json"
    if not human_review_path.exists():
        atomic_write_json(
            human_review_path,
            {
                "schema_version": VC_EVALUATION_SCHEMA_VERSION,
                "status": "pending target-machine validation",
                "reviewer": None,
                "samples": human_samples,
            },
        )
    report["human_listening"]["generated_review_form"] = _persona_relative(paths, human_review_path)
    report["report"] = _persona_relative(paths, json_path)
    report["report_markdown"] = _persona_relative(paths, markdown_path)
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, _report_markdown(report))
    return report


__all__ = [
    "VC_BUCKETS",
    "VC_EVALUATION_SCHEMA_VERSION",
    "VC_MANIFEST_FILENAME",
    "build_vc_evaluation_manifest",
    "evaluate_vc",
    "load_vc_manifest",
]
