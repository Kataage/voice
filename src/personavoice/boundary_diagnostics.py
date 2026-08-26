"""Deterministic, inference-only diagnostics for Irodori boundary defects.

The diagnostic runner deliberately lives outside the prepare/training pipeline.
It can therefore compare already-trained checkpoints without changing a
prepare fingerprint, latent manifest, or training family fingerprint.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from personavoice.atomic import atomic_write_json
from personavoice.config import PersonaConfig
from personavoice.evaluation_metrics import (
    character_error_rate,
    normalize_text,
    tokenize_words,
    word_error_rate,
)
from personavoice.irodori import reference_files, speaker_embedding_complete
from personavoice.model_assets import IRODORI_SOURCE_REVISION
from personavoice.project import PersonaPaths
from personavoice.workers import worker

BOUNDARY_DIAGNOSTIC_SCHEMA = 1
_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _safe_id(value: str) -> str:
    normalized = _ID_RE.sub("-", value.strip()).strip("-.")
    return normalized or "item"


def _sha256_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_scale(value: float, *, label: str) -> float:
    result = _finite(value)
    if result is None or result <= 0:
        raise ValueError(f"{label} must be a finite value greater than zero")
    return result


def build_inference_diagnostic_matrix(
    *,
    margin_scale: float = 1.10,
) -> tuple[dict[str, Any], ...]:
    """Build the deterministic A/B duration and tail matrix from Issue #33.

    The margin is an evaluated candidate, not a runtime policy. The returned
    order is stable and intentionally follows A/B/C/D from the issue:
    ``1.00 + trim``, ``1.00 - trim``, ``margin + trim``, ``margin - trim``.
    """

    margin = _validate_scale(margin_scale, label="margin_scale")
    if math.isclose(margin, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("margin_scale must differ from the baseline 1.0")
    return (
        {"id": "A", "duration_scale": 1.0, "trim_tail": True},
        {"id": "B", "duration_scale": 1.0, "trim_tail": False},
        {"id": "C", "duration_scale": margin, "trim_tail": True},
        {"id": "D", "duration_scale": margin, "trim_tail": False},
    )


def build_leading_artifact_condition_matrix(
    paths: PersonaPaths,
) -> tuple[dict[str, Any], ...]:
    """Build the fixed conditioning matrix used for onset isolation.

    The four baseline conditions always exist. Reference-specific conditions
    are added only when the corresponding trained/reference artifact is
    actually available, so a missing optional conditioning path is reported as
    unavailable rather than being silently substituted.
    """

    conditions: list[dict[str, Any]] = [
        {
            "id": "persona-auto-reference-caption",
            "base_only": False,
            "reference_mode": "auto",
            "caption_conditioning": True,
        },
        {
            "id": "persona-no-reference-caption",
            "base_only": False,
            "reference_mode": "none",
            "caption_conditioning": True,
        },
        {
            "id": "persona-auto-reference-no-caption",
            "base_only": False,
            "reference_mode": "auto",
            "caption_conditioning": False,
        },
        {
            "id": "base-no-reference-no-caption",
            "base_only": True,
            "reference_mode": "none",
            "caption_conditioning": False,
        },
    ]
    if any(path.is_file() and path.stat().st_size > 0 for path in reference_files(paths.references)):
        conditions.append(
            {
                "id": "persona-audio-reference-caption",
                "base_only": False,
                "reference_mode": "audio",
                "caption_conditioning": True,
            }
        )
    speaker = paths.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors"
    if speaker_embedding_complete(speaker):
        conditions.append(
            {
                "id": "persona-speaker-embedding-caption",
                "base_only": False,
                "reference_mode": "speaker-embed",
                "caption_conditioning": True,
            }
        )
    return tuple(conditions)


DEFAULT_BOUNDARY_EVALUATION_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "onset-probe",
        "text": "おはよう、今日もよろしくね。",
        "emotion": "NEUTRAL",
        "events": [],
    },
    {
        "id": "short",
        "text": "はい。",
        "emotion": "NEUTRAL",
        "events": [],
    },
    {
        "id": "long",
        "text": "遠くの町で見つけた小さな灯りについて、ゆっくり詳しく説明するね。",
        "emotion": "NEUTRAL",
        "events": [],
    },
    {
        "id": "final-vowel",
        "text": "そうだよ、これで大丈夫。",
        "emotion": "HAPPY",
        "events": [],
    },
    {
        "id": "closure",
        "text": "きっと最後までやり切ります。",
        "emotion": "DETERMINED",
        "events": [],
    },
    {
        "id": "fast-emotional",
        "text": "本当に見つかったの、すごい！",
        "emotion": "SURPRISED",
        "events": [],
    },
    {
        "id": "punctuation",
        "text": "えっ、今の音はなに？……もう一度聞いてみよう。",
        "emotion": "SURPRISED",
        "events": [],
    },
    {
        "id": "breath",
        "text": "少し息を整えてから、続けるね。",
        "emotion": "CALM",
        "events": ["breath"],
    },
    {
        "id": "laughter",
        "text": "あはは、そんなこともあるよね。",
        "emotion": "HAPPY",
        "events": ["laugh"],
    },
)


def _decode_pcm(raw: bytes, *, sample_width: int, channels: int) -> list[float]:
    if channels <= 0:
        raise ValueError("WAV channel count must be positive")
    frame_width = sample_width * channels
    if frame_width <= 0 or len(raw) % frame_width:
        raise ValueError("WAV payload has an incomplete PCM frame")
    values: list[float] = []
    for offset in range(0, len(raw), frame_width):
        total = 0.0
        for channel in range(channels):
            start = offset + channel * sample_width
            chunk = raw[start : start + sample_width]
            if sample_width == 1:
                sample = (chunk[0] - 128) / 128.0
            elif sample_width == 2:
                sample = struct.unpack_from("<h", chunk)[0] / 32768.0
            elif sample_width == 3:
                unsigned = int.from_bytes(chunk, "little", signed=False)
                if unsigned & 0x800000:
                    unsigned -= 1 << 24
                sample = unsigned / 8388608.0
            elif sample_width == 4:
                sample = struct.unpack_from("<i", chunk)[0] / 2147483648.0
            else:
                raise ValueError(f"Unsupported PCM sample width: {sample_width}")
            total += sample
        values.append(total / channels)
    return values


def _rms_envelope(samples: list[float], sample_rate: int) -> tuple[list[float], int, int]:
    if sample_rate <= 0:
        raise ValueError("WAV sample rate must be positive")
    if not samples:
        raise ValueError("WAV has no PCM samples")
    window = max(1, round(sample_rate * 0.020))
    hop = max(1, round(sample_rate * 0.010))
    envelope: list[float] = []
    for start in range(0, len(samples), hop):
        chunk = samples[start : start + window]
        if not chunk:
            break
        envelope.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)))
    return envelope, window, hop


def analyze_wav_boundary(path: Path) -> dict[str, Any]:
    """Extract deterministic onset/offset energy evidence from a PCM WAV."""

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    samples = _decode_pcm(raw, sample_width=sample_width, channels=channels)
    envelope, window, hop = _rms_envelope(samples, sample_rate)
    ordered = sorted(envelope)
    noise_floor = ordered[max(0, int((len(ordered) - 1) * 0.10))]
    peak = max(envelope)
    threshold = max(1e-4, noise_floor * 3.0, peak * 0.05)
    voiced = [index for index, value in enumerate(envelope) if value >= threshold]
    first = voiced[0] if voiced else None
    last = voiced[-1] if voiced else None

    def seconds(index: int | None) -> float | None:
        return None if index is None else round(index * hop / sample_rate, 6)

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "sample_count": len(samples),
        "duration_seconds": round(len(samples) / sample_rate, 6),
        "frame_window_samples": window,
        "frame_hop_samples": hop,
        "voiced_threshold_rms": round(threshold, 8),
        "peak_rms": round(peak, 8),
        "noise_floor_rms": round(noise_floor, 8),
        "first_voiced_frame": first,
        "last_voiced_frame": last,
        "first_voiced_seconds": seconds(first),
        "last_voiced_seconds": seconds(last),
        "energy_envelope_frames": len(envelope),
        "final_energy_envelope": [round(value, 8) for value in envelope[-32:]],
    }


def _asr_bounds(asr: dict[str, Any]) -> tuple[float | None, float | None]:
    starts: list[float] = []
    ends: list[float] = []
    for segment in asr.get("segments", []):
        if not isinstance(segment, dict):
            continue
        words = segment.get("words")
        rows = words if isinstance(words, list) and words else [segment]
        for row in rows:
            if not isinstance(row, dict):
                continue
            start = _finite(row.get("start"))
            end = _finite(row.get("end"))
            if start is not None and end is not None and end >= start:
                starts.append(start)
                ends.append(end)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _transcript(asr: dict[str, Any]) -> str:
    return "".join(
        str(segment.get("text") or "")
        for segment in asr.get("segments", [])
        if isinstance(segment, dict)
    ).strip()


def _final_token_evidence(expected: str, hypothesis: str) -> dict[str, Any]:
    expected_normalized = normalize_text(expected, remove_whitespace=True)
    hypothesis_normalized = normalize_text(hypothesis, remove_whitespace=True)
    expected_words = tokenize_words(expected)
    expected_token = expected_words[-1] if expected_words else ""
    final_token_preserved = bool(
        expected_token and hypothesis_normalized.endswith(normalize_text(expected_token, remove_whitespace=True))
    )
    suffix_length = min(3, len(expected_normalized))
    expected_suffix = expected_normalized[-suffix_length:] if suffix_length else ""
    return {
        "expected_final_token": expected_token,
        "expected_final_characters": expected_suffix,
        "hypothesis_final_characters": hypothesis_normalized[-suffix_length:]
        if suffix_length
        else "",
        "final_token_preserved": final_token_preserved,
        "final_character_suffix_preserved": bool(
            expected_suffix and hypothesis_normalized.endswith(expected_suffix)
        ),
    }


def _parse_duration_log(stdout: str) -> dict[str, Any]:
    prediction = re.search(
        r"predicted duration frames=([0-9.eE+-]+), scale=([0-9.eE+-]+), "
        r"using_frames=([0-9]+) \(([0-9.eE+-]+)s\)",
        stdout,
    )
    if prediction:
        return {
            "kind": "predicted",
            "predicted_frames": float(prediction.group(1)),
            "duration_scale": float(prediction.group(2)),
            "used_frames": int(prediction.group(3)),
            "requested_duration_seconds": float(prediction.group(4)),
        }
    manual = re.search(r"using manual duration ([0-9.eE+-]+)s", stdout)
    if manual:
        return {"kind": "manual", "requested_duration_seconds": float(manual.group(1))}
    fallback = "checkpoint has no duration predictor" in stdout
    return {"kind": "fallback"} if fallback else {"kind": "unavailable"}


def _parse_final_loss_evidence(stdout: str) -> dict[str, Any] | None:
    """Extract an optional final-loss value if the backend reports one."""

    matches = re.findall(
        r"(?:final[\s_-]+)?loss\s*(?:[:=]|is)\s*([0-9.eE+-]+)",
        stdout,
        flags=re.IGNORECASE,
    )
    for raw in reversed(matches):
        try:
            value = _finite(float(raw))
        except (TypeError, ValueError, OverflowError):
            value = None
        if value is not None:
            return {"value": value, "source": "inference-stdout"}
    return None


def _relative_path(path: str | Path, root: Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return value.name


def _record_output(
    *,
    root: Path,
    output: Path,
    metadata: dict[str, Any],
    expected_text: str,
    asr: dict[str, Any] | None,
    asr_error: str | None,
    sense: dict[str, Any] | None,
    sense_error: str | None,
) -> dict[str, Any]:
    audio = analyze_wav_boundary(output)
    stdout = str(metadata.get("stdout") or "")
    record: dict[str, Any] = {
        "status": "ok",
        "requested_text": expected_text,
        "seed": metadata.get("seed"),
        "checkpoint": _relative_path(str(metadata.get("checkpoint") or ""), root),
        "method": metadata.get("method"),
        "reference_mode": metadata.get("reference_mode"),
        "reference_fingerprint": metadata.get("reference_fingerprint"),
        "duration_scale": metadata.get("duration_scale"),
        "trim_tail": metadata.get("trim_tail"),
        "tail_window_size": metadata.get("tail_window_size"),
        "tail_std_threshold": metadata.get("tail_std_threshold"),
        "tail_mean_threshold": metadata.get("tail_mean_threshold"),
        "duration": _parse_duration_log(stdout),
        "final_loss_evidence": _parse_final_loss_evidence(stdout),
        "output": _relative_path(output, root),
        "output_sha256": _sha256_file(output),
        "audio_boundary": audio,
        "asr_error": asr_error,
        "sense_error": sense_error,
    }
    if asr is not None:
        transcript = _transcript(asr)
        first_onset, last_end = _asr_bounds(asr)
        evidence = _final_token_evidence(expected_text, transcript)
        cer = character_error_rate(expected_text, transcript)
        wer = word_error_rate(expected_text, transcript)
        first_voiced = audio.get("first_voiced_seconds")
        pre_speech = None
        if first_onset is not None and isinstance(first_voiced, (int, float)):
            pre_speech = max(0.0, float(first_onset) - float(first_voiced))
        record["asr"] = {
            "transcript": transcript,
            "first_word_onset_seconds": first_onset,
            "last_word_end_seconds": last_end,
            "cer": cer,
            "wer": wer,
            "pre_speech_voiced_seconds": round(pre_speech, 6)
            if pre_speech is not None
            else None,
            "leading_artifact": bool(pre_speech is not None and pre_speech >= 0.08),
            **evidence,
        }
    else:
        record["asr"] = None
    if sense is not None:
        events = sense.get("events") if isinstance(sense.get("events"), list) else []
        record["sense"] = {
            "emotion": sense.get("emotion"),
            "events": [str(event) for event in events],
            "non_verbal_events": [str(event) for event in events if str(event) != "Speech"],
        }
    else:
        record["sense"] = None
    return record


def _run_asr(
    repo_root: Path,
    cfg: PersonaConfig,
    output: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        result = worker(repo_root, "asr").call(
            repo_root,
            "transcribe",
            {
                "audio": str(output.resolve()),
                "model": cfg.prepare.asr_model,
                "compute_type": cfg.prepare.asr_compute_type,
                "device": cfg.prepare.asr_device,
                "dtype": cfg.prepare.asr_dtype,
                "language": cfg.language,
            },
        )
        if not isinstance(result, dict):
            return None, "invalid ASR response type"
        return result, None
    except Exception as exc:  # diagnostics preserve generation evidence if ASR is unavailable
        return None, f"{type(exc).__name__}: {exc}"


def _run_sense(
    repo_root: Path,
    cfg: PersonaConfig,
    output: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        result = worker(repo_root, "sense").call(
            repo_root,
            "analyze",
            {"audio": str(output.resolve()), "language": cfg.language},
        )
        if not isinstance(result, dict):
            return None, "invalid SenseVoice response type"
        return result, None
    except Exception as exc:  # diagnostics preserve generation evidence if SenseVoice is unavailable
        return None, f"{type(exc).__name__}: {exc}"


def _run_batch_worker(
    repo_root: Path,
    cfg: PersonaConfig,
    rows: list[dict[str, Any]],
    *,
    worker_name: str,
    command: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Analyze all generated files in one model-worker process."""

    items = [
        {
            "id": str(index),
            "audio": str(row["_diagnostic_output_path"]),
        }
        for index, row in enumerate(rows)
        if row.get("status") == "ok" and row.get("_diagnostic_output_path")
    ]
    if not items:
        return {}, {}
    payload: dict[str, Any] = {
        "items": items,
        "language": cfg.language,
    }
    if worker_name == "asr":
        payload.update(
            {
                "model": cfg.prepare.asr_model,
                "compute_type": cfg.prepare.asr_compute_type,
                "device": cfg.prepare.asr_device,
                "dtype": cfg.prepare.asr_dtype,
            }
        )
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    try:
        response = worker(repo_root, worker_name).call(repo_root, command, payload)
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise RuntimeError(f"{worker_name} batch response did not contain results")
        for item in response["results"]:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id"))
            if item.get("ok") is True and isinstance(item.get("result"), dict):
                results[item_id] = item["result"]
            else:
                errors[item_id] = str(item.get("error") or "worker item failed")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.update({str(index): message for index in range(len(rows))})
    return results, errors


def _enrich_deferred_rows(
    repo_root: Path,
    cfg: PersonaConfig,
    root: Path,
    rows: list[dict[str, Any]],
    *,
    include_asr: bool,
    include_sense: bool,
) -> list[dict[str, Any]]:
    asr_results: dict[str, dict[str, Any]] = {}
    asr_errors: dict[str, str] = {}
    sense_results: dict[str, dict[str, Any]] = {}
    sense_errors: dict[str, str] = {}
    if include_asr:
        asr_results, asr_errors = _run_batch_worker(
            repo_root,
            cfg,
            rows,
            worker_name="asr",
            command="batch_transcribe",
        )
    if include_sense:
        sense_results, sense_errors = _run_batch_worker(
            repo_root,
            cfg,
            rows,
            worker_name="sense",
            command="batch_analyze",
        )

    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("status") != "ok":
            enriched.append(row)
            continue
        output = Path(str(row.pop("_diagnostic_output_path")))
        metadata = row.pop("_diagnostic_metadata")
        expected_text = str(row.pop("_diagnostic_expected_text"))
        context = {
            key: value
            for key, value in row.items()
            if key in {"phase", "case_id", "variant", "variant_spec", "condition"}
        }
        record = _record_output(
            root=root,
            output=output,
            metadata=metadata,
            expected_text=expected_text,
            asr=asr_results.get(str(index)),
            asr_error=asr_errors.get(str(index)),
            sense=sense_results.get(str(index)),
            sense_error=sense_errors.get(str(index)),
        )
        record.update(context)
        enriched.append(record)
    return enriched


def _run_one(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    case: dict[str, Any],
    output: Path,
    seed: int,
    duration_scale: float | None = None,
    trim_tail: bool | None = None,
    base_only: bool = False,
    reference_mode: str | None = None,
    caption_conditioning: bool = True,
    include_asr: bool = True,
    include_sense: bool = True,
    defer_workers: bool = False,
) -> dict[str, Any]:
    from personavoice.inference import synthesize

    metadata: dict[str, Any] = {}
    try:
        outputs = synthesize(
            repo_root,
            paths,
            cfg,
            str(case["text"]),
            emotion=str(case.get("emotion") or "NEUTRAL"),
            events=[str(item) for item in case.get("events") or []],
            candidates=1,
            seed=seed,
            output=output,
            base_only=base_only,
            reference_mode=reference_mode,
            caption_conditioning=caption_conditioning,
            duration_scale=duration_scale,
            trim_tail=trim_tail,
            metadata=metadata,
            capture_logs=True,
        )
        if len(outputs) != 1:
            raise RuntimeError("diagnostic synthesis must produce exactly one output")
        output = outputs[0]
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "requested_text": str(case["text"]),
            "seed": seed,
            "output": _relative_path(output, paths.root),
        }
    asr, asr_error = (
        _run_asr(repo_root, cfg, output) if include_asr and not defer_workers else (None, None)
    )
    sense, sense_error = (
        _run_sense(repo_root, cfg, output)
        if include_sense and not defer_workers
        else (None, None)
    )
    try:
        record = _record_output(
            root=paths.root,
            output=output,
            metadata=metadata,
            expected_text=str(case["text"]),
            asr=asr,
            asr_error=asr_error,
            sense=sense,
            sense_error=sense_error,
        )
        if defer_workers:
            record.update(
                {
                    "_diagnostic_output_path": str(output),
                    "_diagnostic_metadata": metadata,
                    "_diagnostic_expected_text": str(case["text"]),
                }
            )
        return record
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "requested_text": str(case["text"]),
            "seed": seed,
            "output": _relative_path(output, paths.root),
            "asr_error": asr_error,
            "sense_error": sense_error,
        }


def _metric_incidence(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if isinstance(row.get(field), bool)]
    return None if not values else sum(value is True for value in values) / len(values)


def _recommend_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        variant = str(row.get("variant") or "")
        groups.setdefault(variant, []).append(row)
    summary: dict[str, Any] = {}
    for variant, values in groups.items():
        asr_rows = [row["asr"] for row in values if isinstance(row.get("asr"), dict)]
        summary[variant] = {
            "samples": len(values),
            "leading_artifact_incidence": _metric_incidence(asr_rows, "leading_artifact"),
            "trailing_cutoff_incidence": (
                None
                if not asr_rows
                else sum(not bool(row.get("final_token_preserved")) for row in asr_rows)
                / len(asr_rows)
            ),
            "mean_cer": (
                None
                if not asr_rows
                else sum(float(row["cer"]) for row in asr_rows) / len(asr_rows)
            ),
        }
    complete = [
        (variant, value)
        for variant, value in summary.items()
        if value["trailing_cutoff_incidence"] is not None and value["mean_cer"] is not None
    ]
    if not complete:
        return {
            "status": "insufficient-evidence",
            "selected_variant": None,
            "requires_manual_listening": True,
            "variants": summary,
            "reason": "ASR/CER evidence is unavailable for the evaluated matrix.",
        }
    baseline = summary.get("A")
    if (
        not isinstance(baseline, dict)
        or baseline.get("trailing_cutoff_incidence") is None
        or baseline.get("mean_cer") is None
    ):
        return {
            "status": "insufficient-evidence",
            "selected_variant": None,
            "requires_manual_listening": True,
            "variants": summary,
            "reason": "The baseline A row is missing from the diagnostic report.",
        }
    improvements = [
        item
        for item in complete
        if item[0] != "A"
        and item[1]["trailing_cutoff_incidence"] < baseline["trailing_cutoff_incidence"]
        and item[1]["mean_cer"] <= baseline["mean_cer"] + 0.05
    ]
    if not improvements:
        return {
            "status": "no-safe-candidate",
            "selected_variant": "A",
            "requires_manual_listening": True,
            "variants": summary,
            "reason": (
                "No variant reduced final-token loss without a material CER regression. "
                "Keep the configured policy and investigate the evidence further."
            ),
        }
    selected_variant, selected = min(
        improvements,
        key=lambda item: (
            item[1]["trailing_cutoff_incidence"],
            item[1]["mean_cer"],
            item[0],
        ),
    )
    return {
        "status": "candidate",
        "selected_variant": selected_variant,
        "requires_manual_listening": True,
        "variants": summary,
        "reason": (
            "Selected by lowest observed final-token loss and CER. This is evidence for review, "
            "not an automatic change to the configured runtime policy."
        ),
        "selected_metrics": selected,
    }


def run_boundary_diagnostics(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    seed: int = 20260826,
    margin_scale: float = 1.10,
    cases: tuple[dict[str, Any], ...] = DEFAULT_BOUNDARY_EVALUATION_CASES,
    output_dir: Path | None = None,
    include_asr: bool = True,
    include_sense: bool = True,
) -> dict[str, Any]:
    """Run the Issue #33 inference-only diagnosis and write ``report.json``."""

    if not cases:
        raise ValueError("diagnostic evaluation set must not be empty")
    variants = build_inference_diagnostic_matrix(margin_scale=margin_scale)
    conditions = build_leading_artifact_condition_matrix(paths)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_dir = output_dir or paths.outputs / "boundary_diagnostics" / stamp
    report_dir.mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = _safe_id(str(case["id"]))
        for variant in variants:
            output = report_dir / "duration_tail" / f"{case_id}_{variant['id']}.wav"
            row = _run_one(
                repo_root,
                paths,
                cfg,
                case=case,
                output=output,
                seed=int(seed),
                duration_scale=float(variant["duration_scale"]),
                trim_tail=bool(variant["trim_tail"]),
                include_asr=include_asr,
                include_sense=include_sense,
                defer_workers=True,
            )
            row.update(
                {
                    "phase": "duration-tail",
                    "case_id": case_id,
                    "variant": variant["id"],
                    "variant_spec": dict(variant),
                }
            )
            matrix_rows.append(row)

    leading_rows: list[dict[str, Any]] = []
    onset_case = cases[0]
    leading_seeds = (int(seed), int(seed) + 1, int(seed) + 2)
    for condition in conditions:
        for condition_seed in leading_seeds:
            condition_id = _safe_id(str(condition["id"]))
            output = (
                report_dir
                / "leading-isolation"
                / f"{_safe_id(str(onset_case['id']))}_{condition_id}_{condition_seed}.wav"
            )
            row = _run_one(
                repo_root,
                paths,
                cfg,
                case=onset_case,
                output=output,
                seed=condition_seed,
                base_only=bool(condition["base_only"]),
                reference_mode=str(condition["reference_mode"]),
                caption_conditioning=bool(condition["caption_conditioning"]),
                include_asr=include_asr,
                include_sense=include_sense,
                defer_workers=True,
            )
            row.update(
                {
                    "phase": "leading-isolation",
                    "case_id": str(onset_case["id"]),
                    "condition": dict(condition),
                }
            )
            leading_rows.append(row)

    combined_rows = matrix_rows + leading_rows
    if include_asr or include_sense:
        combined_rows = _enrich_deferred_rows(
            repo_root,
            cfg,
            paths.root,
            combined_rows,
            include_asr=include_asr,
            include_sense=include_sense,
        )
    else:
        for row in combined_rows:
            row.pop("_diagnostic_output_path", None)
            row.pop("_diagnostic_metadata", None)
            row.pop("_diagnostic_expected_text", None)
    matrix_count = len(matrix_rows)
    matrix_rows = combined_rows[:matrix_count]
    leading_rows = combined_rows[matrix_count:]

    report = {
        "schema_version": BOUNDARY_DIAGNOSTIC_SCHEMA,
        "created_at": _now(),
        "persona": cfg.name,
        "issue": 33,
        "provenance": {
            "irodori_source_revision": IRODORI_SOURCE_REVISION,
            "diagnostic_schema": BOUNDARY_DIAGNOSTIC_SCHEMA,
        },
        "policy": {
            "configured_duration_scale": cfg.inference.duration_scale,
            "configured_trim_tail": cfg.inference.trim_tail,
            "inference_only": True,
            "training_data_modified": False,
            "prepare_rerun_required": False,
            "training_rerun_required": False,
        },
        "evaluation_set": [dict(case) for case in cases],
        "duration_tail_matrix": [dict(variant) for variant in variants],
        "leading_condition_matrix": [dict(condition) for condition in conditions],
        "duration_tail_records": matrix_rows,
        "leading_isolation_records": leading_rows,
        "policy_recommendation": _recommend_policy(matrix_rows),
        "summary": {
            "duration_tail_records": len(matrix_rows),
            "leading_isolation_records": len(leading_rows),
            "generation_errors": sum(
                row.get("status") == "error" for row in matrix_rows + leading_rows
            ),
            "asr_unavailable": sum(
                bool(row.get("asr_error")) for row in matrix_rows + leading_rows
            ),
            "sense_unavailable": sum(
                bool(row.get("sense_error")) for row in matrix_rows + leading_rows
            ),
        },
    }
    atomic_write_json(report_dir / "report.json", report)
    report["report"] = str((report_dir / "report.json").resolve())
    return report
