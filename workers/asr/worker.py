from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from uuid import uuid4

import ctranslate2
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from runtime_policy import choose_compute_type

PINNED_MODEL_NAME = "large-v3"
PINNED_MODEL_ID = "Systran/faster-whisper-large-v3"
PINNED_MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
PINNED_MODEL_WEIGHT_SHA256 = "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"
REVISION_MARKER = ".personavoice-revision"
QWEN_MODEL_NAME = "qwen3-asr-1.7b"
QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
QWEN_REQUIRED_MODEL_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
QWEN_ALIGNER_NAME = "qwen3-forced-aligner-0.6b"
QWEN_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
QWEN_ALIGNER_REQUIRED_MODEL_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
QWEN_ASR_LFS_OIDS = {
    "model-00001-of-00002.safetensors":
        "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
    "model-00002-of-00002.safetensors":
        "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
}
QWEN_ALIGNER_LFS_OIDS = {
    "model.safetensors":
        "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7",
}
ALIGNMENT_CONTRACT_VERSION = "alignment-v1"
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)
ASR_HEARTBEAT_SECONDS = 5.0
ASR_HEARTBEAT_MEDIA_SECONDS = 60.0
ProgressCallback = Callable[[float, int], None]


def request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_CHECKPOINT_ALLOWED = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _checkpoint_item_id(value: object) -> str:
    item_id = str(value)
    if (
        not item_id
        or len(item_id) > 128
        or item_id == "progress"
        or not item_id[0].isalnum()
        or any(char not in _CHECKPOINT_ALLOWED for char in item_id)
    ):
        raise ValueError(f"Unsafe prepare checkpoint item id: {item_id!r}")
    return item_id


def _checkpoint_directory(payload: dict) -> Path | None:
    raw = payload.get("checkpoint_dir")
    if raw is None:
        return None
    target = Path(str(raw))
    if not target.is_absolute():
        raise ValueError("Prepare checkpoint directory must be an absolute path")
    root = Path(os.environ["PERSONAVOICE_ROOT"]).resolve()
    resolved = target.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Prepare checkpoint directory escapes PERSONAVOICE_ROOT") from exc
    parts = relative.parts
    base_layout = len(parts) >= 5 and parts[0] == "personas" and parts[2] == "cache"
    lineage_layout = (
        len(parts) >= 8
        and parts[0] == "personas"
        and parts[2:4] == ("generations", "prepare")
        and re.fullmatch(r"pl-[0-9a-f]{32}", parts[4] or "") is not None
        and parts[5] == "cache"
    )
    if (not base_layout and not lineage_layout) or parts[-1] != ".checkpoints":
        raise ValueError(
            "Prepare checkpoint directory must be personas/<name>/cache/<kind>/.checkpoints "
            "or personas/<name>/generations/prepare/<lineage>/cache/<kind>/.checkpoints"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    verified = resolved.resolve()
    try:
        verified.relative_to(root)
    except ValueError as exc:
        raise ValueError("Prepare checkpoint directory resolves outside PERSONAVOICE_ROOT") from exc
    return verified


def _atomic_checkpoint_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_item_checkpoint(directory: Path | None, item_id: str, result: dict) -> None:
    if directory is None:
        return
    safe_id = _checkpoint_item_id(item_id)
    _atomic_checkpoint_json(
        directory / f"{safe_id}.json",
        {"schema": 1, "id": safe_id, "result": result},
    )


def _write_batch_progress(
    directory: Path | None,
    *,
    worker_name: str,
    command: str,
    phase: str,
    completed: int,
    total: int,
    failed: int,
    current_id: str | None,
    state: str,
    detail: dict | None = None,
) -> None:
    if directory is None:
        return
    value = {
        "schema": 1,
        "worker": worker_name,
        "command": command,
        "phase": phase,
        "completed": completed,
        "total": total,
        "failed": failed,
        "current_id": current_id,
        "state": state,
    }
    if detail:
        value.update(detail)
    _atomic_checkpoint_json(directory / "progress.json", value)


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_weight(local: Path) -> None:
    path = local / "model.bin"
    actual = _sha256(path)
    if actual != PINNED_MODEL_WEIGHT_SHA256:
        raise RuntimeError(
            "ASR model.bin checksum mismatch: "
            f"expected {PINNED_MODEL_WEIGHT_SHA256}, got {actual}. "
            "Re-run `persona setup --download-models` to restore the audited snapshot."
        )


def _read_revision(local: Path) -> str | None:
    marker = local / REVISION_MARKER
    try:
        return marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    except OSError:
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _materialization_complete(local: Path) -> bool:
    return all(_nonempty_file(local / relative) for relative in REQUIRED_MODEL_FILES)


def _qwen_materialization_complete(
    local: Path,
    required_files: tuple[str, ...],
    revision: str,
    integrity_ids: dict[str, str],
) -> bool:
    try:
        recorded_ids = json.loads((local / "integrity_ids.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        recorded_ids = None
    return (
        all(_nonempty_file(local / relative) for relative in required_files)
        and _read_revision(local) == revision
        and recorded_ids == integrity_ids
    )


def _qwen_model_path(model: str) -> str:
    if model not in {QWEN_MODEL_NAME, QWEN_MODEL_ID, "qwen"}:
        raise ValueError(
            f"This audited ASR worker supports only {QWEN_MODEL_NAME!r}; got {model!r}."
        )
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "asr" / QWEN_MODEL_NAME
    if not _qwen_materialization_complete(
        local,
        QWEN_REQUIRED_MODEL_FILES,
        QWEN_MODEL_REVISION,
        QWEN_ASR_LFS_OIDS,
    ):
        missing = [
            relative
            for relative in QWEN_REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned Qwen ASR model is missing or incomplete: "
            f"{local} (invalid: {', '.join(missing) or 'revision marker'}). "
            "Run `persona setup --asr-backend qwen3-asr-1.7b --download-models`."
        )
    return str(local)


def _qwen_aligner_path() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "asr" / QWEN_ALIGNER_NAME
    if not _qwen_materialization_complete(
        local,
        QWEN_ALIGNER_REQUIRED_MODEL_FILES,
        QWEN_ALIGNER_REVISION,
        QWEN_ALIGNER_LFS_OIDS,
    ):
        missing = [
            relative
            for relative in QWEN_ALIGNER_REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned Qwen forced aligner is missing or incomplete: "
            f"{local} (invalid: {', '.join(missing) or 'revision marker'}). "
            "Run `persona setup --asr-backend qwen3-asr-1.7b --download-models`."
        )
    return str(local)


def model_path(model: str) -> str:
    if model in {QWEN_MODEL_NAME, QWEN_MODEL_ID, "qwen"}:
        return _qwen_model_path(model)
    if model != PINNED_MODEL_NAME:
        raise ValueError(
            f"This audited ASR worker supports only {PINNED_MODEL_NAME!r}; got {model!r}."
        )
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "asr" / PINNED_MODEL_NAME
    if not _materialization_complete(local):
        missing = [
            relative
            for relative in REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned ASR model is missing or incomplete: "
            f"{local} (invalid: {', '.join(missing)}). "
            "Run `persona setup --download-models`."
        )
    actual_revision = _read_revision(local)
    if actual_revision != PINNED_MODEL_REVISION:
        raise RuntimeError(
            "Local ASR snapshot does not match the audited revision: "
            f"expected {PINNED_MODEL_REVISION}, got {actual_revision!r}. "
            "Re-run `persona setup --download-models`."
        )
    _verify_weight(local)
    return str(local)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The Qwen ASR worker requires its optional qwen environment; "
            "run `persona setup --asr-backend qwen3-asr-1.7b`."
        ) from exc
    return torch


def qwen_runtime_config(
    requested_device: str = "auto",
    requested_dtype: str = "auto",
) -> dict[str, object]:
    """Resolve Qwen's device/dtype without assuming BF16 or FlashAttention.

    Pascal is intentionally pinned to eager attention and fp16.  ``auto`` may
    choose CPU, but the choice is returned to the caller so a fallback is
    inspectable rather than silent.
    """

    device_request = str(requested_device or "auto").strip().lower()
    dtype_request = str(requested_dtype or "auto").strip().lower()
    if device_request not in {"auto", "cpu", "cuda"}:
        raise ValueError("Qwen ASR device must be auto, cpu, or cuda")
    if dtype_request not in {"auto", "fp16", "fp32"}:
        raise ValueError("Qwen ASR dtype must be auto, fp16, or fp32; BF16 is not assumed")

    torch = _torch()
    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if device_request == "cuda" and not cuda_available:
        raise RuntimeError("Qwen ASR requested CUDA, but torch.cuda.is_available() is false")
    device = "cuda" if device_request == "cuda" or (device_request == "auto" and cuda_available) else "cpu"
    if device == "cpu":
        if dtype_request == "fp16":
            raise ValueError("Qwen ASR fp16 was requested on CPU; use dtype=fp32 or auto")
        return {
            "device": "cpu",
            "dtype": "fp32",
            "torch_dtype": "float32",
            "cuda_detected": cuda_available,
            "attention_implementation": "eager",
            "flash_attention": False,
            "compute_capability": None,
            "fallback_reason": "cuda_unavailable" if device_request == "auto" else None,
        }

    try:
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("Qwen ASR CUDA capability could not be determined; refusing implicit fallback") from exc
    if len(capability) != 2:
        raise RuntimeError(f"Qwen ASR returned an invalid CUDA capability: {capability!r}")
    dtype = "fp16" if dtype_request == "auto" else dtype_request
    return {
        "device": "cuda",
        "dtype": dtype,
        "torch_dtype": "float16" if dtype == "fp16" else "float32",
        "cuda_detected": True,
        "attention_implementation": "eager",
        "flash_attention": False,
        "compute_capability": list(capability),
        "pascal_safe": capability < (7, 0),
        "fallback_reason": None,
    }


def _qwen_device_map(runtime: dict[str, object]) -> str:
    return "cuda:0" if runtime["device"] == "cuda" else "cpu"


def _qwen_torch_dtype(runtime: dict[str, object]):
    torch = _torch()
    return torch.float16 if runtime["dtype"] == "fp16" else torch.float32


def _load_qwen_model(runtime: dict[str, object]):
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "qwen-asr is not installed in the ASR worker environment; "
            "run `persona setup --asr-backend qwen3-asr-1.7b`."
        ) from exc
    return Qwen3ASRModel.from_pretrained(
        _qwen_model_path(QWEN_MODEL_NAME),
        dtype=_qwen_torch_dtype(runtime),
        device_map=_qwen_device_map(runtime),
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=256,
    )


def _load_qwen_aligner(runtime: dict[str, object]):
    try:
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as exc:
        raise RuntimeError(
            "qwen-asr is not installed in the ASR worker environment; "
            "run `persona setup --asr-backend qwen3-asr-1.7b`."
        ) from exc
    return Qwen3ForcedAligner.from_pretrained(
        _qwen_aligner_path(),
        dtype=_qwen_torch_dtype(runtime),
        device_map=_qwen_device_map(runtime),
        attn_implementation="eager",
    )


def cuda_device_count() -> int:
    try:
        return int(ctranslate2.get_cuda_device_count())
    except (RuntimeError, OSError):
        return 0


def cuda_available() -> bool:
    return cuda_device_count() > 0


def runtime_config(compute_type: str = "auto") -> tuple[str, str, set[str]]:
    """Resolve a device/compute pair from CTranslate2's actual runtime capabilities."""

    if cuda_available():
        try:
            cuda_types = set(ctranslate2.get_supported_compute_types("cuda", 0))
            selected = choose_compute_type("cuda", cuda_types, compute_type)
            return "cuda", selected, cuda_types
        except (RuntimeError, OSError, ValueError):
            # A CUDA device can be enumerated even when this CTranslate2 build or
            # GPU architecture cannot execute the requested type. Auto mode must
            # remain usable, so it falls through to the CPU capability query.
            if compute_type != "auto":
                raise

    cpu_types = set(ctranslate2.get_supported_compute_types("cpu"))
    selected = choose_compute_type("cpu", cpu_types, compute_type)
    return "cpu", selected, cpu_types


def _make_model_with_runtime(name: str, *, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(
        model_path(name),
        device=device,
        compute_type=compute_type,
        download_root=str(Path(os.environ["HF_HOME"]) / "faster-whisper"),
    )


def make_model(name: str, compute_type: str = "auto") -> WhisperModel:
    device, selected_compute_type, _supported = runtime_config(compute_type)
    return _make_model_with_runtime(
        name,
        device=device,
        compute_type=selected_compute_type,
    )


def transcribe_with_model(
    model: WhisperModel,
    audio: str,
    *,
    language: str,
    progress: ProgressCallback | None = None,
) -> dict:
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250},
        condition_on_previous_text=True,
    )
    rows = []
    for seg in segments:
        rows.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "words": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in (seg.words or [])
                ],
            }
        )
        if progress is not None:
            progress(float(seg.end), len(rows))
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": rows,
        "backend": PINNED_MODEL_NAME,
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "backend_metadata": {"runtime": "faster-whisper"},
        "provenance": {
            "contract_version": "asr-normalized-v1",
            "backend": PINNED_MODEL_NAME,
            "model_id": PINNED_MODEL_ID,
            "model_revision": PINNED_MODEL_REVISION,
            "source_audio": str(Path(audio).resolve()),
        },
    }


def _object_value(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _finite_float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def _audio_duration(audio: str) -> float:
    try:
        import soundfile as sf

        duration = _finite_float(sf.info(audio).duration)
        if duration is not None and duration >= 0:
            return duration
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    try:
        import wave

        with wave.open(audio, "rb") as handle:
            rate = handle.getframerate()
            return (handle.getnframes() / rate) if rate else 0.0
    except (OSError, wave.Error, ValueError):
        return 0.0


_LANGUAGE_NAMES = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


def _qwen_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = str(language).strip()
    return _LANGUAGE_NAMES.get(normalized.lower(), normalized) or None


def _qwen_timestamps(value: object) -> list[dict[str, object]]:
    raw = _object_value(value, "time_stamps", "timestamps", default=None)
    if not isinstance(raw, (list, tuple)):
        return []
    rows = []
    for item in raw:
        text = _object_value(item, "text", "word", default="")
        start = _finite_float(_object_value(item, "start_time", "start", default=None))
        end = _finite_float(_object_value(item, "end_time", "end", default=None))
        if not isinstance(text, str) or not text.strip() or start is None or end is None:
            continue
        if start < 0 or end < start:
            continue
        confidence = _finite_float(_object_value(item, "confidence", "probability", default=None))
        row: dict[str, object] = {
            "start": start,
            "end": end,
            "text": text,
            # Qwen timestamps are alignment units, not Whisper word objects.
            # Keep the normalized ASR segment shape explicit without inventing
            # word probabilities.
            "words": [],
        }
        if confidence is not None:
            # Only preserve a value explicitly emitted by qwen-asr.  This is
            # intentionally not inferred from Whisper-style log probabilities.
            row["confidence"] = confidence
        rows.append(row)
    return rows


def _qwen_transcribe_with_model(
    model: object,
    audio: str,
    *,
    language: str,
    runtime: dict[str, object],
) -> dict:
    result = model.transcribe(
        audio=audio,
        language=_qwen_language(language),
        return_time_stamps=True,
    )
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("Qwen ASR returned no result")
        result = result[0]
    text = _object_value(result, "text", default="")
    detected_language = _object_value(result, "language", default=None)
    language_probability = _finite_float(
        _object_value(result, "language_probability", "language_prob", default=None)
    )
    duration = _finite_float(_object_value(result, "duration", default=None))
    if duration is None or duration < 0:
        duration = _audio_duration(audio)
    if not isinstance(text, str):
        raise RuntimeError("Qwen ASR returned a non-string transcript")
    timestamps = _qwen_timestamps(result)
    segments = timestamps or [{"start": 0.0, "end": duration, "text": text, "words": []}]
    output: dict[str, object] = {
        "language": str(detected_language or language or "und"),
        "language_probability": language_probability,
        "duration": duration,
        "segments": segments,
        "backend": QWEN_MODEL_NAME,
        "model_id": QWEN_MODEL_ID,
        "model_revision": QWEN_MODEL_REVISION,
        "backend_metadata": {
            "runtime": runtime,
            "timestamps_native": bool(timestamps),
        },
        "provenance": {
            "contract_version": "asr-normalized-v1",
            "backend": QWEN_MODEL_NAME,
            "model_id": QWEN_MODEL_ID,
            "model_revision": QWEN_MODEL_REVISION,
            "source_audio": str(Path(audio).resolve()),
        },
    }
    return output  # type: ignore[return-value]


def _fake_qwen_result(audio: str, *, language: str, runtime: dict[str, object]) -> dict:
    """Return a deterministic fixture only when explicitly enabled by CI/tests."""

    duration = _audio_duration(audio) or 1.0
    return {
        "language": language or "ja",
        "language_probability": None,
        "duration": duration,
        "segments": [{"start": 0.0, "end": duration, "text": "テスト", "words": []}],
        "backend": QWEN_MODEL_NAME,
        "model_id": QWEN_MODEL_ID,
        "model_revision": QWEN_MODEL_REVISION,
        "backend_metadata": {"fixture": True, "runtime": runtime},
        "provenance": {
            "contract_version": "asr-normalized-v1",
            "backend": QWEN_MODEL_NAME,
            "model_id": QWEN_MODEL_ID,
            "model_revision": QWEN_MODEL_REVISION,
            "source_audio": str(Path(audio).resolve()),
            "fixture": True,
        },
    }


def _is_qwen_model(model: object) -> bool:
    return str(model or "").strip().lower() in {
        QWEN_MODEL_NAME,
        QWEN_MODEL_ID.lower(),
        "qwen",
    }


def _batch_qwen_transcribe(payload: dict) -> dict:
    items = payload.get("items") or []
    checkpoint = _checkpoint_directory(payload)
    language = payload.get("language") or "ja"
    total = len(items)
    runtime = qwen_runtime_config(payload.get("device", "auto"), payload.get("dtype", "auto"))
    detail = {key: value for key, value in runtime.items() if key != "torch_dtype"}
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="runtime_config",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
        detail={"backend": QWEN_MODEL_NAME, **detail},
    )
    fixture = os.getenv("PERSONAVOICE_FAKE_ASR") == "1"
    model = None if fixture else _load_qwen_model(runtime)
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="model_load",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
        detail={"backend": QWEN_MODEL_NAME, **detail},
    )
    results = []
    failed = 0
    for index, item in enumerate(items):
        item_id = _checkpoint_item_id(item["id"])
        _write_batch_progress(
            checkpoint,
            worker_name="asr",
            command="batch_transcribe",
            phase="transcribe",
            completed=index,
            total=total,
            failed=failed,
            current_id=item_id,
            state="running",
            detail={"backend": QWEN_MODEL_NAME, **detail},
        )
        try:
            audio = str(item["audio"])
            value = (
                _fake_qwen_result(audio, language=language, runtime=detail)
                if fixture
                else _qwen_transcribe_with_model(
                    model,
                    audio,
                    language=language,
                    runtime=detail,
                )
            )
            _write_item_checkpoint(checkpoint, item_id, value)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            failed += 1
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="transcribe",
        completed=total,
        total=total,
        failed=failed,
        current_id=None,
        state="finished",
        detail={"backend": QWEN_MODEL_NAME, **detail},
    )
    return {"results": results}


def transcribe(payload: dict) -> dict:
    name = payload.get("model", PINNED_MODEL_NAME)
    if _is_qwen_model(name):
        runtime = qwen_runtime_config(payload.get("device", "auto"), payload.get("dtype", "auto"))
        if os.getenv("PERSONAVOICE_FAKE_ASR") == "1":
            return _fake_qwen_result(
                str(payload["audio"]),
                language=payload.get("language") or "ja",
                runtime=runtime,
            )
        model = _load_qwen_model(runtime)
        return _qwen_transcribe_with_model(
            model,
            str(payload["audio"]),
            language=payload.get("language") or "ja",
            runtime={key: value for key, value in runtime.items() if key != "torch_dtype"},
        )
    model = make_model(name, payload.get("compute_type", "auto"))
    return transcribe_with_model(model, payload["audio"], language=payload.get("language") or "ja")


def batch_transcribe(payload: dict) -> dict:
    items = payload.get("items") or []
    checkpoint = _checkpoint_directory(payload)
    name = payload.get("model", PINNED_MODEL_NAME)
    if _is_qwen_model(name):
        return _batch_qwen_transcribe(payload)
    language = payload.get("language") or "ja"
    requested_compute_type = payload.get("compute_type", "auto")
    results = []
    failed = 0
    total = len(items)

    # Publish progress before any runtime query, 3 GB model checksum, or model
    # construction so the parent can distinguish slow initialization from a
    # worker that stopped making forward progress.
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="runtime_config",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
        detail={"requested_compute_type": str(requested_compute_type)},
    )
    device, selected_compute_type, supported = runtime_config(str(requested_compute_type))
    runtime_detail = {
        "device": device,
        "compute_type": selected_compute_type,
        "supported_compute_types": sorted(supported),
    }
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="model_load",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
        detail=runtime_detail,
    )
    model = _make_model_with_runtime(
        name,
        device=device,
        compute_type=selected_compute_type,
    )
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="transcribe",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
        detail=runtime_detail,
    )

    for index, item in enumerate(items):
        item_id = _checkpoint_item_id(item["id"])
        _write_batch_progress(
            checkpoint,
            worker_name="asr",
            command="batch_transcribe",
            phase="transcribe",
            completed=index,
            total=total,
            failed=failed,
            current_id=item_id,
            state="running",
            detail={
                **runtime_detail,
                "current_processed_seconds": 0.0,
                "current_segments": 0,
            },
        )
        last_heartbeat = monotonic()
        last_media_heartbeat = 0.0

        def heartbeat(
            processed_seconds: float,
            segment_count: int,
            completed_index: int = index,
            failed_count: int = failed,
            source_id: str = item_id,
        ) -> None:
            nonlocal last_heartbeat, last_media_heartbeat
            now = monotonic()
            if (
                now - last_heartbeat < ASR_HEARTBEAT_SECONDS
                and processed_seconds - last_media_heartbeat < ASR_HEARTBEAT_MEDIA_SECONDS
            ):
                return
            _write_batch_progress(
                checkpoint,
                worker_name="asr",
                command="batch_transcribe",
                phase="transcribe",
                completed=completed_index,
                total=total,
                failed=failed_count,
                current_id=source_id,
                state="running",
                detail={
                    **runtime_detail,
                    "current_processed_seconds": round(processed_seconds, 3),
                    "current_segments": segment_count,
                },
            )
            last_heartbeat = now
            last_media_heartbeat = processed_seconds

        try:
            value = transcribe_with_model(
                model,
                str(item["audio"]),
                language=language,
                progress=heartbeat,
            )
            _write_item_checkpoint(checkpoint, item_id, value)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            failed += 1
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        _write_batch_progress(
            checkpoint,
            worker_name="asr",
            command="batch_transcribe",
            phase="transcribe",
            completed=index + 1,
            total=total,
            failed=failed,
            current_id=None,
            state="running",
            detail=runtime_detail,
        )
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="transcribe",
        completed=total,
        total=total,
        failed=failed,
        current_id=None,
        state="finished",
        detail=runtime_detail,
    )
    return {"results": results}


def _validate_alignment_request(payload: dict) -> dict:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        contract = {}
    if contract.get("key") == "domain-ctc-aligner" or payload.get("backend") == "domain-ctc-aligner":
        raise RuntimeError(
            "Domain CTC alignment is disabled; its head may not be attached to the general Qwen encoder"
        )
    model_id = payload.get("model") or contract.get("model_id")
    revision = payload.get("revision") or contract.get("revision")
    if model_id != QWEN_ALIGNER_ID or revision != QWEN_ALIGNER_REVISION:
        raise RuntimeError(
            "Forced alignment request does not match the pinned Qwen3-ForcedAligner contract"
        )
    if contract.get("asr_backend") not in {None, QWEN_MODEL_NAME}:
        raise RuntimeError("Qwen forced alignment is only coupled to the general Qwen ASR backend")
    return contract


def _alignment_units(value: object) -> list[dict[str, object]]:
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        return []
    units = []
    for item in value:
        text = _object_value(item, "text", "word", "unit", default="")
        start = _finite_float(_object_value(item, "start_time", "start", default=None))
        end = _finite_float(_object_value(item, "end_time", "end", default=None))
        if not isinstance(text, str) or not text.strip() or start is None or end is None:
            continue
        if start < 0 or end < start:
            continue
        confidence = _finite_float(_object_value(item, "confidence", "probability", default=None))
        row: dict[str, object] = {"unit": text, "start": start, "end": end}
        if confidence is not None:
            row["confidence"] = confidence
        units.append(row)
    return units


def _qwen_align_with_model(
    model: object,
    payload: dict,
    *,
    runtime: dict[str, object],
) -> dict:
    contract = _validate_alignment_request(payload)
    text = str(payload.get("text") or "")
    if not text.strip():
        raw_units: list[dict[str, object]] = []
    else:
        result = model.align(
            audio=str(payload["audio"]),
            text=text,
            language=_qwen_language(payload.get("language") or "ja"),
        )
        raw = result[0] if isinstance(result, (list, tuple)) and result else result
        raw_units = _alignment_units(raw)
    return {
        "contract_version": ALIGNMENT_CONTRACT_VERSION,
        "key": "qwen3-forced-aligner-0.6b",
        "units_kind": "character_or_word",
        "units": raw_units,
        "backend": "qwen3-forced-aligner-0.6b",
        "model_id": QWEN_ALIGNER_ID,
        "model_revision": QWEN_ALIGNER_REVISION,
        "revision": QWEN_ALIGNER_REVISION,
        "transcript_hash": contract.get("transcript_hash"),
        "asr_backend": contract.get("asr_backend", QWEN_MODEL_NAME),
        "asr_model_id": contract.get("asr_model_id", QWEN_MODEL_ID),
        "asr_model_revision": contract.get("asr_model_revision", QWEN_MODEL_REVISION),
        "runtime": {key: value for key, value in runtime.items() if key != "torch_dtype"},
    }


def _fake_qwen_alignment(payload: dict, *, runtime: dict[str, object]) -> dict:
    contract = _validate_alignment_request(payload)
    text = str(payload.get("text") or "")
    duration = _audio_duration(str(payload["audio"])) or 1.0
    characters = [char for char in text if not char.isspace()]
    width = duration / len(characters) if characters else duration
    units = [
        {"unit": char, "start": index * width, "end": min(duration, (index + 1) * width)}
        for index, char in enumerate(characters)
    ]
    return {
        "contract_version": ALIGNMENT_CONTRACT_VERSION,
        "key": "qwen3-forced-aligner-0.6b",
        "units_kind": "character",
        "units": units,
        "backend": "qwen3-forced-aligner-0.6b",
        "model_id": QWEN_ALIGNER_ID,
        "model_revision": QWEN_ALIGNER_REVISION,
        "revision": QWEN_ALIGNER_REVISION,
        "transcript_hash": contract.get("transcript_hash"),
        "asr_backend": contract.get("asr_backend", QWEN_MODEL_NAME),
        "asr_model_id": contract.get("asr_model_id", QWEN_MODEL_ID),
        "asr_model_revision": contract.get("asr_model_revision", QWEN_MODEL_REVISION),
        "runtime": {"fixture": True, **{key: value for key, value in runtime.items() if key != "torch_dtype"}},
    }


def align(payload: dict) -> dict:
    _validate_alignment_request(payload)
    runtime = qwen_runtime_config(payload.get("device", "auto"), payload.get("dtype", "auto"))
    if os.getenv("PERSONAVOICE_FAKE_ASR") == "1":
        return _fake_qwen_alignment(
            payload,
            runtime={key: value for key, value in runtime.items() if key != "torch_dtype"},
        )
    model = _load_qwen_aligner(runtime)
    return _qwen_align_with_model(
        model,
        payload,
        runtime={key: value for key, value in runtime.items() if key != "torch_dtype"},
    )


def batch_align(payload: dict) -> dict:
    items = payload.get("items") or []
    checkpoint = _checkpoint_directory(payload)
    total = len(items)
    runtime = qwen_runtime_config(payload.get("device", "auto"), payload.get("dtype", "auto"))
    detail = {key: value for key, value in runtime.items() if key != "torch_dtype"}
    fixture = os.getenv("PERSONAVOICE_FAKE_ASR") == "1"
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_align",
        phase="runtime_config",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
        detail={"backend": "qwen3-forced-aligner-0.6b", **detail},
    )
    model = None if fixture else _load_qwen_aligner(runtime)
    results = []
    failed = 0
    for index, item in enumerate(items):
        item_id = _checkpoint_item_id(item["id"])
        try:
            request_payload = {
                **item,
                "model": payload.get("model", QWEN_ALIGNER_ID),
                "revision": payload.get("revision", QWEN_ALIGNER_REVISION),
                "backend": payload.get("backend", "qwen3-forced-aligner-0.6b"),
            }
            value = (
                _fake_qwen_alignment(request_payload, runtime=detail)
                if fixture
                else _qwen_align_with_model(model, request_payload, runtime=detail)
            )
            _write_item_checkpoint(checkpoint, item_id, value)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            failed += 1
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        _write_batch_progress(
            checkpoint,
            worker_name="asr",
            command="batch_align",
            phase="align",
            completed=index + 1,
            total=total,
            failed=failed,
            current_id=None,
            state="running" if index + 1 < total else "finished",
            detail={"backend": "qwen3-forced-aligner-0.6b", **detail},
        )
    return {"results": results}


def download(payload: dict) -> dict:
    requested = payload.get("model", PINNED_MODEL_NAME)
    if _is_qwen_model(requested):
        root = Path(os.environ["PERSONAVOICE_ROOT"])
        cache_dir = Path(os.environ["HF_HOME"])
        downloaded = []
        for name, model_id, revision, required in (
            (QWEN_MODEL_NAME, QWEN_MODEL_ID, QWEN_MODEL_REVISION, QWEN_REQUIRED_MODEL_FILES),
            (QWEN_ALIGNER_NAME, QWEN_ALIGNER_ID, QWEN_ALIGNER_REVISION, QWEN_ALIGNER_REQUIRED_MODEL_FILES),
        ):
            local = root / "models" / "asr" / name
            integrity_ids = (
                QWEN_ASR_LFS_OIDS if name == QWEN_MODEL_NAME else QWEN_ALIGNER_LFS_OIDS
            )
            if _qwen_materialization_complete(local, required, revision, integrity_ids):
                downloaded.append({"model": model_id, "revision": revision, "reused": True})
                continue
            incoming = local.with_name(f".{name}.incoming")
            shutil.rmtree(incoming, ignore_errors=True)
            incoming.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                model_id,
                revision=revision,
                local_dir=incoming,
                cache_dir=cache_dir,
            )
            missing = [
                relative for relative in required if not _nonempty_file(incoming / relative)
            ]
            if missing:
                shutil.rmtree(incoming, ignore_errors=True)
                raise FileNotFoundError(
                    f"Pinned Qwen download completed without required model files: {', '.join(missing)}"
                )
            _atomic_write_text(
                incoming / "integrity_ids.json",
                json.dumps(integrity_ids, sort_keys=True, separators=(",", ":")) + "\n",
            )
            _atomic_write_text(incoming / REVISION_MARKER, revision + "\n")
            shutil.rmtree(local, ignore_errors=True)
            incoming.replace(local)
            downloaded.append({"model": model_id, "revision": revision, "reused": False})
        return {"backend": QWEN_MODEL_NAME, "models": downloaded}
    if requested != PINNED_MODEL_NAME:
        raise ValueError(
            f"This audited ASR worker can download only {PINNED_MODEL_NAME!r}; got {requested!r}."
        )
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "asr" / PINNED_MODEL_NAME
    snapshot_download(
        PINNED_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        local_dir=local,
        cache_dir=Path(os.environ["HF_HOME"]),
    )
    if not _materialization_complete(local):
        missing = [
            relative
            for relative in REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned ASR download completed without required model files: "
            f"{', '.join(missing)}"
        )
    _verify_weight(local)
    _atomic_write_text(local / REVISION_MARKER, PINNED_MODEL_REVISION + "\n")
    return {
        "model": PINNED_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "path": str(local),
    }


def health(payload: dict) -> dict:
    requested_model = payload.get("model", PINNED_MODEL_NAME)
    if _is_qwen_model(requested_model):
        runtime = qwen_runtime_config(payload.get("device", "auto"), payload.get("dtype", "auto"))
        result = {
            "ok": True,
            "backend": QWEN_MODEL_NAME,
            "model": QWEN_MODEL_ID,
            "revision": QWEN_MODEL_REVISION,
            "cuda": runtime["device"] == "cuda",
            "cuda_detected": runtime["cuda_detected"],
            **{key: value for key, value in runtime.items() if key != "torch_dtype"},
        }
        try:
            result["model_materialized"] = bool(_qwen_model_path(QWEN_MODEL_NAME))
        except (FileNotFoundError, RuntimeError):
            result["model_materialized"] = False
        if payload.get("deep"):
            model = None if os.getenv("PERSONAVOICE_FAKE_ASR") == "1" else _load_qwen_model(runtime)
            result["model_loaded"] = model is not None or os.getenv("PERSONAVOICE_FAKE_ASR") == "1"
        return result
    requested = payload.get("compute_type", "auto")
    device, compute_type, supported = runtime_config(requested)
    result = {
        "ok": True,
        "cuda": device == "cuda",
        "cuda_detected": cuda_available(),
        "device": device,
        "compute_type": compute_type,
        "supported_compute_types": sorted(supported),
    }
    if payload.get("deep"):
        model = make_model(payload.get("model", PINNED_MODEL_NAME), requested)
        result["model_loaded"] = model is not None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["transcribe", "batch_transcribe", "align", "batch_align", "download", "health"],
    )
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = request(args.request)
    if args.command == "transcribe":
        result = transcribe(payload)
    elif args.command == "batch_transcribe":
        result = batch_transcribe(payload)
    elif args.command == "align":
        result = align(payload)
    elif args.command == "batch_align":
        result = batch_align(payload)
    elif args.command == "download":
        result = download(payload)
    else:
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
