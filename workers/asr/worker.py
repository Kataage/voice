from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    if (
        len(parts) < 5
        or parts[0] != "personas"
        or parts[2] != "cache"
        or parts[-1] != ".checkpoints"
    ):
        raise ValueError(
            "Prepare checkpoint directory must be personas/<name>/cache/<kind>/.checkpoints"
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


def model_path(model: str) -> str:
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
    }


def transcribe(payload: dict) -> dict:
    name = payload.get("model", PINNED_MODEL_NAME)
    model = make_model(name, payload.get("compute_type", "auto"))
    return transcribe_with_model(model, payload["audio"], language=payload.get("language") or "ja")


def batch_transcribe(payload: dict) -> dict:
    items = payload.get("items") or []
    checkpoint = _checkpoint_directory(payload)
    name = payload.get("model", PINNED_MODEL_NAME)
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


def download(payload: dict) -> dict:
    requested = payload.get("model", PINNED_MODEL_NAME)
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
    parser.add_argument("command", choices=["transcribe", "batch_transcribe", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = request(args.request)
    if args.command == "transcribe":
        result = transcribe(payload)
    elif args.command == "batch_transcribe":
        result = batch_transcribe(payload)
    elif args.command == "download":
        result = download(payload)
    else:
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
