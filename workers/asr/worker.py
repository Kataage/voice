from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

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


def request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def cuda_available() -> bool:
    import ctranslate2

    return ctranslate2.get_cuda_device_count() > 0


def make_model(name: str, compute_type: str = "auto") -> WhisperModel:
    device = "cuda" if cuda_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(
        model_path(name),
        device=device,
        compute_type=compute_type,
        download_root=str(Path(os.environ["HF_HOME"]) / "faster-whisper"),
    )


def transcribe_with_model(model: WhisperModel, audio: str, *, language: str) -> dict:
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
    name = payload.get("model", PINNED_MODEL_NAME)
    model = make_model(name, payload.get("compute_type", "auto"))
    language = payload.get("language") or "ja"
    results = []
    for item in items:
        item_id = str(item["id"])
        try:
            value = transcribe_with_model(model, str(item["audio"]), language=language)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
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
    has_cuda = cuda_available()
    result = {"ok": True, "cuda": has_cuda, "device": "cuda" if has_cuda else "cpu"}
    if payload.get("deep"):
        model = make_model(payload.get("model", PINNED_MODEL_NAME), payload.get("compute_type", "auto"))
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
