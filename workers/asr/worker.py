from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download


def request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_path(model: str) -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "asr" / model
    return str(local) if local.exists() else model


def make_model(name: str, compute_type: str = "auto") -> WhisperModel:
    import ctranslate2

    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
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
    name = payload.get("model", "large-v3")
    model = make_model(name, payload.get("compute_type", "auto"))
    return transcribe_with_model(model, payload["audio"], language=payload.get("language") or "ja")


def batch_transcribe(payload: dict) -> dict:
    items = payload.get("items") or []
    name = payload.get("model", "large-v3")
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
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    name = payload.get("model", "large-v3")
    repo_id = f"Systran/faster-whisper-{name}"
    local = root / "models" / "asr" / name
    snapshot_download(repo_id, local_dir=local, cache_dir=Path(os.environ["HF_HOME"]))
    return {"model": repo_id, "path": str(local)}


def health(payload: dict) -> dict:
    result = {"ok": True}
    if payload.get("deep"):
        model = make_model(payload.get("model", "large-v3"), payload.get("compute_type", "auto"))
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
