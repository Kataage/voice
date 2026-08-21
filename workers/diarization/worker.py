from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from pyannote.audio import Pipeline

MODEL_ID = "pyannote/speaker-diarization-community-1"


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_source() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "pyannote" / "community-1"
    return str(local) if (local / "config.yaml").exists() else MODEL_ID


def load_pipeline() -> Pipeline:
    token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(local_source(), token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def annotation_rows(annotation) -> list[dict]:
    if annotation is None:
        return []
    return [
        {
            "start": round(float(segment.start), 4),
            "end": round(float(segment.end), 4),
            "speaker": str(speaker),
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]


def diarize_with_pipeline(pipeline: Pipeline, audio: str, *, force_one: bool = False) -> dict:
    output = pipeline(audio, num_speakers=1 if force_one else None)
    diarization = output.speaker_diarization
    labels = [str(label) for label in diarization.labels()]
    embeddings = getattr(output, "speaker_embeddings", None)
    mapped = {}
    if embeddings is not None:
        for index, label in enumerate(labels):
            if index < len(embeddings):
                mapped[label] = [float(v) for v in embeddings[index].tolist()]
    exclusive = getattr(output, "exclusive_speaker_diarization", None) or diarization
    return {
        "turns": annotation_rows(diarization),
        "exclusive_turns": annotation_rows(exclusive),
        "speaker_embeddings": mapped,
    }


def diarize(payload: dict, *, force_one: bool = False) -> dict:
    pipeline = load_pipeline()
    return diarize_with_pipeline(pipeline, payload["audio"], force_one=force_one)


def embed_with_pipeline(pipeline: Pipeline, audio: str) -> dict:
    result = diarize_with_pipeline(pipeline, audio, force_one=True)
    embeddings = result["speaker_embeddings"]
    if not embeddings:
        raise RuntimeError("No speaker embedding could be extracted from identity audio")
    return {"embedding": next(iter(embeddings.values()))}


def embed(payload: dict) -> dict:
    pipeline = load_pipeline()
    return embed_with_pipeline(pipeline, payload["audio"])


def batch(payload: dict) -> dict:
    pipeline = load_pipeline()
    results: dict[str, list[dict]] = {"embeddings": [], "diarizations": []}
    for item in payload.get("embeddings") or []:
        item_id = str(item["id"])
        try:
            value = embed_with_pipeline(pipeline, str(item["audio"]))
            results["embeddings"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results["embeddings"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    for item in payload.get("diarizations") or []:
        item_id = str(item["id"])
        try:
            value = diarize_with_pipeline(pipeline, str(item["audio"]))
            results["diarizations"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results["diarizations"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    return results


def download(payload: dict) -> dict:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "pyannote" / "community-1"
    token = os.getenv("HF_TOKEN")
    snapshot_download(
        MODEL_ID,
        local_dir=local,
        cache_dir=Path(os.environ["HF_HOME"]),
        token=token,
    )
    return {"model": MODEL_ID, "path": str(local)}


def health(payload: dict) -> dict:
    result = {"ok": True, "cuda": torch.cuda.is_available()}
    if payload.get("deep"):
        pipeline = load_pipeline()
        result["model_loaded"] = pipeline is not None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["diarize", "embed", "batch", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "diarize":
        result = diarize(payload)
    elif args.command == "embed":
        result = embed(payload)
    elif args.command == "batch":
        result = batch(payload)
    elif args.command == "download":
        result = download(payload)
    else:
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
