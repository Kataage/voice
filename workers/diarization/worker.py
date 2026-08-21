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
    return [
        {"start": round(float(segment.start), 4), "end": round(float(segment.end), 4), "speaker": str(speaker)}
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]


def diarize(payload: dict, *, force_one: bool = False) -> dict:
    pipeline = load_pipeline()
    output = pipeline(payload["audio"], num_speakers=1 if force_one else None)
    labels = [str(label) for label in output.speaker_diarization.labels()]
    embeddings = output.speaker_embeddings
    mapped = {}
    if embeddings is not None:
        for index, label in enumerate(labels):
            if index < len(embeddings):
                mapped[label] = [float(v) for v in embeddings[index].tolist()]
    return {
        "turns": annotation_rows(output.speaker_diarization),
        "exclusive_turns": annotation_rows(output.exclusive_speaker_diarization),
        "speaker_embeddings": mapped,
    }


def embed(payload: dict) -> dict:
    result = diarize(payload, force_one=True)
    embeddings = result["speaker_embeddings"]
    if not embeddings:
        raise RuntimeError("No speaker embedding could be extracted from identity audio")
    return {"embedding": next(iter(embeddings.values()))}


def download(payload: dict) -> dict:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "pyannote" / "community-1"
    token = os.getenv("HF_TOKEN")
    snapshot_download(MODEL_ID, local_dir=local, cache_dir=Path(os.environ["HF_HOME"]), token=token)
    return {"model": MODEL_ID, "path": str(local)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["diarize", "embed", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "diarize":
        result = diarize(payload)
    elif args.command == "embed":
        result = embed(payload)
    elif args.command == "download":
        result = download(payload)
    else:
        result = {"ok": True, "cuda": torch.cuda.is_available()}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
