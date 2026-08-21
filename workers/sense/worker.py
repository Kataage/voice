from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from funasr import AutoModel
from modelscope import snapshot_download

MODEL_ID = "iic/SenseVoiceSmall"
EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
EVENTS = {"BGM", "Speech", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough", "Sing", "Speech_Noise"}
TAG_RE = re.compile(r"<\|([^|]+)\|>")


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_model() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "sense" / "SenseVoiceSmall"
    return str(local) if local.exists() else MODEL_ID


def analyze(payload: dict) -> dict:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModel(model=local_model(), trust_remote_code=True, device=device)
    result = model.generate(
        input=payload["audio"],
        cache={},
        language=payload.get("language", "ja"),
        use_itn=True,
        batch_size=1,
    )
    raw = result[0].get("text", "") if result else ""
    tags = TAG_RE.findall(raw)
    emotion = next((tag for tag in tags if tag in EMOTIONS), "UNKNOWN")
    events = []
    for tag in tags:
        if tag in EVENTS and tag not in events:
            events.append(tag)
    return {"raw": raw, "emotion": emotion, "events": events, "tags": tags}


def download_model(payload: dict) -> dict:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "sense" / "SenseVoiceSmall"
    local.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(MODEL_ID, local_dir=str(local))
    return {"model": MODEL_ID, "path": str(resolved)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["analyze", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "analyze":
        result = analyze(payload)
    elif args.command == "download":
        result = download_model(payload)
    else:
        result = {"ok": True, "cuda": torch.cuda.is_available()}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
