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
EVENTS = {
    "BGM",
    "Speech",
    "Applause",
    "Laughter",
    "Cry",
    "Sneeze",
    "Breath",
    "Cough",
    "Sing",
    "Speech_Noise",
}
TAG_RE = re.compile(r"<\|([^|]+)\|>")


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_model() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "sense" / "SenseVoiceSmall"
    return str(local) if local.exists() else MODEL_ID


def load_model() -> AutoModel:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return AutoModel(model=local_model(), trust_remote_code=True, device=device)


def parse_result(result) -> dict:
    raw = result[0].get("text", "") if result else ""
    tags = TAG_RE.findall(raw)
    emotion = next((tag for tag in tags if tag in EMOTIONS), "UNKNOWN")
    events = []
    for tag in tags:
        if tag in EVENTS and tag not in events:
            events.append(tag)
    return {"raw": raw, "emotion": emotion, "events": events, "tags": tags}


def analyze_with_model(model: AutoModel, audio: str, *, language: str) -> dict:
    result = model.generate(
        input=audio,
        cache={},
        language=language,
        use_itn=True,
        batch_size=1,
    )
    return parse_result(result)


def analyze(payload: dict) -> dict:
    model = load_model()
    return analyze_with_model(model, payload["audio"], language=payload.get("language", "ja"))


def batch_analyze(payload: dict) -> dict:
    model = load_model()
    language = payload.get("language", "ja")
    results = []
    for item in payload.get("items") or []:
        item_id = str(item["id"])
        try:
            value = analyze_with_model(model, str(item["audio"]), language=language)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"results": results}


def download_model(payload: dict) -> dict:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "sense" / "SenseVoiceSmall"
    local.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(MODEL_ID, local_dir=str(local))
    return {"model": MODEL_ID, "path": str(resolved)}


def health(payload: dict) -> dict:
    result = {"ok": True, "cuda": torch.cuda.is_available()}
    if payload.get("deep"):
        model = load_model()
        result["model_loaded"] = model is not None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["analyze", "batch_analyze", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "analyze":
        result = analyze(payload)
    elif args.command == "batch_analyze":
        result = batch_analyze(payload)
    elif args.command == "download":
        result = download_model(payload)
    else:
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
