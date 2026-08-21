from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_path() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "lfm" / "base"
    return str(local) if (local / "config.json").exists() else MODEL_ID


def infer(payload: dict) -> dict:
    base = base_path()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=os.getenv("HF_HUB_OFFLINE") == "1")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=dtype,
        device_map="auto",
        local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
    )
    adapter = payload.get("adapter")
    if adapter and Path(adapter).exists():
        model = PeftModel.from_pretrained(model, adapter)
    messages = payload["messages"]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=float(payload.get("temperature", 0.35)),
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.05,
        max_new_tokens=int(payload.get("max_new_tokens", 384)),
    )
    generated = output[0, input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {"text": text}


def download_model(payload: dict) -> dict:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "lfm" / "base"
    snapshot_download(MODEL_ID, local_dir=local, cache_dir=Path(os.environ["HF_HOME"]))
    return {"model": MODEL_ID, "path": str(local)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["infer", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "infer":
        result = infer(payload)
    elif args.command == "download":
        result = download_model(payload)
    else:
        result = {"ok": True, "cuda": torch.cuda.is_available()}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
