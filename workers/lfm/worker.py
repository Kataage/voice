from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from model_contract import audited_attention_lora_targets
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
MODEL_REVISION = "b31023f2d69b95fbd7876898f8de9fae90e8afbd"
REVISION_MARKER = ".personavoice-revision"
ADAPTER_REVISION_MARKER = ".personavoice-base-revision"


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_path() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "lfm" / "base"
    config = local / "config.json"
    marker = local / REVISION_MARKER
    if not config.is_file():
        raise FileNotFoundError(
            f"Pinned LFM model is missing: {local}. Run `persona setup --download-models`."
        )
    actual_revision = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if actual_revision != MODEL_REVISION:
        raise RuntimeError(
            "Local LFM snapshot does not match the audited revision: "
            f"expected {MODEL_REVISION}, got {actual_revision!r}. "
            "Re-run `persona setup --download-models`."
        )
    return str(local)


def model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_base():
    base = base_path()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=model_dtype(),
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def _adapter_weight(adapter: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def verify_adapter(adapter: Path) -> None:
    if not (adapter / "adapter_config.json").is_file() or _adapter_weight(adapter) is None:
        raise FileNotFoundError(f"LFM adapter is incomplete: {adapter}")
    marker = adapter / ADAPTER_REVISION_MARKER
    revision = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if revision != MODEL_REVISION:
        raise RuntimeError(
            "LFM adapter was not finalized against the audited JP-202606 base revision: "
            f"expected {MODEL_REVISION}, got {revision!r}. Retrain the persona adapter."
        )


def infer(payload: dict) -> dict:
    tokenizer, model = load_base()
    adapter = payload.get("adapter")
    if adapter:
        adapter_path = Path(adapter)
        verify_adapter(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    messages = payload["messages"]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=float(payload.get("temperature", 0.1)),
        top_k=50,
        repetition_penalty=1.05,
        max_new_tokens=int(payload.get("max_new_tokens", 384)),
    )
    generated = output[0, input_ids.shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {"text": text}


def download_model(payload: dict) -> dict:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "lfm" / "base"
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=local,
        cache_dir=Path(os.environ["HF_HOME"]),
    )
    (local / REVISION_MARKER).write_text(MODEL_REVISION + "\n", encoding="utf-8")
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "path": str(local)}


def health(payload: dict) -> dict:
    result = {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "dtype": str(model_dtype()).removeprefix("torch."),
    }
    if payload.get("deep"):
        tokenizer, model = load_base()
        lora_targets = audited_attention_lora_targets(model)
        result.update(
            {
                "model_loaded": model is not None,
                "tokenizer_loaded": tokenizer is not None,
                "lora_targets_ok": True,
                "lora_target_count": len(lora_targets),
            }
        )
    return result


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
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
