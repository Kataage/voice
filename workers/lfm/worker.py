from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import torch
from huggingface_hub import snapshot_download
from model_contract import audited_attention_lora_targets
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
MODEL_REVISION = "b31023f2d69b95fbd7876898f8de9fae90e8afbd"
MODEL_WEIGHT_SHA256 = "abf38960d3f37c2be7c946a9b6b06d23ed04a1afb8ac192aa3b491e3dcdcf325"
REVISION_MARKER = ".personavoice-revision"
ADAPTER_REVISION_MARKER = ".personavoice-base-revision"
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


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
    path = local / "model.safetensors"
    actual = _sha256(path)
    if actual != MODEL_WEIGHT_SHA256:
        raise RuntimeError(
            "LFM model.safetensors checksum mismatch: "
            f"expected {MODEL_WEIGHT_SHA256}, got {actual}. "
            "Re-run `persona setup --download-models` to restore the audited base."
        )


def _read_marker(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None
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


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_path() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "lfm" / "base"
    marker = local / REVISION_MARKER
    if not _materialization_complete(local):
        missing = [
            relative
            for relative in REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned LFM model is missing or incomplete: "
            f"{local} (invalid: {', '.join(missing)}). "
            "Run `persona setup --download-models`."
        )
    actual_revision = _read_marker(marker)
    if actual_revision != MODEL_REVISION:
        raise RuntimeError(
            "Local LFM snapshot does not match the audited revision: "
            f"expected {MODEL_REVISION}, got {actual_revision!r}. "
            "Re-run `persona setup --download-models`."
        )
    _verify_weight(local)
    return str(local)


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability(0)
    except (RuntimeError, AssertionError):
        return False
    return capability >= (8, 0) and bool(torch.cuda.is_bf16_supported())


def model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if _bf16_supported() else torch.float16


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
        if _nonempty_file(candidate):
            return candidate
    return None


def verify_adapter(adapter: Path) -> None:
    if not _nonempty_file(adapter / "adapter_config.json") or _adapter_weight(adapter) is None:
        raise FileNotFoundError(f"LFM adapter is incomplete: {adapter}")
    revision = _read_marker(adapter / ADAPTER_REVISION_MARKER)
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
    if not _materialization_complete(local):
        missing = [
            relative
            for relative in REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned LFM download completed without required model files: "
            f"{', '.join(missing)}"
        )
    _verify_weight(local)
    _atomic_write_text(local / REVISION_MARKER, MODEL_REVISION + "\n")
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "path": str(local)}


def health(payload: dict) -> dict:
    result = {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "dtype": str(model_dtype()).removeprefix("torch."),
    }
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            result["compute_capability"] = f"{major}.{minor}"
        except (RuntimeError, AssertionError):
            result["compute_capability"] = None
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
