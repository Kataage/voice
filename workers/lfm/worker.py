from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from uuid import uuid4

import torch
from huggingface_hub import snapshot_download
from model_contract import audited_attention_lora_targets, json_contains_absolute_local_path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
MODEL_REVISION = "b31023f2d69b95fbd7876898f8de9fae90e8afbd"
MODEL_WEIGHT_SHA256 = "abf38960d3f37c2be7c946a9b6b06d23ed04a1afb8ac192aa3b491e3dcdcf325"
MODEL_ASSET_SHA256 = {
    "chat_template.jinja": "89e790f027916b5a2bca145a6a8454e06ffc7a5043bf3b6d97829aff86bb543f",
    "config.json": "df8dac1ebef28c06a010be6353e7dd2d0a3ff9c2ca23591bb8ced252d74510a1",
    "model.safetensors": MODEL_WEIGHT_SHA256,
    "special_tokens_map.json": "742aefe2b7dec496e8caffdba03a75d0c1a9925d53bd3f3e0d388c96b591b6f4",
    "tokenizer.json": "d7a0ab0fc22e41ec8c6d7450a9ff9ce40e196ec5e5a2fa6a2105e064e0514ed7",
    "tokenizer_config.json": "8cba5b0c7acab23a0d4cc9ac587346c9220a1b6d288fc5346fe118202fd6f43e",
}
REVISION_MARKER = ".personavoice-revision"
ADAPTER_REVISION_MARKER = ".personavoice-base-revision"
TRAINING_METHOD_MARKER = ".personavoice-training-method"
PROVENANCE_FILE = "provenance.json"
REQUIRED_MODEL_FILES = (
    "chat_template.jinja",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    for name, expected in MODEL_ASSET_SHA256.items():
        actual = _sha256(local / name)
        if actual != expected:
            raise RuntimeError(
                "LFM base asset checksum mismatch: "
                f"{name}: expected {expected}, got {actual}. "
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
            relative for relative in REQUIRED_MODEL_FILES if not _nonempty_file(local / relative)
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


def _safe_artifact_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate.as_posix()


def verify_full_model(full_model: Path) -> None:
    """Verify a standalone persona model without confusing it with the base.

    Fine-tuned weights cannot equal the pinned base checksum. Their provenance
    instead binds every portable file to a SHA-256 inventory and binds the
    training ancestry to the exact audited base revision and checksum.
    """

    if not full_model.is_dir():
        raise FileNotFoundError(f"LFM full model directory is missing: {full_model}")
    if _nonempty_file(full_model / "adapter_config.json") or _adapter_weight(full_model):
        raise RuntimeError(f"LFM full model directory contains PEFT adapter files: {full_model}")
    provenance_path = full_model / PROVENANCE_FILE
    if not _nonempty_file(provenance_path):
        raise FileNotFoundError(f"LFM full model provenance is missing: {provenance_path}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LFM full model provenance is unreadable: {provenance_path}") from exc
    if not isinstance(provenance, dict):
        raise RuntimeError(f"LFM full model provenance root must be a mapping: {provenance_path}")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("family") != "lfm"
        or provenance.get("method") != "full"
    ):
        raise RuntimeError("LFM full model provenance does not declare schema 1 / lfm / full")
    plan_fingerprint = provenance.get("training_plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or _SHA256_RE.fullmatch(plan_fingerprint) is None:
        raise RuntimeError("LFM full model provenance has an invalid training plan fingerprint")
    best_validation_loss = provenance.get("best_validation_loss")
    if (
        not isinstance(best_validation_loss, (int, float))
        or isinstance(best_validation_loss, bool)
        or not math.isfinite(float(best_validation_loss))
    ):
        raise RuntimeError("LFM full model provenance has no finite best validation loss")
    expected_base = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
    }
    if provenance.get("base_model") != expected_base:
        raise RuntimeError("LFM full model provenance does not match the audited base contract")
    revision = _read_marker(full_model / ADAPTER_REVISION_MARKER)
    if revision != MODEL_REVISION:
        raise RuntimeError(
            "LFM full model was not finalized against the audited JP-202606 base revision: "
            f"expected {MODEL_REVISION}, got {revision!r}. Retrain the persona full model."
        )
    method = _read_marker(full_model / TRAINING_METHOD_MARKER)
    if method != "full":
        raise RuntimeError(f"LFM full model method marker is invalid: {method!r}")
    raw_files = provenance.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("LFM full model provenance files must be a list")
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict):
            raise RuntimeError("LFM full model provenance contains an invalid file entry")
        relative = _safe_artifact_path(item.get("path"))
        if relative is None or relative in seen:
            raise RuntimeError("LFM full model provenance contains an unsafe or duplicate path")
        seen.add(relative)
        path = full_model / PurePosixPath(relative)
        if not _nonempty_file(path):
            raise FileNotFoundError(f"LFM full model artifact file is missing: {relative}")
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != _sha256(path):
            raise RuntimeError(f"LFM full model artifact checksum mismatch: {relative}")
    required = set(REQUIRED_MODEL_FILES) | {
        ADAPTER_REVISION_MARKER,
        TRAINING_METHOD_MARKER,
    }
    missing = sorted(required - seen)
    if missing:
        raise FileNotFoundError(
            "LFM full model provenance does not cover required portable files: "
            + ", ".join(missing)
        )
    actual = {
        path.relative_to(full_model).as_posix()
        for path in full_model.rglob("*")
        if path.is_file() and path != provenance_path
    }
    if actual != seen:
        raise RuntimeError("LFM full model contains files outside its checksum inventory")
    for relative in REQUIRED_MODEL_FILES:
        if not relative.endswith(".json"):
            continue
        path = full_model / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LFM full model JSON is unreadable: {relative}") from exc
        if json_contains_absolute_local_path(payload):
            raise RuntimeError(f"LFM full model JSON contains an absolute local path: {relative}")


def load_full_model(full_model: Path):
    verify_full_model(full_model)
    tokenizer = AutoTokenizer.from_pretrained(full_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        full_model,
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


def _generation_kwargs(payload: dict) -> dict:
    raw_temperature = payload.get("temperature", 0.1)
    if isinstance(raw_temperature, bool) or not isinstance(raw_temperature, (int, float)):
        raise ValueError("LFM temperature must be a finite number")
    temperature = float(raw_temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("LFM temperature must be finite and non-negative")
    raw_max_new_tokens = payload.get("max_new_tokens", 384)
    if (
        isinstance(raw_max_new_tokens, bool)
        or not isinstance(raw_max_new_tokens, int)
        or not 1 <= raw_max_new_tokens <= 4096
    ):
        raise ValueError("LFM max_new_tokens must be an integer between 1 and 4096")
    options = {
        "do_sample": temperature > 0,
        "repetition_penalty": 1.05,
        "max_new_tokens": raw_max_new_tokens,
    }
    if temperature > 0:
        options.update({"temperature": temperature, "top_k": 50})
    return options


def infer(payload: dict) -> dict:
    full_model = payload.get("full_model")
    adapter = payload.get("adapter")
    if full_model and adapter:
        raise ValueError("LFM full_model and adapter are mutually exclusive")
    if full_model:
        tokenizer, model = load_full_model(Path(full_model))
    else:
        tokenizer, model = load_base()
    if adapter:
        from peft import PeftModel

        adapter_path = Path(adapter)
        verify_adapter(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    messages = payload["messages"]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_ids = inputs.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise RuntimeError("LFM chat template did not return tensor input_ids")
    output = model.generate(
        **inputs,
        **_generation_kwargs(payload),
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
            relative for relative in REQUIRED_MODEL_FILES if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            f"Pinned LFM download completed without required model files: {', '.join(missing)}"
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
