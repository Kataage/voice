from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from uuid import uuid4

import torch
from checkpoint_contract import (
    checkpoint_complete,
    latest_complete_checkpoint,
    prune_incomplete_checkpoints,
    seal_checkpoint,
)
from datasets import load_dataset
from model_contract import audited_attention_lora_targets, json_contains_absolute_local_path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

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
MAX_SEQUENCE_LENGTH = 2048
FULL_REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_METHODS = {"full", "lora"}
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


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability(0)
    except (RuntimeError, AssertionError):
        return False
    # BF16 tensor-core execution is an Ampere-or-newer capability. Some CUDA
    # builds can report a broad bf16 API capability even when the selected GPU
    # architecture cannot execute the required kernels, so gate both signals.
    return capability >= (8, 0) and bool(torch.cuda.is_bf16_supported())


def _model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if _bf16_supported() else torch.float16


def _verify_base(base: Path) -> None:
    missing = [name for name in MODEL_ASSET_SHA256 if not _nonempty_file(base / name)]
    if missing:
        raise FileNotFoundError(
            f"Pinned LFM base model is missing or incomplete: {base} ({', '.join(missing)})"
        )
    marker = base / REVISION_MARKER
    actual = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if actual != MODEL_REVISION:
        raise RuntimeError(
            "LFM fine-tuning base does not match the audited revision: "
            f"expected {MODEL_REVISION}, got {actual!r}. Run `persona setup --download-models`."
        )
    for name, expected_sha256 in MODEL_ASSET_SHA256.items():
        actual_sha256 = _sha256(base / name)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "LFM fine-tuning base asset checksum mismatch: "
                f"{name}: expected {expected_sha256}, got {actual_sha256}. "
                "Run `persona setup --download-models`."
            )


def _adapter_weight(output: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = output / name
        if _nonempty_file(candidate):
            return candidate
    return None


def _plan_fingerprint(value: str | None, *, required: bool) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        if required:
            raise ValueError("Full LFM training requires --plan-fingerprint")
        return None
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError("--plan-fingerprint must be a lowercase SHA-256 hex digest")
    return normalized


def _split_dataset(dataset, *, validation_ratio: float, seed: int):
    if len(dataset) < 2:
        raise RuntimeError("LFM fine-tuning needs at least two conversational examples")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be greater than 0 and less than 1")
    columns = set(getattr(dataset, "column_names", ()) or ())
    if columns and not {"prompt", "completion"}.issubset(columns):
        raise ValueError(
            "LFM dataset must preserve the conversational prompt/completion contract"
        )
    valid_count = max(1, int(len(dataset) * validation_ratio))
    valid_count = min(valid_count, len(dataset) - 1)
    split = dataset.train_test_split(test_size=valid_count, seed=int(seed), shuffle=True)
    return split["train"], split["test"]


def _token_ids(value: object, *, label: str) -> list[int]:
    """Normalize one tokenizer result without silently accepting batched data."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"LFM {label} tokenization did not return a non-empty token list")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise RuntimeError(f"LFM {label} tokenization returned invalid token IDs")
    return value


def _raw_completion_shapes(
    dataset: Iterable[object],
    tokenizer,
    *,
    dataset_name: str,
    max_length: int,
) -> Counter[tuple[int, int]]:
    """Prove every raw response remains wholly inside the TRL token window.

    TRL 0.29.1 creates ``completion_mask`` before applying ``max_length``.
    Consequently, a long prompt can truncate some or all response tokens while
    completion-only training still starts. Reproduce the pinned trainer's
    conversational tokenization here and fail closed before loading/training the
    model. The returned multiset is later compared with TRL's processed rows so
    tokenizer or trainer drift cannot silently weaken this proof.
    """

    if max_length <= 0:
        raise ValueError("LFM max_length must be greater than 0")
    shapes: Counter[tuple[int, int]] = Counter()
    count = 0
    for index, raw_example in enumerate(dataset):
        count += 1
        if not isinstance(raw_example, Mapping):
            raise RuntimeError(f"LFM {dataset_name} example {index} is not a mapping")
        prompt = raw_example.get("prompt")
        completion = raw_example.get("completion")
        if not isinstance(prompt, list) or not prompt:
            raise RuntimeError(f"LFM {dataset_name} example {index} has no prompt messages")
        if not isinstance(completion, list) or not completion:
            raise RuntimeError(
                f"LFM {dataset_name} example {index} has no completion messages"
            )
        for label, messages, allowed_roles in (
            ("prompt", prompt, {"system", "user"}),
            ("completion", completion, {"assistant"}),
        ):
            for message in messages:
                if not isinstance(message, Mapping):
                    raise RuntimeError(
                        f"LFM {dataset_name} example {index} has an invalid {label} message"
                    )
                role = message.get("role")
                content = message.get("content")
                if (
                    not isinstance(role, str)
                    or role not in allowed_roles
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    expected = "/".join(sorted(allowed_roles))
                    raise RuntimeError(
                        f"LFM {dataset_name} example {index} {label} messages require "
                        f"non-empty content and role {expected}"
                    )
        tools = raw_example.get("tools")
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"LFM {dataset_name} example {index} has invalid tools JSON"
                ) from exc
        chat_template_kwargs = raw_example.get("chat_template_kwargs", {})
        if chat_template_kwargs is None:
            chat_template_kwargs = {}
        if not isinstance(chat_template_kwargs, Mapping):
            raise RuntimeError(
                f"LFM {dataset_name} example {index} has invalid chat_template_kwargs"
            )
        prompt_ids = _token_ids(
            tokenizer.apply_chat_template(
                prompt,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **dict(chat_template_kwargs),
            ),
            label=f"{dataset_name} example {index} prompt",
        )
        processed = tokenizer.apply_chat_template(
            prompt + completion,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=False,
            **dict(chat_template_kwargs),
        )
        if not isinstance(processed, Mapping) or "input_ids" not in processed:
            raise RuntimeError(
                f"LFM {dataset_name} example {index} full tokenization has no input_ids"
            )
        full_ids = _token_ids(
            processed["input_ids"],
            label=f"{dataset_name} example {index} prompt+completion",
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"LFM {dataset_name} example {index} prompt tokenization is not a prefix "
                "of prompt+completion tokenization; completion-only labels would be unsafe"
            )
        completion_tokens = len(full_ids) - len(prompt_ids)
        if completion_tokens <= 0:
            raise RuntimeError(
                f"LFM {dataset_name} example {index} has no completion tokens"
            )
        if len(full_ids) > max_length:
            raise RuntimeError(
                f"LFM {dataset_name} example {index} needs {len(full_ids)} tokens, exceeding "
                f"max_length={max_length}; shorten or split it so the response is not truncated"
            )
        shapes[(len(full_ids), completion_tokens)] += 1
    if count == 0:
        raise RuntimeError(f"LFM {dataset_name} dataset is empty")
    return shapes


def _verify_processed_completion_shapes(
    dataset: Iterable[object],
    *,
    dataset_name: str,
    expected: Counter[tuple[int, int]],
) -> None:
    """Verify TRL retained the exact completion-only labels proven above."""

    actual: Counter[tuple[int, int]] = Counter()
    for index, raw_example in enumerate(dataset):
        if not isinstance(raw_example, Mapping):
            raise RuntimeError(
                f"LFM processed {dataset_name} example {index} is not a mapping"
            )
        input_ids = _token_ids(
            raw_example.get("input_ids"),
            label=f"processed {dataset_name} example {index}",
        )
        raw_mask = raw_example.get("completion_mask")
        if hasattr(raw_mask, "tolist"):
            raw_mask = raw_mask.tolist()
        if not isinstance(raw_mask, list) or len(raw_mask) != len(input_ids):
            raise RuntimeError(
                f"LFM processed {dataset_name} example {index} has an invalid completion mask"
            )
        if any(value not in (0, 1, False, True) for value in raw_mask):
            raise RuntimeError(
                f"LFM processed {dataset_name} example {index} has non-binary completion labels"
            )
        completion_tokens = sum(int(value) for value in raw_mask)
        if completion_tokens <= 0:
            raise RuntimeError(
                f"LFM processed {dataset_name} example {index} has no completion labels"
            )
        first_completion = next(i for i, value in enumerate(raw_mask) if int(value) == 1)
        if any(int(value) == 0 for value in raw_mask[first_completion:]):
            raise RuntimeError(
                f"LFM processed {dataset_name} example {index} has a non-contiguous completion mask"
            )
        actual[(len(input_ids), completion_tokens)] += 1
    if actual != expected:
        raise RuntimeError(
            f"LFM processed {dataset_name} completion labels differ from the audited raw dataset"
        )


def _safe_artifact_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate.as_posix()


def _sanitize_known_model_identity(path: Path) -> None:
    if not _nonempty_file(path):
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LFM portable JSON is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"LFM portable JSON root must be a mapping: {path.name}")
    changed = False
    for key in ("_name_or_path", "name_or_path"):
        if key in payload:
            payload[key] = MODEL_ID
            changed = True
    if changed:
        _atomic_write_json(path, payload)
    if json_contains_absolute_local_path(payload):
        raise RuntimeError(f"LFM portable JSON contains an absolute local path: {path.name}")


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == PROVENANCE_FILE:
            continue
        if not _nonempty_file(path):
            raise RuntimeError(f"LFM full artifact contains an empty file: {relative}")
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def full_artifact_complete(output: Path, *, plan_fingerprint: str | None = None) -> bool:
    if not output.is_dir():
        return False
    provenance_path = output / PROVENANCE_FILE
    if not _nonempty_file(provenance_path):
        return False
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(provenance, dict):
        return False
    if (
        provenance.get("schema_version") != 1
        or provenance.get("family") != "lfm"
        or provenance.get("method") != "full"
    ):
        return False
    recorded_plan = provenance.get("training_plan_fingerprint")
    if not isinstance(recorded_plan, str) or _SHA256_RE.fullmatch(recorded_plan) is None:
        return False
    if plan_fingerprint is not None and recorded_plan != plan_fingerprint:
        return False
    best_validation_loss = provenance.get("best_validation_loss")
    if (
        not isinstance(best_validation_loss, (int, float))
        or isinstance(best_validation_loss, bool)
        or not math.isfinite(float(best_validation_loss))
    ):
        return False
    base = provenance.get("base_model")
    if not isinstance(base, dict) or base != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
    }:
        return False
    try:
        revision = (output / ADAPTER_REVISION_MARKER).read_text(encoding="utf-8").strip()
        method = (output / TRAINING_METHOD_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if revision != MODEL_REVISION or method != "full":
        return False
    raw_files = provenance.get("files")
    if not isinstance(raw_files, list):
        return False
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict):
            return False
        relative = _safe_artifact_path(item.get("path"))
        if relative is None or relative in seen:
            return False
        seen.add(relative)
        path = output / PurePosixPath(relative)
        if not _nonempty_file(path):
            return False
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != _sha256(path):
            return False
    required = set(FULL_REQUIRED_MODEL_FILES) | {
        ADAPTER_REVISION_MARKER,
        TRAINING_METHOD_MARKER,
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != provenance_path
    }
    return required.issubset(seen) and seen == actual


def _best_checkpoint(trainer, run_dir: Path, *, method: str) -> Path:
    raw = getattr(getattr(trainer, "state", None), "best_model_checkpoint", None)
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("LFM validation did not select a best checkpoint")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(run_dir.resolve(strict=False))
    except ValueError as exc:
        raise RuntimeError("LFM best checkpoint escaped the configured run directory") from exc
    if not checkpoint_complete(candidate, method=method):
        raise RuntimeError(f"LFM best checkpoint is incomplete for method={method}: {candidate}")
    return candidate


def _mark_complete_checkpoints(run_dir: Path, *, method: str, training_args) -> None:
    for path in run_dir.glob("checkpoint-*"):
        if checkpoint_complete(path, method=method):
            seal_checkpoint(path, method=method, training_args=training_args)


class _CheckpointMethodMarkerCallback(TrainerCallback):
    """Safely load-verify and attest each fully saved Trainer checkpoint.

    ``on_save`` runs after Transformers has written model/adapter, optimizer,
    scheduler, trainer, RNG and optional scaler state. The attestation is
    written atomically only after those native payloads safely load and binds
    their digests, method, step and precision for the dependency-light Modal
    observer. The final sweep remains a compatibility fallback for older
    Trainer callback implementations.
    """

    def __init__(self, method: str) -> None:
        if method not in _METHODS:
            raise ValueError(f"Unsupported LFM training method: {method!r}")
        self.method = method

    def on_save(self, args, state, control, **_kwargs):
        step = getattr(state, "global_step", None)
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise RuntimeError("LFM Trainer callback reported an invalid checkpoint step")
        checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
        try:
            seal_checkpoint(checkpoint, method=self.method, training_args=args)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "LFM Trainer on_save fired before a safely loadable, fully resumable "
                "checkpoint was durable"
            ) from exc
        if not checkpoint_complete(checkpoint, method=self.method):
            raise RuntimeError("LFM checkpoint attestation failed verification")
        return control


def _trainer_best_validation_loss(trainer) -> float:
    raw = getattr(getattr(trainer, "state", None), "best_metric", None)
    if (
        not isinstance(raw, (int, float))
        or isinstance(raw, bool)
        or not math.isfinite(float(raw))
    ):
        raise RuntimeError("LFM validation did not report a finite best validation loss")
    return float(raw)


def _sft_config(
    *,
    run_dir: Path,
    epochs: float,
    batch: int,
    grad_accum: int,
    learning_rate: float,
    has_cuda: bool,
    use_bf16: bool,
    save_steps: int,
) -> SFTConfig:
    """Build the exact pinned TRL/Transformers training contract."""

    return SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        use_cpu=not has_cuda,
        bf16=bool(has_cuda and use_bf16),
        fp16=bool(has_cuda and not use_bf16),
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=False,
        enable_jit_checkpoint=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=MAX_SEQUENCE_LENGTH,
        completion_only_loss=True,
        report_to="none",
        gradient_checkpointing=True,
    )


def _finalize_full_artifact(
    *,
    trainer,
    tokenizer,
    base: Path,
    output: Path,
    plan_fingerprint: str,
    best_checkpoint: Path,
) -> Path:
    if output.exists():
        if full_artifact_complete(output, plan_fingerprint=plan_fingerprint):
            return output
        raise FileExistsError(
            f"Refusing to overwrite an existing incomplete or different LFM full artifact: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        # load_best_model_at_end has already restored the validation winner into
        # trainer.model. Save inference assets only; checkpoints and
        # training_args may contain machine-local paths and stay in run_dir.
        model_config = getattr(trainer.model, "config", None)
        if model_config is not None:
            model_config._name_or_path = MODEL_ID
        if hasattr(tokenizer, "name_or_path"):
            tokenizer.name_or_path = MODEL_ID
        tokenizer_init = getattr(tokenizer, "init_kwargs", None)
        if isinstance(tokenizer_init, dict):
            tokenizer_init["name_or_path"] = MODEL_ID
        # Transformers 5 writes native model checkpoints as safetensors and no
        # longer exposes the legacy ``safe_serialization`` TrainingArguments or
        # save_pretrained keyword. The required file inventory below verifies
        # that the portable artifact is in fact safetensors-backed.
        trainer.model.save_pretrained(str(staging))
        tokenizer.save_pretrained(str(staging))
        # Transformers 5 may fold the special-token declarations into
        # tokenizer_config.json and omit special_tokens_map.json on save. Keep
        # the exact checksum-verified base declaration in the portable bundle
        # so its required tokenizer inventory is stable across library versions.
        special_tokens = staging / "special_tokens_map.json"
        if not _nonempty_file(special_tokens):
            shutil.copy2(base / "special_tokens_map.json", special_tokens)
        for relative in FULL_REQUIRED_MODEL_FILES:
            if relative.endswith(".json"):
                _sanitize_known_model_identity(staging / relative)
        _atomic_write_text(staging / ADAPTER_REVISION_MARKER, MODEL_REVISION + "\n")
        _atomic_write_text(staging / TRAINING_METHOD_MARKER, "full\n")
        best_validation_loss = _trainer_best_validation_loss(trainer)
        provenance = {
            "schema_version": 1,
            "family": "lfm",
            "method": "full",
            "training_plan_fingerprint": plan_fingerprint,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_weight_sha256": MODEL_WEIGHT_SHA256,
            },
            "best_checkpoint": best_checkpoint.name,
            "best_validation_loss": best_validation_loss,
            "files": _artifact_inventory(staging),
        }
        _atomic_write_json(staging / PROVENANCE_FILE, provenance)
        if not full_artifact_complete(staging, plan_fingerprint=plan_fingerprint):
            raise RuntimeError("LFM full artifact failed portable integrity verification")
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _write_lora_provenance(
    output: Path,
    *,
    plan_fingerprint: str | None,
    best_checkpoint: Path,
    best_validation_loss: float,
) -> None:
    _atomic_write_text(output / ADAPTER_REVISION_MARKER, MODEL_REVISION + "\n")
    _atomic_write_text(output / TRAINING_METHOD_MARKER, "lora\n")
    _atomic_write_json(
        output / PROVENANCE_FILE,
        {
            "schema_version": 1,
            "family": "lfm",
            "method": "lora",
            "training_plan_fingerprint": plan_fingerprint,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_weight_sha256": MODEL_WEIGHT_SHA256,
            },
            "best_checkpoint": best_checkpoint.name,
            "best_validation_loss": best_validation_loss,
        },
    )


def run_training(args: argparse.Namespace) -> dict[str, object]:
    method = str(args.method).strip().lower()
    if method not in _METHODS:
        raise ValueError(f"Unsupported LFM training method: {method!r}")
    plan_fingerprint = _plan_fingerprint(args.plan_fingerprint, required=method == "full")
    base = Path(args.base).resolve()
    output = Path(args.output).resolve()
    if method == "full":
        if args.run_dir:
            run_dir = Path(args.run_dir).resolve()
        else:
            assert plan_fingerprint is not None
            run_dir = output.parent / ".runs" / f"full-{plan_fingerprint}"
        if full_artifact_complete(output, plan_fingerprint=plan_fingerprint):
            provenance = json.loads((output / PROVENANCE_FILE).read_text(encoding="utf-8"))
            return {
                "method": "full",
                "artifact": str(output),
                "best_validation_loss": float(provenance["best_validation_loss"]),
                "training_plan_fingerprint": plan_fingerprint,
                "reused": True,
            }
    else:
        run_dir = Path(args.run_dir).resolve() if args.run_dir else output

    _verify_base(base)
    run_dir.mkdir(parents=True, exist_ok=True)
    prune_incomplete_checkpoints(run_dir, method=method)
    resume = latest_complete_checkpoint(run_dir, method=method)

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    train_dataset, eval_dataset = _split_dataset(
        dataset,
        validation_ratio=float(args.validation_ratio),
        seed=int(args.validation_seed),
    )
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    expected_train_shapes = _raw_completion_shapes(
        train_dataset,
        tokenizer,
        dataset_name="train",
        max_length=MAX_SEQUENCE_LENGTH,
    )
    expected_eval_shapes = _raw_completion_shapes(
        eval_dataset,
        tokenizer,
        dataset_name="validation",
        max_length=MAX_SEQUENCE_LENGTH,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=_model_dtype(),
        local_files_only=True,
    )
    if method == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    has_cuda = torch.cuda.is_available()
    use_bf16 = _bf16_supported()
    batch = (
        2
        if has_cuda and torch.cuda.get_device_properties(0).total_memory >= 16 * 1024**3
        else 1
    )
    grad_accum = 8 if batch == 1 else 4
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-5 if method == "full" else 2e-4
    # ``run_training`` remains callable by the v0.3 worker harnesses that
    # construct a Namespace directly and therefore do not know this new CLI
    # field yet. The command-line parser and schema-aware orchestrator both
    # pass it explicitly.
    save_steps = int(getattr(args, "save_steps", 25))
    if save_steps <= 0:
        raise ValueError("--save-steps must be greater than 0")
    config = _sft_config(
        run_dir=run_dir,
        epochs=float(args.epochs),
        batch=batch,
        grad_accum=grad_accum,
        learning_rate=float(learning_rate),
        has_cuda=has_cuda,
        use_bf16=use_bf16,
        save_steps=save_steps,
    )
    trainer_kwargs = {
        "model": model,
        "args": config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "callbacks": [_CheckpointMethodMarkerCallback(method)],
    }
    if method == "lora":
        from peft import LoraConfig

        trainer_kwargs["peft_config"] = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=audited_attention_lora_targets(model),
            task_type="CAUSAL_LM",
        )
    trainer = SFTTrainer(**trainer_kwargs)
    _verify_processed_completion_shapes(
        trainer.train_dataset,
        dataset_name="train",
        expected=expected_train_shapes,
    )
    _verify_processed_completion_shapes(
        trainer.eval_dataset,
        dataset_name="validation",
        expected=expected_eval_shapes,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    _mark_complete_checkpoints(run_dir, method=method, training_args=config)
    best_checkpoint = _best_checkpoint(trainer, run_dir, method=method)
    best_validation_loss = _trainer_best_validation_loss(trainer)

    if method == "full":
        assert plan_fingerprint is not None
        artifact = _finalize_full_artifact(
            trainer=trainer,
            tokenizer=tokenizer,
            base=base,
            output=output,
            plan_fingerprint=plan_fingerprint,
            best_checkpoint=best_checkpoint,
        )
        return {
            "method": method,
            "artifact": str(artifact),
            "run_dir": str(run_dir),
            "resumed_from": str(resume) if resume is not None else None,
            "best_checkpoint": str(best_checkpoint),
            "best_validation_loss": best_validation_loss,
            "training_plan_fingerprint": plan_fingerprint,
            "reused": False,
        }

    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    if not _nonempty_file(output / "adapter_config.json") or _adapter_weight(output) is None:
        raise RuntimeError("LFM fine-tuning completed without a complete PEFT adapter")
    _write_lora_provenance(
        output,
        plan_fingerprint=plan_fingerprint,
        best_checkpoint=best_checkpoint,
        best_validation_loss=best_validation_loss,
    )
    return {
        "method": method,
        "artifact": str(output),
        "run_dir": str(run_dir),
        "resumed_from": str(resume) if resume is not None else None,
        "best_checkpoint": str(best_checkpoint),
        "best_validation_loss": best_validation_loss,
        "training_plan_fingerprint": plan_fingerprint,
        "reused": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", default=None)
    # Keep the worker's legacy default for callers from v0.3. New schema-aware
    # orchestration always passes the explicit method; its user-facing default
    # is full and must never be changed by this worker.
    parser.add_argument("--method", choices=sorted(_METHODS), default="lora")
    parser.add_argument("--plan-fingerprint", default=None)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=20260824)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    result = run_training(parse_args())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
