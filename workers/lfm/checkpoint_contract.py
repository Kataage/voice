from __future__ import annotations

import hashlib
import json
import math
import os
import pickletools
import re
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INCOMPLETE_SENTINEL = "checkpoint-is-incomplete.txt"
_CHECKPOINT_ATTESTATION = ".personavoice-checkpoint.json"
_ADAPTER_CONFIG = "adapter_config.json"
_FULL_CONFIG = "config.json"
_ADAPTER_WEIGHT_CHOICES = (
    ("adapter_model.safetensors",),
    ("adapter_model.bin",),
)
_FULL_WEIGHT_CHOICES = (
    ("model.safetensors",),
    ("pytorch_model.bin",),
    ("model.safetensors.index.json",),
    ("pytorch_model.bin.index.json",),
)
_TRAINING_METHOD_MARKER = ".personavoice-training-method"
_REQUIRED_TRAINER_STATE = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
)
_RNG_STATE = "rng_state.pth"
_SCALER_STATE = "scaler.pt"
_METHODS = {"full", "lora"}
_ATTESTATION_SCHEMA = 1
_ATTESTATION_EXCLUSIONS = {
    _CHECKPOINT_ATTESTATION,
    # The orchestrator adds these only after it has independently verified a
    # native checkpoint on a Modal Volume. They are not Trainer resume state.
    "checkpoint.complete.json",
    "checkpoint-family.json",
}
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_TRAINING_ARGS_PICKLE_BYTES = 32 * 1024 * 1024
_SAFETENSORS_DTYPES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False


def _json_object(path: Path) -> dict[str, Any]:
    if not _nonempty(path):
        raise ValueError(f"missing or empty JSON file: {path.name}")
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"JSON file is unexpectedly large: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _strict_relative_file(root: Path, raw: str) -> Path:
    relative = PurePosixPath(raw)
    if (
        not raw
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe checkpoint inventory path: {raw!r}")
    candidate = root.joinpath(*relative.parts)
    if not _nonempty(candidate):
        raise ValueError(f"checkpoint inventory file is missing or empty: {raw}")
    return candidate


def _safetensors_complete(path: Path) -> bool:
    """Validate the complete safetensors container without importing a framework."""

    try:
        size = path.stat().st_size
        if not _nonempty(path) or size < 12:
            return False
        with path.open("rb") as stream:
            header_size_raw = stream.read(8)
            if len(header_size_raw) != 8:
                return False
            header_size = struct.unpack("<Q", header_size_raw)[0]
            if header_size <= 2 or header_size > min(size - 8, _MAX_JSON_BYTES):
                return False
            header = json.loads(stream.read(header_size).decode("utf-8"))
        if not isinstance(header, dict):
            return False
        data_size = size - 8 - header_size
        ranges: list[tuple[int, int]] = []
        has_nonempty_tensor = False
        for name, metadata in header.items():
            if name == "__metadata__":
                if not isinstance(metadata, dict):
                    return False
                continue
            if not isinstance(name, str) or not name or not isinstance(metadata, dict):
                return False
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if dtype not in _SAFETENSORS_DTYPES or not isinstance(shape, list):
                return False
            if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
                return False
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(item, int) or isinstance(item, bool) for item in offsets)
            ):
                return False
            start, end = offsets
            if start < 0 or end < start or end > data_size:
                return False
            elements = math.prod(shape)
            if end - start != elements * _SAFETENSORS_DTYPES[dtype]:
                return False
            has_nonempty_tensor = has_nonempty_tensor or elements > 0
            ranges.append((start, end))
        if not ranges or not has_nonempty_tensor:
            return False
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                return False
            cursor = end
        return cursor == data_size
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, struct.error):
        return False


def _torch_archive_pickle(path: Path) -> bytes:
    """Read only the pickle bytecode from a modern torch.save ZIP.

    No pickle is executed. This is used for ``training_args.bin`` because it
    contains a TrainingArguments instance and therefore must never be loaded by
    a general-purpose unpickler merely to discover FP16 settings.
    """

    if not _nonempty(path) or not zipfile.is_zipfile(path):
        raise ValueError(f"not a modern torch.save ZIP archive: {path.name}")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members or any(
            info.is_dir()
            or info.file_size < 0
            or info.file_size > _MAX_TRAINING_ARGS_PICKLE_BYTES
            for info in members
        ):
            raise ValueError(f"unsafe training-arguments archive: {path.name}")
        names = [info.filename for info in members]
        if any(
            PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(name).parts)
            for name in names
        ):
            raise ValueError(f"unsafe member in training-arguments archive: {path.name}")
        pickle_names = [name for name in names if name == "data.pkl" or name.endswith("/data.pkl")]
        version_names = [name for name in names if name == "version" or name.endswith("/version")]
        if len(pickle_names) != 1 or len(version_names) != 1:
            raise ValueError(f"incomplete training-arguments archive: {path.name}")
        if archive.testzip() is not None:
            raise ValueError(f"CRC failure in training-arguments archive: {path.name}")
        payload = archive.read(pickle_names[0])
    # ``genops`` parses the complete stream but never imports or constructs the
    # GLOBAL objects named by pickle opcodes.
    list(pickletools.genops(payload))
    return payload


def _pickle_boolean_fields(payload: bytes, names: set[str]) -> dict[str, bool]:
    values: dict[str, bool] = {}
    pending: str | None = None
    memo_opcodes = {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}
    string_opcodes = {
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "UNICODE",
        "BINSTRING",
        "SHORT_BINSTRING",
        "STRING",
    }
    for opcode, argument, _position in pickletools.genops(payload):
        if opcode.name in string_opcodes:
            pending = argument if isinstance(argument, str) and argument in names else None
            continue
        if pending is not None and opcode.name in memo_opcodes:
            continue
        if pending is not None and opcode.name in {"NEWTRUE", "NEWFALSE"}:
            values[pending] = opcode.name == "NEWTRUE"
        pending = None
    return values


def _training_precision_from_file(path: Path) -> dict[str, bool]:
    values = _pickle_boolean_fields(
        _torch_archive_pickle(path),
        {"fp16", "bf16", "use_cpu", "no_cuda"},
    )
    if "fp16" not in values or "bf16" not in values:
        raise ValueError("training_args.bin does not encode fp16 and bf16")
    if "use_cpu" not in values:
        if "no_cuda" not in values:
            raise ValueError("training_args.bin does not encode use_cpu/no_cuda")
        values["use_cpu"] = values["no_cuda"]
    precision = {name: values[name] for name in ("fp16", "bf16", "use_cpu")}
    if precision["fp16"] and precision["bf16"]:
        raise ValueError("checkpoint cannot be both fp16 and bf16")
    if precision["use_cpu"] and (precision["fp16"] or precision["bf16"]):
        raise ValueError("CPU checkpoint cannot declare fp16 or bf16 training")
    return precision


def _training_precision_from_object(training_args: Any) -> dict[str, bool]:
    kwargs = getattr(training_args, "kwargs", None)

    def value(name: str) -> Any:
        if hasattr(training_args, name):
            return getattr(training_args, name)
        if isinstance(kwargs, Mapping):
            return kwargs.get(name)
        return None

    precision: dict[str, bool] = {}
    for name in ("fp16", "bf16", "use_cpu"):
        raw = value(name)
        if not isinstance(raw, bool):
            raise ValueError(f"training arguments do not expose boolean {name}")
        precision[name] = raw
    if precision["fp16"] and precision["bf16"]:
        raise ValueError("training cannot enable fp16 and bf16 together")
    if precision["use_cpu"] and (precision["fp16"] or precision["bf16"]):
        raise ValueError("CPU training cannot enable fp16 or bf16")
    return precision


def _safe_torch_load(path: Path) -> Any:
    """Load Trainer-native state through the same restricted path as Transformers."""

    import torch
    from transformers.trainer_pt_utils import safe_globals
    from transformers.utils import check_torch_load_is_safe

    with safe_globals():
        check_torch_load_is_safe()
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, TypeError):
            # ``mmap`` can be unavailable for a valid ZIP on a platform/build;
            # the fallback remains restricted by weights_only=True.
            return torch.load(path, map_location="cpu", weights_only=True)


def _load_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    value = _safe_torch_load(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} state is not a mapping")
    return value


def _validate_optimizer(path: Path) -> None:
    state = _load_mapping(path, label="optimizer")
    optimizer_state = state.get("state")
    param_groups = state.get("param_groups")
    if not isinstance(optimizer_state, Mapping) or not optimizer_state:
        raise ValueError("optimizer state contains no parameter moments")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("optimizer state contains no parameter groups")
    if any(not isinstance(group, Mapping) or not group.get("params") for group in param_groups):
        raise ValueError("optimizer parameter group is incomplete")


def _validate_scheduler(path: Path, *, step: int) -> None:
    state = _load_mapping(path, label="scheduler")
    last_epoch = state.get("last_epoch")
    if not isinstance(last_epoch, int) or isinstance(last_epoch, bool) or last_epoch != step:
        raise ValueError("scheduler state does not match checkpoint step")


def _validate_rng(path: Path, *, use_cpu: bool) -> None:
    state = _load_mapping(path, label="RNG")
    if not {"python", "numpy", "cpu"}.issubset(state):
        raise ValueError("RNG state is missing Python, NumPy, or CPU state")
    if not use_cpu and "cuda" not in state:
        raise ValueError("CUDA training checkpoint is missing CUDA RNG state")


def _validate_scaler(path: Path) -> None:
    state = _load_mapping(path, label="FP16 scaler")
    required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if not required.issubset(state):
        raise ValueError("FP16 scaler state is incomplete")


def _weight_files(path: Path, *, method: str) -> tuple[Path, ...]:
    choices = _ADAPTER_WEIGHT_CHOICES if method == "lora" else _FULL_WEIGHT_CHOICES
    present = [choice for choice in choices if all(_nonempty(path / name) for name in choice)]
    if len(present) != 1:
        raise ValueError("checkpoint has missing or ambiguous model weight payload")
    primary = present[0][0]
    if not primary.endswith(".index.json"):
        return (path / primary,)
    index = _json_object(path / primary)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("sharded model index has no weight map")
    suffix = ".safetensors" if primary.endswith("safetensors.index.json") else ".bin"
    shard_names: set[str] = set()
    for raw in weight_map.values():
        if (
            not isinstance(raw, str)
            or PurePosixPath(raw).name != raw
            or not raw.endswith(suffix)
        ):
            raise ValueError("sharded model index contains an unsafe shard path")
        shard_names.add(raw)
    shards = tuple(path / name for name in sorted(shard_names))
    if not shards or not all(_nonempty(shard) for shard in shards):
        raise ValueError("sharded model payload is incomplete")
    return (path / primary, *shards)


def _validate_model_payload(path: Path, *, method: str) -> tuple[Path, ...]:
    config_name = _ADAPTER_CONFIG if method == "lora" else _FULL_CONFIG
    if not _json_object(path / config_name):
        raise ValueError(f"{config_name} must not be empty")
    files = _weight_files(path, method=method)
    for weight in files:
        if weight.name.endswith(".index.json"):
            continue
        if weight.suffix == ".safetensors":
            if not _safetensors_complete(weight):
                raise ValueError(f"invalid safetensors payload: {weight.name}")
        else:
            state = _load_mapping(weight, label="model")
            if not state or any(not isinstance(name, str) or not name for name in state):
                raise ValueError(f"invalid PyTorch model state: {weight.name}")
    return files


def _validate_native_checkpoint(
    path: Path,
    *,
    method: str,
    expected_precision: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    step = checkpoint_step(path)
    if step is None or not path.is_dir() or path.is_symlink():
        raise ValueError("checkpoint directory is not a numeric real directory")
    if (path / _INCOMPLETE_SENTINEL).exists():
        raise ValueError("checkpoint incomplete sentinel is present")
    marker = path / _TRAINING_METHOD_MARKER
    if marker.exists() and (
        not _nonempty(marker) or marker.read_text(encoding="utf-8").strip() != method
    ):
        raise ValueError("checkpoint training-method marker does not match")
    trainer_state = _json_object(path / "trainer_state.json")
    global_step = trainer_state.get("global_step")
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step != step:
        raise ValueError("trainer_state global_step does not match checkpoint directory")
    max_steps = trainer_state.get("max_steps")
    if (
        max_steps is not None
        and (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < global_step
        )
    ):
        raise ValueError("trainer_state max_steps is inconsistent")
    precision = _training_precision_from_file(path / "training_args.bin")
    if expected_precision is not None and dict(expected_precision) != precision:
        raise ValueError("saved training arguments do not match active precision settings")
    _validate_model_payload(path, method=method)
    _validate_optimizer(path / "optimizer.pt")
    _validate_scheduler(path / "scheduler.pt", step=step)
    rng_files = sorted(path.glob("rng_state*.pth"))
    if len(rng_files) != 1 or rng_files[0].name != _RNG_STATE:
        raise ValueError("single-process LFM checkpoint must contain exactly rng_state.pth")
    _validate_rng(rng_files[0], use_cpu=precision["use_cpu"])
    scaler = path / _SCALER_STATE
    if precision["fp16"] and not _nonempty(scaler):
        raise ValueError("FP16 checkpoint is missing scaler.pt")
    if not precision["fp16"] and scaler.exists():
        raise ValueError("non-FP16 checkpoint contains an unexpected scaler.pt")
    if precision["fp16"]:
        if not _nonempty(scaler):
            raise ValueError("scaler.pt is empty")
        _validate_scaler(scaler)
    return precision


def _inventory(path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path).as_posix()
        if relative in _ATTESTATION_EXCLUSIONS:
            continue
        if candidate.is_symlink() or candidate.is_dir():
            if candidate.is_symlink():
                raise ValueError(f"checkpoint payload contains a link: {relative}")
            continue
        if not _nonempty(candidate):
            raise ValueError(f"checkpoint payload contains an empty or irregular file: {relative}")
        files.append(
            {
                "path": relative,
                "size": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    if not files:
        raise ValueError("checkpoint payload inventory is empty")
    return files


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_attestation(path: Path, *, method: str) -> dict[str, bool]:
    document = _json_object(path / _CHECKPOINT_ATTESTATION)
    step = checkpoint_step(path)
    if (
        document.get("schema_version") != _ATTESTATION_SCHEMA
        or document.get("step") != step
        or document.get("method") != method
    ):
        raise ValueError("checkpoint attestation identity does not match")
    precision = document.get("precision")
    if not isinstance(precision, dict) or set(precision) != {"fp16", "bf16", "use_cpu"}:
        raise ValueError("checkpoint attestation precision is invalid")
    if any(not isinstance(value, bool) for value in precision.values()):
        raise ValueError("checkpoint attestation precision must be boolean")
    if precision["fp16"] and precision["bf16"]:
        raise ValueError("checkpoint attestation enables fp16 and bf16")
    if precision["use_cpu"] and (precision["fp16"] or precision["bf16"]):
        raise ValueError("checkpoint attestation has invalid CPU precision")
    expected_files = document.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("checkpoint attestation inventory is empty")
    seen: set[str] = set()
    required = {
        *_REQUIRED_TRAINER_STATE,
        _RNG_STATE,
        _TRAINING_METHOD_MARKER,
        _ADAPTER_CONFIG if method == "lora" else _FULL_CONFIG,
    }
    for item in expected_files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError("checkpoint attestation inventory row is invalid")
        raw = item["path"]
        size = item["size"]
        digest = item["sha256"]
        if not isinstance(raw, str) or raw in seen:
            raise ValueError("checkpoint attestation has a duplicate path")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("checkpoint attestation has an invalid size")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("checkpoint attestation has an invalid digest")
        candidate = _strict_relative_file(path, raw)
        if candidate.stat().st_size != size or _sha256_file(candidate) != digest:
            raise ValueError(f"checkpoint payload changed after attestation: {raw}")
        seen.add(raw)
    actual = {item["path"] for item in _inventory(path)}
    if seen != actual or not required.issubset(seen):
        raise ValueError("checkpoint attestation does not cover the exact payload")
    if precision["fp16"] != (_SCALER_STATE in seen):
        raise ValueError("checkpoint scaler inventory does not match fp16 mode")
    # JSON and safetensors remain cheap and safe to parse in the orchestrator;
    # torch states are bound to the safe worker load above by their digests.
    trainer_state = _json_object(path / "trainer_state.json")
    if trainer_state.get("global_step") != step:
        raise ValueError("attested trainer state step changed")
    config_name = _ADAPTER_CONFIG if method == "lora" else _FULL_CONFIG
    if not _json_object(path / config_name):
        raise ValueError("attested model config is empty")
    weights = _weight_files(path, method=method)
    for weight in weights:
        if weight.suffix == ".safetensors" and not _safetensors_complete(weight):
            raise ValueError("attested safetensors payload is invalid")
    return dict(precision)


def seal_checkpoint(path: Path, *, method: str, training_args: Any) -> None:
    """Safely load-validate and atomically attest one Trainer checkpoint.

    Existing native files are never rewritten. If an attestation already
    exists but does not verify, the function fails closed instead of blessing
    the changed payload with a new digest.
    """

    if method not in _METHODS:
        raise ValueError(f"Unsupported LFM training method: {method!r}")
    attestation = path / _CHECKPOINT_ATTESTATION
    if attestation.exists():
        if checkpoint_complete(path, method=method):
            return
        raise RuntimeError("existing checkpoint attestation failed verification")
    expected_precision = _training_precision_from_object(training_args)
    precision = _validate_native_checkpoint(
        path,
        method=method,
        expected_precision=expected_precision,
    )
    marker = path / _TRAINING_METHOD_MARKER
    if not marker.exists():
        _atomic_write_text(marker, method + "\n")
    elif marker.read_text(encoding="utf-8").strip() != method:
        raise RuntimeError("existing checkpoint method marker does not match")
    document = {
        "schema_version": _ATTESTATION_SCHEMA,
        "step": checkpoint_step(path),
        "method": method,
        "precision": precision,
        "files": _inventory(path),
    }
    _atomic_write_json(attestation, document)
    if not checkpoint_complete(path, method=method):
        raise RuntimeError("checkpoint attestation failed final verification")


def checkpoint_complete(path: Path, *, method: str = "lora") -> bool:
    """Return whether a method-specific Trainer checkpoint can resume exactly.

    New checkpoints carry a digest-bound attestation written only after safe
    native loads succeed. A markerless v0.3 checkpoint can still resume inside
    the isolated LFM worker when the installed pinned runtime can perform the
    same safe validation; an environment lacking those loaders fails closed.
    """

    if method not in _METHODS:
        raise ValueError(f"Unsupported LFM training method: {method!r}")
    try:
        if not path.is_dir() or path.is_symlink() or checkpoint_step(path) is None:
            return False
        if (path / _INCOMPLETE_SENTINEL).exists():
            return False
        if (path / _CHECKPOINT_ATTESTATION).exists():
            _verify_attestation(path, method=method)
            return True
        _validate_native_checkpoint(path, method=method)
        return True
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile):
        return False


def latest_complete_checkpoint(output: Path, *, method: str = "lora") -> Path | None:
    if not output.is_dir():
        return None
    candidates = [
        (step, path)
        for path in output.glob("checkpoint-*")
        if (step := checkpoint_step(path)) is not None
        and checkpoint_complete(path, method=method)
    ]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def prune_incomplete_checkpoints(output: Path, *, method: str = "lora") -> list[Path]:
    """Return unusable numeric checkpoints without deleting or rewriting them.

    The historical function name is retained for worker API compatibility.
    Trainer receives an explicit, fully verified resume path, so an unverifiable
    newer directory cannot override it and does not need destructive pruning.
    """

    if method not in _METHODS:
        raise ValueError(f"Unsupported LFM training method: {method!r}")
    other_method = "full" if method == "lora" else "lora"
    rejected: list[Path] = []
    if not output.is_dir():
        return rejected
    for path in output.glob("checkpoint-*"):
        if not path.is_dir() or checkpoint_step(path) is None:
            continue
        if checkpoint_complete(path, method=method):
            continue
        if checkpoint_complete(path, method=other_method):
            continue
        rejected.append(path)
    return sorted(rejected, key=lambda path: path.name)
