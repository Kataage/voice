from __future__ import annotations

import json
import math
import pickletools
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

import yaml
from huggingface_hub import hf_hub_download

from personavoice.atomic import atomic_write_json, atomic_write_text
from personavoice.environment_contract import require_current_environment
from personavoice.hardware import irodori_training_precision, safe_batch_profile
from personavoice.media import sha256_file
from personavoice.model_assets import (
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_ID,
    IRODORI_MODEL_REVISION,
    IRODORI_MODEL_SHA256,
    IRODORI_SOURCE_REVISION,
    IRODORI_TEXT_ENCODER_ID,
    IRODORI_TEXT_ENCODER_REVISION,
)
from personavoice.process import run
from personavoice.workers import local_model_env

SUPPORTED_BACKENDS = {"cpu", "cu126", "cu128", "rocm", "xpu"}
_CHECKPOINT_STEP_RE = re.compile(r"^checkpoint_(\d+)(?:\.pt|\.speaker\.safetensors)?$")
_BEST_FULL_CHECKPOINT_RE = re.compile(r"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)\.pt$")
_BEST_VALIDATION_CHECKPOINT_RE = re.compile(
    r"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)"
    r"(?:\.pt|\.speaker\.safetensors)?$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LORA_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
_LORA_TRAINER_STATE = "trainer_state.pt"
_LORA_PROVENANCE_NAME = ".personavoice-provenance.json"
_SPEAKER_EMBEDDING_KEY = "speaker_embedding"
_SPEAKER_EMBEDDING_VALIDATION_SCRIPT = r"""
import sys
from pathlib import Path

from safetensors import safe_open

path = Path(sys.argv[1])
if not path.name.endswith(".speaker.safetensors"):
    raise ValueError("speaker inversion checkpoint has the wrong suffix")
with safe_open(str(path), framework="pt", device="cpu") as handle:
    metadata = handle.metadata()
    if metadata not in (None, {}):
        raise ValueError("speaker inversion checkpoint has unexpected metadata")
    keys = list(handle.keys())
    if keys != ["speaker_embedding"]:
        raise ValueError("speaker inversion checkpoint must contain exactly speaker_embedding")
    view = handle.get_slice("speaker_embedding")
    shape = tuple(view.get_shape())
    dtype = str(view.get_dtype()).rsplit(".", 1)[-1]
if len(shape) != 2 or any(not isinstance(size, int) or size <= 0 for size in shape):
    raise ValueError("speaker_embedding must be a non-empty two-dimensional tensor")
if dtype != "F32":
    raise ValueError("speaker_embedding must use the pinned upstream float32 format")
"""
_FULL_CHECKPOINT_VALIDATION_SCRIPT = r"""
import inspect
import re
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
match = re.fullmatch(
    r"checkpoint_(\d+)\.pt|checkpoint_best_val_loss_(\d+)_-?\d+(?:\.\d+)?\.pt",
    path.name,
)
if match is None:
    raise ValueError("checkpoint filename does not encode a native step")
filename_step = int(match.group(1) or match.group(2))
kwargs = {"map_location": "cpu", "weights_only": True}
if "mmap" in inspect.signature(torch.load).parameters:
    kwargs["mmap"] = True
payload = torch.load(path, **kwargs)
required = {
    "step",
    "model",
    "optimizer",
    "scheduler",
    "model_config",
    "train_config",
    "dataloader_state",
    "runtime_state",
}
if not isinstance(payload, dict):
    raise TypeError("checkpoint payload is not a mapping")
missing = sorted(required - set(payload))
if missing:
    raise ValueError(f"checkpoint is missing resumable state: {missing}")
if (
    not isinstance(payload["step"], int)
    or isinstance(payload["step"], bool)
    or payload["step"] < 0
):
    raise ValueError("checkpoint step is invalid")
if payload["step"] != filename_step:
    raise ValueError("checkpoint filename step does not match native trainer step")
if not isinstance(payload["model"], dict) or not payload["model"]:
    raise ValueError("checkpoint model state is empty")
if not isinstance(payload["model_config"], dict) or not payload["model_config"]:
    raise ValueError("checkpoint model config is invalid")
if not isinstance(payload["train_config"], dict) or not payload["train_config"]:
    raise ValueError("checkpoint model/train config is invalid")
train_config = payload["train_config"]
optimizer = payload["optimizer"]

def standard_optimizer_complete(state):
    return (
        isinstance(state, dict)
        and isinstance(state.get("state"), dict)
        and bool(state["state"])
        and isinstance(state.get("param_groups"), list)
        and bool(state["param_groups"])
        and all(
            isinstance(group, dict)
            and isinstance(group.get("params"), list)
            and bool(group["params"])
            for group in state["param_groups"]
        )
    )

optimizer_name = str(train_config.get("optimizer", "")).strip().lower()
if optimizer_name == "adamw":
    optimizer_complete = standard_optimizer_complete(optimizer)
elif optimizer_name == "muon":
    optimizer_complete = (
        isinstance(optimizer, dict)
        and standard_optimizer_complete(optimizer.get("muon"))
        and (
            optimizer.get("aux") is None
            or standard_optimizer_complete(optimizer.get("aux"))
        )
    )
else:
    optimizer_complete = False
if not optimizer_complete:
    raise ValueError("checkpoint optimizer state is incomplete")
scheduler_name = str(train_config.get("lr_scheduler", "")).strip().lower()
scheduler = payload["scheduler"]
if scheduler_name == "none":
    if scheduler is not None:
        raise ValueError("disabled scheduler unexpectedly has state")
elif scheduler_name in {"cosine", "wsd"}:
    if (
        not isinstance(scheduler, dict)
        or not isinstance(scheduler.get("base_lrs"), list)
        or not scheduler["base_lrs"]
        or not isinstance(scheduler.get("last_step"), int)
        or isinstance(scheduler["last_step"], bool)
        # ScalarLRScheduler starts at -1 and upstream calls scheduler.step()
        # before incrementing the public training step.  A checkpoint saved
        # after N optimizer updates therefore carries last_step == N - 1.
        or scheduler["last_step"] != filename_step - 1
    ):
        raise ValueError("checkpoint scheduler state does not match native step")
else:
    raise ValueError("checkpoint scheduler kind is unsupported")
dataloader = payload["dataloader_state"]
if (
    not isinstance(dataloader, dict)
    or dataloader.get("version") != 1
    or dataloader.get("world_size") != 1
    or not isinstance(dataloader.get("rank_states"), list)
    or len(dataloader["rank_states"]) != 1
    or not isinstance(dataloader["rank_states"][0], dict)
    or not dataloader["rank_states"][0]
):
    raise ValueError("checkpoint single-process dataloader state is incomplete")
runtime = payload["runtime_state"]
if not isinstance(runtime, dict):
    raise ValueError("checkpoint runtime state is invalid")
for key in ("epoch", "sampler_epoch", "epoch_step"):
    if not isinstance(runtime.get(key), int) or isinstance(runtime[key], bool):
        raise ValueError("checkpoint runtime state contains a non-integer counter")
if (
    runtime["epoch"] < 1
    or runtime["sampler_epoch"] != max(0, runtime["epoch"] - 1)
    or runtime["epoch_step"] < 0
):
    raise ValueError("checkpoint runtime epoch state is inconsistent")
max_steps = train_config.get("max_steps")
if (
    not isinstance(max_steps, int)
    or isinstance(max_steps, bool)
    or max_steps < filename_step
):
    raise ValueError("checkpoint train_config max_steps is inconsistent")
"""
_LORA_CHECKPOINT_VALIDATION_SCRIPT = r"""
import inspect
import json
import re
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
match = re.fullmatch(
    r"checkpoint_(\d+)|checkpoint_best_val_loss_(\d+)_-?\d+(?:\.\d+)?",
    checkpoint.name,
)
if match is None:
    raise ValueError("checkpoint directory does not encode a native step")
filename_step = int(match.group(1) or match.group(2))
config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
if not isinstance(config, dict) or not config:
    raise ValueError("adapter config is invalid")
safe_weight = checkpoint / "adapter_model.safetensors"
binary_weight = checkpoint / "adapter_model.bin"
if safe_weight.is_file() and binary_weight.is_file():
    raise ValueError("adapter weight payload is ambiguous")
if safe_weight.is_file():
    from safetensors import safe_open

    with safe_open(str(safe_weight), framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise ValueError("adapter safetensors contains no tensors")
elif binary_weight.is_file():
    weights = torch.load(binary_weight, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not weights:
        raise ValueError("adapter binary state is invalid")
else:
    raise ValueError("adapter weights are missing")
kwargs = {"map_location": "cpu", "weights_only": True}
if "mmap" in inspect.signature(torch.load).parameters:
    kwargs["mmap"] = True
payload = torch.load(checkpoint / "trainer_state.pt", **kwargs)
required = {
    "step",
    "optimizer",
    "scheduler",
    "model_config",
    "train_config",
    "dataloader_state",
    "runtime_state",
}
if not isinstance(payload, dict) or not required.issubset(payload):
    raise ValueError("LoRA trainer state is incomplete")
if (
    not isinstance(payload["step"], int)
    or isinstance(payload["step"], bool)
    or payload["step"] != filename_step
):
    raise ValueError("checkpoint directory step does not match native trainer step")
optimizer = payload["optimizer"]

def standard_optimizer_complete(state):
    return (
        isinstance(state, dict)
        and isinstance(state.get("state"), dict)
        and bool(state["state"])
        and isinstance(state.get("param_groups"), list)
        and bool(state["param_groups"])
        and all(
            isinstance(group, dict)
            and isinstance(group.get("params"), list)
            and bool(group["params"])
            for group in state["param_groups"]
        )
    )

train_config = payload["train_config"]
optimizer_name = str(train_config.get("optimizer", "")).strip().lower()
if optimizer_name == "adamw":
    optimizer_complete = standard_optimizer_complete(optimizer)
elif optimizer_name == "muon":
    optimizer_complete = (
        isinstance(optimizer, dict)
        and standard_optimizer_complete(optimizer.get("muon"))
        and (
            optimizer.get("aux") is None
            or standard_optimizer_complete(optimizer.get("aux"))
        )
    )
else:
    optimizer_complete = False
if not optimizer_complete:
    raise ValueError("LoRA optimizer state is incomplete")
if not isinstance(payload["model_config"], dict) or not payload["model_config"]:
    raise ValueError("LoRA model config is invalid")
if not isinstance(train_config, dict) or not train_config:
    raise ValueError("LoRA train config is invalid")
scheduler_name = str(train_config.get("lr_scheduler", "")).strip().lower()
scheduler = payload["scheduler"]
if scheduler_name == "none":
    if scheduler is not None:
        raise ValueError("disabled LoRA scheduler unexpectedly has state")
elif scheduler_name in {"cosine", "wsd"}:
    if (
        not isinstance(scheduler, dict)
        or not isinstance(scheduler.get("base_lrs"), list)
        or not scheduler["base_lrs"]
        or not isinstance(scheduler.get("last_step"), int)
        or isinstance(scheduler["last_step"], bool)
        # Keep this bound to the pinned upstream update order: the scalar
        # scheduler is advanced before the public step counter, so its native
        # state is exactly one behind the checkpoint filename/trainer step.
        or scheduler["last_step"] != filename_step - 1
    ):
        raise ValueError("LoRA scheduler state does not match native step")
else:
    raise ValueError("LoRA scheduler kind is unsupported")
dataloader = payload["dataloader_state"]
if (
    not isinstance(dataloader, dict)
    or dataloader.get("version") != 1
    or dataloader.get("world_size") != 1
    or not isinstance(dataloader.get("rank_states"), list)
    or len(dataloader["rank_states"]) != 1
    or not isinstance(dataloader["rank_states"][0], dict)
    or not dataloader["rank_states"][0]
):
    raise ValueError("LoRA single-process dataloader state is incomplete")
runtime = payload["runtime_state"]
if not isinstance(runtime, dict):
    raise ValueError("LoRA runtime state is invalid")
for key in ("epoch", "sampler_epoch", "epoch_step"):
    if not isinstance(runtime.get(key), int) or isinstance(runtime[key], bool):
        raise ValueError("LoRA runtime state contains a non-integer counter")
if (
    runtime["epoch"] < 1
    or runtime["sampler_epoch"] != max(0, runtime["epoch"] - 1)
    or runtime["epoch_step"] < 0
):
    raise ValueError("LoRA runtime epoch state is inconsistent")
max_steps = train_config.get("max_steps")
if (
    not isinstance(max_steps, int)
    or isinstance(max_steps, bool)
    or max_steps < filename_step
):
    raise ValueError("LoRA train_config max_steps is inconsistent")
"""


def vendor_dir(repo_root: Path) -> Path:
    path = repo_root / "vendor" / "Irodori-TTS"
    if not (path / "infer.py").is_file() or not (path / ".git").exists():
        raise FileNotFoundError("Irodori-TTS is not installed. Run `persona setup` first.")
    try:
        head = run(["git", "rev-parse", "HEAD"], cwd=path, capture=True).stdout.strip()
        status = run(
            ["git", "status", "--porcelain"],
            cwd=path,
            capture=True,
        ).stdout.strip()
    except Exception as exc:
        raise RuntimeError("Irodori vendor checkout integrity could not be verified") from exc
    if head != IRODORI_SOURCE_REVISION:
        raise RuntimeError(
            f"Irodori vendor HEAD mismatch: expected {IRODORI_SOURCE_REVISION}, got {head}. "
            "Run `persona setup` to restore the audited checkout."
        )
    if status:
        raise RuntimeError(
            "Irodori vendor checkout has local modifications or untracked files. "
            "Restore the checkout and run `persona setup` before model work."
        )
    return path


def configured_backend(repo_root: Path) -> str:
    value = require_current_environment(repo_root)
    if value.get("irodori_backend") is None:
        setup = repo_root / ".runtime" / "setup.json"
        raise RuntimeError(
            f"PersonaVoice setup state does not record irodori_backend: {setup}. "
            "Re-run `persona setup`."
        )
    backend = str(value["irodori_backend"])
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimeError(f"Unsupported recorded Irodori backend: {backend!r}")
    return backend


def backend_device(backend: str) -> str:
    if backend in {"cu126", "cu128", "rocm"}:
        return "cuda"
    if backend == "xpu":
        return "xpu"
    if backend == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported Irodori backend: {backend!r}")


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise RuntimeError(f"{label} checksum could not be read: {path}") from exc
    if actual != expected:
        raise RuntimeError(
            f"{label} checksum mismatch: expected {expected}, got {actual}. "
            "Run `persona setup --download-models` to restore the audited asset."
        )


def speaker_embedding_complete(path: Path) -> bool:
    return _nonempty_file(path)


def lora_adapter_weight(path: Path) -> Path | None:
    for name in _LORA_WEIGHT_NAMES:
        candidate = path / name
        if _nonempty_file(candidate):
            return candidate
    return None


def lora_adapter_complete(path: Path) -> bool:
    return _nonempty_file(path / "adapter_config.json") and lora_adapter_weight(path) is not None


def _lora_native_trainer_step(path: Path) -> int | None:
    """Statically read the first top-level ``step`` scalar from torch.save.

    Only pickle bytecode is parsed; no GLOBAL is imported or executed. The
    isolated pinned Irodori runtime subsequently performs the authoritative
    ``weights_only=True`` load before a checkpoint is resumed or observed.
    """

    state = path / _LORA_TRAINER_STATE
    try:
        if not _nonempty_file(state) or not zipfile.is_zipfile(state):
            return None
        with zipfile.ZipFile(state) as archive:
            pickle_names = [
                name
                for name in archive.namelist()
                if name == "data.pkl" or name.endswith("/data.pkl")
            ]
            if len(pickle_names) != 1:
                return None
            if archive.getinfo(pickle_names[0]).file_size > 64 * 1024 * 1024:
                return None
            payload = archive.read(pickle_names[0])
        pending_step = False
        for opcode, argument, _position in pickletools.genops(payload):
            if opcode.name in {
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "UNICODE",
                "BINSTRING",
                "SHORT_BINSTRING",
                "STRING",
            }:
                pending_step = argument == "step"
                continue
            if pending_step and opcode.name in {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}:
                continue
            if pending_step and opcode.name in {"BININT", "BININT1", "BININT2", "INT", "LONG"}:
                if isinstance(argument, int) and not isinstance(argument, bool) and argument >= 0:
                    return argument
                return None
            pending_step = False
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    return None


def lora_resume_checkpoint_complete(path: Path) -> bool:
    if not lora_adapter_complete(path) or not _nonempty_file(path / _LORA_TRAINER_STATE):
        return False
    match = _CHECKPOINT_STEP_RE.fullmatch(path.name)
    if match is None:
        match = _BEST_VALIDATION_CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        # ``checkpoint_final`` remains a v0.3 inference/candidate compatibility
        # path. Numeric resume and validation directories are step-bound below.
        return True
    return _lora_native_trainer_step(path) == int(match.group(1))


def codec_checkpoint(repo_root: Path) -> Path:
    expected = repo_root / "models" / "irodori" / "dacvae" / IRODORI_DACVAE_FILENAME
    if not _nonempty_file(expected):
        raise FileNotFoundError(
            f"Irodori DACVAE is not materialized at {expected}. "
            "Run `persona setup --download-models`."
        )
    _verify_sha256(expected, IRODORI_DACVAE_SHA256, label="Irodori DACVAE")
    return expected


def base_checkpoint(repo_root: Path, *, online: bool = False) -> Path:
    env = local_model_env(repo_root, offline=not online)
    local_dir = repo_root / "models" / "irodori" / "v4.1-small"
    expected = local_dir / IRODORI_MODEL_FILENAME
    if _nonempty_file(expected):
        try:
            _verify_sha256(expected, IRODORI_MODEL_SHA256, label="Irodori base checkpoint")
            return expected
        except RuntimeError:
            if not online:
                raise
            expected.unlink(missing_ok=True)
    elif not online:
        raise FileNotFoundError(
            f"Irodori base checkpoint is not materialized at {expected}. "
            "Run `persona setup --download-models`."
        )
    expected.unlink(missing_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=IRODORI_MODEL_ID,
        filename=IRODORI_MODEL_FILENAME,
        revision=IRODORI_MODEL_REVISION,
        local_dir=local_dir,
        cache_dir=Path(env["HUGGINGFACE_HUB_CACHE"]),
    )
    if not _nonempty_file(expected):
        raise FileNotFoundError(f"Irodori download completed but {expected} was not created")
    try:
        _verify_sha256(expected, IRODORI_MODEL_SHA256, label="Irodori base checkpoint")
    except Exception:
        expected.unlink(missing_ok=True)
        raise
    return expected


def materialize_base(repo_root: Path) -> Path:
    return base_checkpoint(repo_root, online=True)


def prepare_manifest(
    repo_root: Path,
    source_jsonl: Path,
    output_manifest: Path,
    latent_dir: Path,
) -> None:
    vendor = vendor_dir(repo_root)
    backend = configured_backend(repo_root)
    device = backend_device(backend)
    codec = codec_checkpoint(repo_root)
    env = local_model_env(repo_root)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "uv",
            "run",
            "--project",
            vendor,
            "--no-sync",
            "python",
            vendor / "prepare_manifest.py",
            "--dataset",
            "json",
            "--data-files",
            source_jsonl,
            "--audio-column",
            "audio",
            "--text-column",
            "text",
            "--caption-column",
            "caption",
            "--speaker-column",
            "speaker",
            "--output-manifest",
            output_manifest,
            "--latent-dir",
            latent_dir,
            "--codec-repo",
            codec,
            "--device",
            device,
        ],
        cwd=vendor,
        env=env,
    )
    if not output_manifest.exists() or output_manifest.stat().st_size == 0:
        raise RuntimeError("Irodori manifest preparation produced no training examples")


def _validate_pinned_model_config(data: dict, *, source: Path) -> None:
    model_cfg = data.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Irodori training config has invalid model section: {source}")
    actual_repo = model_cfg.get("text_tokenizer_repo")
    actual_revision = model_cfg.get("text_encoder_revision")
    if actual_repo != IRODORI_TEXT_ENCODER_ID or actual_revision != IRODORI_TEXT_ENCODER_REVISION:
        raise RuntimeError(
            "Pinned Irodori training config no longer matches the audited text encoder: "
            f"repo={actual_repo!r}, revision={actual_revision!r}; expected "
            f"{IRODORI_TEXT_ENCODER_ID!r}@{IRODORI_TEXT_ENCODER_REVISION}. "
            "Update PersonaVoice asset pins and re-audit before training."
        )
    caption_repo = model_cfg.get("caption_tokenizer_repo")
    if caption_repo not in {None, IRODORI_TEXT_ENCODER_ID}:
        raise RuntimeError(
            "Pinned Irodori caption tokenizer no longer matches the audited text encoder: "
            f"{caption_repo!r}"
        )


def _patched_config(
    source: Path,
    destination: Path,
    *,
    max_steps: int,
    backend: str,
    validation_ratio: float | None = None,
    validation_every: int | None = None,
    checkpoint_best_n: int | None = None,
) -> Path:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Irodori training config is not a YAML mapping: {source}")
    _validate_pinned_model_config(data, source=source)
    profile = safe_batch_profile(backend=backend)
    train_cfg = data.setdefault("train", {})
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Irodori training config has invalid train section: {source}")
    train_cfg["batch_size"] = int(profile["batch_size"])
    train_cfg["gradient_accumulation_steps"] = int(profile["gradient_accumulation_steps"])
    train_cfg["num_workers"] = int(profile["num_workers"])
    train_cfg["max_steps"] = int(max_steps)
    train_cfg["stable_steps"] = max(0, int(max_steps * 0.75))
    train_cfg["warmup_steps"] = min(250, max(20, int(max_steps * 0.05)))
    train_cfg["save_every"] = max(100, min(500, max_steps // 4 or 100))
    train_cfg["valid_every"] = train_cfg["save_every"]
    if validation_ratio is not None:
        if not 0.0 < float(validation_ratio) < 1.0:
            raise ValueError("Irodori validation_ratio must be greater than 0 and less than 1")
        train_cfg["valid_ratio"] = float(validation_ratio)
    if validation_every is not None:
        if int(validation_every) <= 0:
            raise ValueError("Irodori validation_every must be greater than 0")
        train_cfg["valid_every"] = int(validation_every)
        train_cfg["save_every"] = int(validation_every)
    if checkpoint_best_n is not None:
        if int(checkpoint_best_n) <= 0:
            raise ValueError("Irodori checkpoint_best_n must be greater than 0")
        train_cfg["checkpoint_best_n"] = int(checkpoint_best_n)
    if backend in {"cu126", "cu128"}:
        precision_policy = irodori_training_precision(backend)
        train_cfg["precision"] = str(precision_policy["precision"])
        train_cfg["allow_tf32"] = bool(precision_policy["allow_tf32"])
    elif backend == "cpu":
        train_cfg["dataloader_cuda_prefetch"] = False
        train_cfg["precision"] = "fp32"
        train_cfg["allow_tf32"] = False
    if "gradient_checkpointing" in train_cfg or profile["gradient_checkpointing"]:
        train_cfg["gradient_checkpointing"] = bool(profile["gradient_checkpointing"])
    atomic_write_text(
        destination,
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
    )
    return destination


def _patched_full_config(
    source: Path,
    destination: Path,
    *,
    max_steps: int,
    backend: str,
    validation_ratio: float,
    validation_every: int,
    checkpoint_best_n: int,
) -> Path:
    """Patch only runtime sizing while retaining the pinned v4 full contract."""

    if source.name != "train_v4_small.yaml":
        raise ValueError(f"Irodori full training requires train_v4_small.yaml, got {source.name}")
    _patched_config(
        source,
        destination,
        max_steps=max_steps,
        backend=backend,
        validation_ratio=validation_ratio,
        validation_every=validation_every,
        checkpoint_best_n=checkpoint_best_n,
    )
    data = yaml.safe_load(destination.read_text(encoding="utf-8"))
    train_cfg = data.get("train") if isinstance(data, dict) else None
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Irodori full training config has invalid train section: {destination}")
    valid_ratio = float(train_cfg.get("valid_ratio", 0.0))
    valid_every = int(train_cfg.get("valid_every", 0))
    checkpoint_best_n = int(train_cfg.get("checkpoint_best_n", 0))
    if not 0.0 < valid_ratio < 1.0:
        raise RuntimeError("Pinned Irodori full config must enable a held-out validation split")
    if valid_every <= 0 or checkpoint_best_n <= 0:
        raise RuntimeError(
            "Pinned Irodori full config must enable validation and best-checkpoint retention"
        )
    return destination


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _latest_numeric_checkpoint(paths: list[Path]) -> Path | None:
    ranked = [(step, path) for path in paths if (step := _checkpoint_step(path)) is not None]
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _latest_resume(output_dir: Path, *, verify=None) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        path
        for path in output_dir.glob("checkpoint_*")
        if path.is_dir() and lora_resume_checkpoint_complete(path)
    ]
    if verify is None:
        return _latest_numeric_checkpoint(candidates)
    ranked = [
        (step, path)
        for path in candidates
        if (step := _checkpoint_step(path)) is not None
    ]
    for _, path in sorted(ranked, key=lambda item: item[0], reverse=True):
        if verify(path):
            return path
    if ranked:
        raise RuntimeError(
            "Irodori LoRA run contains numeric checkpoints, but none have complete "
            "step-bound native trainer state"
        )
    return None


def _latest_verified_speaker_embedding(
    output_dir: Path,
    *,
    verify,
) -> Path | None:
    """Select the newest safely loadable upstream speaker checkpoint.

    Invalid files remain untouched. If at least one numeric checkpoint exists
    but none can be verified, fail instead of silently warm-starting from a
    partial payload or overwriting the only recovery evidence.
    """

    if not output_dir.is_dir():
        return None
    candidates = [
        (step, path)
        for path in output_dir.glob("checkpoint_*.speaker.safetensors")
        if (step := _checkpoint_step(path)) is not None
    ]
    for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        if verify(path):
            return path
    if candidates:
        raise RuntimeError(
            "Irodori Speaker Inversion run contains numeric checkpoints, but none "
            "is a complete upstream speaker_embedding safetensors payload"
        )
    return None


def _validate_plan_fingerprint(value: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError("Irodori training plan fingerprint must be a SHA-256 hex digest")
    return normalized


def _verify_speaker_embedding_checkpoint(
    vendor: Path,
    checkpoint: Path,
    *,
    env: dict[str, str],
) -> bool:
    """Validate an upstream Speaker Inversion payload in its pinned runtime."""

    if (
        not _nonempty_file(checkpoint)
        or not checkpoint.name.endswith(".speaker.safetensors")
    ):
        return False
    completed = run(
        [
            "uv",
            "run",
            "--project",
            vendor,
            "--no-sync",
            "python",
            "-c",
            _SPEAKER_EMBEDDING_VALIDATION_SCRIPT,
            checkpoint,
        ],
        cwd=vendor,
        env=env,
        capture=True,
        check=False,
    )
    return completed.returncode == 0


def _verify_full_training_checkpoint(
    vendor: Path,
    checkpoint: Path,
    *,
    env: dict[str, str],
) -> bool:
    if not _nonempty_file(checkpoint) or checkpoint.suffix.lower() != ".pt":
        return False
    completed = run(
        [
            "uv",
            "run",
            "--project",
            vendor,
            "--no-sync",
            "python",
            "-c",
            _FULL_CHECKPOINT_VALIDATION_SCRIPT,
            checkpoint,
        ],
        cwd=vendor,
        env=env,
        capture=True,
        check=False,
    )
    return completed.returncode == 0


def _verify_lora_training_checkpoint(
    vendor: Path,
    checkpoint: Path,
    *,
    env: dict[str, str],
) -> bool:
    """Safely load a LoRA checkpoint and bind its directory step to trainer state."""

    if not checkpoint.is_dir() or not lora_resume_checkpoint_complete(checkpoint):
        return False
    if _checkpoint_step(checkpoint) is None and irodori_validation_checkpoint_metadata(
        checkpoint
    ) is None:
        return False
    completed = run(
        [
            "uv",
            "run",
            "--project",
            vendor,
            "--no-sync",
            "python",
            "-c",
            _LORA_CHECKPOINT_VALIDATION_SCRIPT,
            checkpoint,
        ],
        cwd=vendor,
        env=env,
        capture=True,
        check=False,
    )
    return completed.returncode == 0


def _latest_verified_full_resume(
    output_dir: Path,
    *,
    verify,
) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = [
        (step, path)
        for path in output_dir.glob("checkpoint_*.pt")
        if (step := _checkpoint_step(path)) is not None
    ]
    if not candidates:
        return None
    for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        if verify(path):
            return path
    raise RuntimeError(
        "Irodori full run contains numeric checkpoints, but none have complete resumable state"
    )


def _best_verified_full_checkpoint(
    output_dir: Path,
    *,
    verify,
) -> tuple[Path, float, int]:
    candidates: list[tuple[float, int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint_best_val_loss_*.pt"):
            match = _BEST_FULL_CHECKPOINT_RE.fullmatch(path.name)
            if match is None:
                continue
            loss = float(match.group(2))
            if math.isfinite(loss):
                candidates.append((loss, int(match.group(1)), path))
    if not candidates:
        raise RuntimeError("Irodori full training produced no best validation checkpoint")
    for loss, step, path in sorted(candidates, key=lambda item: (item[0], item[1])):
        if verify(path):
            return path, loss, step
    raise RuntimeError("Irodori best validation checkpoints are all incomplete or unreadable")


def irodori_validation_checkpoint_metadata(path: Path) -> tuple[int, float] | None:
    """Return the upstream step/loss encoded in a best-validation checkpoint name."""

    match = _BEST_VALIDATION_CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        return None
    loss = float(match.group(2))
    if not math.isfinite(loss):
        return None
    return int(match.group(1)), loss


def _best_validation_checkpoint(
    output_dir: Path,
    *,
    complete,
) -> tuple[Path, float, int] | None:
    candidates: list[tuple[float, int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint_best_val_loss_*"):
            metadata = irodori_validation_checkpoint_metadata(path)
            if metadata is None or not complete(path):
                continue
            step, loss = metadata
            candidates.append((loss, step, path))
    if not candidates:
        return None
    loss, step, path = min(candidates, key=lambda item: (item[0], item[1]))
    return path, loss, step


def _best_validation_loss(output_dir: Path) -> float | None:
    """Retain the v0.3 filename-only metric lookup for legacy direct callers."""

    values = [
        loss
        for path in output_dir.glob("checkpoint_best_val_loss_*")
        if (metadata := irodori_validation_checkpoint_metadata(path)) is not None
        for _, loss in (metadata,)
    ]
    return min(values) if values else None


def irodori_lora_candidate_complete(
    adapter_dir: Path,
    *,
    plan_fingerprint: str,
) -> bool:
    """Verify the extra provenance required only for schema-v9 candidates.

    ``lora_adapter_complete`` intentionally remains the v0.3 structural
    contract. Keeping this verifier separate lets the orchestrator adopt a
    completed legacy adapter without writing into or invalidating it.
    """

    normalized = str(plan_fingerprint).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None or not lora_adapter_complete(adapter_dir):
        return False
    provenance_path = adapter_dir / _LORA_PROVENANCE_NAME
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
        or provenance.get("family") != "irodori"
        or provenance.get("method") != "lora"
        or provenance.get("training_plan_fingerprint") != normalized
        or provenance.get("upstream_source_revision") != IRODORI_SOURCE_REVISION
        or provenance.get("base_model")
        != {
            "id": IRODORI_MODEL_ID,
            "revision": IRODORI_MODEL_REVISION,
            "sha256": IRODORI_MODEL_SHA256,
        }
    ):
        return False
    best_validation_loss = provenance.get("best_validation_loss")
    if (
        not isinstance(best_validation_loss, (int, float))
        or isinstance(best_validation_loss, bool)
        or not math.isfinite(float(best_validation_loss))
    ):
        return False
    best_step = provenance.get("best_step")
    selected_checkpoint = provenance.get("selected_checkpoint")
    if (
        not isinstance(best_step, int)
        or isinstance(best_step, bool)
        or best_step <= 0
        or not isinstance(selected_checkpoint, str)
        or Path(selected_checkpoint).name != selected_checkpoint
        or "\\" in selected_checkpoint
    ):
        return False
    metadata = irodori_validation_checkpoint_metadata(Path(selected_checkpoint))
    if metadata != (best_step, float(best_validation_loss)):
        return False
    try:
        adapter_config = json.loads(
            (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(adapter_config, dict) or _json_contains_absolute_path(adapter_config):
        return False
    weight = lora_adapter_weight(adapter_dir)
    if weight is None:
        return False
    actual_files = {
        path.relative_to(adapter_dir).as_posix(): path
        for path in adapter_dir.rglob("*")
        if path.is_file() and path != provenance_path
    }
    if set(actual_files) != {"adapter_config.json", weight.name}:
        return False
    raw_files = provenance.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != 2:
        return False
    recorded_files: dict[str, tuple[int, str]] = {}
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            return False
        relative = _safe_artifact_path(item.get("path"))
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            relative is None
            or relative in recorded_files
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            return False
        recorded_files[relative] = (size, digest)
    actual_contract = {
        relative: (path.stat().st_size, sha256_file(path))
        for relative, path in actual_files.items()
    }
    return (
        recorded_files == actual_contract
        and not _json_contains_absolute_path(provenance)
    )


def _write_irodori_lora_candidate_provenance(
    adapter_dir: Path,
    *,
    plan_fingerprint: str,
    best_validation_loss: float,
    best_step: int,
    selected_checkpoint: str,
) -> Path:
    normalized = _validate_plan_fingerprint(plan_fingerprint)
    if not math.isfinite(float(best_validation_loss)):
        raise ValueError("Irodori LoRA best validation loss must be finite")
    if not isinstance(best_step, int) or isinstance(best_step, bool) or best_step <= 0:
        raise ValueError("Irodori LoRA best checkpoint step must be a positive integer")
    metadata = irodori_validation_checkpoint_metadata(Path(selected_checkpoint))
    if metadata != (best_step, float(best_validation_loss)):
        raise ValueError(
            "Irodori LoRA selected checkpoint does not encode its best step/loss"
        )
    if not lora_adapter_complete(adapter_dir):
        raise RuntimeError("Irodori LoRA portable adapter is incomplete")
    weight = lora_adapter_weight(adapter_dir)
    assert weight is not None
    payload_paths = [adapter_dir / "adapter_config.json", weight]
    actual_files = {
        path.relative_to(adapter_dir).as_posix()
        for path in adapter_dir.rglob("*")
        if path.is_file() and path.name != _LORA_PROVENANCE_NAME
    }
    if actual_files != {path.name for path in payload_paths}:
        raise RuntimeError("Irodori LoRA portable adapter contains unexpected files")
    provenance_path = adapter_dir / _LORA_PROVENANCE_NAME
    if provenance_path.exists():
        if irodori_lora_candidate_complete(
            adapter_dir,
            plan_fingerprint=normalized,
        ):
            return provenance_path
        raise FileExistsError(
            "Refusing to overwrite invalid or different Irodori LoRA candidate provenance: "
            f"{provenance_path}"
        )
    provenance = {
        "schema_version": 1,
        "family": "irodori",
        "method": "lora",
        "training_plan_fingerprint": normalized,
        "upstream_source_revision": IRODORI_SOURCE_REVISION,
        "base_model": {
            "id": IRODORI_MODEL_ID,
            "revision": IRODORI_MODEL_REVISION,
            "sha256": IRODORI_MODEL_SHA256,
        },
        "best_validation_loss": float(best_validation_loss),
        "best_step": best_step,
        "selected_checkpoint": selected_checkpoint,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(payload_paths, key=lambda item: item.name)
        ],
    }
    if _json_contains_absolute_path(provenance):
        raise RuntimeError("Irodori LoRA candidate provenance contains an absolute path")
    atomic_write_json(provenance_path, provenance)
    if not irodori_lora_candidate_complete(
        adapter_dir,
        plan_fingerprint=normalized,
    ):
        raise RuntimeError("Irodori LoRA candidate provenance failed integrity verification")
    return provenance_path


def _finalize_irodori_lora_candidate(
    source_checkpoint: Path,
    artifact_dir: Path,
    *,
    plan_fingerprint: str,
    best_validation_loss: float,
    best_step: int,
) -> Path:
    """Publish only the upstream-selected adapter weights, never trainer state."""

    normalized = _validate_plan_fingerprint(plan_fingerprint)
    if artifact_dir.exists():
        if irodori_lora_candidate_complete(
            artifact_dir,
            plan_fingerprint=normalized,
        ):
            return artifact_dir
        raise FileExistsError(
            "Refusing to overwrite an incomplete or different Irodori LoRA candidate: "
            f"{artifact_dir}"
        )
    source_weight = lora_adapter_weight(source_checkpoint)
    if source_weight is None or not _nonempty_file(source_checkpoint / "adapter_config.json"):
        raise RuntimeError("Irodori best-validation LoRA checkpoint has no complete adapter")
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = artifact_dir.parent / f".{artifact_dir.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        config_path = staging / "adapter_config.json"
        shutil.copy2(source_checkpoint / "adapter_config.json", config_path)
        try:
            adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Irodori LoRA adapter config is unreadable") from exc
        if not isinstance(adapter_config, dict):
            raise RuntimeError("Irodori LoRA adapter config root must be a mapping")
        for key in ("_name_or_path", "name_or_path", "base_model_name_or_path"):
            if key in adapter_config:
                adapter_config[key] = IRODORI_MODEL_ID
        if _json_contains_absolute_path(adapter_config):
            raise RuntimeError("Irodori LoRA adapter config contains an absolute local path")
        atomic_write_json(config_path, adapter_config)
        shutil.copy2(source_weight, staging / source_weight.name)
        _write_irodori_lora_candidate_provenance(
            staging,
            plan_fingerprint=normalized,
            best_validation_loss=best_validation_loss,
            best_step=best_step,
            selected_checkpoint=source_checkpoint.name,
        )
        if not irodori_lora_candidate_complete(
            staging,
            plan_fingerprint=normalized,
        ):
            raise RuntimeError("Irodori LoRA portable candidate failed final verification")
        staging.replace(artifact_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return artifact_dir


def _safe_artifact_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate.as_posix()


def _json_contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_json_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_json_contains_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    return value.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value) is not None


def _portable_inventory(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        if not _nonempty_file(path):
            raise RuntimeError(f"Irodori full artifact contains an empty file: {relative}")
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def irodori_full_artifact_complete(
    artifact_dir: Path,
    *,
    plan_fingerprint: str | None = None,
) -> bool:
    manifest_path = artifact_dir / "manifest.json"
    provenance_path = artifact_dir / "provenance.json"
    if not artifact_dir.is_dir() or not _nonempty_file(manifest_path):
        return False
    if not _nonempty_file(provenance_path):
        return False
    try:
        artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(artifact_manifest, dict) or not isinstance(provenance, dict):
        return False
    if (
        artifact_manifest.get("schema_version") != 1
        or artifact_manifest.get("family") != "irodori"
        or artifact_manifest.get("method") != "full"
        or provenance.get("schema_version") != 1
        or provenance.get("family") != "irodori"
        or provenance.get("method") != "full"
    ):
        return False
    recorded_plan = provenance.get("training_plan_fingerprint")
    if not isinstance(recorded_plan, str) or _SHA256_RE.fullmatch(recorded_plan) is None:
        return False
    if artifact_manifest.get("training_plan_fingerprint") != recorded_plan:
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
    if provenance.get("base_model") != {
        "id": IRODORI_MODEL_ID,
        "revision": IRODORI_MODEL_REVISION,
        "sha256": IRODORI_MODEL_SHA256,
    }:
        return False
    if provenance.get("upstream_source_revision") != IRODORI_SOURCE_REVISION:
        return False
    if _json_contains_absolute_path(artifact_manifest) or _json_contains_absolute_path(provenance):
        return False
    raw_files = artifact_manifest.get("files")
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
        path = artifact_dir / PurePosixPath(relative)
        if not _nonempty_file(path):
            return False
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            return False
    if not {"model.safetensors", "tokenizer/tokenizer_config.json", "provenance.json"}.issubset(
        seen
    ):
        return False
    actual = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    return seen == actual


def _finalize_irodori_full_artifact(
    *,
    vendor: Path,
    env: dict[str, str],
    artifact_dir: Path,
    selected_checkpoint: Path,
    selected_loss: float,
    selected_step: int,
    source_manifest: Path,
    patched_config: Path,
    plan_fingerprint: str,
) -> Path:
    if artifact_dir.exists():
        if irodori_full_artifact_complete(
            artifact_dir,
            plan_fingerprint=plan_fingerprint,
        ):
            return artifact_dir
        raise FileExistsError(
            "Refusing to overwrite an existing incomplete or different Irodori full artifact: "
            f"{artifact_dir}"
        )
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = artifact_dir.parent / f".{artifact_dir.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        run(
            [
                "uv",
                "run",
                "--project",
                vendor,
                "--no-sync",
                "python",
                vendor / "convert_checkpoint_to_safetensors.py",
                selected_checkpoint,
                "--output",
                staging / "model.safetensors",
            ],
            cwd=vendor,
            env=env,
        )
        if not _nonempty_file(staging / "model.safetensors"):
            raise RuntimeError("Irodori checkpoint conversion produced no model.safetensors")
        if not _nonempty_file(staging / "tokenizer" / "tokenizer_config.json"):
            raise RuntimeError("Irodori checkpoint conversion did not export the tokenizer")
        provenance = {
            "schema_version": 1,
            "family": "irodori",
            "method": "full",
            "training_plan_fingerprint": plan_fingerprint,
            "upstream_source_revision": IRODORI_SOURCE_REVISION,
            "base_model": {
                "id": IRODORI_MODEL_ID,
                "revision": IRODORI_MODEL_REVISION,
                "sha256": IRODORI_MODEL_SHA256,
            },
            "text_encoder": {
                "id": IRODORI_TEXT_ENCODER_ID,
                "revision": IRODORI_TEXT_ENCODER_REVISION,
            },
            "training_config": {
                "source": "configs/train_v4_small.yaml",
                "patched_sha256": sha256_file(patched_config),
            },
            "source_manifest_sha256": sha256_file(source_manifest),
            "selected_checkpoint": selected_checkpoint.name,
            "best_validation_loss": selected_loss,
            "best_step": selected_step,
        }
        if _json_contains_absolute_path(provenance):
            raise RuntimeError("Irodori full provenance contains a machine-local absolute path")
        atomic_write_json(staging / "provenance.json", provenance)
        artifact_manifest = {
            "schema_version": 1,
            "family": "irodori",
            "method": "full",
            "training_plan_fingerprint": plan_fingerprint,
            "files": _portable_inventory(staging),
        }
        atomic_write_json(staging / "manifest.json", artifact_manifest)
        if not irodori_full_artifact_complete(
            staging,
            plan_fingerprint=plan_fingerprint,
        ):
            raise RuntimeError("Irodori full artifact failed portable integrity verification")
        staging.replace(artifact_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return artifact_dir


def train_irodori_full(
    repo_root: Path,
    manifest: Path,
    artifact_dir: Path,
    run_dir: Path,
    *,
    max_steps: int,
    plan_fingerprint: str,
    validation_ratio: float,
    validation_every: int,
    checkpoint_best_n: int,
    backend_override: str | None = None,
) -> dict[str, object]:
    """Run the pinned upstream v4 full trainer and export its best checkpoint."""

    plan_fingerprint = _validate_plan_fingerprint(plan_fingerprint)
    if irodori_full_artifact_complete(
        artifact_dir,
        plan_fingerprint=plan_fingerprint,
    ):
        provenance = json.loads((artifact_dir / "provenance.json").read_text(encoding="utf-8"))
        return {
            "method": "full",
            "full_model": str(artifact_dir / "model.safetensors"),
            "artifact": str(artifact_dir),
            "best_validation_loss": float(provenance["best_validation_loss"]),
            "training_plan_fingerprint": plan_fingerprint,
            "reused": True,
        }
    if artifact_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing incomplete or different Irodori full artifact: "
            f"{artifact_dir}"
        )
    if not _nonempty_file(manifest):
        raise FileNotFoundError(f"Irodori latent manifest is missing or empty: {manifest}")
    if artifact_dir.resolve(strict=False) == run_dir.resolve(strict=False):
        raise ValueError("Irodori full artifact_dir and run_dir must be separate")

    vendor = vendor_dir(repo_root)
    base = base_checkpoint(repo_root)
    backend = backend_override or configured_backend(repo_root)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported Irodori training backend override: {backend!r}")
    device = backend_device(backend)
    env = local_model_env(repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _patched_full_config(
        vendor / "configs" / "train_v4_small.yaml",
        run_dir / "personavoice_train_v4_small.yaml",
        max_steps=max_steps,
        backend=backend,
        validation_ratio=validation_ratio,
        validation_every=validation_every,
        checkpoint_best_n=checkpoint_best_n,
    )

    def verify(path: Path) -> bool:
        return _verify_full_training_checkpoint(vendor, path, env=env)

    resume = _latest_verified_full_resume(run_dir, verify=verify)
    args: list[str | Path] = [
        "uv",
        "run",
        "--project",
        vendor,
        "--no-sync",
        "python",
        vendor / "train.py",
        "--config",
        config,
        "--manifest",
        manifest,
        "--output-dir",
        run_dir,
        "--device",
        device,
    ]
    if resume is None:
        args += ["--init-checkpoint", base]
    else:
        # Upstream explicitly rejects combining --resume and --init-checkpoint
        # for non-LoRA training. A verified full .pt restores all optimizer,
        # scheduler, dataloader and runtime state by itself.
        args += ["--resume", resume]
    run(args, cwd=vendor, env=env)
    best, best_loss, best_step = _best_verified_full_checkpoint(run_dir, verify=verify)
    artifact = _finalize_irodori_full_artifact(
        vendor=vendor,
        env=env,
        artifact_dir=artifact_dir,
        selected_checkpoint=best,
        selected_loss=best_loss,
        selected_step=best_step,
        source_manifest=manifest,
        patched_config=config,
        plan_fingerprint=plan_fingerprint,
    )
    return {
        "method": "full",
        "full_model": str(artifact / "model.safetensors"),
        "artifact": str(artifact),
        "run_dir": str(run_dir),
        "resumed_from": str(resume) if resume is not None else None,
        "best_checkpoint": str(best),
        "checkpoint_step": best_step,
        "best_validation_loss": best_loss,
        "training_plan_fingerprint": plan_fingerprint,
        "reused": False,
    }


def train_irodori(
    repo_root: Path,
    manifest: Path,
    models_dir: Path,
    cache_dir: Path,
    *,
    speaker_steps: int,
    lora_steps: int,
    do_speaker: bool,
    do_lora: bool,
    validation_ratio: float | None = None,
    validation_every: int | None = None,
    checkpoint_best_n: int | None = None,
    plan_fingerprint: str | None = None,
    backend_override: str | None = None,
) -> dict[str, object]:
    vendor = vendor_dir(repo_root)
    base = base_checkpoint(repo_root)
    backend = backend_override or configured_backend(repo_root)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported Irodori training backend override: {backend!r}")
    device = backend_device(backend)
    env = local_model_env(repo_root)
    outputs: dict[str, object] = {
        "base": str(base),
        "best_validation_loss": None,
    }
    if do_speaker:
        out = models_dir / "irodori" / "speaker"
        final = out / "checkpoint_final.speaker.safetensors"

        def verify_speaker(checkpoint: Path) -> bool:
            return _verify_speaker_embedding_checkpoint(
                vendor,
                checkpoint,
                env=env,
            )

        speaker_was_complete = verify_speaker(final)
        if not speaker_was_complete:
            cfg = _patched_config(
                vendor / "configs" / "train_v4_small_speaker_inversion.yaml",
                cache_dir / "irodori_speaker.yaml",
                max_steps=speaker_steps,
                backend=backend,
                validation_ratio=validation_ratio,
                validation_every=validation_every,
                checkpoint_best_n=checkpoint_best_n,
            )
            args = [
                "uv",
                "run",
                "--project",
                vendor,
                "--no-sync",
                "python",
                vendor / "train.py",
                "--config",
                cfg,
                "--manifest",
                manifest,
                "--init-checkpoint",
                base,
                "--output-dir",
                out,
                "--device",
                device,
            ]
            checkpoint = _latest_verified_speaker_embedding(
                out,
                verify=verify_speaker,
            )
            if checkpoint is not None:
                args += ["--speaker-inversion-init-embedding", checkpoint]
            run(args, cwd=vendor, env=env)
        if not verify_speaker(final):
            raise RuntimeError("Irodori Speaker Inversion did not produce a valid checkpoint_final")
        outputs["speaker_embedding"] = str(final)
        best_speaker = _best_validation_checkpoint(
            out,
            complete=lambda path: (
                path.name.endswith(".speaker.safetensors")
                and verify_speaker(path)
            ),
        )
        validation_required = plan_fingerprint is not None
        if best_speaker is None:
            if validation_required:
                raise RuntimeError(
                    "Irodori Speaker Inversion produced no complete best validation checkpoint"
                )
            outputs["artifact"] = str(final)
            outputs["best_checkpoint"] = str(final)
            outputs["checkpoint_step"] = speaker_steps
        else:
            best_checkpoint, best_loss, best_step = best_speaker
            # Preserve the v0.3 final-embedding API while letting the schema-v9
            # orchestrator evaluate and publish the upstream-selected best run.
            outputs["artifact"] = str(best_checkpoint if validation_required else final)
            outputs["best_checkpoint"] = str(best_checkpoint)
            outputs["checkpoint_step"] = best_step
            outputs["best_validation_loss"] = best_loss
        outputs["reused"] = speaker_was_complete
    if do_lora:
        out = models_dir / "irodori" / "lora"
        final = out / "checkpoint_final"
        selected = out / "selected"
        candidate_was_complete = bool(
            plan_fingerprint is not None
            and irodori_lora_candidate_complete(
                selected,
                plan_fingerprint=plan_fingerprint,
            )
        )
        if candidate_was_complete:
            provenance_path = selected / _LORA_PROVENANCE_NAME
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            outputs.update(
                {
                    "artifact": str(selected),
                    "lora_adapter": str(selected),
                    "best_validation_loss": float(provenance["best_validation_loss"]),
                    "checkpoint_step": int(provenance["best_step"]),
                    "provenance": str(provenance_path),
                    "reused": True,
                }
            )
        elif not lora_adapter_complete(final):
            cfg = _patched_config(
                vendor / "configs" / "train_v4_small_lora.yaml",
                cache_dir / "irodori_lora.yaml",
                max_steps=lora_steps,
                backend=backend,
                validation_ratio=validation_ratio,
                validation_every=validation_every,
                checkpoint_best_n=checkpoint_best_n,
            )
            args = [
                "uv",
                "run",
                "--project",
                vendor,
                "--no-sync",
                "python",
                vendor / "train.py",
                "--config",
                cfg,
                "--manifest",
                manifest,
                "--init-checkpoint",
                base,
                "--output-dir",
                out,
                "--device",
                device,
            ]
            resume = _latest_resume(
                out,
                verify=lambda checkpoint: _verify_lora_training_checkpoint(
                    vendor,
                    checkpoint,
                    env=env,
                ),
            )
            if resume:
                args += ["--resume", resume]
            run(args, cwd=vendor, env=env)
        if not candidate_was_complete:
            if not lora_adapter_complete(final):
                raise RuntimeError(
                    "Irodori LoRA training did not produce a complete PEFT adapter"
                )
            best_lora = _best_validation_checkpoint(
                out,
                complete=lambda path: _verify_lora_training_checkpoint(
                    vendor,
                    path,
                    env=env,
                ),
            )
            best_validation_loss = best_lora[1] if best_lora is not None else None
            if plan_fingerprint is None and best_validation_loss is None:
                best_validation_loss = _best_validation_loss(out)
            outputs["best_validation_loss"] = best_validation_loss
            if best_lora is not None:
                best_checkpoint, _, best_step = best_lora
                outputs["best_checkpoint"] = str(best_checkpoint)
                outputs["checkpoint_step"] = best_step
            if plan_fingerprint is None:
                outputs["lora_adapter"] = str(final)
            else:
                if best_lora is None or best_validation_loss is None:
                    raise RuntimeError(
                        "Irodori LoRA candidate has no finite best validation checkpoint"
                    )
                best_checkpoint, _, best_step = best_lora
                artifact = _finalize_irodori_lora_candidate(
                    best_checkpoint,
                    selected,
                    plan_fingerprint=plan_fingerprint,
                    best_validation_loss=best_validation_loss,
                    best_step=best_step,
                )
                outputs["artifact"] = str(artifact)
                outputs["lora_adapter"] = str(artifact)
                outputs["provenance"] = str(artifact / _LORA_PROVENANCE_NAME)
                outputs["reused"] = False
    return outputs


def train_irodori_method(
    repo_root: Path,
    manifest: Path,
    models_dir: Path,
    cache_dir: Path,
    *,
    method: str,
    max_steps: int,
    plan_fingerprint: str,
    validation_ratio: float,
    validation_every: int,
    checkpoint_best_n: int,
    run_dir: Path | None = None,
    backend_override: str | None = None,
) -> dict[str, object]:
    """Method-aware entry point while retaining the v0.3 legacy API above."""

    normalized = str(method).strip().lower()
    plan_fingerprint = _validate_plan_fingerprint(plan_fingerprint)
    if normalized == "full":
        selected_run_dir = run_dir or (cache_dir / "irodori_full_runs" / plan_fingerprint)
        return train_irodori_full(
            repo_root,
            manifest,
            models_dir / "irodori" / "full",
            selected_run_dir,
            max_steps=max_steps,
            plan_fingerprint=plan_fingerprint,
            validation_ratio=validation_ratio,
            validation_every=validation_every,
            checkpoint_best_n=checkpoint_best_n,
            backend_override=backend_override,
        )
    if run_dir is not None:
        raise ValueError("run_dir is only supported by the Irodori full method")
    if normalized == "lora":
        return train_irodori(
            repo_root,
            manifest,
            models_dir,
            cache_dir,
            speaker_steps=max_steps,
            lora_steps=max_steps,
            do_speaker=False,
            do_lora=True,
            validation_ratio=validation_ratio,
            validation_every=validation_every,
            checkpoint_best_n=checkpoint_best_n,
            plan_fingerprint=plan_fingerprint,
            backend_override=backend_override,
        )
    if normalized == "speaker-inversion":
        return train_irodori(
            repo_root,
            manifest,
            models_dir,
            cache_dir,
            speaker_steps=max_steps,
            lora_steps=max_steps,
            do_speaker=True,
            do_lora=False,
            validation_ratio=validation_ratio,
            validation_every=validation_every,
            checkpoint_best_n=checkpoint_best_n,
            plan_fingerprint=plan_fingerprint,
            backend_override=backend_override,
        )
    raise ValueError(
        "Unsupported Irodori training method: "
        f"{method!r}; expected full, lora, or speaker-inversion"
    )


def reference_files(references_dir: Path) -> list[Path]:
    bank = references_dir / "bank.json"
    if bank.exists():
        data = json.loads(bank.read_text(encoding="utf-8"))
        files = [Path(path) for path in data.get("files", [])]
        return [path for path in files if path.exists()]
    return sorted(references_dir.glob("*.flac"))
