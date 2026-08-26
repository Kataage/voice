from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from threading import Event, Lock, Thread
from typing import Any
from uuid import uuid4

from personavoice.artifacts import verify_training_candidate
from personavoice.config import PersonaConfig
from personavoice.environment import load_root_environment
from personavoice.environment_contract import require_current_environment
from personavoice.executors import (
    LocalExecutor,
    LocalRunner,
    ModalExecutor,
    ModalUnavailableError,
    RemoteConsent,
    dispatch_training,
    inspect_local_resources,
    select_executor,
)
from personavoice.irodori import (
    _verify_full_training_checkpoint,
    _verify_lora_training_checkpoint,
    _verify_speaker_embedding_checkpoint,
    irodori_lora_candidate_complete,
    irodori_validation_checkpoint_metadata,
    lora_resume_checkpoint_complete,
    speaker_embedding_complete,
    train_irodori_method,
)
from personavoice.modal_transport import (
    CHECKPOINT_COMPLETION_NAME,
    CHECKPOINT_FAMILY_NAME,
    RESULT_COMPLETION_NAME,
    TRAINING_RESULT_NAME,
    DownloadedTrainingResult,
    ModalSettings,
    ModalTerminalCallRecoveredError,
    ModalTransport,
    RemoteSubmission,
    ResultCandidate,
    ResultFamily,
    detect_modal_auth,
    latest_verified_family_checkpoint,
    training_result_from_value,
    verify_completed_directory,
    write_checkpoint_family_contract,
    write_completion_manifest,
)
from personavoice.model_assets import (
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_SHA256,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_ASSET_SHA256,
    LFM_MODEL_REVISION,
)
from personavoice.pipeline import _prepare_fingerprint
from personavoice.process import run, run_json
from personavoice.project import PersonaPaths
from personavoice.setup_env import IRODORI_REVISION, SEED_VC_REVISION
from personavoice.state import StateStore
from personavoice.training_bundle import (
    IRODORI_SANITIZED_MANIFEST_PATH,
    build_training_bundle,
    canonical_plan_bytes,
    verify_training_bundle,
)
from personavoice.training_inputs import ensure_irodori_manifest
from personavoice.training_plan import (
    EVALUATION_CONTRACT_FILES,
    EXECUTOR_CONTRACT_FILES,
    FamilyPlan,
    TrainingPlan,
    build_training_plan,
    normalized_source_sha256,
    sha256_file,
    verify_plan_files,
)
from personavoice.workers import local_model_env, worker

TRAIN_SCHEMA_VERSION = 9
_SEED_VC_STEP_RE = re.compile(r"_step_(\d+)\.pth$")
_LFM_ADAPTER_REVISION_MARKER = ".personavoice-base-revision"
_LFM_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_LFM_CHECKPOINT_METHOD_MARKER = ".personavoice-training-method"
_LFM_CHECKPOINT_ATTESTATION = ".personavoice-checkpoint.json"
_LFM_CHECKPOINT_ATTESTATION_SCHEMA = 1
_LFM_CHECKPOINT_REQUIRED_STATE = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
)
_EXTERNAL_CANDIDATE_PREFIX = "__candidate__/"
_IRODORI_PERIODIC_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)(?:\.pt|\.speaker\.safetensors)?$")
_REMOTE_CHECKPOINT_OBSERVER_SECONDS = 5.0
_LEGACY_V03_TRAIN_SCHEMA = 8
_LEGACY_V03_MODEL_CONTRACT = {
    "irodori_source_revision": "8224dafb46d0aba89209a8f905f1cb7e3299d9c1",
    "irodori_model_sha256": "c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6",
    "irodori_dacvae_sha256": "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5",
    "irodori_text_encoder_revision": "77675fc96a7e445e982e2ba90246b816efc74ec6",
    "lfm_revision": "b31023f2d69b95fbd7876898f8de9fae90e8afbd",
    "seed_vc_source_revision": "51383efd921027683c89e5348211d93ff12ac2a8",
}
# Exact SHA-256 values of the v0.3.0 tag (commit
# 939c341f76424ee6d88ef8bc62cc9aba8beada27) used by its train-stage
# fingerprint.  Both Git LF blobs and Windows core.autocrlf CRLF worktrees are
# accepted; no current implementation byte is substituted into this proof.
_LEGACY_V03_FILE_CONTRACTS = (
    {
        "irodori_lock_sha256": "ccbde5b872e9541e27149864764d3fd0387069dfe1131f63e3fc7e9e05356210",
        "lfm_lock_sha256": "fa5ce727daa08338a00f3aea7a79d063a14ae9d77cf81cafadd85ff0e8e87b6a",
        "seed_vc_lock_sha256": "aa143ab4f09c8d95e5ec152c91737549269fcf7965d757fc6db079f180a3e0d2",
        "training_code_sha256": "fed1978d8f39f344a1285012f9dfcb0eeaa78d94f4487fd43ee30c91e7a07b9a",
        "irodori_code_sha256": "74054532a359ae47f70aad053f8347ce9d6a68a7ec8ff9ea7e4df9b4df38a5eb",
        "lfm_train_code_sha256": "d9e3f28d00a6691b08f47e3c446307cf10cc70ec13908f0322d1253bf652ded1",
        "lfm_checkpoint_contract_code_sha256": "695389bb0bbfe5c9b0d7227931ad1b5b43356660ea11c1db15f480716269b1a0",
        "lfm_model_contract_code_sha256": "b0aea088a454238386ed5bfdaedc931ae6ca990f36db3af2cf089f07a8263ed1",
        "seed_vc_worker_code_sha256": "850151618613d3de3aed39110f3b4e83e4a00f0d39f9b055ef05449546823479",
    },
    {
        "irodori_lock_sha256": "40b3b94f981251afb4ea92dedea47a4006264044dc8512532c6a46c14a2fec16",
        "lfm_lock_sha256": "9f71bea5b26dd7a4eea0f892e840d593cfaafaae7d2bcf09f59818945edaf3b6",
        "seed_vc_lock_sha256": "38fe5bfd6aa572555ddf1e85e6f9376d404c3d09f0c863a0e12336b48562caf4",
        "training_code_sha256": "e5d088cf3aa59b852d88b09bbd23eba80cc71ade5d9176556dc36e9733c3496c",
        "irodori_code_sha256": "88fd483d7f493346d9ceb6c224e9c754711267423c37ea6142f7782840ccbc76",
        "lfm_train_code_sha256": "65e5ef2b12590b281e7a615f53b5d686efc8d28e98c290b52cf0ace2a31c998c",
        "lfm_checkpoint_contract_code_sha256": "91375f0db8d3deae1d669440cd3e8daa482ed6fc20b43772043b73605b306077",
        "lfm_model_contract_code_sha256": "4e6689dd4505c524bb313f8844454ced54918bf5c5db7f82700069d2c0a3aee6",
        "seed_vc_worker_code_sha256": "544ae892ba7256fc7ea6a0d3797cf12d3e3e9a139c0cdc0548f623ca48ae8354",
    },
)


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _line_count(path: Path) -> int:
    try:
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _file_contract(path: Path) -> str:
    try:
        if not path.is_file():
            return "missing"
        return sha256_file(path)
    except OSError:
        return "unreadable"


def _source_contract(path: Path) -> str:
    """Hash Python policy sources without invalidating on line-ending changes."""

    try:
        if not path.is_file():
            return "missing"
        return normalized_source_sha256(path)
    except OSError:
        return "unreadable"


def _fingerprint(paths: PersonaPaths, cfg: PersonaConfig) -> str:
    digest = hashlib.sha256()
    digest.update(f"train-schema:{TRAIN_SCHEMA_VERSION}".encode())
    repo_root = paths.root.parents[1]
    model_contract = {
        "irodori_source_revision": IRODORI_REVISION,
        "irodori_model_sha256": IRODORI_MODEL_SHA256,
        "irodori_dacvae_sha256": IRODORI_DACVAE_SHA256,
        "irodori_text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
        "lfm_revision": LFM_MODEL_REVISION,
        "seed_vc_source_revision": SEED_VC_REVISION,
        "irodori_lock_sha256": _file_contract(repo_root / "locks" / "Irodori-TTS.uv.lock"),
        "lfm_lock_sha256": _file_contract(repo_root / "workers" / "lfm" / "uv.lock"),
        "seed_vc_lock_sha256": _file_contract(repo_root / "workers" / "seed_vc" / "uv.lock"),
        "training_code_sha256": _file_contract(repo_root / "src" / "personavoice" / "training.py"),
        "irodori_code_sha256": _file_contract(repo_root / "src" / "personavoice" / "irodori.py"),
        "lfm_train_code_sha256": _file_contract(repo_root / "workers" / "lfm" / "train.py"),
        "lfm_checkpoint_contract_code_sha256": _file_contract(
            repo_root / "workers" / "lfm" / "checkpoint_contract.py"
        ),
        "lfm_model_contract_code_sha256": _file_contract(
            repo_root / "workers" / "lfm" / "model_contract.py"
        ),
        "seed_vc_worker_code_sha256": _file_contract(
            repo_root / "workers" / "seed_vc" / "worker.py"
        ),
        "training_plan_code_sha256": _file_contract(
            repo_root / "src" / "personavoice" / "training_plan.py"
        ),
        "artifact_contract_code_sha256": _file_contract(
            repo_root / "src" / "personavoice" / "artifacts.py"
        ),
        # Re-enter the train stage whenever the shared execution/security
        # boundary changes.  These hashes intentionally do not participate in
        # a family fingerprint, so compatible prepared inputs and native
        # optimizer checkpoints are re-verified/re-attested instead of being
        # discarded merely because routing or transport code was hardened.
        "executor_contract": {
            relative: _file_contract(repo_root.joinpath(*PurePosixPath(relative).parts))
            for relative in EXECUTOR_CONTRACT_FILES
        },
        # Re-enter the train stage so an already verified candidate is checked
        # against the current local publication contract.  Family fingerprints
        # deliberately exclude this policy, preserving compatible optimizer
        # checkpoints and method-native candidate bytes.
        "evaluation_contract": {
            relative: _source_contract(repo_root.joinpath(*PurePosixPath(relative).parts))
            for relative in EVALUATION_CONTRACT_FILES
        },
    }
    digest.update(json.dumps(model_contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for path in (
        paths.dataset / "irodori_source.jsonl",
        paths.dataset / "lfm_train.jsonl",
        paths.dataset / "seed_vc" / "manifest.jsonl",
    ):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    training_contract = cfg.training.model_dump(mode="json")
    # Routing and credentials do not change optimization semantics. Switching
    # local/Modal for an identical request must preserve prepared data and
    # resumable family checkpoints.
    training_contract.pop("executor", None)
    training_contract.pop("remote_data_authorized", None)
    digest.update(json.dumps(training_contract, sort_keys=True).encode())
    return digest.hexdigest()


def _legacy_v03_training_contract(cfg: PersonaConfig) -> dict[str, Any] | None:
    """Reconstruct only the schema-v0.3 optimization request represented by v2."""

    irodori = cfg.training.irodori
    lfm = cfg.training.lfm
    if irodori.enabled and irodori.method not in {"lora", "speaker-inversion"}:
        return None
    if lfm.enabled and lfm.method != "lora":
        return None
    # These fields did not exist in v0.3.  Adoption is safe only at the exact
    # migration defaults that describe the old trainer's behavior; otherwise
    # the caller requested a genuinely new family contract.
    if irodori.enabled and (
        irodori.conditioning != "speaker"
        or irodori.validation_ratio != 0.0005
        or irodori.validation_every != 1000
        or irodori.checkpoint_best_n != 5
    ):
        return None
    if lfm.enabled and (lfm.validation_ratio != 0.1 or lfm.save_steps != 25):
        return None
    return {
        "irodori_speaker_inversion": bool(
            irodori.enabled
            and (irodori.method == "speaker-inversion" or irodori.auxiliary_speaker_inversion)
        ),
        "irodori_lora": bool(irodori.enabled and irodori.method == "lora"),
        "lfm_lora": bool(lfm.enabled),
        "seed_vc_finetune": bool(cfg.training.seed_vc.finetune),
        "irodori_max_steps": int(irodori.max_steps),
        "speaker_inversion_max_steps": int(irodori.speaker_inversion_max_steps),
        "lfm_epochs": float(lfm.epochs),
        "lfm_learning_rate": float(lfm.learning_rate),
        "lfm_lora_r": int(lfm.lora_r),
        "lfm_lora_alpha": int(lfm.lora_alpha),
        "seed_vc_max_steps": int(cfg.training.seed_vc.max_steps),
    }


def _legacy_v03_fingerprints(paths: PersonaPaths, cfg: PersonaConfig) -> frozenset[str]:
    training_contract = _legacy_v03_training_contract(cfg)
    if training_contract is None:
        return frozenset()
    fingerprints: set[str] = set()
    for file_contract in _LEGACY_V03_FILE_CONTRACTS:
        digest = hashlib.sha256()
        digest.update(f"train-schema:{_LEGACY_V03_TRAIN_SCHEMA}".encode())
        model_contract = {
            **_LEGACY_V03_MODEL_CONTRACT,
            **file_contract,
        }
        digest.update(
            json.dumps(model_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for path in (
            paths.dataset / "irodori_source.jsonl",
            paths.dataset / "lfm_train.jsonl",
            paths.dataset / "seed_vc" / "manifest.jsonl",
        ):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
        digest.update(json.dumps(training_contract, sort_keys=True).encode())
        fingerprints.add(digest.hexdigest())
    return frozenset(fingerprints)


def _legacy_v03_checkpoint_lineage_verified(
    previous_stage: dict[str, Any],
    paths: PersonaPaths,
    cfg: PersonaConfig,
) -> bool:
    """Authorize only native Irodori checkpoint seeding from an exact v0.3 stage.

    A failed stage, or a ``running`` record whose OS stage lock has since been
    acquired by the caller, can still contain a complete periodic checkpoint.
    Final artifacts require the stricter completed-result lineage below.
    """

    recorded = previous_stage.get("fingerprint")
    return (
        previous_stage.get("status") in {"complete", "failed", "running"}
        and isinstance(recorded, str)
        and recorded in _legacy_v03_fingerprints(paths, cfg)
    )


def _legacy_v03_final_lineage_verified(
    previous_stage: dict[str, Any],
    paths: PersonaPaths,
    cfg: PersonaConfig,
) -> bool:
    """Authorize legacy final-artifact adoption only from a complete exact result."""

    result = previous_stage.get("result")
    if (
        previous_stage.get("status") != "complete"
        or not isinstance(result, dict)
        or result.get("train_schema") != _LEGACY_V03_TRAIN_SCHEMA
        or not {"irodori", "lfm_adapter", "seed_vc_cfm"}.issubset(result)
    ):
        return False
    recorded = result.get("fingerprint")
    return (
        isinstance(recorded, str)
        and previous_stage.get("fingerprint") == recorded
        and recorded in _legacy_v03_fingerprints(paths, cfg)
    )


def _legacy_v03_lineage_verified(
    previous_stage: dict[str, Any],
    paths: PersonaPaths,
    cfg: PersonaConfig,
) -> bool:
    """Backward-compatible name for the strict final-artifact lineage check."""

    return _legacy_v03_final_lineage_verified(previous_stage, paths, cfg)


def _invalidate_training_artifacts(paths: PersonaPaths) -> None:
    for target in (
        paths.models / "irodori",
        paths.models / "lfm",
        paths.models / "seed_vc",
        paths.cache / "irodori_latents",
    ):
        shutil.rmtree(target, ignore_errors=True)
    (paths.dataset / "irodori_manifest.jsonl").unlink(missing_ok=True)
    for config in paths.cache.glob("irodori_*.yaml"):
        config.unlink(missing_ok=True)


def _has_training_artifacts(paths: PersonaPaths) -> bool:
    markers = (
        paths.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors",
        paths.models / "irodori" / "lora" / "checkpoint_final",
        paths.models / "lfm" / "adapter" / "adapter_config.json",
        paths.models / "seed_vc" / "cfm.pth",
        paths.dataset / "irodori_manifest.jsonl",
    )
    if any(path.exists() for path in markers):
        return True
    latents = paths.cache / "irodori_latents"
    return latents.is_dir() and any(latents.iterdir())


def _lfm_adapter_weight(output: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = output / name
        if _nonempty_file(candidate):
            return candidate
    return None


def _lfm_adapter_complete(output: Path) -> bool:
    if not _nonempty_file(output / "adapter_config.json") or _lfm_adapter_weight(output) is None:
        return False
    marker = output / _LFM_ADAPTER_REVISION_MARKER
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == LFM_MODEL_REVISION
    except OSError:
        return False


def _lfm_native_checkpoint_step(path: Path) -> int | None:
    match = _LFM_CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _lfm_native_checkpoint_complete(path: Path, *, method: str) -> bool:
    """Verify the worker's digest-bound native checkpoint attestation.

    PyTorch is intentionally absent from the root orchestrator. The isolated
    worker writes this file only after restricted ``weights_only`` loads of all
    Trainer state succeed. Re-hashing the exact native payload here makes Modal
    observation fail closed on a partial upload or post-save mutation.
    """

    if method not in {"full", "lora"} or not path.is_dir():
        return False
    step = _lfm_native_checkpoint_step(path)
    if step is None or (path / "checkpoint-is-incomplete.txt").exists():
        return False
    marker = path / _LFM_CHECKPOINT_METHOD_MARKER
    attestation_path = path / _LFM_CHECKPOINT_ATTESTATION
    try:
        if (
            _is_link_or_junction(path)
            or not marker.is_file()
            or _is_link_or_junction(marker)
            or marker.read_text(encoding="utf-8").strip() != method
            or not attestation_path.is_file()
            or _is_link_or_junction(attestation_path)
        ):
            return False
        document = json.loads(attestation_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or (
            document.get("schema_version") != _LFM_CHECKPOINT_ATTESTATION_SCHEMA
            or document.get("step") != step
            or document.get("method") != method
        ):
            return False
        precision = document.get("precision")
        if (
            not isinstance(precision, dict)
            or set(precision) != {"fp16", "bf16", "use_cpu"}
            or any(not isinstance(value, bool) for value in precision.values())
            or (precision["fp16"] and precision["bf16"])
            or (precision["use_cpu"] and (precision["fp16"] or precision["bf16"]))
        ):
            return False
        rows = document.get("files")
        if not isinstance(rows, list) or not rows:
            return False
        recorded: dict[str, tuple[int, str]] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
                return False
            raw = row["path"]
            size = row["size"]
            digest = row["sha256"]
            relative = PurePosixPath(raw) if isinstance(raw, str) else None
            if (
                relative is None
                or relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or raw in recorded
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                return False
            candidate = path.joinpath(*relative.parts)
            if (
                not candidate.is_file()
                or _is_link_or_junction(candidate)
                or candidate.stat().st_size != size
                or sha256_file(candidate) != digest
            ):
                return False
            recorded[raw] = (size, digest)
        excluded = {
            _LFM_CHECKPOINT_ATTESTATION,
            CHECKPOINT_COMPLETION_NAME,
            CHECKPOINT_FAMILY_NAME,
        }
        actual: set[str] = set()
        for candidate in path.rglob("*"):
            relative = candidate.relative_to(path).as_posix()
            if relative in excluded:
                continue
            if _is_link_or_junction(candidate):
                return False
            if candidate.is_dir():
                continue
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                return False
            actual.add(relative)
        required = {
            *_LFM_CHECKPOINT_REQUIRED_STATE,
            "rng_state.pth",
            _LFM_CHECKPOINT_METHOD_MARKER,
            "adapter_config.json" if method == "lora" else "config.json",
        }
        if set(recorded) != actual or not required.issubset(recorded):
            return False
        if precision["fp16"] != ("scaler.pt" in recorded):
            return False
        if any(name.startswith("rng_state") and name != "rng_state.pth" for name in recorded):
            return False
        trainer_state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        if not isinstance(trainer_state, dict) or trainer_state.get("global_step") != step:
            return False
        config_name = "adapter_config.json" if method == "lora" else "config.json"
        config = json.loads((path / config_name).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config:
            return False
        if method == "lora":
            return (
                len({"adapter_model.safetensors", "adapter_model.bin"}.intersection(recorded)) == 1
            )
        primary = {
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }.intersection(recorded)
        return len(primary) == 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _seed_vc_checkpoint_step(path: Path) -> int | None:
    match = _SEED_VC_STEP_RE.search(path.name)
    return int(match.group(1)) if match else None


def _latest_seed_vc_checkpoint(source_dir: Path) -> Path | None:
    candidates = [
        (step, path)
        for path in source_dir.glob("CFM_*_step_*.pth")
        if _nonempty_file(path) and (step := _seed_vc_checkpoint_step(path)) is not None
    ]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _seed_vc_training_progress(vendor: Path, persona_name: str) -> tuple[int, Path | None]:
    """Return cumulative completed CFM update steps across PersonaVoice stages."""

    runs = vendor / "runs"
    prefix = f"personavoice_{persona_name}_stage_"
    best_step = 0
    best_checkpoint: Path | None = None
    if not runs.exists():
        return best_step, best_checkpoint
    for directory in runs.glob(f"{prefix}*"):
        if not directory.is_dir():
            continue
        suffix = directory.name[len(prefix) :]
        if not suffix.isdigit():
            continue
        checkpoint = _latest_seed_vc_checkpoint(directory)
        if checkpoint is None:
            continue
        local_step = _seed_vc_checkpoint_step(checkpoint)
        if local_step is None or local_step <= 0:
            continue
        cumulative = int(suffix) + local_step
        if cumulative > best_step:
            best_step = cumulative
            best_checkpoint = checkpoint
    return best_step, best_checkpoint


def _clear_seed_vc_runs(repo_root: Path, persona_name: str) -> None:
    runs = repo_root / "vendor" / "seed-vc" / "runs"
    shutil.rmtree(runs / f"personavoice_{persona_name}", ignore_errors=True)
    if runs.exists():
        for path in runs.glob(f"personavoice_{persona_name}_stage_*"):
            shutil.rmtree(path, ignore_errors=True)


def train_lfm(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    method: str | None = None,
    output: Path | None = None,
    run_dir: Path | None = None,
    plan_fingerprint: str | None = None,
    dataset_override: Path | None = None,
) -> dict[str, Any]:
    dataset = dataset_override or (paths.dataset / "lfm_train.jsonl")
    example_count = _line_count(dataset)
    if example_count < 2:
        raise RuntimeError(
            "training.lfm is enabled, but fewer than two valid conversational "
            f"examples were exported ({example_count}). Add source conversations containing "
            "the authorized speaker responding to another speaker, rerun `persona prepare`, "
            "or deliberately set training.lfm.enabled: false."
        )
    selected_method = method or cfg.training.lfm.method
    if selected_method not in {"full", "lora"}:
        raise ValueError(f"Unsupported LFM training method: {selected_method!r}")
    base = repo_root / "models" / "lfm" / "base"
    if not _nonempty_file(base / "config.json"):
        raise FileNotFoundError("LFM base model is missing. Run `persona setup --download-models`.")
    output = output or (paths.models / "lfm" / ("full" if selected_method == "full" else "adapter"))
    output.parent.mkdir(parents=True, exist_ok=True)
    project = repo_root / "workers" / "lfm"
    args: list[str | Path] = [
        "uv",
        "run",
        "--project",
        project,
        "--no-sync",
        "python",
        project / "train.py",
        "--base",
        base,
        "--dataset",
        dataset,
        "--output",
        output,
        "--method",
        selected_method,
        "--epochs",
        str(cfg.training.lfm.epochs),
        "--learning-rate",
        str(cfg.training.lfm.learning_rate),
        "--validation-ratio",
        str(cfg.training.lfm.validation_ratio),
        "--save-steps",
        str(cfg.training.lfm.save_steps),
        "--lora-r",
        str(cfg.training.lfm.lora_r),
        "--lora-alpha",
        str(cfg.training.lfm.lora_alpha),
    ]
    if plan_fingerprint is not None:
        args += ["--plan-fingerprint", plan_fingerprint]
    if run_dir is not None:
        args += ["--run-dir", run_dir]
    result = run_json(
        args,
        cwd=repo_root,
        env=local_model_env(repo_root),
    )
    if not isinstance(result, dict) or result.get("method") != selected_method:
        raise RuntimeError("LFM trainer returned an invalid method-aware result contract")
    if selected_method == "lora" and not _lfm_adapter_complete(output):
        raise RuntimeError(
            "LFM fine-tuning completed without a complete adapter for the audited base revision"
        )
    return result


def _run_seed_vc_stage(
    repo_root: Path,
    *,
    project: Path,
    vendor: Path,
    audio_dir: Path,
    persona_name: str,
    completed_steps: int,
    desired_steps: int,
    initial_checkpoint: Path | None,
) -> tuple[int, Path]:
    remaining_steps = desired_steps - completed_steps
    stage_name = f"personavoice_{persona_name}_stage_{completed_steps:010d}"
    stage_dir = vendor / "runs" / stage_name
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    args: list[str | Path] = [
        "uv",
        "run",
        "--project",
        project,
        "--no-sync",
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "fp16",
        vendor / "train_v2.py",
        "--dataset-dir",
        audio_dir,
        "--run-name",
        stage_name,
        "--batch-size",
        "2",
        "--max-steps",
        str(remaining_steps),
        "--max-epochs",
        str(max(1000, remaining_steps + 10)),
        "--save-every",
        str(max(25, min(500, max(1, remaining_steps // 2)))),
        "--num-workers",
        "0",
        "--train-cfm",
    ]
    if initial_checkpoint is not None:
        args += ["--pretrained-cfm-ckpt", initial_checkpoint]
    run(args, cwd=vendor, env=local_model_env(repo_root))

    checkpoint = _latest_seed_vc_checkpoint(stage_dir)
    if checkpoint is None:
        raise RuntimeError(
            "Seed-VC fine-tuning stage completed without a non-empty CFM checkpoint: "
            f"stage={stage_name}"
        )
    local_steps = _seed_vc_checkpoint_step(checkpoint)
    if local_steps is None or local_steps <= 0:
        raise RuntimeError(f"Seed-VC produced an invalid checkpoint step: {checkpoint.name}")
    total_steps = completed_steps + local_steps
    if total_steps <= completed_steps:
        raise RuntimeError(
            "Seed-VC staged fine-tuning made no forward progress; refusing an automatic retry loop"
        )
    if total_steps > desired_steps:
        raise RuntimeError(
            "Seed-VC staged fine-tuning exceeded the requested cumulative step count: "
            f"completed={total_steps}, expected<={desired_steps}"
        )
    return total_steps, checkpoint


def train_seed_vc(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> str | None:
    if not cfg.training.seed_vc_finetune:
        return None
    audio_dir = paths.dataset / "seed_vc" / "audio"
    audio_files = (
        [path for path in audio_dir.glob("*.flac") if _nonempty_file(path)]
        if audio_dir.exists()
        else []
    )
    if len(audio_files) < 2:
        raise RuntimeError(
            "training.seed_vc_finetune is enabled, but fewer than two valid target-speaker "
            f"audio clips were exported ({len(audio_files)}). Add usable target audio, "
            "rerun `persona prepare`, or deliberately set training.seed_vc_finetune: false."
        )

    health = worker(repo_root, "seed_vc").call(repo_root, "health", {"deep": False})
    if not bool(health.get("ok", True)) or not bool(health.get("cuda")):
        raise RuntimeError(
            "Seed-VC fine-tuning requires a healthy CUDA-enabled Seed-VC worker. "
            "Re-run `persona setup` on a supported NVIDIA system, inspect `persona doctor --deep`, "
            "or leave training.seed_vc_finetune=false and use zero-shot reenactment."
        )

    vendor = repo_root / "vendor" / "seed-vc"
    project = repo_root / "workers" / "seed_vc"
    completed_steps, checkpoint = _seed_vc_training_progress(vendor, cfg.name)
    desired_steps = cfg.training.seed_vc_max_steps
    if completed_steps > desired_steps:
        raise RuntimeError(
            "Existing staged Seed-VC progress exceeds the configured max steps: "
            f"completed={completed_steps}, configured={desired_steps}. "
            "Run `persona train --force` to restart with the current training configuration."
        )

    target = paths.models / "seed_vc" / "cfm.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    while completed_steps < desired_steps:
        completed_steps, checkpoint = _run_seed_vc_stage(
            repo_root,
            project=project,
            vendor=vendor,
            audio_dir=audio_dir,
            persona_name=cfg.name,
            completed_steps=completed_steps,
            desired_steps=desired_steps,
            initial_checkpoint=checkpoint,
        )
    if checkpoint is None or not _nonempty_file(checkpoint):
        raise RuntimeError(
            "Seed-VC reached the requested step count without a usable CFM checkpoint"
        )
    shutil.copy2(checkpoint, target)
    if not _nonempty_file(target):
        raise RuntimeError("Seed-VC final CFM checkpoint copy is missing or empty")
    return str(target)


def _persona_relative(paths: PersonaPaths, path: Path | str) -> str:
    root = paths.root.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Training artifact escaped the persona directory: {resolved}") from exc
    return relative.as_posix()


def _runner_artifact_path(
    paths: PersonaPaths,
    path: Path | str,
    *,
    external_candidate_root: Path | None,
) -> str:
    """Serialize a runner-only path without exposing a machine-local absolute path."""

    if external_candidate_root is None:
        return _persona_relative(paths, path)
    root = external_candidate_root.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return _persona_relative(paths, resolved)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Training candidate path is not portable: {resolved}")
    return _EXTERNAL_CANDIDATE_PREFIX + relative.as_posix()


def _legacy_lfm_lora_checkpoint_complete(path: Path) -> bool:
    """Recognize the exact resumable Trainer payload written by v0.3."""

    if not path.is_dir() or _lfm_native_checkpoint_step(path) is None:
        return False
    if (path / "checkpoint-is-incomplete.txt").exists():
        return False
    if not all(_nonempty_file(path / name) for name in _LFM_CHECKPOINT_REQUIRED_STATE):
        return False
    if not any(_nonempty_file(candidate) for candidate in path.glob("rng_state*.pth")):
        return False
    return _nonempty_file(path / "adapter_config.json") and any(
        _nonempty_file(path / name) for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _migrate_legacy_checkpoint_directory(
    source: Path,
    destination: Path,
    *,
    destination_complete: Callable[[Path], bool],
) -> bool:
    """Losslessly seed a family namespace without changing the v0.3 payload."""

    source_inventory = _checkpoint_payload_inventory(source)
    if destination.exists() and destination_complete(destination):
        return False
    staging = destination.with_name(f".{destination.name}.{uuid4().hex}.legacy-staging")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        _materialize_checkpoint_payload(source, staging)
        if _checkpoint_payload_inventory(staging) != source_inventory:
            raise RuntimeError("Legacy checkpoint migration was not lossless")
        if destination.exists():
            if _is_link_or_junction(destination) or not destination.is_dir():
                raise RuntimeError("Legacy checkpoint migration destination is unsafe")
            shutil.rmtree(destination)
        staging.replace(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return True


def _migrate_legacy_checkpoint_file(source: Path, destination: Path) -> bool:
    if not _nonempty_file(source) or _is_link_or_junction(source):
        return False
    if _nonempty_file(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.legacy-tmp")
    try:
        _link_or_copy_checkpoint_file(source, temporary)
        if temporary.stat().st_size != source.stat().st_size or sha256_file(
            temporary
        ) != sha256_file(source):
            raise RuntimeError("Legacy checkpoint file migration was not lossless")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _best_lora_validation_loss(root: Path) -> float | None:
    values: list[float] = []
    for path in root.glob("checkpoint_best_val_loss_*"):
        try:
            value = float(path.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if value >= 0:
            values.append(value)
    return min(values) if values else None


def _family_result(
    plan: TrainingPlan,
    paths: PersonaPaths,
    *,
    family_name: str,
    artifact: Path | None,
    validation_loss: float | None = None,
    reused: bool = False,
    auxiliary_artifact: Path | None = None,
    external_candidate_root: Path | None = None,
) -> dict[str, Any]:
    family = plan.family(family_name)
    value: dict[str, Any] = {
        "enabled": family.enabled,
        "method": family.method,
        "family_fingerprint": family.fingerprint,
        "artifact": (
            _runner_artifact_path(
                paths,
                artifact,
                external_candidate_root=external_candidate_root,
            )
            if artifact is not None
            else None
        ),
        "validation": {"loss": validation_loss},
        "reused": bool(reused),
    }
    if auxiliary_artifact is not None:
        auxiliary_fingerprint = family.auxiliary_fingerprint
        if auxiliary_fingerprint is None:
            raise RuntimeError("Unexpected auxiliary artifact for this family contract")
        value["auxiliary_speaker_embedding"] = _runner_artifact_path(
            paths,
            auxiliary_artifact,
            external_candidate_root=external_candidate_root,
        )
        value["auxiliary_family_fingerprint"] = auxiliary_fingerprint
    return value


def _completed_family_checkpoint(
    result: dict[str, Any],
    *,
    family: str,
    method: str,
    run_root: Path,
) -> tuple[Path, int] | None:
    """Validate a freshly produced native checkpoint before retaining it in memory."""

    if result.get("reused") is True:
        return None
    raw_checkpoint = result.get("best_checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise RuntimeError(f"Fresh {family}/{method} training returned no best checkpoint")
    try:
        resolved_root = run_root.resolve(strict=True)
        unresolved = Path(raw_checkpoint)
        candidate = unresolved if unresolved.is_absolute() else resolved_root / unresolved
        lexical_relative = candidate.relative_to(resolved_root)
        if not lexical_relative.parts or any(
            part in {"", ".", ".."} for part in lexical_relative.parts
        ):
            raise ValueError("checkpoint path is not a strict child of its run directory")
        cursor = resolved_root
        for part in lexical_relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise RuntimeError(f"Fresh {family}/{method} checkpoint path contains a symlink")
        checkpoint = candidate.resolve(strict=True)
        checkpoint.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Fresh {family}/{method} checkpoint escaped its run directory") from exc

    raw_loss = result.get("best_validation_loss")
    if (
        not isinstance(raw_loss, (int, float))
        or isinstance(raw_loss, bool)
        or not math.isfinite(float(raw_loss))
    ):
        raise RuntimeError(f"Fresh {family}/{method} checkpoint has no finite validation loss")
    if family == "irodori":
        metadata = irodori_validation_checkpoint_metadata(checkpoint)
        if metadata is None or metadata[1] != float(raw_loss):
            raise RuntimeError("Fresh Irodori checkpoint does not encode its validation loss")
        step = metadata[0]
        if method == "full":
            complete = checkpoint.suffix == ".pt" and _nonempty_file(checkpoint)
        elif method == "lora":
            complete = lora_resume_checkpoint_complete(checkpoint)
        elif method == "speaker-inversion":
            complete = checkpoint.name.endswith(
                ".speaker.safetensors"
            ) and speaker_embedding_complete(checkpoint)
        else:
            raise RuntimeError(f"Unsupported Irodori checkpoint method: {method}")
    elif family == "lfm":
        parsed_step = _lfm_native_checkpoint_step(checkpoint)
        if parsed_step is None:
            raise RuntimeError("Fresh LFM trainer returned a non-numeric checkpoint")
        step = parsed_step
        complete = _lfm_native_checkpoint_complete(checkpoint, method=method)
    else:
        raise RuntimeError(f"Unsupported family checkpoint: {family}/{method}")
    raw_step = result.get("checkpoint_step")
    if raw_step is not None and (
        not isinstance(raw_step, int) or isinstance(raw_step, bool) or raw_step != step
    ):
        raise RuntimeError(f"Fresh {family}/{method} checkpoint step is inconsistent")
    if not complete:
        raise RuntimeError(f"Fresh {family}/{method} checkpoint is not exactly resumable")
    return checkpoint, step


class PersonaLocalTrainingRunner(LocalRunner):
    """Execute every family from the immutable plan in isolated existing runtimes."""

    def __init__(
        self,
        repo_root: Path,
        paths: PersonaPaths,
        cfg: PersonaConfig,
        *,
        legacy_result: dict[str, Any] | None,
        legacy_checkpoint_lineage_verified: bool = False,
        irodori_manifest: Path | None = None,
        lfm_dataset: Path | None = None,
        candidate_root: Path | None = None,
        checkpoint_root: Path | None = None,
        backend_override: str | None = None,
        verify_local_plan_files: bool = True,
    ) -> None:
        self.repo_root = repo_root
        self.paths = paths
        self.cfg = cfg
        self.legacy_result = legacy_result
        self.legacy_checkpoint_lineage_verified = legacy_checkpoint_lineage_verified
        self.irodori_manifest = irodori_manifest
        self.lfm_dataset = lfm_dataset
        self.candidate_root = candidate_root
        self.checkpoint_root = checkpoint_root
        self.backend_override = backend_override
        self.verify_local_plan_files = verify_local_plan_files
        self.completed_checkpoints: dict[str, tuple[Path, int]] = {}

    def _manifest(self, plan: TrainingPlan) -> Path:
        if self.irodori_manifest is not None:
            return self.irodori_manifest
        return next(
            self.paths.root / item.path
            for item in plan.files
            if item.role == "irodori-latent-manifest"
        )

    def _migrate_legacy_irodori_resumes(
        self,
        *,
        method: str,
        candidate_models: Path,
    ) -> None:
        if not self.legacy_checkpoint_lineage_verified:
            return
        if method == "lora":
            source_root = self.paths.models / "irodori" / "lora"
            destination_root = candidate_models / "irodori" / "lora"
            if not source_root.is_dir():
                return
            for source in sorted(source_root.glob("checkpoint_*")):
                if (
                    not source.is_dir()
                    or _irodori_periodic_checkpoint_step(source) is None
                    or not lora_resume_checkpoint_complete(source)
                ):
                    continue
                _migrate_legacy_checkpoint_directory(
                    source,
                    destination_root / source.name,
                    destination_complete=lora_resume_checkpoint_complete,
                )
            return
        if method != "speaker-inversion":
            return
        source_root = self.paths.models / "irodori" / "speaker"
        destination_root = candidate_models / "irodori" / "speaker"
        if not source_root.is_dir():
            return
        for source in sorted(source_root.glob("checkpoint_*.speaker.safetensors")):
            if _irodori_periodic_checkpoint_step(source) is not None and speaker_embedding_complete(
                source
            ):
                _migrate_legacy_checkpoint_file(source, destination_root / source.name)

    def _status(
        self,
        callback,
        *,
        model: str,
        remote_state: str,
    ) -> None:
        if callback is not None:
            callback(
                {
                    "executor": "local",
                    "remote_state": remote_state,
                    "model": model,
                    "step": 0,
                    "checkpoint": None,
                }
            )

    def _irodori(self, plan: TrainingPlan, callback) -> dict[str, Any]:
        family = plan.family("irodori")
        if not family.enabled:
            return _family_result(plan, self.paths, family_name="irodori", artifact=None)
        self._status(callback, model="irodori", remote_state="running")
        adopted: Path | None = None
        auxiliary: Path | None = None
        auxiliary_requested = family.training.get("auxiliary_speaker_inversion") is True
        if auxiliary_requested:
            legacy_irodori = (
                self.legacy_result.get("irodori") if isinstance(self.legacy_result, dict) else None
            )
            if isinstance(legacy_irodori, dict):
                raw_auxiliary = legacy_irodori.get("speaker_embedding")
                if isinstance(raw_auxiliary, str) and speaker_embedding_complete(
                    Path(raw_auxiliary)
                ):
                    auxiliary = Path(raw_auxiliary)

        candidate_models = (
            self.candidate_root / "irodori" / family.fingerprint / "models"
            if self.candidate_root is not None
            else self.paths.models / ".candidates" / "irodori" / family.fingerprint / "models"
        )
        run_cache = (
            self.checkpoint_root / "irodori" / family.fingerprint
            if self.checkpoint_root is not None
            else self.paths.cache / "training_runs" / "irodori" / family.fingerprint
        )
        auxiliary_family = family.auxiliary_family
        auxiliary_candidate_models = (
            (
                self.candidate_root / "irodori" / auxiliary_family.fingerprint / "models"
                if self.candidate_root is not None
                else self.paths.models
                / ".candidates"
                / "irodori"
                / auxiliary_family.fingerprint
                / "models"
            )
            if auxiliary_family is not None
            else None
        )
        auxiliary_run_cache = (
            (
                self.checkpoint_root / "irodori" / auxiliary_family.fingerprint
                if self.checkpoint_root is not None
                else self.paths.cache
                / "training_runs"
                / "irodori"
                / auxiliary_family.fingerprint
            )
            if auxiliary_family is not None
            else None
        )
        self._migrate_legacy_irodori_resumes(
            method=family.method,
            candidate_models=candidate_models,
        )
        if auxiliary_requested:
            assert auxiliary_candidate_models is not None
            self._migrate_legacy_irodori_resumes(
                method="speaker-inversion",
                candidate_models=auxiliary_candidate_models,
            )
        if auxiliary_requested and auxiliary is None:
            assert auxiliary_family is not None
            assert auxiliary_candidate_models is not None
            assert auxiliary_run_cache is not None
            auxiliary_result = train_irodori_method(
                self.repo_root,
                self._manifest(plan),
                auxiliary_candidate_models,
                auxiliary_run_cache,
                method="speaker-inversion",
                max_steps=int(family.training["speaker_inversion_max_steps"]),
                plan_fingerprint=auxiliary_family.fingerprint,
                validation_ratio=float(family.training["validation_ratio"]),
                validation_every=int(family.training["validation_every"]),
                checkpoint_best_n=int(family.training["checkpoint_best_n"]),
                backend_override=self.backend_override,
            )
            raw_auxiliary = auxiliary_result.get("artifact") or auxiliary_result.get(
                "speaker_embedding"
            )
            auxiliary = Path(str(raw_auxiliary)) if raw_auxiliary else None
            auxiliary_checkpoint = _completed_family_checkpoint(
                auxiliary_result,
                family="irodori",
                method="speaker-inversion",
                run_root=auxiliary_candidate_models / "irodori" / "speaker",
            )
            if auxiliary_checkpoint is not None:
                self.completed_checkpoints["irodori-auxiliary"] = auxiliary_checkpoint

        validation_loss: float | None = None
        reused = adopted is not None
        artifact = adopted
        if artifact is None:
            manifest = self._manifest(plan)
            result = train_irodori_method(
                self.repo_root,
                manifest,
                candidate_models,
                run_cache,
                method=family.method,
                max_steps=int(
                    family.training["speaker_inversion_max_steps"]
                    if family.method == "speaker-inversion"
                    else family.training["max_steps"]
                ),
                plan_fingerprint=family.fingerprint,
                validation_ratio=float(family.training["validation_ratio"]),
                validation_every=int(family.training["validation_every"]),
                checkpoint_best_n=int(family.training["checkpoint_best_n"]),
                run_dir=(run_cache if family.method == "full" else None),
                backend_override=self.backend_override,
            )
            raw_artifact = (
                result.get("artifact")
                or result.get("lora_adapter")
                or result.get("speaker_embedding")
            )
            if not isinstance(raw_artifact, str) or not raw_artifact:
                raise RuntimeError("Irodori trainer returned no method-specific artifact")
            artifact = Path(raw_artifact)
            raw_loss = result.get("best_validation_loss")
            if isinstance(raw_loss, (int, float)) and not isinstance(raw_loss, bool):
                validation_loss = float(raw_loss)
            elif family.method == "lora":
                validation_loss = _best_lora_validation_loss(artifact.parent)
            reused = result.get("reused") is True
            checkpoint = _completed_family_checkpoint(
                result,
                family="irodori",
                method=family.method,
                run_root=(
                    run_cache
                    if family.method == "full"
                    else candidate_models
                    / "irodori"
                    / ("lora" if family.method == "lora" else "speaker")
                ),
            )
            if checkpoint is not None:
                self.completed_checkpoints["irodori"] = checkpoint
        self._status(callback, model="irodori", remote_state="complete")
        return _family_result(
            plan,
            self.paths,
            family_name="irodori",
            artifact=artifact,
            validation_loss=validation_loss,
            reused=reused,
            auxiliary_artifact=auxiliary,
            external_candidate_root=self.candidate_root,
        )

    def _lfm(self, plan: TrainingPlan, callback) -> dict[str, Any]:
        family = plan.family("lfm")
        if not family.enabled:
            return _family_result(plan, self.paths, family_name="lfm", artifact=None)
        self._status(callback, model="lfm", remote_state="running")
        adopted: Path | None = None
        validation_loss: float | None = None
        reused = adopted is not None
        artifact = adopted
        if artifact is None:
            root = (
                self.candidate_root / "lfm" / family.fingerprint
                if self.candidate_root is not None
                else self.paths.models / ".candidates" / "lfm" / family.fingerprint
            )
            output = root / ("full" if family.method == "full" else "adapter")
            run_dir = (
                self.checkpoint_root / "lfm" / family.fingerprint
                if self.checkpoint_root is not None
                else self.paths.cache / "training_runs" / "lfm" / family.fingerprint
            )
            # v0.3 LFM checkpoints were created before the deterministic held-out
            # split and best-validation selection contract.  Their optimizer and
            # dataloader state is intentionally left untouched at the legacy path;
            # only checkpoints produced by this family fingerprint may resume.
            result = train_lfm(
                self.repo_root,
                self.paths,
                self.cfg,
                method=family.method,
                output=output,
                run_dir=run_dir,
                plan_fingerprint=family.fingerprint,
                dataset_override=self.lfm_dataset,
            )
            raw_artifact = result.get("artifact")
            if not isinstance(raw_artifact, str) or not raw_artifact:
                raise RuntimeError("LFM trainer returned no method-specific artifact")
            artifact = Path(raw_artifact)
            raw_loss = result.get("best_validation_loss")
            if isinstance(raw_loss, (int, float)) and not isinstance(raw_loss, bool):
                validation_loss = float(raw_loss)
            reused = result.get("reused") is True
            checkpoint = _completed_family_checkpoint(
                result,
                family="lfm",
                method=family.method,
                run_root=run_dir,
            )
            if checkpoint is not None:
                self.completed_checkpoints["lfm"] = checkpoint
        self._status(callback, model="lfm", remote_state="complete")
        return _family_result(
            plan,
            self.paths,
            family_name="lfm",
            artifact=artifact,
            validation_loss=validation_loss,
            reused=reused,
            external_candidate_root=self.candidate_root,
        )

    def _seed_vc(self, plan: TrainingPlan, callback) -> dict[str, Any]:
        family = plan.family("seed-vc")
        if not family.enabled:
            return _family_result(plan, self.paths, family_name="seed-vc", artifact=None)
        self._status(callback, model="seed-vc", remote_state="running")
        artifact = Path(str(train_seed_vc(self.repo_root, self.paths, self.cfg)))
        self._status(callback, model="seed-vc", remote_state="complete")
        return _family_result(
            plan,
            self.paths,
            family_name="seed-vc",
            artifact=artifact,
            reused=False,
        )

    def run(
        self,
        *,
        plan: TrainingPlan,
        plan_bytes: bytes,
        status_callback,
    ) -> dict[str, Any]:
        del plan_bytes
        if self.verify_local_plan_files:
            verify_plan_files(plan, self.paths.root)
        checkpoint_observer: _RemoteCheckpointObserver | None = None
        if self.checkpoint_root is None and status_callback is not None:

            def local_checkpoint_status(model: str, step: float, checkpoint: str) -> None:
                status_callback(
                    {
                        "executor": "local",
                        "remote_state": "running",
                        "model": model,
                        "step": int(step),
                        "checkpoint": checkpoint,
                    }
                )

            checkpoint_observer = _RemoteCheckpointObserver(
                plan,
                checkpoint_root=self.paths.cache / "training_runs",
                candidate_root=self.paths.models / ".candidates",
                runtime_repo=self.repo_root,
                status_callback=local_checkpoint_status,
                attest=False,
                status_path_root=self.paths.root,
            )
            checkpoint_observer.start()
        try:
            result = {
                "families": {
                    "irodori": self._irodori(plan, status_callback),
                    "lfm": self._lfm(plan, status_callback),
                    "seed-vc": self._seed_vc(plan, status_callback),
                }
            }
        except BaseException:
            if checkpoint_observer is not None:
                with suppress(OSError, RuntimeError, ValueError):
                    checkpoint_observer.stop(flush=True)
            raise
        if checkpoint_observer is not None:
            checkpoint_observer.stop(flush=True)
        return result


def _config_from_training_plan(plan: TrainingPlan) -> PersonaConfig:
    family_values = {family.family: family for family in plan.families}
    irodori = family_values["irodori"]
    lfm = family_values["lfm"]
    seed = family_values["seed-vc"]
    quality = dict(irodori.evaluation_policy or lfm.evaluation_policy)
    return PersonaConfig.model_validate(
        {
            "name": plan.persona,
            "consent": {"authorized": True, "scope": "remote-authorized-training"},
            "training": {
                "schema_version": 2,
                "executor": "local",
                "remote_data_authorized": False,
                "irodori": {
                    "enabled": irodori.enabled,
                    "method": irodori.method,
                    **irodori.as_dict()["training"],
                },
                "lfm": {
                    "enabled": lfm.enabled,
                    "method": lfm.method,
                    **lfm.as_dict()["training"],
                },
                "seed_vc": {
                    "finetune": seed.enabled,
                    **seed.as_dict()["training"],
                },
                "quality_gate": quality,
            },
        }
    )


def _verify_plan_implementation(plan: TrainingPlan, code_root: Path) -> None:
    for relative, expected in plan.executor_contract.items():
        path = code_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or normalized_source_sha256(path) != expected:
            raise RuntimeError(
                "Runtime executor implementation does not match the immutable TrainingPlan: "
                f"{relative}"
            )
    for family in plan.families:
        if not family.enabled:
            continue
        for relative, expected in family.implementation_contract.items():
            path = code_root.joinpath(*PurePosixPath(relative).parts)
            if not path.is_file() or normalized_source_sha256(path) != expected:
                raise RuntimeError(
                    f"Runtime implementation does not match {family.family} TrainingPlan: "
                    f"{relative}"
                )


def _ensure_runtime_link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise RuntimeError(f"Runtime link points to a different source: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Runtime layout contains an unexpected path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _remote_asset_paths(
    plan: TrainingPlan,
    asset_root: Path,
    family_name: str,
) -> dict[str, Path]:
    family = plan.family(family_name)
    index_path = (
        asset_root / "plans" / plan.plan_id / family_name / family.fingerprint / "asset-index.json"
    )
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Pinned {family_name} asset index is unreadable") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "status",
        "plan_fingerprint",
        "family",
        "family_fingerprint",
        "assets",
        "fingerprint",
    }:
        raise RuntimeError(f"Pinned {family_name} asset index is invalid")
    unsigned = dict(value)
    recorded_fingerprint = unsigned.pop("fingerprint")
    actual_fingerprint = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        value.get("schema_version") != 1
        or value.get("status") != "complete"
        or value.get("plan_fingerprint") != plan.fingerprint
        or value.get("family") != family_name
        or value.get("family_fingerprint") != family.fingerprint
        or not isinstance(value.get("assets"), list)
        or recorded_fingerprint != actual_fingerprint
    ):
        raise RuntimeError(f"Pinned {family_name} asset index is invalid")

    # Import lazily to keep the local-only training module independent from the
    # optional Modal SDK while reusing the exact materializer specification.
    from personavoice.modal_app import asset_specs_for_plan, verify_asset_cache

    expected_specs = {
        spec.name: spec for spec in asset_specs_for_plan(plan) if spec.family == family_name
    }
    resolved: dict[str, Path] = {}
    for raw in value["assets"]:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "repo_id",
            "cache_key",
            "revision",
            "runtime_path",
        }:
            raise RuntimeError(f"Pinned {family_name} asset record is invalid")
        name = raw.get("name")
        cache_key = raw.get("cache_key")
        spec = expected_specs.get(name) if isinstance(name, str) else None
        if (
            spec is None
            or name in resolved
            or not isinstance(cache_key, str)
            or not re.fullmatch(r"[0-9a-f]{64}", cache_key)
            or raw
            != {
                "name": spec.name,
                "repo_id": spec.repo_id,
                "cache_key": spec.cache_key,
                "revision": spec.revision,
                "runtime_path": spec.runtime_path,
            }
        ):
            raise RuntimeError(f"Pinned {family_name} asset identity is invalid")
        path = asset_root / "cache" / cache_key
        try:
            verify_asset_cache(path, spec)
        except ValueError as exc:
            raise RuntimeError(f"Pinned {family_name}/{name} asset is incomplete") from exc
        resolved[name] = path
    if set(resolved) != set(expected_specs):
        raise RuntimeError(f"Pinned {family_name} asset set is incomplete")
    return resolved


def _materialize_lfm_runtime_base(source: Path, destination: Path) -> None:
    marker = source / ".personavoice-revision"
    if (
        any(
            not (source / relative).is_file() or _file_contract(source / relative) != expected
            for relative, expected in LFM_MODEL_ASSET_SHA256.items()
        )
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != LFM_MODEL_REVISION
    ):
        raise RuntimeError("Pinned LFM runtime asset checksum contract is invalid")
    _ensure_runtime_link(source, destination)


def _remote_runtime_repo(
    plan: TrainingPlan,
    *,
    checkpoint_root: Path,
    asset_root: Path,
) -> Path:
    code_root = Path(__file__).resolve().parents[2]
    _verify_plan_implementation(plan, code_root)
    runtime = checkpoint_root / ".runtime-repos" / plan.plan_id
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("src", "workers", "locks", "config"):
        source = code_root / name
        if source.exists():
            _ensure_runtime_link(source, runtime / name)
    for name in ("pyproject.toml", "uv.lock"):
        source = code_root / name
        if source.is_file():
            _ensure_runtime_link(source, runtime / name)
    if plan.family("irodori").enabled:
        _ensure_runtime_link(
            code_root / "vendor" / "Irodori-TTS",
            runtime / "vendor" / "Irodori-TTS",
        )
        assets = _remote_asset_paths(plan, asset_root, "irodori")
        if set(assets) != {"base", "dacvae", "text-encoder"}:
            raise RuntimeError("Pinned Irodori asset set is incomplete")
        _ensure_runtime_link(assets["base"], runtime / "models" / "irodori" / "v4.1-small")
        _ensure_runtime_link(assets["dacvae"], runtime / "models" / "irodori" / "dacvae")
        base = runtime / "models" / "irodori" / "v4.1-small" / IRODORI_MODEL_FILENAME
        dacvae = runtime / "models" / "irodori" / "dacvae" / IRODORI_DACVAE_FILENAME
        if _file_contract(base) != IRODORI_MODEL_SHA256:
            raise RuntimeError("Pinned Irodori base checksum is invalid")
        if _file_contract(dacvae) != IRODORI_DACVAE_SHA256:
            raise RuntimeError("Pinned Irodori DACVAE checksum is invalid")
    if plan.family("lfm").enabled:
        assets = _remote_asset_paths(plan, asset_root, "lfm")
        if set(assets) != {"base"}:
            raise RuntimeError("Pinned LFM asset set is incomplete")
        _materialize_lfm_runtime_base(assets["base"], runtime / "models" / "lfm" / "base")
    if plan.family("irodori").enabled:
        hub_cache = asset_root / "models" / "hf-cache" / "hub"
        if not hub_cache.is_dir():
            raise RuntimeError("Pinned Irodori text-encoder cache projection is missing")
        _ensure_runtime_link(hub_cache, runtime / "models" / "hf-cache" / "hub")
    return runtime


def _verify_common_training_candidate(
    artifact: Path,
    *,
    family: str,
    method: str,
    family_fingerprint: str,
):
    """Verify method-native bytes and return their recorded validation loss.

    The portable result contract protects the transport bytes, while this
    check binds those bytes back to the exact family training contract.  The
    latter is especially important for adapters, whose tensor/config shape is
    not sufficient evidence that they belong to this dataset and base model.
    """

    verification = verify_training_candidate(
        artifact,
        family=family,
        method=method,
        family_fingerprint=family_fingerprint,
    )
    if family == "irodori" and method == "speaker-inversion":
        metadata = irodori_validation_checkpoint_metadata(artifact)
        return verification, (
            metadata[1] if metadata is not None else None
        ), (metadata[0] if metadata is not None else None)
    if family == "seed-vc" and method == "finetune":
        return verification, None, None
    if family == "irodori" and method == "lora":
        if not irodori_lora_candidate_complete(
            artifact,
            plan_fingerprint=family_fingerprint,
        ):
            raise RuntimeError("Irodori LoRA candidate provenance is incomplete")
        provenance_path = artifact / ".personavoice-provenance.json"
    else:
        provenance_path = artifact / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{family}/{method} candidate provenance is unreadable") from exc
    if not isinstance(provenance, dict) or (
        provenance.get("schema_version") != 1
        or provenance.get("family") != family
        or provenance.get("method") != method
        or provenance.get("training_plan_fingerprint") != family_fingerprint
    ):
        raise RuntimeError(f"{family}/{method} candidate provenance changed the family plan")
    raw_loss = provenance.get("best_validation_loss")
    if (
        not isinstance(raw_loss, (int, float))
        or isinstance(raw_loss, bool)
        or not math.isfinite(float(raw_loss))
    ):
        raise RuntimeError(f"{family}/{method} candidate provenance has no finite validation loss")
    if family == "irodori":
        raw_step = provenance.get("best_step")
    elif family == "lfm":
        raw_checkpoint = provenance.get("best_checkpoint")
        raw_step = (
            _lfm_native_checkpoint_step(Path(raw_checkpoint))
            if isinstance(raw_checkpoint, str)
            else None
        )
    else:
        raw_step = None
    if (
        family in {"irodori", "lfm"}
        and (
            not isinstance(raw_step, int)
            or isinstance(raw_step, bool)
            or raw_step <= 0
        )
    ):
        raise RuntimeError(f"{family}/{method} candidate provenance has no valid best step")
    return verification, float(raw_loss), raw_step


def _copy_remote_artifact(
    source: Path,
    destination: Path,
    *,
    family: str,
    method: str,
    family_fingerprint: str,
) -> tuple[Path, Path]:
    """Copy a verified candidate into a result-only directory.

    The first returned path is the directory represented by ResultCandidate;
    the second is the method-specific artifact used by local publication.
    """

    source_verification, source_loss, source_step = _verify_common_training_candidate(
        source,
        family=family,
        method=method,
        family_fingerprint=family_fingerprint,
    )
    if destination.exists():
        raise FileExistsError(f"Remote result candidate already exists: {destination}")
    if source.is_file():
        destination.mkdir(parents=True)
        copied = destination / source.name
        shutil.copy2(source, copied)
        artifact = copied
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        artifact = destination
    copied_verification, copied_loss, copied_step = _verify_common_training_candidate(
        artifact,
        family=family,
        method=method,
        family_fingerprint=family_fingerprint,
    )
    if (
        copied_verification.digest != source_verification.digest
        or copied_verification.files != source_verification.files
        or copied_loss != source_loss
        or copied_step != source_step
    ):
        raise RuntimeError("Remote candidate copy was not lossless")
    return destination, artifact


def _is_link_or_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(checker is not None and checker())


def _checkpoint_payload_inventory(path: Path) -> tuple[tuple[str, int, str], ...]:
    """Hash only native resume payload, excluding PersonaVoice attestations."""

    if _is_link_or_junction(path) or not path.exists():
        raise RuntimeError("Native checkpoint payload is missing or is a link")
    if path.is_file():
        if path.stat().st_size <= 0:
            raise RuntimeError("Native checkpoint payload is empty")
        return ((path.name, path.stat().st_size, sha256_file(path)),)
    if not path.is_dir():
        raise RuntimeError("Native checkpoint payload is not a regular file or directory")
    inventory: list[tuple[str, int, str]] = []
    excluded = {CHECKPOINT_COMPLETION_NAME, CHECKPOINT_FAMILY_NAME}
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path).as_posix()
        if _is_link_or_junction(candidate):
            raise RuntimeError(f"Native checkpoint payload contains a link: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError(f"Native checkpoint payload is not regular: {relative}")
        if relative in excluded:
            continue
        size = candidate.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Native checkpoint payload contains an empty file: {relative}")
        inventory.append((relative, size, sha256_file(candidate)))
    if not inventory:
        raise RuntimeError("Native checkpoint directory contains no resumable payload")
    return tuple(inventory)


def _link_or_copy_checkpoint_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as link_error:
        shutil.copy2(source, destination)
        if destination.stat().st_size != source.stat().st_size or sha256_file(
            destination
        ) != sha256_file(source):
            raise RuntimeError("Native checkpoint fallback copy was not lossless") from link_error


def _materialize_checkpoint_payload(source: Path, destination: Path) -> None:
    if source.is_file():
        _link_or_copy_checkpoint_file(source, destination / source.name)
        return
    for candidate in sorted(source.rglob("*")):
        relative = candidate.relative_to(source)
        target = destination / relative
        if _is_link_or_junction(candidate):
            raise RuntimeError(f"Native checkpoint payload contains a link: {relative.as_posix()}")
        if candidate.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif candidate.is_file():
            _link_or_copy_checkpoint_file(candidate, target)
        else:
            raise RuntimeError(f"Native checkpoint payload is not regular: {relative.as_posix()}")


def _write_remote_checkpoint_marker(
    plan: TrainingPlan,
    family_name: str,
    checkpoint_root: Path,
    *,
    native_checkpoint: Path,
    step: int,
    method: str,
    family_contract: FamilyPlan | None = None,
) -> Path:
    """Finalize an actual method-native resume payload, writing the marker last."""

    primary_family = plan.family(family_name)
    if family_contract is None:
        family = primary_family
    else:
        expected_auxiliary = primary_family.auxiliary_family
        if (
            expected_auxiliary is None
            or family_contract.as_dict() != expected_auxiliary.as_dict()
            or family_contract.fingerprint != expected_auxiliary.fingerprint
        ):
            raise RuntimeError("Remote checkpoint auxiliary family contract is unauthorized")
        family = family_contract
    if not family.enabled or family.method != method:
        raise RuntimeError("Remote checkpoint method changed the immutable family contract")
    namespace = checkpoint_root / family_name / family.fingerprint
    namespace.mkdir(parents=True, exist_ok=True)
    if _is_link_or_junction(namespace) or not namespace.is_dir():
        raise RuntimeError("Remote checkpoint namespace must be a real directory")
    native_checkpoint = native_checkpoint.resolve(strict=True)
    expected_inventory = _checkpoint_payload_inventory(native_checkpoint)
    directory = namespace / f"checkpoint-{step}"
    same_directory = (
        native_checkpoint.is_dir()
        and directory.exists()
        and native_checkpoint == directory.resolve(strict=True)
    )

    if directory.exists() and not _is_link_or_junction(directory) and directory.is_dir():
        compatible = latest_verified_family_checkpoint(
            namespace,
            plan_fingerprint=plan.fingerprint,
            family=family,
        )
        try:
            existing = verify_completed_directory(
                directory,
                expected_plan_fingerprint=plan.fingerprint,
                completion_name=CHECKPOINT_COMPLETION_NAME,
                expected_kind="checkpoint",
                expected_model=family_name,
            )
            actual_inventory = _checkpoint_payload_inventory(directory)
        except (OSError, RuntimeError, ValueError):
            pass
        else:
            if (
                existing.step == step
                and actual_inventory == expected_inventory
                and compatible is not None
                and compatible[0].resolve(strict=True) == directory.resolve(strict=True)
            ):
                return directory

    if same_directory:
        for name in (CHECKPOINT_COMPLETION_NAME, CHECKPOINT_FAMILY_NAME):
            marker = directory / name
            if marker.exists() and (_is_link_or_junction(marker) or not marker.is_file()):
                raise RuntimeError("Remote checkpoint attestation path is unsafe")
            marker.unlink(missing_ok=True)
        write_checkpoint_family_contract(directory, family)
        write_completion_manifest(
            directory,
            kind="checkpoint",
            plan_fingerprint=plan.fingerprint,
            model=family_name,
            step=step,
            quality_gate_passed=False,
            completion_name=CHECKPOINT_COMPLETION_NAME,
        )
    else:
        staging = namespace / f".{directory.name}.{uuid4().hex}.staging"
        staging.mkdir()
        try:
            _materialize_checkpoint_payload(native_checkpoint, staging)
            if _checkpoint_payload_inventory(staging) != expected_inventory:
                raise RuntimeError("Remote checkpoint materialization was not lossless")
            write_checkpoint_family_contract(staging, family)
            write_completion_manifest(
                staging,
                kind="checkpoint",
                plan_fingerprint=plan.fingerprint,
                model=family_name,
                step=step,
                quality_gate_passed=False,
                completion_name=CHECKPOINT_COMPLETION_NAME,
            )
            if directory.exists():
                if _is_link_or_junction(directory) or not directory.is_dir():
                    raise RuntimeError("Remote checkpoint destination is unsafe")
                shutil.rmtree(directory)
            staging.replace(directory)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    verified = verify_completed_directory(
        directory,
        expected_plan_fingerprint=plan.fingerprint,
        completion_name=CHECKPOINT_COMPLETION_NAME,
        expected_kind="checkpoint",
        expected_model=family_name,
    )
    if verified.step != step or _checkpoint_payload_inventory(directory) != expected_inventory:
        raise RuntimeError("Remote checkpoint final verification failed")
    compatible = latest_verified_family_checkpoint(
        namespace,
        plan_fingerprint=plan.fingerprint,
        family=family,
    )
    if compatible is None:
        raise RuntimeError("Remote checkpoint family contract failed final verification")
    return directory


def _real_directory_root(path: Path, *, label: str) -> Path:
    """Resolve a caller-owned root only after rejecting a leaf symlink."""

    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Common runner {label} root must be a real directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Common runner {label} root could not be resolved") from exc


def _checkpoint_stability_signature(
    path: Path,
) -> tuple[tuple[str, int, int], ...] | None:
    """Return a cheap stability signature without treating partial files as complete."""

    excluded = {CHECKPOINT_COMPLETION_NAME, CHECKPOINT_FAMILY_NAME}
    if _is_link_or_junction(path):
        raise RuntimeError("Periodic native checkpoint path contains a link")
    if path.is_file():
        stat = path.stat()
        return ((path.name, stat.st_size, stat.st_mtime_ns),) if stat.st_size > 0 else None
    if not path.is_dir():
        return None
    signature: list[tuple[str, int, int]] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path).as_posix()
        if _is_link_or_junction(candidate):
            raise RuntimeError(f"Periodic native checkpoint payload contains a link: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError(f"Periodic native checkpoint payload is not regular: {relative}")
        if relative in excluded:
            continue
        stat = candidate.stat()
        if stat.st_size <= 0:
            return None
        signature.append((relative, stat.st_size, stat.st_mtime_ns))
    return tuple(signature) if signature else None


def _irodori_periodic_checkpoint_step(path: Path) -> int | None:
    metadata = irodori_validation_checkpoint_metadata(path)
    if metadata is not None:
        return metadata[0]
    match = _IRODORI_PERIODIC_CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


class _RemoteCheckpointObserver:
    """Attest and report complete native checkpoints while training is active.

    Modal only makes Volume writes durable after ``volume.commit()``.  The
    callback supplied by the Modal app commits at exactly this observer's
    boundary, after method-native payload verification and completion-marker
    finalization.  A checkpoint must have an unchanged stat inventory across
    two observations (or pass the final forced scan) before the more expensive
    method-specific validation runs.
    """

    def __init__(
        self,
        plan: TrainingPlan,
        *,
        checkpoint_root: Path,
        runtime_repo: Path,
        status_callback: Callable[[str, float, str], None] | None,
        candidate_root: Path | None = None,
        attest: bool = True,
        status_path_root: Path | None = None,
        interval_seconds: float = _REMOTE_CHECKPOINT_OBSERVER_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Remote checkpoint observer interval must be positive")
        self.plan = plan
        self.checkpoint_root = checkpoint_root
        self.runtime_repo = runtime_repo
        self.status_callback = status_callback
        self.candidate_root = candidate_root or (checkpoint_root / "candidates")
        self.attest = attest
        self.status_path_root = status_path_root or checkpoint_root
        self.interval_seconds = interval_seconds
        self._observed: dict[str, tuple[tuple[str, int, int], ...]] = {}
        self._persisted: dict[str, tuple[tuple[str, int, int], ...]] = {}
        self._persisted_labels: dict[str, tuple[str, int, str]] = {}
        self._stable_failures: set[str] = set()
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._background_failed = False

    def _candidates(self) -> tuple[tuple[str, FamilyPlan, int, Path], ...]:
        candidates: list[tuple[str, FamilyPlan, int, Path]] = []
        for family in self.plan.families:
            if not family.enabled or family.family == "seed-vc":
                continue
            root: Path
            paths: list[Path]
            if family.family == "lfm":
                root = self.checkpoint_root / "lfm" / family.fingerprint
                paths = list(root.glob("checkpoint-*")) if root.is_dir() else []
                for path in paths:
                    step = _lfm_native_checkpoint_step(path)
                    if step is not None:
                        candidates.append((family.family, family, step, path))
                continue
            if family.family != "irodori":
                continue
            if family.method == "full":
                root = self.checkpoint_root / "irodori" / family.fingerprint
                paths = list(root.glob("checkpoint_*.pt")) if root.is_dir() else []
            elif family.method == "lora":
                root = (
                    self.candidate_root
                    / "irodori"
                    / family.fingerprint
                    / "models"
                    / "irodori"
                    / "lora"
                )
                paths = list(root.glob("checkpoint_*")) if root.is_dir() else []
            elif family.method == "speaker-inversion":
                root = (
                    self.candidate_root
                    / "irodori"
                    / family.fingerprint
                    / "models"
                    / "irodori"
                    / "speaker"
                )
                paths = list(root.glob("checkpoint_*.speaker.safetensors")) if root.is_dir() else []
            else:
                continue
            for path in paths:
                step = _irodori_periodic_checkpoint_step(path)
                if step is not None:
                    candidates.append((family.family, family, step, path))
            auxiliary = family.auxiliary_family
            if auxiliary is not None:
                auxiliary_root = (
                    self.candidate_root
                    / "irodori"
                    / auxiliary.fingerprint
                    / "models"
                    / "irodori"
                    / "speaker"
                )
                auxiliary_paths = (
                    list(auxiliary_root.glob("checkpoint_*.speaker.safetensors"))
                    if auxiliary_root.is_dir()
                    else []
                )
                for path in auxiliary_paths:
                    step = _irodori_periodic_checkpoint_step(path)
                    if step is not None:
                        candidates.append(("irodori-auxiliary", auxiliary, step, path))
        return tuple(sorted(candidates, key=lambda item: (item[0], item[2], item[3].name)))

    def _method_complete(self, family: str, method: str, path: Path) -> bool:
        if family == "lfm":
            return _lfm_native_checkpoint_complete(path, method=method)
        if method == "full":
            return _verify_full_training_checkpoint(
                self.runtime_repo / "vendor" / "Irodori-TTS",
                path,
                env=local_model_env(self.runtime_repo),
            )
        if method == "lora":
            return _verify_lora_training_checkpoint(
                self.runtime_repo / "vendor" / "Irodori-TTS",
                path,
                env=local_model_env(self.runtime_repo),
            )
        if method == "speaker-inversion":
            return _verify_speaker_embedding_checkpoint(
                self.runtime_repo / "vendor" / "Irodori-TTS",
                path,
                env=local_model_env(self.runtime_repo),
            )
        return False

    def scan(self, *, force: bool = False) -> None:
        with self._lock:
            candidates = self._candidates()
            current_keys = {
                f"{label}:{family.fingerprint}:{path.absolute()}"
                for label, family, _, path in candidates
            }
            self._stable_failures.intersection_update(current_keys)
            for status_label, family, step, path in candidates:
                key = f"{status_label}:{family.fingerprint}:{path.absolute()}"
                try:
                    signature = _checkpoint_stability_signature(path)
                except (OSError, RuntimeError):
                    if force:
                        self._stable_failures.add(key)
                    continue
                if signature is None:
                    continue
                previous = self._observed.get(key)
                self._observed[key] = signature
                if not force and previous != signature:
                    continue
                if self._persisted.get(key) == signature:
                    if force and self.status_callback is not None:
                        saved_family, saved_step, saved_label = self._persisted_labels[key]
                        self.status_callback(
                            saved_family,
                            float(saved_step),
                            saved_label,
                        )
                    continue
                try:
                    if not self._method_complete(family.family, family.method, path):
                        if force:
                            self._stable_failures.add(key)
                        continue
                    completed = (
                        _write_remote_checkpoint_marker(
                            self.plan,
                            family.family,
                            self.checkpoint_root,
                            native_checkpoint=path,
                            step=step,
                            method=family.method,
                            family_contract=(
                                family if status_label == "irodori-auxiliary" else None
                            ),
                        )
                        if self.attest
                        else path.resolve(strict=True)
                    )
                    label = (
                        completed.resolve(strict=True)
                        .relative_to(self.status_path_root.resolve(strict=True))
                        .as_posix()
                    )
                    if self.status_callback is not None:
                        self.status_callback(status_label, float(step), label)
                except (OSError, RuntimeError, ValueError):
                    try:
                        changed = _checkpoint_stability_signature(path) != signature
                    except (OSError, RuntimeError):
                        changed = True
                    if not changed:
                        self._stable_failures.add(key)
                    continue
                self._persisted[key] = signature
                self._persisted_labels[key] = (status_label, step, label)
                self._stable_failures.discard(key)
            if force and (self._stable_failures or self._background_failed):
                raise RuntimeError(
                    "A stable periodic checkpoint could not be verified and committed"
                )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.scan()
            except BaseException:
                self._background_failed = True
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Remote checkpoint observer was already started")
        self._thread = Thread(
            target=self._run,
            name="personavoice-checkpoint-observer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, flush: bool) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
            if self._thread.is_alive():
                raise RuntimeError("Remote checkpoint observer did not stop")
        if flush:
            self.scan(force=True)


def run_training_plan(
    plan_bytes: bytes,
    bundle_root: Path,
    run_root: Path,
    checkpoint_root: Path,
    asset_root: Path,
    *,
    status_callback=None,
) -> tuple[ResultFamily, ...]:
    """Run the same family trainers used locally from a verified remote bundle."""

    plan = TrainingPlan.from_bytes(plan_bytes)
    if canonical_plan_bytes(plan) != plan_bytes:
        raise ValueError("Common runner received non-canonical TrainingPlan bytes")
    bundle = verify_training_bundle(
        bundle_root,
        expected_plan_fingerprint=plan.fingerprint,
    )
    bundled_plan = bundle.root / "contracts" / "training-plan.json"
    if bundled_plan.read_bytes() != plan_bytes:
        raise ValueError("Common runner bundle contains a different TrainingPlan")
    if plan.family("seed-vc").enabled:
        raise RuntimeError("Seed-VC fine-tuning is not authorized in a remote training bundle")
    run_root = _real_directory_root(run_root, label="result")
    checkpoint_root = _real_directory_root(checkpoint_root, label="checkpoint")
    asset_root = _real_directory_root(asset_root, label="asset")
    runtime_repo = _remote_runtime_repo(
        plan,
        checkpoint_root=checkpoint_root,
        asset_root=asset_root,
    )
    inventory_by_role = {item.role: item for item in bundle.inventory.files}
    manifest_item = inventory_by_role.get("irodori-latent-manifest")
    irodori_manifest = (
        bundle.root.joinpath(*PurePosixPath(IRODORI_SANITIZED_MANIFEST_PATH).parts)
        if manifest_item is not None
        else None
    )
    lfm_item = inventory_by_role.get("lfm-conversations")
    lfm_dataset = (
        bundle.root.joinpath(*PurePosixPath(lfm_item.path).parts) if lfm_item is not None else None
    )
    workspace = checkpoint_root / "workspaces" / plan.plan_id
    paths = PersonaPaths(workspace / "persona")
    for directory in (paths.dataset, paths.models, paths.cache, run_root / "models"):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = _config_from_training_plan(plan)
    runner = PersonaLocalTrainingRunner(
        runtime_repo,
        paths,
        cfg,
        legacy_result=None,
        irodori_manifest=irodori_manifest,
        lfm_dataset=lfm_dataset,
        candidate_root=checkpoint_root / "candidates",
        checkpoint_root=checkpoint_root,
        backend_override="cu128",
        verify_local_plan_files=False,
    )

    def relay(progress: dict[str, Any]) -> None:
        if status_callback is None:
            return
        model = progress.get("model")
        step = progress.get("step", 0)
        checkpoint = progress.get("checkpoint") or ""
        status_callback(str(model), float(step), str(checkpoint))

    checkpoint_observer = _RemoteCheckpointObserver(
        plan,
        checkpoint_root=checkpoint_root,
        runtime_repo=runtime_repo,
        status_callback=status_callback,
    )
    checkpoint_observer.start()
    try:
        result = runner.run(plan=plan, plan_bytes=plan_bytes, status_callback=relay)
    except BaseException:
        # Preserve any checkpoint that completed just before a trainer error.
        # The original training failure remains the primary diagnostic; Modal
        # retries will rediscover only fully attested payloads.
        with suppress(OSError, RuntimeError, ValueError):
            checkpoint_observer.stop(flush=True)
        raise
    else:
        checkpoint_observer.stop(flush=True)
    raw_families = result.get("families")
    if not isinstance(raw_families, dict):
        raise RuntimeError("Common family runner returned no result mapping")
    remote_families: list[ResultFamily] = []
    for family in plan.families:
        if not family.enabled:
            continue
        raw = raw_families.get(family.family)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Common runner omitted {family.family}")
        if (
            raw.get("enabled") is not True
            or raw.get("method") != family.method
            or raw.get("family_fingerprint") != family.fingerprint
        ):
            raise RuntimeError(f"Common runner changed the {family.family} family contract")
        source = _family_artifact_path(
            paths,
            raw.get("artifact"),
            external_candidate_root=runner.candidate_root,
        )
        destination = run_root / "models" / family.family / family.method
        result_directory, copied_artifact = _copy_remote_artifact(
            source,
            destination,
            family=family.family,
            method=family.method,
            family_fingerprint=family.fingerprint,
        )
        _, recorded_loss, recorded_step = _verify_common_training_candidate(
            copied_artifact,
            family=family.family,
            method=family.method,
            family_fingerprint=family.fingerprint,
        )
        validation = raw.get("validation")
        if not isinstance(validation, dict) or "loss" not in validation:
            raise RuntimeError(f"{family.family} candidate omitted its validation result")
        loss = validation["loss"]
        finite_loss = (
            isinstance(loss, (int, float))
            and not isinstance(loss, bool)
            and math.isfinite(float(loss))
        )
        if not finite_loss or float(loss) != recorded_loss:
            raise RuntimeError(
                f"{family.family} candidate validation loss does not match its artifact"
            )
        completed_checkpoint = runner.completed_checkpoints.get(family.family)
        if completed_checkpoint is not None:
            native_checkpoint, step = completed_checkpoint
            if recorded_step is not None and step != recorded_step:
                raise RuntimeError(
                    f"{family.family} selected artifact and native best checkpoint steps disagree"
                )
        else:
            native_checkpoint = None
            step = int(recorded_step or family.training.get("max_steps", 0) or 0)
            if family.family == "lfm" and recorded_step is None:
                step = max(step, int(float(family.training.get("epochs", 0)) * 1000))
        if family.family == "irodori" and family.method == "speaker-inversion":
            metadata = irodori_validation_checkpoint_metadata(copied_artifact)
            if metadata is None:
                raise RuntimeError("Irodori Speaker Inversion artifact is not a best checkpoint")
            if native_checkpoint is not None and step != metadata[0]:
                raise RuntimeError(
                    "Irodori Speaker Inversion candidate and native checkpoint steps disagree"
                )
            step = metadata[0]
        candidate_validation: dict[str, Any] = {
            "passed": True,
            "validation_loss": float(loss) if finite_loss else None,
            "step": step,
        }
        auxiliary = raw.get("auxiliary_speaker_embedding")
        auxiliary_fingerprint = raw.get("auxiliary_family_fingerprint")
        auxiliary_requested = (
            family.family == "irodori"
            and family.training.get("auxiliary_speaker_inversion") is True
        )
        if auxiliary is None:
            if auxiliary_requested or auxiliary_fingerprint is not None:
                raise RuntimeError("Common runner omitted the requested auxiliary speaker")
        else:
            auxiliary_family = family.auxiliary_family
            if (
                not auxiliary_requested
                or auxiliary_family is None
                or not isinstance(auxiliary, str)
                or not auxiliary
                or auxiliary_fingerprint != auxiliary_family.fingerprint
            ):
                raise RuntimeError("Common runner returned an unexpected auxiliary speaker")
            auxiliary_source = _family_artifact_path(
                paths,
                auxiliary,
                external_candidate_root=runner.candidate_root,
            )
            auxiliary_destination = run_root / "auxiliary" / "irodori-speaker"
            _, copied_auxiliary = _copy_remote_artifact(
                auxiliary_source,
                auxiliary_destination,
                family="irodori",
                method="speaker-inversion",
                family_fingerprint=auxiliary_family.fingerprint,
            )
            candidate_validation["auxiliary_speaker_path"] = copied_auxiliary.relative_to(
                run_root
            ).as_posix()
            candidate_validation["auxiliary_family_fingerprint"] = (
                auxiliary_family.fingerprint
            )
            auxiliary_checkpoint = runner.completed_checkpoints.get("irodori-auxiliary")
            if auxiliary_checkpoint is not None:
                native_auxiliary_checkpoint, auxiliary_step = auxiliary_checkpoint
                completed_auxiliary = _write_remote_checkpoint_marker(
                    plan,
                    "irodori",
                    checkpoint_root,
                    native_checkpoint=native_auxiliary_checkpoint,
                    step=auxiliary_step,
                    method="speaker-inversion",
                    family_contract=auxiliary_family,
                )
                auxiliary_checkpoint_label = completed_auxiliary.relative_to(
                    checkpoint_root
                ).as_posix()
            else:
                reused_auxiliary = latest_verified_family_checkpoint(
                    checkpoint_root / "irodori" / auxiliary_family.fingerprint,
                    plan_fingerprint=plan.fingerprint,
                    family=auxiliary_family,
                )
                if reused_auxiliary is None:
                    auxiliary_step = 0
                    auxiliary_checkpoint_label = ""
                else:
                    auxiliary_step = reused_auxiliary[1].step
                    auxiliary_checkpoint_label = reused_auxiliary[0].relative_to(
                        checkpoint_root
                    ).as_posix()
            candidate_validation["auxiliary_step"] = auxiliary_step
            if status_callback is not None and auxiliary_checkpoint_label:
                status_callback(
                    "irodori-auxiliary",
                    float(auxiliary_step),
                    auxiliary_checkpoint_label,
                )
        relative = result_directory.relative_to(run_root).as_posix()
        checkpoint_label: str | None = None
        if native_checkpoint is not None:
            completed_directory = _write_remote_checkpoint_marker(
                plan,
                family.family,
                checkpoint_root,
                native_checkpoint=native_checkpoint,
                step=step,
                method=family.method,
            )
            checkpoint_label = completed_directory.relative_to(checkpoint_root).as_posix()
        else:
            reused_checkpoint = latest_verified_family_checkpoint(
                checkpoint_root / family.family / family.fingerprint,
                plan_fingerprint=plan.fingerprint,
                family=family,
            )
            if reused_checkpoint is not None:
                step = reused_checkpoint[1].step
                checkpoint_label = reused_checkpoint[0].relative_to(checkpoint_root).as_posix()
        candidate_validation["step"] = step
        remote_families.append(
            ResultFamily(
                family=family.family,
                method=family.method,
                family_fingerprint=family.fingerprint,
                selected_artifact_path=relative,
                candidates=(
                    ResultCandidate(
                        artifact_path=relative,
                        validation=candidate_validation,
                    ),
                ),
            )
        )
        if status_callback is not None:
            status_callback(family.family, float(step), checkpoint_label or "")
    return tuple(sorted(remote_families, key=lambda item: item.family))


def _family_artifact_path(
    paths: PersonaPaths,
    value: Any,
    *,
    external_candidate_root: Path | None = None,
) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("Common runner family artifact path is missing")
    if (
        "\\" in value
        or re.match(r"^[A-Za-z]:", value) is not None
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise RuntimeError("Common runner family artifact path is not portable")
    portable = PurePosixPath(value)
    if portable.is_absolute() or portable.as_posix() != value:
        raise RuntimeError("Common runner family artifact path is not portable")
    if value.startswith(_EXTERNAL_CANDIDATE_PREFIX):
        if external_candidate_root is None:
            raise RuntimeError("Common runner returned an unexpected external candidate path")
        relative = PurePosixPath(value[len(_EXTERNAL_CANDIDATE_PREFIX) :])
        root = external_candidate_root
    else:
        relative = portable
        root = paths.root
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError("Common runner family artifact path is not portable")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("Common runner family artifact root is unavailable") from exc
    path = resolved_root.joinpath(*relative.parts)
    cursor = resolved_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("Common runner family artifact path contains a symlink")
    try:
        path.resolve(strict=True).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Common runner family artifact escaped its workspace") from exc
    return path


def _positive_timeout_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of seconds") from exc
    if not 0 < value <= 7 * 24 * 60 * 60:
        raise ValueError(f"{name} must be greater than zero and at most seven days")
    return value


def _load_downloaded_result(
    root: Path,
    *,
    submission: RemoteSubmission,
    plan: TrainingPlan,
) -> DownloadedTrainingResult:
    completion = verify_completed_directory(
        root,
        expected_plan_fingerprint=plan.fingerprint,
        completion_name=RESULT_COMPLETION_NAME,
        expected_kind="result",
        expected_model=submission.model,
        # A completed remote download is a verified candidate cache, not a
        # publication. The remote marker must remain false until the held-out
        # gate runs locally, including when this cache is reused after an
        # interruption between download and state adoption.
        require_quality_gate=False,
    )
    try:
        value = json.loads((root / TRAINING_RESULT_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Downloaded Modal training-result.json is unreadable") from exc
    expected = {
        family: (fingerprint, method) for family, fingerprint, method in submission.family_contracts
    }
    contract = training_result_from_value(
        value,
        expected_plan_fingerprint=plan.fingerprint,
        expected_families=expected,
    )
    completed_paths = {item.path: item for item in completion.files}
    for family in contract.families:
        for candidate in family.candidates:
            artifact_prefix = candidate.artifact_path.rstrip("/") + "/"
            expected_files = {item.path: item for item in candidate.files}
            if not expected_files or any(
                not relative.startswith(artifact_prefix) for relative in expected_files
            ):
                raise ValueError(
                    f"Downloaded {family.family} candidate inventory escapes its artifact"
                )
            for relative, member in expected_files.items():
                completed = completed_paths.get(relative)
                if completed is None or (
                    completed.sha256 != member.sha256 or completed.size != member.size
                ):
                    raise ValueError(
                        f"Downloaded {family.family} candidate inventory is inconsistent"
                    )
    return DownloadedTrainingResult(completion=completion, contract=contract)


def _remote_families(
    downloaded: DownloadedTrainingResult,
    *,
    plan: TrainingPlan,
    paths: PersonaPaths,
    root: Path,
) -> dict[str, Any]:
    selected = {family.family: family for family in downloaded.contract.families}
    families: dict[str, Any] = {}
    for family_plan in plan.families:
        family = selected.get(family_plan.family)
        if not family_plan.enabled:
            families[family_plan.family] = _family_result(
                plan,
                paths,
                family_name=family_plan.family,
                artifact=None,
            )
            continue
        if family is None:
            raise ValueError(f"Modal result omitted enabled family {family_plan.family}")
        matching = [
            candidate
            for candidate in family.candidates
            if candidate.artifact_path == family.selected_artifact_path
        ]
        if len(matching) != 1:
            raise ValueError(f"Modal result selected an ambiguous {family.family} candidate")
        validation = matching[0].validation
        if validation.get("passed") is not True or "validation_loss" not in validation:
            raise RuntimeError(f"Modal {family.family} selected candidate did not pass validation")
        raw_loss = validation.get("validation_loss")
        artifact_root = root.joinpath(*family.selected_artifact_path.split("/"))
        artifact = artifact_root
        if family.method == "speaker-inversion":
            speakers = sorted(artifact_root.glob("*.speaker.safetensors"))
            if len(speakers) != 1:
                raise ValueError(
                    "Modal Irodori speaker-inversion result must contain exactly one "
                    "speaker embedding"
                )
            artifact = speakers[0]
        _, recorded_loss, recorded_step = _verify_common_training_candidate(
            artifact,
            family=family.family,
            method=family.method,
            family_fingerprint=family_plan.fingerprint,
        )
        if (
            not isinstance(raw_loss, (int, float))
            or isinstance(raw_loss, bool)
            or not math.isfinite(float(raw_loss))
            or float(raw_loss) != recorded_loss
        ):
            raise RuntimeError(
                f"Modal {family.family} validation loss does not match candidate provenance"
            )
        validation_loss = float(raw_loss)
        raw_step = validation.get("step")
        if (
            recorded_step is not None
            and (
                not isinstance(raw_step, int)
                or isinstance(raw_step, bool)
                or raw_step != recorded_step
            )
        ):
            raise RuntimeError(
                f"Modal {family.family} validation step does not match candidate provenance"
            )
        auxiliary_artifact: Path | None = None
        raw_auxiliary = validation.get("auxiliary_speaker_path")
        raw_auxiliary_fingerprint = validation.get("auxiliary_family_fingerprint")
        auxiliary_requested = (
            family_plan.family == "irodori"
            and family_plan.training.get("auxiliary_speaker_inversion") is True
        )
        if raw_auxiliary is None:
            if auxiliary_requested or raw_auxiliary_fingerprint is not None:
                raise ValueError("Modal result omitted the requested auxiliary speaker embedding")
        else:
            auxiliary_family = family_plan.auxiliary_family
            if (
                not auxiliary_requested
                or auxiliary_family is None
                or family.family != "irodori"
                or not isinstance(raw_auxiliary, str)
                or not raw_auxiliary
                or raw_auxiliary_fingerprint != auxiliary_family.fingerprint
            ):
                raise ValueError("Modal result contains an invalid auxiliary speaker path")
            auxiliary_path = PurePosixPath(raw_auxiliary)
            if (
                auxiliary_path.is_absolute()
                or any(part in {"", ".", ".."} for part in raw_auxiliary.split("/"))
                or auxiliary_path.as_posix() != raw_auxiliary
            ):
                raise ValueError("Modal result auxiliary speaker path is not portable")
            auxiliary_artifact = root.joinpath(*auxiliary_path.parts)
            try:
                auxiliary_artifact.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError("Modal result auxiliary speaker path is invalid") from exc
            try:
                verify_training_candidate(
                    auxiliary_artifact,
                    family="irodori",
                    method="speaker-inversion",
                    family_fingerprint=auxiliary_family.fingerprint,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Modal result auxiliary speaker embedding is incomplete"
                ) from exc
        families[family.family] = _family_result(
            plan,
            paths,
            family_name=family.family,
            artifact=artifact,
            validation_loss=validation_loss,
            auxiliary_artifact=auxiliary_artifact,
        )
    return families


def _wait_for_modal_result(
    transport: ModalTransport,
    submission: RemoteSubmission,
    plan: TrainingPlan,
    paths: PersonaPaths,
    *,
    status_callback,
) -> dict[str, Any]:
    """Resume/poll one durable Modal call and verify every byte before adoption."""

    if submission.plan_fingerprint != plan.fingerprint:
        raise ValueError("Saved Modal submission belongs to a different TrainingPlan")
    timeout = _positive_timeout_from_env("PERSONAVOICE_MODAL_WAIT_SECONDS", 24 * 60 * 60)
    poll_seconds = _positive_timeout_from_env("PERSONAVOICE_MODAL_POLL_SECONDS", 30.0)
    deadline = time.monotonic() + timeout
    result = None
    while result is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "Modal training is still running. The durable call ID was retained; rerun "
                "`persona train --executor modal` to continue polling without resubmitting."
            )
        result = transport.poll(
            submission,
            timeout=min(poll_seconds, remaining),
            status_callback=status_callback,
        )
        if result is None and poll_seconds < 1:
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    destination = (
        paths.models
        / ".remote_candidates"
        / plan.fingerprint
        / result.completion_manifest_sha256[:24]
    )
    if destination.exists():
        downloaded = _load_downloaded_result(
            destination,
            submission=submission,
            plan=plan,
        )
    else:
        downloaded = transport.download_result(
            result,
            destination,
            expected_plan_fingerprint=plan.fingerprint,
        )
    return _remote_families(
        downloaded,
        plan=plan,
        paths=paths,
        root=destination,
    )


def train_persona(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    force: bool = False,
    executor: str | None = None,
) -> dict:
    if not cfg.consent.authorized:
        raise PermissionError("Training is blocked because consent.authorized is not true.")

    load_root_environment(repo_root)

    store = StateStore(paths.state)
    current_prepare_fingerprint = _prepare_fingerprint(paths, cfg)
    if not store.is_complete("prepare", current_prepare_fingerprint):
        raise RuntimeError(
            "Prepared dataset is missing, stale, or incomplete for the current inputs. "
            "Run `persona prepare` before training."
        )

    source = paths.dataset / "irodori_source.jsonl"
    if cfg.training.irodori.enabled and _line_count(source) < 2:
        raise RuntimeError(
            "Prepared Irodori dataset is missing or too small. Run `persona prepare` first."
        )

    fingerprint = _fingerprint(paths, cfg)
    previous = store.stage("train")
    if not force and store.is_complete("train", fingerprint):
        return previous.get("result", {})
    if not force and store.is_trained(fingerprint):
        return previous.get("result", {})

    if cfg.training.irodori.enabled:
        manifest = ensure_irodori_manifest(
            repo_root,
            paths,
            conditioning=cfg.training.irodori.conditioning,
        )
    else:
        # The builder ignores this placeholder when the family is disabled.
        manifest = paths.dataset / "irodori_manifest.jsonl"
    plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    verify_plan_files(plan, paths.root)
    requested_executor = executor or cfg.training.executor
    if requested_executor not in {"auto", "local", "modal"}:
        raise ValueError(f"Unsupported training executor: {requested_executor!r}")

    setup_current = False
    backend = "unknown"
    try:
        setup = require_current_environment(repo_root)
    except (FileNotFoundError, RuntimeError, ValueError):
        setup_path = repo_root / ".runtime" / "setup.json"
        try:
            recorded = json.loads(setup_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            recorded = {}
        if isinstance(recorded, dict) and isinstance(recorded.get("irodori_backend"), str):
            backend = str(recorded["irodori_backend"])
    else:
        setup_current = True
        backend = str(setup.get("irodori_backend") or "unknown")
    resources = inspect_local_resources(
        paths.root,
        backend=backend,
        setup_current=setup_current,
    )
    remote_consent = RemoteConsent(
        remote_data_authorized=cfg.training.remote_data_authorized,
        scopes=frozenset({cfg.consent.scope}),
    )
    modal_auth_probe = detect_modal_auth
    preselected = select_executor(
        requested_executor,
        plan,
        resources,
        consent=remote_consent,
        modal_auth_probe=modal_auth_probe,
    )
    if preselected.executor == "modal" and plan.family("seed-vc").enabled:
        raise ModalUnavailableError(
            "Seed-VC fine-tuning is local-only because its source audio is not an approved "
            "remote bundle member. Disable Seed-VC fine-tuning or run a separate authorized "
            "local training plan."
        )
    modal_transport = ModalTransport(ModalSettings.from_env())

    active_submission: RemoteSubmission | None = None

    def record_progress(progress: dict[str, Any]) -> None:
        value = dict(progress)
        if active_submission is not None:
            value["submission"] = active_submission.resume_dict()
        store.set_progress("train", value)

    def bundle_factory():
        destination = paths.cache / "training_bundles" / plan.fingerprint
        if destination.exists():
            return verify_training_bundle(
                destination,
                expected_plan_fingerprint=plan.fingerprint,
            )
        return build_training_bundle(plan, paths.root, destination)

    with store.running(
        "train",
        fingerprint,
        force=force,
        success_status="trained",
    ) as locked_previous:
        # Both checks intentionally occur after the OS stage lock is acquired.
        # A stale v0.3 running/failed record may seed exact Irodori native
        # checkpoints, while only a complete result may provide a final artifact.
        legacy_checkpoint_lineage_verified = _legacy_v03_checkpoint_lineage_verified(
            locked_previous,
            paths,
            cfg,
        )
        legacy_final_lineage_verified = _legacy_v03_final_lineage_verified(
            locked_previous,
            paths,
            cfg,
        )
        previous_result = locked_previous.get("result")
        legacy_result = (
            previous_result
            if legacy_final_lineage_verified and isinstance(previous_result, dict)
            else None
        )
        local_runner = PersonaLocalTrainingRunner(
            repo_root,
            paths,
            cfg,
            legacy_result=legacy_result,
            legacy_checkpoint_lineage_verified=legacy_checkpoint_lineage_verified,
        )
        staged_progress = locked_previous.get("progress")
        resume_payload = (
            staged_progress.get("submission")
            if isinstance(staged_progress, dict) and staged_progress.get("executor") == "modal"
            else None
        )
        if (
            isinstance(resume_payload, dict)
            and resume_payload.get("plan_fingerprint") == plan.fingerprint
        ):
            decision = preselected
            if decision.executor != "modal":
                raise RuntimeError(
                    "A resumable Modal call exists for this plan; rerun with --executor modal "
                    "or wait for it before choosing local execution."
                )
            active_submission = RemoteSubmission.from_resume_dict(resume_payload)
            dispatched_result: Any = active_submission
        else:
            dispatched = dispatch_training(
                requested_executor,
                plan,
                resources,
                local_executor=LocalExecutor(local_runner),
                modal_executor=ModalExecutor(modal_transport),
                bundle_factory=bundle_factory,
                consent=remote_consent,
                modal_auth_probe=modal_auth_probe,
                status_callback=record_progress,
            )
            decision = dispatched.decision
            dispatched_result = dispatched.result
            if decision.executor == "modal":
                if not isinstance(dispatched_result, RemoteSubmission):
                    raise RuntimeError("Modal executor returned no durable submission contract")
                active_submission = dispatched_result
                record_progress(
                    {
                        "executor": "modal",
                        "remote_state": "running",
                        "model": active_submission.model,
                        "step": 0,
                        "checkpoint": None,
                    }
                )

        if decision.executor == "local":
            if not isinstance(dispatched_result, dict) or not isinstance(
                dispatched_result.get("families"), dict
            ):
                raise RuntimeError("Local executor returned an invalid family result contract")
            families = dispatched_result["families"]
            remote_candidate_verified = False
        else:
            assert active_submission is not None
            try:
                families = _wait_for_modal_result(
                    modal_transport,
                    active_submission,
                    plan,
                    paths,
                    status_callback=record_progress,
                )
            except ModalTerminalCallRecoveredError as exc:
                # Do not leave the terminal call as a resumable submission.
                # The deployed recovery function already removed only claims
                # owned by that independently verified failed call; a rerun can
                # now dispatch a replacement that adopts verified checkpoints.
                failed_model = active_submission.model
                active_submission = None
                store.set_progress(
                    "train",
                    {
                        "executor": "modal",
                        "remote_state": "failed-recoverable",
                        "model": failed_model,
                        "step": 0,
                        "checkpoint": None,
                        "terminal_call_id": exc.call_id,
                        "claims_released": True,
                    },
                )
                raise
            remote_candidate_verified = True

        preflight = decision.local_preflight
        result = {
            "train_schema": TRAIN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "plan_fingerprint": plan.fingerprint,
            "executor": {
                "kind": decision.executor,
                "reason": decision.reason,
                "local_preflight": {
                    "ok": preflight.ok,
                    "full_families": list(preflight.full_families),
                    "failures": [asdict(item) for item in preflight.failures],
                    "required_gpu_total_mib": preflight.required_gpu_total_mib,
                    "required_gpu_free_mib": preflight.required_gpu_free_mib,
                    "required_ram_available_bytes": preflight.required_ram_available_bytes,
                    "required_disk_free_bytes": preflight.required_disk_free_bytes,
                },
            },
            "families": families,
            "download_verified": True,
            "quality_gate": {
                "passed": False,
                "pending_local_evaluation": True,
                # Modal verifies candidate selection, provenance and bytes. It
                # never authorizes publication; only the subsequent local
                # held-out evaluation can make `passed` true.
                "remote_candidate_verified": remote_candidate_verified,
            },
        }
        store.set_result("train", result)
    if not store.is_trained(fingerprint):
        raise RuntimeError("Training completed without a fully verified candidate artifact set")
    return store.stage("train").get("result", {})
