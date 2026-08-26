from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from personavoice.config import PersonaConfig
from personavoice.irodori import base_checkpoint, prepare_manifest, train_irodori
from personavoice.lfm_contract import LFM_CONTRACT_FINGERPRINT, LFM_CONTRACT_SCHEMA_VERSION
from personavoice.model_assets import (
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_SHA256,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_REVISION,
)
from personavoice.pipeline import _prepare_fingerprint
from personavoice.process import run
from personavoice.project import PersonaPaths
from personavoice.setup_env import IRODORI_REVISION, SEED_VC_REVISION
from personavoice.state import StateStore
from personavoice.workers import local_model_env, worker

TRAIN_SCHEMA_VERSION = 8
_SEED_VC_STEP_RE = re.compile(r"_step_(\d+)\.pth$")
_LFM_ADAPTER_REVISION_MARKER = ".personavoice-base-revision"


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
        return hashlib.sha256(path.read_bytes()).hexdigest()
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
    digest.update(json.dumps(cfg.training.model_dump(mode="json"), sort_keys=True).encode())
    return digest.hexdigest()


def _non_lfm_fingerprint(paths: PersonaPaths, cfg: PersonaConfig) -> str:
    """Fingerprint Irodori/Seed-VC inputs independently from LFM export inputs."""

    repo_root = paths.root.parents[1]
    contract = {
        "scope": "non-lfm",
        "train_schema": TRAIN_SCHEMA_VERSION,
        "irodori_source_revision": IRODORI_REVISION,
        "irodori_model_sha256": IRODORI_MODEL_SHA256,
        "irodori_dacvae_sha256": IRODORI_DACVAE_SHA256,
        "irodori_text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
        "seed_vc_source_revision": SEED_VC_REVISION,
        "irodori_lock_sha256": _file_contract(repo_root / "locks" / "Irodori-TTS.uv.lock"),
        "seed_vc_lock_sha256": _file_contract(repo_root / "workers" / "seed_vc" / "uv.lock"),
        "training_code_sha256": _file_contract(repo_root / "src" / "personavoice" / "training.py"),
        "irodori_code_sha256": _file_contract(repo_root / "src" / "personavoice" / "irodori.py"),
        "seed_vc_worker_sha256": _file_contract(repo_root / "workers" / "seed_vc" / "worker.py"),
        "training": {
            key: value
            for key, value in cfg.training.model_dump(mode="json").items()
            if not key.startswith("lfm_")
        },
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    for path in (
        paths.dataset / "irodori_source.jsonl",
        paths.dataset / "seed_vc" / "manifest.jsonl",
    ):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _lfm_fingerprint(paths: PersonaPaths, cfg: PersonaConfig) -> str:
    """Fingerprint only the LFM contract, export, model and LoRA settings."""

    repo_root = paths.root.parents[1]
    contract = {
        "scope": "lfm",
        "train_schema": TRAIN_SCHEMA_VERSION,
        "lfm_contract_schema_version": LFM_CONTRACT_SCHEMA_VERSION,
        "lfm_contract_fingerprint": LFM_CONTRACT_FINGERPRINT,
        "lfm_revision": LFM_MODEL_REVISION,
        "lfm_lock_sha256": _file_contract(repo_root / "workers" / "lfm" / "uv.lock"),
        "lfm_worker_sha256": _file_contract(repo_root / "workers" / "lfm" / "worker.py"),
        "lfm_train_code_sha256": _file_contract(repo_root / "workers" / "lfm" / "train.py"),
        "lfm_checkpoint_contract_code_sha256": _file_contract(
            repo_root / "workers" / "lfm" / "checkpoint_contract.py"
        ),
        "lfm_model_contract_code_sha256": _file_contract(
            repo_root / "workers" / "lfm" / "model_contract.py"
        ),
        "training": {
            key: value
            for key, value in cfg.training.model_dump(mode="json").items()
            if key.startswith("lfm_")
        },
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    path = paths.dataset / "lfm_train.jsonl"
    if path.is_file():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def _invalidate_lfm_artifacts(paths: PersonaPaths) -> None:
    """Invalidate only derived LFM artifacts when the LFM contract changes."""

    shutil.rmtree(paths.models / "lfm", ignore_errors=True)


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


def train_lfm(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> str:
    dataset = paths.dataset / "lfm_train.jsonl"
    example_count = _line_count(dataset)
    if example_count < 2:
        raise RuntimeError(
            "training.lfm_lora is enabled, but fewer than two valid conversational "
            f"examples were exported ({example_count}). Add source conversations containing "
            "the authorized speaker responding to another speaker, rerun `persona prepare`, "
            "or deliberately set training.lfm_lora: false."
        )
    base = repo_root / "models" / "lfm" / "base"
    if not _nonempty_file(base / "config.json"):
        raise FileNotFoundError("LFM base model is missing. Run `persona setup --download-models`.")
    output = paths.models / "lfm" / "adapter"
    output.parent.mkdir(parents=True, exist_ok=True)
    if _lfm_adapter_complete(output):
        return str(output)
    project = repo_root / "workers" / "lfm"
    run(
        [
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
            "--epochs",
            str(cfg.training.lfm_epochs),
            "--learning-rate",
            str(cfg.training.lfm_learning_rate),
            "--lora-r",
            str(cfg.training.lfm_lora_r),
            "--lora-alpha",
            str(cfg.training.lfm_lora_alpha),
        ],
        cwd=repo_root,
        env=local_model_env(repo_root),
    )
    if not _lfm_adapter_complete(output):
        raise RuntimeError(
            "LFM fine-tuning completed without a complete adapter for the audited base revision"
        )
    return str(output)


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
    audio_files = [path for path in audio_dir.glob("*.flac") if _nonempty_file(path)] if audio_dir.exists() else []
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
        raise RuntimeError("Seed-VC reached the requested step count without a usable CFM checkpoint")
    shutil.copy2(checkpoint, target)
    if not _nonempty_file(target):
        raise RuntimeError("Seed-VC final CFM checkpoint copy is missing or empty")
    return str(target)


def train_persona(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    force: bool = False,
) -> dict:
    if not cfg.consent.authorized:
        raise PermissionError("Training is blocked because consent.authorized is not true.")

    store = StateStore(paths.state)
    current_prepare_fingerprint = _prepare_fingerprint(paths, cfg)
    if not store.is_complete("prepare", current_prepare_fingerprint):
        raise RuntimeError(
            "Prepared dataset is missing, stale, or incomplete for the current inputs. "
            "Run `persona prepare` before training."
        )

    source = paths.dataset / "irodori_source.jsonl"
    if _line_count(source) < 2:
        raise RuntimeError(
            "Prepared Irodori dataset is missing or too small. Run `persona prepare` first."
        )

    fingerprint = _fingerprint(paths, cfg)
    non_lfm_fingerprint = _non_lfm_fingerprint(paths, cfg)
    lfm_fingerprint = _lfm_fingerprint(paths, cfg)
    previous = store.stage("train")
    if not force and store.is_complete("train", fingerprint):
        return previous.get("result", {})

    previous_fingerprint = previous.get("fingerprint")
    inputs_changed = bool(previous_fingerprint and previous_fingerprint != fingerprint)
    untracked_artifacts = previous_fingerprint is None and _has_training_artifacts(paths)
    previous_result = previous.get("result")
    lfm_only_change = bool(
        not force
        and inputs_changed
        and isinstance(previous_result, dict)
        and previous_result.get("non_lfm_fingerprint") == non_lfm_fingerprint
        and previous_result.get("lfm_fingerprint") != lfm_fingerprint
    )
    if force or inputs_changed or untracked_artifacts:
        if lfm_only_change:
            _invalidate_lfm_artifacts(paths)
        else:
            _invalidate_training_artifacts(paths)
        if cfg.training.seed_vc_finetune and not lfm_only_change:
            _clear_seed_vc_runs(repo_root, cfg.name)

    with store.running("train", fingerprint):
        base_checkpoint(repo_root)
        manifest = paths.dataset / "irodori_manifest.jsonl"
        latents = paths.cache / "irodori_latents"
        if not _nonempty_file(manifest):
            prepare_manifest(repo_root, source, manifest, latents)
        irodori = train_irodori(
            repo_root,
            manifest,
            paths.models,
            paths.cache,
            speaker_steps=cfg.training.speaker_inversion_max_steps,
            lora_steps=cfg.training.irodori_max_steps,
            do_speaker=cfg.training.irodori_speaker_inversion,
            do_lora=cfg.training.irodori_lora,
        )
        lfm = train_lfm(repo_root, paths, cfg) if cfg.training.lfm_lora else None
        seed = train_seed_vc(repo_root, paths, cfg)
        result = {
            "train_schema": TRAIN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "non_lfm_fingerprint": non_lfm_fingerprint,
            "lfm_fingerprint": lfm_fingerprint,
            "irodori": irodori,
            "lfm_adapter": lfm,
            "seed_vc_cfm": seed,
        }
        store.set_result("train", result)
        return result
