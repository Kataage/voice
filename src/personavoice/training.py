from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from personavoice.config import PersonaConfig
from personavoice.irodori import base_checkpoint, prepare_manifest, train_irodori
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

TRAIN_SCHEMA_VERSION = 4


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _file_contract(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        # Dependency-graph changes can alter optimization/inference behavior even
        # when model weights and user settings are identical. Treat the audited
        # lockfiles as part of training provenance.
        "irodori_lock_sha256": _file_contract(repo_root / "locks" / "Irodori-TTS.uv.lock"),
        "lfm_lock_sha256": _file_contract(repo_root / "workers" / "lfm" / "uv.lock"),
        "seed_vc_lock_sha256": _file_contract(repo_root / "workers" / "seed_vc" / "uv.lock"),
    }
    digest.update(
        json.dumps(model_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for path in (
        paths.dataset / "irodori_source.jsonl",
        paths.dataset / "lfm_train.jsonl",
        paths.dataset / "seed_vc" / "manifest.jsonl",
    ):
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    digest.update(json.dumps(cfg.training.model_dump(mode="json"), sort_keys=True).encode())
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


def _has_training_artifacts(paths: PersonaPaths) -> bool:
    """Return True when derived training state exists without relying on directories alone."""

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


def train_lfm(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> str | None:
    dataset = paths.dataset / "lfm_train.jsonl"
    if _line_count(dataset) < 2:
        return None
    base = repo_root / "models" / "lfm" / "base"
    if not (base / "config.json").exists():
        raise FileNotFoundError("LFM base model is missing. Run `persona setup --download-models`.")
    output = paths.models / "lfm" / "adapter"
    output.parent.mkdir(parents=True, exist_ok=True)
    if (output / "adapter_config.json").exists():
        return str(output)
    project = repo_root / "workers" / "lfm"
    env = local_model_env(repo_root)
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
        env=env,
    )
    if not (output / "adapter_config.json").exists():
        raise RuntimeError("LFM fine-tuning completed without adapter_config.json")
    return str(output)


def train_seed_vc(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> str | None:
    if not cfg.training.seed_vc_finetune:
        return None
    audio_dir = paths.dataset / "seed_vc" / "audio"
    audio_files = list(audio_dir.glob("*.flac")) if audio_dir.exists() else []
    if len(audio_files) < 2:
        return None

    health = worker(repo_root, "seed_vc").call(repo_root, "health", {"deep": False})
    if not bool(health.get("cuda")):
        raise RuntimeError(
            "Seed-VC fine-tuning requires a CUDA-enabled Seed-VC worker. "
            "Re-run `persona setup` on a supported NVIDIA system, or leave "
            "training.seed_vc_finetune=false and use zero-shot reenactment."
        )

    vendor = repo_root / "vendor" / "seed-vc"
    project = repo_root / "workers" / "seed_vc"
    run_name = f"personavoice_{cfg.name}"
    run(
        [
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
            run_name,
            "--batch-size",
            "2",
            "--max-steps",
            str(cfg.training.seed_vc_max_steps),
            "--max-epochs",
            "1000",
            "--save-every",
            str(max(100, cfg.training.seed_vc_max_steps // 2)),
            "--num-workers",
            "0",
            "--train-cfm",
        ],
        cwd=vendor,
        env=local_model_env(repo_root),
    )
    source_dir = vendor / "runs" / run_name
    checkpoints = sorted(source_dir.glob("CFM_*_step_*.pth"))
    if not checkpoints:
        raise RuntimeError("Seed-VC fine-tuning completed without a CFM checkpoint")
    target = paths.models / "seed_vc" / "cfm.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoints[-1], target)
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
            "Prepared dataset is missing or stale for the current raw/identity/config inputs. "
            "Run `persona prepare` before training."
        )

    source = paths.dataset / "irodori_source.jsonl"
    if _line_count(source) < 2:
        raise RuntimeError(
            "Prepared Irodori dataset is missing or too small. Run `persona prepare` first."
        )

    fingerprint = _fingerprint(paths, cfg)
    previous = store.stage("train")
    if not force and store.is_complete("train", fingerprint):
        return previous.get("result", {})

    previous_fingerprint = previous.get("fingerprint")
    inputs_changed = bool(previous_fingerprint and previous_fingerprint != fingerprint)
    untracked_artifacts = previous_fingerprint is None and _has_training_artifacts(paths)
    if force or inputs_changed or untracked_artifacts:
        _invalidate_training_artifacts(paths)
        if cfg.training.seed_vc_finetune:
            shutil.rmtree(
                repo_root / "vendor" / "seed-vc" / "runs" / f"personavoice_{cfg.name}",
                ignore_errors=True,
            )

    with store.running("train", fingerprint):
        base_checkpoint(repo_root)
        manifest = paths.dataset / "irodori_manifest.jsonl"
        latents = paths.cache / "irodori_latents"
        if not manifest.exists():
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
            "irodori": irodori,
            "lfm_adapter": lfm,
            "seed_vc_cfm": seed,
        }
        store.set_result("train", result)
        return result
