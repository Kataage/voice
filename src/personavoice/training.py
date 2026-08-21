from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from personavoice.config import PersonaConfig
from personavoice.irodori import base_checkpoint, prepare_manifest, train_irodori
from personavoice.process import run
from personavoice.project import PersonaPaths
from personavoice.state import StateStore
from personavoice.workers import local_model_env


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _fingerprint(paths: PersonaPaths, cfg: PersonaConfig) -> str:
    source = paths.dataset / "irodori_source.jsonl"
    brain = paths.dataset / "lfm_train.jsonl"
    digest = hashlib.sha256()
    for path in (source, brain):
        if path.exists():
            digest.update(path.read_bytes())
    digest.update(json.dumps(cfg.training.model_dump(), sort_keys=True).encode())
    return digest.hexdigest()


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
            "uv", "run", "--project", project, "--no-sync", "python", project / "train.py",
            "--base", base, "--dataset", dataset, "--output", output,
            "--epochs", str(cfg.training.lfm_epochs),
            "--learning-rate", str(cfg.training.lfm_learning_rate),
            "--lora-r", str(cfg.training.lfm_lora_r),
            "--lora-alpha", str(cfg.training.lfm_lora_alpha),
        ],
        cwd=repo_root,
        env=env,
    )
    return str(output)


def train_seed_vc(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> str | None:
    if not cfg.training.seed_vc_finetune:
        return None
    audio_dir = paths.dataset / "seed_vc" / "audio"
    audio_files = list(audio_dir.glob("*.flac")) if audio_dir.exists() else []
    if len(audio_files) < 2:
        return None
    vendor = repo_root / "vendor" / "seed-vc"
    project = repo_root / "workers" / "seed_vc"
    run_name = f"personavoice_{cfg.name}"
    run(
        [
            "uv", "run", "--project", project, "--no-sync", "accelerate", "launch",
            "--num_processes", "1", "--mixed_precision", "fp16",
            vendor / "train_v2.py",
            "--dataset-dir", audio_dir,
            "--run-name", run_name,
            "--batch-size", "2",
            "--max-steps", str(cfg.training.seed_vc_max_steps),
            "--max-epochs", "1000",
            "--save-every", str(max(100, cfg.training.seed_vc_max_steps // 2)),
            "--num-workers", "0",
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
    repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig, *, force: bool = False
) -> dict:
    if not cfg.consent.authorized:
        raise PermissionError("Training is blocked because consent.authorized is not true.")
    source = paths.dataset / "irodori_source.jsonl"
    if _line_count(source) < 2:
        raise RuntimeError("Prepared Irodori dataset is missing or too small. Run `persona prepare` first.")
    store = StateStore(paths.state)
    fingerprint = _fingerprint(paths, cfg)
    if not force and store.is_complete("train", fingerprint):
        return store.stage("train").get("result", {})
    with store.running("train", fingerprint):
        base_checkpoint(repo_root)
        manifest = paths.dataset / "irodori_manifest.jsonl"
        latents = paths.cache / "irodori_latents"
        if force or not manifest.exists():
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
        result = {"irodori": irodori, "lfm_adapter": lfm, "seed_vc_cfm": seed}
        store.set_result("train", result)
        return result
