from __future__ import annotations

import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from personavoice.hardware import detect_irodori_backend, safe_batch_profile
from personavoice.process import run
from personavoice.workers import local_model_env

MODEL_ID = "Aratako/Irodori-TTS-v4.1-Small"
SUPPORTED_BACKENDS = {"cpu", "cu128", "rocm", "xpu"}


def vendor_dir(repo_root: Path) -> Path:
    path = repo_root / "vendor" / "Irodori-TTS"
    if not (path / "infer.py").exists():
        raise FileNotFoundError("Irodori-TTS is not installed. Run `persona setup` first.")
    return path


def configured_backend(repo_root: Path) -> str:
    setup = repo_root / ".runtime" / "setup.json"
    if setup.exists():
        try:
            value = json.loads(setup.read_text(encoding="utf-8")).get("irodori_backend")
        except (json.JSONDecodeError, OSError):
            value = None
        if value is not None:
            backend = str(value)
            if backend not in SUPPORTED_BACKENDS:
                raise RuntimeError(f"Unsupported recorded Irodori backend: {backend!r}")
            return backend
    return detect_irodori_backend()


def backend_device(backend: str) -> str:
    if backend in {"cu128", "rocm"}:
        return "cuda"
    if backend == "xpu":
        return "xpu"
    if backend == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported Irodori backend: {backend!r}")


def base_checkpoint(repo_root: Path, *, online: bool = False) -> Path:
    env = local_model_env(repo_root, offline=not online)
    local_dir = repo_root / "models" / "irodori" / "v4.1-small"
    expected = local_dir / "model.safetensors"
    if expected.exists():
        return expected
    if not online:
        raise FileNotFoundError(
            f"Irodori base checkpoint is not materialized at {expected}. "
            "Run `persona setup --download-models`."
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        local_dir=local_dir,
        cache_dir=Path(env["HUGGINGFACE_HUB_CACHE"]),
    )
    if not expected.exists():
        raise FileNotFoundError(f"Irodori download completed but {expected} was not created")
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
            "--device",
            device,
        ],
        cwd=vendor,
        env=env,
    )
    if not output_manifest.exists() or output_manifest.stat().st_size == 0:
        raise RuntimeError("Irodori manifest preparation produced no training examples")


def _patched_config(
    source: Path,
    destination: Path,
    *,
    max_steps: int,
    backend: str,
) -> Path:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Irodori training config is not a YAML mapping: {source}")
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
    if "gradient_checkpointing" in train_cfg or profile["gradient_checkpointing"]:
        train_cfg["gradient_checkpointing"] = bool(profile["gradient_checkpointing"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def _latest_resume(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        path
        for path in output_dir.glob("checkpoint_*")
        if "best_val_loss" not in path.name and "final" not in path.name
    ]
    return sorted(candidates, key=lambda path: path.name)[-1] if candidates else None


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
) -> dict:
    vendor = vendor_dir(repo_root)
    base = base_checkpoint(repo_root)
    backend = configured_backend(repo_root)
    env = local_model_env(repo_root)
    outputs: dict[str, str] = {"base": str(base)}
    if do_speaker:
        out = models_dir / "irodori" / "speaker"
        final = out / "checkpoint_final.speaker.safetensors"
        if not final.exists():
            cfg = _patched_config(
                vendor / "configs" / "train_v4_small_speaker_inversion.yaml",
                cache_dir / "irodori_speaker.yaml",
                max_steps=speaker_steps,
                backend=backend,
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
            ]
            checkpoints = (
                sorted(out.glob("checkpoint_*.speaker.safetensors"))
                if out.exists()
                else []
            )
            if checkpoints:
                args += ["--speaker-inversion-init-embedding", checkpoints[-1]]
            run(args, cwd=vendor, env=env)
        if not final.exists():
            raise RuntimeError("Irodori Speaker Inversion did not produce checkpoint_final")
        outputs["speaker_embedding"] = str(final)
    if do_lora:
        out = models_dir / "irodori" / "lora"
        final = out / "checkpoint_final"
        if not final.exists():
            cfg = _patched_config(
                vendor / "configs" / "train_v4_small_lora.yaml",
                cache_dir / "irodori_lora.yaml",
                max_steps=lora_steps,
                backend=backend,
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
            ]
            resume = _latest_resume(out)
            if resume:
                args += ["--resume", resume]
            run(args, cwd=vendor, env=env)
        if not final.exists():
            raise RuntimeError("Irodori LoRA training did not produce checkpoint_final")
        outputs["lora_adapter"] = str(final)
    return outputs


def reference_files(references_dir: Path) -> list[Path]:
    bank = references_dir / "bank.json"
    if bank.exists():
        data = json.loads(bank.read_text(encoding="utf-8"))
        files = [Path(path) for path in data.get("files", [])]
        return [path for path in files if path.exists()]
    return sorted(references_dir.glob("*.flac"))
