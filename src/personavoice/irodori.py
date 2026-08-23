from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from personavoice.atomic import atomic_write_text
from personavoice.environment_contract import require_current_environment
from personavoice.hardware import safe_batch_profile
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
_CHECKPOINT_STEP_RE = re.compile(r"^checkpoint_(\d+)(?:\.speaker\.safetensors)?$")
_LORA_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
_LORA_TRAINER_STATE = "trainer_state.pt"


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


def lora_resume_checkpoint_complete(path: Path) -> bool:
    return lora_adapter_complete(path) and _nonempty_file(path / _LORA_TRAINER_STATE)


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
    if backend == "cu126":
        train_cfg["precision"] = "fp32"
        train_cfg["allow_tf32"] = False
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


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _latest_numeric_checkpoint(paths: list[Path]) -> Path | None:
    ranked = [(step, path) for path in paths if (step := _checkpoint_step(path)) is not None]
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _latest_resume(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        path
        for path in output_dir.glob("checkpoint_*")
        if path.is_dir() and lora_resume_checkpoint_complete(path)
    ]
    return _latest_numeric_checkpoint(candidates)


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
    device = backend_device(backend)
    env = local_model_env(repo_root)
    outputs: dict[str, str] = {"base": str(base)}
    if do_speaker:
        out = models_dir / "irodori" / "speaker"
        final = out / "checkpoint_final.speaker.safetensors"
        if not speaker_embedding_complete(final):
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
                "--device",
                device,
            ]
            checkpoint = (
                _latest_numeric_checkpoint(
                    [
                        path
                        for path in out.glob("checkpoint_*.speaker.safetensors")
                        if speaker_embedding_complete(path)
                    ]
                )
                if out.exists()
                else None
            )
            if checkpoint is not None:
                args += ["--speaker-inversion-init-embedding", checkpoint]
            run(args, cwd=vendor, env=env)
        if not speaker_embedding_complete(final):
            raise RuntimeError("Irodori Speaker Inversion did not produce a valid checkpoint_final")
        outputs["speaker_embedding"] = str(final)
    if do_lora:
        out = models_dir / "irodori" / "lora"
        final = out / "checkpoint_final"
        if not lora_adapter_complete(final):
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
                "--device",
                device,
            ]
            resume = _latest_resume(out)
            if resume:
                args += ["--resume", resume]
            run(args, cwd=vendor, env=env)
        if not lora_adapter_complete(final):
            raise RuntimeError("Irodori LoRA training did not produce a complete PEFT adapter")
        outputs["lora_adapter"] = str(final)
    return outputs


def reference_files(references_dir: Path) -> list[Path]:
    bank = references_dir / "bank.json"
    if bank.exists():
        data = json.loads(bank.read_text(encoding="utf-8"))
        files = [Path(path) for path in data.get("files", [])]
        return [path for path in files if path.exists()]
    return sorted(references_dir.glob("*.flac"))
