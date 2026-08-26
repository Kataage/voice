from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from personavoice.atomic import atomic_write_json
from personavoice.config import PersonaConfig
from personavoice.irodori import base_checkpoint, prepare_manifest, train_irodori
from personavoice.lfm_contract import LFM_CONTRACT_FINGERPRINT, LFM_CONTRACT_SCHEMA_VERSION
from personavoice.lineage import active_generation_id, active_lineage_id, load_lineage
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

TRAIN_SCHEMA_VERSION = 9
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


def train_seed_vc(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    run_name: str | None = None,
) -> str | None:
    if not cfg.training.seed_vc_finetune:
        return None
    target = paths.models / "seed_vc" / "cfm.pth"
    if _nonempty_file(target):
        return str(target)
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
            "rerun persona prepare, or use zero-shot reenactment."
        )

    health = worker(repo_root, "seed_vc").call(repo_root, "health", {"deep": False})
    if not bool(health.get("ok", True)) or not bool(health.get("cuda")):
        raise RuntimeError(
            "Seed-VC fine-tuning requires a healthy CUDA-enabled Seed-VC worker. "
            "Re-run persona setup on a supported NVIDIA system, inspect persona doctor --deep, "
            "or leave training.seed_vc_finetune=false and use zero-shot reenactment."
        )

    vendor = repo_root / "vendor" / "seed-vc"
    project = repo_root / "workers" / "seed_vc"
    namespace = run_name or cfg.name
    completed_steps, checkpoint = _seed_vc_training_progress(vendor, namespace)
    desired_steps = cfg.training.seed_vc_max_steps
    if completed_steps > desired_steps:
        raise RuntimeError(
            "Existing staged Seed-VC progress exceeds the configured max steps: "
            f"completed={completed_steps}, configured={desired_steps}. "
            "Use a new candidate generation or lower the configured step count."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    while completed_steps < desired_steps:
        completed_steps, checkpoint = _run_seed_vc_stage(
            repo_root,
            project=project,
            vendor=vendor,
            audio_dir=audio_dir,
            persona_name=namespace,
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


def _lineage_metadata(paths: PersonaPaths) -> dict[str, str]:
    """Return the Prepare identity that every v0.3 family must inherit."""

    if paths.lineage_id is None:
        raise RuntimeError(
            "v0.3 candidate training requires a lineaged Prepare result; run persona prepare."
        )
    record = load_lineage(paths, paths.lineage_id)
    if not isinstance(record, dict):
        raise RuntimeError(f"Prepare lineage record is missing: {paths.lineage_id}")
    lineage_fingerprint = record.get("lineage_fingerprint")
    master_fingerprint = record.get("master_fingerprint")
    if not all(
        isinstance(value, str) and value
        for value in (paths.lineage_id, lineage_fingerprint, master_fingerprint)
    ):
        raise RuntimeError("Prepare lineage record has incomplete identity metadata")
    return {
        "lineage_id": paths.lineage_id,
        "lineage_fingerprint": lineage_fingerprint,
        "master_fingerprint": master_fingerprint,
    }


def _relative_to_persona(paths: PersonaPaths, value: Path) -> str:
    try:
        return value.resolve().relative_to(paths.root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Candidate artifact is outside the persona root: {value}") from exc


def _candidate_artifact(paths: PersonaPaths, value: Path) -> dict[str, object]:
    try:
        relative = value.resolve().relative_to(paths.generation_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Generation artifact is outside its candidate generation: {value}"
        ) from exc
    if not _nonempty_file(value):
        raise RuntimeError(f"Candidate artifact is missing or empty: {value}")
    return {
        "path": relative,
        "size": value.stat().st_size,
        "sha256": _file_contract(value),
    }


def _write_family_marker(
    paths: PersonaPaths,
    family: str,
    *,
    mode: str,
    lineage: dict[str, str],
    source_files: list[str],
) -> Path:
    marker = paths.models / family / "lineage.json"
    atomic_write_json(
        marker,
        {
            "schema_version": 1,
            "kind": "personavoice-v03-family",
            "family": family,
            "mode": mode,
            **lineage,
            "source_files": source_files,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return marker


def _family_record(
    paths: PersonaPaths,
    family: str,
    *,
    enabled: bool,
    mode: str,
    lineage: dict[str, str],
    files: list[Path],
) -> dict[str, object]:
    if not enabled:
        return {"status": "not_requested", "mode": mode, "artifacts": []}
    marker = _write_family_marker(
        paths,
        family,
        mode=mode,
        lineage=lineage,
        source_files=[_relative_to_persona(paths, path) for path in files],
    )
    artifacts = [_candidate_artifact(paths, marker)]
    artifacts.extend(_candidate_artifact(paths, path) for path in files)
    return {"status": "complete", "mode": mode, "artifacts": artifacts}


def _adapter_files(directory: Path) -> list[Path]:
    files = [directory / "adapter_config.json"]
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = directory / name
        if _nonempty_file(candidate):
            files.append(candidate)
            break
    return files


def _copy_candidate_family(source: PersonaPaths, destination: PersonaPaths, family: str) -> None:
    source_dir = source.models / family
    destination_dir = destination.models / family
    if not source_dir.is_dir():
        raise RuntimeError(f"Cannot reuse missing {family} candidate artifacts: {source_dir}")
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def _candidate_from_result(paths: PersonaPaths, result: object) -> PersonaPaths | None:
    if not isinstance(result, dict):
        return None
    lineage_id = result.get("lineage_id") or result.get("prepare_lineage_id")
    generation_id = result.get("generation_id")
    if not isinstance(lineage_id, str) or not isinstance(generation_id, str):
        return None
    try:
        candidate = paths.for_generation(lineage_id, generation_id)
    except ValueError:
        return None
    return candidate if candidate.generation_manifest.is_file() else None


def _generation_fingerprint(
    lineage: dict[str, str],
    *,
    training_fingerprint: str,
    non_lfm_fingerprint: str,
    lfm_fingerprint: str,
    cfg: PersonaConfig,
) -> str:
    payload = {
        "architecture": "v0.3-pre-full-fine-tuning",
        "lineage": lineage,
        "training_fingerprint": training_fingerprint,
        "non_lfm_fingerprint": non_lfm_fingerprint,
        "lfm_fingerprint": lfm_fingerprint,
        "training": cfg.training.model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json_file(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _quality_report(
    paths: PersonaPaths,
    result: dict[str, object],
    key: str,
) -> dict[str, object] | None:
    value = result.get(key)
    if not isinstance(value, str):
        return None
    path = (paths.root / value).resolve()
    try:
        path.relative_to(paths.root.resolve())
    except ValueError:
        return None
    return _read_json_file(path)


def _safe_count(report: dict[str, object] | None, key: str) -> int:
    if report is None:
        return 0
    value = report.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _validation_check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    detail: object,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def validate_generation(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    generation_id: str | None = None,
) -> dict[str, object]:
    """Validate a v0.3 candidate without changing the active runtime pointer."""

    del repo_root, cfg
    store = StateStore(paths.state)
    result = store.stage("train").get("result")
    if not isinstance(result, dict):
        raise RuntimeError("No candidate training result exists; run persona train first.")
    lineage_id = result.get("lineage_id") or result.get("prepare_lineage_id")
    selected_generation = generation_id or result.get("generation_id")
    if not isinstance(lineage_id, str) or not isinstance(selected_generation, str):
        raise RuntimeError("Training result has no candidate lineage/generation identity")
    candidate = paths.for_generation(lineage_id, selected_generation)
    manifest = _read_json_file(candidate.generation_manifest)
    lineage = load_lineage(paths, lineage_id)
    checks: list[dict[str, object]] = []
    _validation_check(
        checks,
        "prepare_lineage_record",
        isinstance(lineage, dict)
        and lineage.get("lineage_fingerprint") == result.get("prepare_lineage_fingerprint")
        and lineage.get("master_fingerprint") == result.get("master_fingerprint"),
        {"lineage_id": lineage_id},
    )
    _validation_check(
        checks,
        "candidate_generation_manifest",
        isinstance(manifest, dict)
        and manifest.get("generation_fingerprint") == result.get("generation_fingerprint")
        and manifest.get("lineage_id") == lineage_id
        and manifest.get("master_fingerprint") == result.get("master_fingerprint"),
        {"generation_id": selected_generation},
    )
    families = manifest.get("families") if isinstance(manifest, dict) else None
    family_results: dict[str, object] = {}
    for family in ("irodori", "lfm", "seed_vc"):
        value = families.get(family) if isinstance(families, dict) else None
        family_ok = isinstance(value, dict) and value.get("status") in {"complete", "not_requested"}
        artifact_results: list[dict[str, object]] = []
        if family_ok and value.get("status") == "complete":
            artifacts = value.get("artifacts")
            family_ok = isinstance(artifacts, list) and bool(artifacts)
            for item in artifacts if isinstance(artifacts, list) else []:
                if not isinstance(item, dict):
                    family_ok = False
                    continue
                relative = item.get("path")
                path = (
                    candidate.generation_root / relative
                    if isinstance(relative, str)
                    else candidate.generation_root / "__invalid__"
                )
                try:
                    path.resolve().relative_to(candidate.generation_root.resolve())
                    safe = True
                except ValueError:
                    safe = False
                item_ok = (
                    safe
                    and _nonempty_file(path)
                    and item.get("size") == path.stat().st_size
                    and item.get("sha256") == _file_contract(path)
                )
                artifact_results.append({"path": relative, "passed": item_ok})
                family_ok = family_ok and item_ok
        family_results[family] = {
            "passed": family_ok,
            "status": value.get("status") if isinstance(value, dict) else None,
            "artifacts": artifact_results,
        }
        _validation_check(checks, f"{family}_artifacts", family_ok, family_results[family])

    prepared = paths.for_lineage(lineage_id)
    irodori_report = _quality_report(prepared, result, "irodori_quality_report")
    lfm_report = _quality_report(prepared, result, "lfm_quality_report")
    irodori_qgate = (
        irodori_report.get("quality_gate") if isinstance(irodori_report, dict) else None
    )
    lfm_qgate = lfm_report.get("quality_gate") if isinstance(lfm_report, dict) else None
    _validation_check(
        checks,
        "irodori_quality_gate",
        _safe_count(irodori_report, "accepted_count") > 0
        and isinstance(irodori_qgate, dict)
        and irodori_qgate.get("passed") is True,
        {"accepted_count": _safe_count(irodori_report, "accepted_count")},
    )
    lfm_required = bool(result.get("lfm_requested"))
    _validation_check(
        checks,
        "lfm_quality_gate",
        not lfm_required
        or (
            _safe_count(lfm_report, "accepted_count") >= 2
            and isinstance(lfm_qgate, dict)
            and lfm_qgate.get("passed") is True
        ),
        {"accepted_count": _safe_count(lfm_report, "accepted_count")},
    )
    seed_manifest = prepared.dataset / "seed_vc" / "manifest.jsonl"
    reference_bank = _read_json_file(prepared.references / "bank.json")
    reference_files = reference_bank.get("files") if isinstance(reference_bank, dict) else None
    seed_ok = (
        _line_count(seed_manifest) >= 2
        if bool(result.get("seed_vc_finetune"))
        else isinstance(reference_files, list) and bool(reference_files)
    )
    _validation_check(
        checks,
        "seed_vc_dependency_manifest",
        seed_ok,
        {
            "fine_tune": bool(result.get("seed_vc_finetune")),
            "seed_examples": _line_count(seed_manifest),
            "references": len(reference_files) if isinstance(reference_files, list) else 0,
        },
    )

    passed = all(bool(check.get("passed")) for check in checks)
    validation = {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "validated_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "acoustic_quality": {
            "status": "not_run",
            "reason": "No authorized persona audio/GPU target was available in hosted CI.",
        },
    }
    if not isinstance(manifest, dict):
        raise RuntimeError("Candidate generation manifest is missing or invalid")
    manifest["validation"] = validation
    atomic_write_json(candidate.generation_manifest, manifest)
    result["validation"] = validation
    store.set_result("train", result)
    store.set_status("train", "trained")
    return validation


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
    prepare_fingerprint = _prepare_fingerprint(paths, cfg)
    if not store.is_complete("prepare", prepare_fingerprint):
        raise RuntimeError(
            "Prepared dataset is missing, stale, or incomplete for the current inputs. "
            "Run persona prepare before training."
        )
    prepare_result = store.stage("prepare").get("result")
    if not isinstance(prepare_result, dict):
        raise RuntimeError("Prepare result is missing; run persona prepare first.")
    lineage_id = prepare_result.get("lineage_id")
    if not isinstance(lineage_id, str):
        raise RuntimeError(
            "v0.3 training refuses the historical root Prepare layout. "
            "Run persona prepare to create an immutable candidate lineage."
        )
    prepared = paths.for_lineage(lineage_id)
    lineage = _lineage_metadata(prepared)
    source = prepared.dataset / "irodori_source.jsonl"
    if _line_count(source) < 2:
        raise RuntimeError(
            "Prepared Irodori dataset is missing or too small. Run persona prepare first."
        )

    fingerprint = _fingerprint(prepared, cfg)
    non_lfm_fingerprint = _non_lfm_fingerprint(prepared, cfg)
    lfm_fingerprint = _lfm_fingerprint(prepared, cfg)
    previous = store.stage("train")
    if not force and store.is_complete("train", fingerprint):
        return previous.get("result", {})
    if not force and store.is_trained(fingerprint):
        return previous.get("result", {})

    previous_result = previous.get("result")
    previous_candidate = _candidate_from_result(paths, previous_result)
    inputs_changed = bool(
        previous.get("fingerprint") and previous.get("fingerprint") != fingerprint
    )
    lfm_only_change = bool(
        not force
        and inputs_changed
        and isinstance(previous_result, dict)
        and previous_result.get("non_lfm_fingerprint") == non_lfm_fingerprint
        and previous_result.get("lfm_fingerprint") != lfm_fingerprint
    )

    generation_fingerprint = _generation_fingerprint(
        lineage,
        training_fingerprint=fingerprint,
        non_lfm_fingerprint=non_lfm_fingerprint,
        lfm_fingerprint=lfm_fingerprint,
        cfg=cfg,
    )
    generation_id = f"gen-{generation_fingerprint[:32]}"
    if (
        force
        and active_lineage_id(paths) == lineage_id
        and active_generation_id(paths) == generation_id
    ):
        generation_id = f"gen-{os.urandom(16).hex()}"
    candidate = paths.for_generation(lineage_id, generation_id)
    candidate.ensure_lineage()

    with store.running("train", fingerprint, success_status="trained"):
        if lfm_only_change and previous_candidate is not None:
            _copy_candidate_family(previous_candidate, candidate, "irodori")
            _copy_candidate_family(previous_candidate, candidate, "seed_vc")
        base_checkpoint(repo_root)
        manifest = prepared.dataset / "irodori_manifest.jsonl"
        latents = prepared.cache / "irodori_latents"
        if not _nonempty_file(manifest):
            prepare_manifest(repo_root, source, manifest, latents)
        if not (lfm_only_change and previous_candidate is not None):
            train_irodori(
                repo_root,
                manifest,
                candidate.models,
                candidate.cache,
                speaker_steps=cfg.training.speaker_inversion_max_steps,
                lora_steps=cfg.training.irodori_max_steps,
                do_speaker=cfg.training.irodori_speaker_inversion,
                do_lora=cfg.training.irodori_lora,
            )
        if cfg.training.lfm_lora:
            train_lfm(repo_root, candidate, cfg)
        seed = (
            train_seed_vc(
                repo_root,
                candidate,
                cfg,
                run_name=f"{cfg.name}_{lineage_id[3:]}_{generation_id[4:]}",
            )
            if not lfm_only_change
            else (
                str(candidate.models / "seed_vc" / "cfm.pth")
                if _nonempty_file(candidate.models / "seed_vc" / "cfm.pth")
                else None
            )
        )

        irodori_files: list[Path] = []
        if cfg.training.irodori_speaker_inversion:
            irodori_files.append(
                candidate.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors"
            )
        if cfg.training.irodori_lora:
            irodori_files.extend(
                _adapter_files(candidate.models / "irodori" / "lora" / "checkpoint_final")
            )
        lfm_files = (
            _adapter_files(candidate.models / "lfm" / "adapter")
            if cfg.training.lfm_lora
            else []
        )        if cfg.training.lfm_lora:
            lfm_files.append(candidate.models / "lfm" / "adapter" / _LFM_ADAPTER_REVISION_MARKER)
        seed_files = [Path(seed)] if seed is not None else []
        family_records = {
            "irodori": _family_record(
                candidate,
                "irodori",
                enabled=bool(irodori_files),
                mode="speaker-inversion+lora",
                lineage=lineage,
                files=irodori_files,
            ),
            "lfm": _family_record(
                candidate,
                "lfm",
                enabled=bool(cfg.training.lfm_lora),
                mode="lora",
                lineage=lineage,
                files=lfm_files,
            ),
            "seed_vc": _family_record(
                candidate,
                "seed_vc",
                enabled=True,
                mode=(
                    "finetuned-cfm"
                    if cfg.training.seed_vc_finetune
                    else "zero-shot-reference-only"
                ),
                lineage=lineage,
                files=seed_files,
            ),
        }
        manifest_value = {
            "schema_version": 1,
            "kind": "personavoice-v03-generation",
            "architecture": "v0.3-pre-full-fine-tuning",
            "persona": cfg.name,
            "lineage_id": lineage_id,
            "lineage_fingerprint": lineage["lineage_fingerprint"],
            "master_fingerprint": lineage["master_fingerprint"],
            "generation_id": generation_id,
            "generation_fingerprint": generation_fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
            "families": family_records,
            "provenance": {
                "prepare_lineage": lineage,
                "irodori_quality_report": _relative_to_persona(
                    paths, prepared.dataset / "irodori_quality_report.json"
                ),
                "lfm_quality_report": _relative_to_persona(
                    paths, prepared.dataset / "lfm_quality_report.json"
                ),
                "seed_manifest": _relative_to_persona(
                    paths, prepared.dataset / "seed_vc" / "manifest.jsonl"
                ),
                "architecture_boundary": "LFM LoRA + Irodori Speaker Inversion/LoRA + Seed-VC",
            },
            "validation": {
                "passed": False,
                "status": "pending",
                "acoustic_quality": {
                    "status": "not_run",
                    "reason": "Target-machine audio/GPU validation is an explicit post-CI step.",
                },
            },
        }
        atomic_write_json(candidate.generation_manifest, manifest_value)
        result = {
            "train_schema": TRAIN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "non_lfm_fingerprint": non_lfm_fingerprint,
            "lfm_fingerprint": lfm_fingerprint,
            "lineage_id": lineage_id,
            "prepare_lineage_id": lineage_id,
            "prepare_lineage_fingerprint": lineage["lineage_fingerprint"],
            "master_fingerprint": lineage["master_fingerprint"],
            "generation_id": generation_id,
            "generation_fingerprint": generation_fingerprint,
            "generation_manifest": _relative_to_persona(paths, candidate.generation_manifest),
            "families": family_records,
            "irodori_quality_report": _relative_to_persona(
                paths, prepared.dataset / "irodori_quality_report.json"
            ),
            "lfm_quality_report": _relative_to_persona(
                paths, prepared.dataset / "lfm_quality_report.json"
            ),
            "seed_manifest": _relative_to_persona(
                paths, prepared.dataset / "seed_vc" / "manifest.jsonl"
            ),
            "lfm_requested": bool(cfg.training.lfm_lora),
            "seed_vc_finetune": bool(cfg.training.seed_vc_finetune),
            "lfm_only_regeneration": lfm_only_change,
            "migration": (
                "lfm-only-regeneration"
                if lfm_only_change
                else "full-dependent-family-migration"
            ),
            "acoustic_validation": {
                "status": "not_run",
                "reason": "Hosted CI has no authorized persona audio or GTX 1080 Ti target.",
            },
        }
        store.set_result("train", result)
        return result
