from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from personavoice.hardware import hardware_report
from personavoice.media import sha256_file
from personavoice.model_assets import (
    ASR_MODEL_REVISION,
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_SHA256,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_REVISION,
    PYANNOTE_MODEL_REVISION,
    SENSE_MODEL_CMVN_SHA256,
    SENSE_MODEL_TOKENIZER_SHA256,
    SENSE_MODEL_WEIGHT_SHA256,
)
from personavoice.process import run
from personavoice.setup_env import IRODORI_REVISION, REVISION_MARKER, SEED_VC_REVISION
from personavoice.workers import local_model_env, worker

WORKER_NAMES = ("asr", "diarization", "sense", "lfm", "seed_vc")


def _setup_state(repo_root: Path) -> dict:
    path = repo_root / ".runtime" / "setup.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _expected_worker_backend(name: str, setup: dict) -> str | None:
    if name == "asr":
        return "cuda" if setup.get("irodori_backend") == "cu128" else "cpu"
    value = (setup.get("worker_backends") or {}).get(name)
    return None if value is None else str(value)


def _requires_cuda(value: str | None) -> bool:
    return bool(value and (value == "cuda" or value.startswith("cu")))


def _irodori_device(backend: str | None) -> str:
    if backend in {"cu128", "rocm"}:
        return "cuda"
    if backend == "xpu":
        return "xpu"
    return "cpu"


def _vendor_integrity(repo_root: Path, relative: str, expected_revision: str) -> dict:
    directory = repo_root / "vendor" / relative
    if not (directory / ".git").exists():
        return {
            "ok": False,
            "expected_revision": expected_revision,
            "error": "vendor checkout is missing or is not a git repository",
        }
    try:
        head = run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture=True,
        ).stdout.strip()
        status = run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture=True,
        ).stdout.strip()
    except Exception as exc:
        return {
            "ok": False,
            "expected_revision": expected_revision,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": head == expected_revision and not status,
        "head": head,
        "expected_revision": expected_revision,
        "clean": not status,
        "status": status,
    }


def _read_revision(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None
    except OSError:
        return None


def _model_asset_integrity(repo_root: Path, setup: dict, *, deep: bool) -> dict:
    expected_setup = {
        "irodori_model_sha256": IRODORI_MODEL_SHA256,
        "irodori_dacvae_sha256": IRODORI_DACVAE_SHA256,
        "irodori_text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
        "lfm_revision": LFM_MODEL_REVISION,
        "asr_revision": ASR_MODEL_REVISION,
    }
    expected_prepare = {
        "pyannote_revision": PYANNOTE_MODEL_REVISION,
        "sense_weight_sha256": SENSE_MODEL_WEIGHT_SHA256,
        "sense_cmvn_sha256": SENSE_MODEL_CMVN_SHA256,
        "sense_tokenizer_sha256": SENSE_MODEL_TOKENIZER_SHA256,
    }
    recorded = setup.get("model_assets") if isinstance(setup.get("model_assets"), dict) else {}
    recorded_prepare = (
        setup.get("prepare_assets") if isinstance(setup.get("prepare_assets"), dict) else {}
    )
    irodori = repo_root / "models" / "irodori" / "v4.1-small" / IRODORI_MODEL_FILENAME
    dacvae = repo_root / "models" / "irodori" / "dacvae" / IRODORI_DACVAE_FILENAME
    lfm_revision = _read_revision(repo_root / "models" / "lfm" / "base" / REVISION_MARKER)
    asr_revision = _read_revision(repo_root / "models" / "asr" / "large-v3" / REVISION_MARKER)
    pyannote_revision = _read_revision(
        repo_root / "models" / "pyannote" / "community-1" / REVISION_MARKER
    )
    sense_marker = _read_revision(repo_root / ".runtime" / "sense-model-ready")

    result = {
        "ok": True,
        "setup_matches": recorded == expected_setup,
        "prepare_setup_matches": recorded_prepare == expected_prepare,
        "expected": expected_setup,
        "recorded": recorded,
        "expected_prepare": expected_prepare,
        "recorded_prepare": recorded_prepare,
        "lfm_revision": lfm_revision,
        "asr_revision": asr_revision,
        "pyannote_revision": pyannote_revision,
        "sense_verified_marker": sense_marker,
        "irodori_sha256": None,
        "dacvae_sha256": None,
    }
    if recorded != expected_setup:
        result["ok"] = False
        result["error"] = "setup model asset pins do not match this PersonaVoice revision"
    if recorded_prepare != expected_prepare:
        result["ok"] = False
        result["error"] = "setup preparation asset pins do not match this PersonaVoice revision"
    if not irodori.is_file() or not dacvae.is_file():
        result["ok"] = False
        result["error"] = "Irodori base checkpoint or DACVAE checkpoint is missing"
    if lfm_revision != LFM_MODEL_REVISION:
        result["ok"] = False
        result["error"] = "LFM materialized revision does not match the audited revision"
    if asr_revision != ASR_MODEL_REVISION:
        result["ok"] = False
        result["error"] = "ASR materialized revision does not match the audited revision"
    if pyannote_revision != PYANNOTE_MODEL_REVISION:
        result["ok"] = False
        result["error"] = "pyannote materialized revision does not match the audited revision"
    if sense_marker != "verified":
        result["ok"] = False
        result["error"] = "SenseVoice assets have not been verified by the current setup"

    if deep and irodori.is_file() and dacvae.is_file():
        irodori_sha = sha256_file(irodori)
        dacvae_sha = sha256_file(dacvae)
        result["irodori_sha256"] = irodori_sha
        result["dacvae_sha256"] = dacvae_sha
        if irodori_sha != IRODORI_MODEL_SHA256:
            result["ok"] = False
            result["error"] = "Irodori checkpoint checksum mismatch"
        if dacvae_sha != IRODORI_DACVAE_SHA256:
            result["ok"] = False
            result["error"] = "Irodori DACVAE checksum mismatch"
    return result


def _irodori_health(repo_root: Path, setup: dict) -> dict:
    vendor = repo_root / "vendor" / "Irodori-TTS"
    checkpoint = repo_root / "models" / "irodori" / "v4.1-small" / IRODORI_MODEL_FILENAME
    codec = repo_root / "models" / "irodori" / "dacvae" / IRODORI_DACVAE_FILENAME
    if not (vendor / "infer.py").exists():
        return {"ok": False, "error": "Irodori vendor checkout is missing"}
    if not checkpoint.exists():
        return {"ok": False, "error": "Irodori base checkpoint is missing"}
    if not codec.exists():
        return {"ok": False, "error": "Irodori DACVAE checkpoint is missing"}

    backend = setup.get("irodori_backend") or hardware_report().get("irodori_backend")
    device = _irodori_device(str(backend))
    output_dir = repo_root / ".runtime" / "doctor"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "irodori-smoke.wav"
    output.unlink(missing_ok=True)
    try:
        run(
            [
                "uv",
                "run",
                "--project",
                vendor,
                "--no-sync",
                "python",
                vendor / "infer.py",
                "--checkpoint",
                checkpoint,
                "--codec-repo",
                codec,
                "--text",
                "動作確認です。",
                "--no-ref",
                "--output-wav",
                output,
                "--num-steps",
                "1",
                "--seconds",
                "0.5",
                "--num-candidates",
                "1",
                "--model-device",
                device,
                "--codec-device",
                device,
                "--model-precision",
                "fp32",
                "--codec-precision",
                "fp32",
                "--no-show-timings",
            ],
            cwd=vendor,
            env=local_model_env(repo_root, offline=True),
            capture=True,
        )
        if not output.exists() or output.stat().st_size <= 44:
            return {"ok": False, "device": device, "error": "Irodori produced no valid WAV"}
        return {
            "ok": True,
            "device": device,
            "model_loaded": True,
            "smoke_inference": True,
        }
    finally:
        output.unlink(missing_ok=True)


def report(
    repo_root: Path,
    *,
    deep: bool = False,
    require_seed_vc: bool = True,
) -> dict:
    required = {name: shutil.which(name) for name in ("uv", "git", "ffmpeg", "ffprobe")}
    runtime = repo_root / ".runtime"
    models = {
        "irodori": (repo_root / "models/irodori/v4.1-small/model.safetensors").exists(),
        "irodori_dacvae": (repo_root / "models/irodori/dacvae/weights.pth").exists(),
        "lfm": (repo_root / "models/lfm/base/config.json").exists(),
        "asr": (repo_root / "models/asr/large-v3/model.bin").exists(),
        "pyannote": (repo_root / "models/pyannote/community-1/config.yaml").exists(),
        "sense": (runtime / "sense-model-ready").exists(),
        "seed_vc_models": (runtime / "seed-vc-models-ready").exists(),
        "seed_vc_vendor": (repo_root / "vendor/seed-vc/inference_v2.py").exists(),
        "irodori_vendor": (repo_root / "vendor/Irodori-TTS/infer.py").exists(),
    }
    workers = {
        name: (repo_root / "workers" / name / ".venv").exists()
        for name in WORKER_NAMES
    }
    active_workers = tuple(
        name for name in WORKER_NAMES if require_seed_vc or name != "seed_vc"
    )
    setup = _setup_state(repo_root)
    model_assets = _model_asset_integrity(repo_root, setup, deep=deep)
    vendor_integrity = {
        "irodori": _vendor_integrity(repo_root, "Irodori-TTS", IRODORI_REVISION),
        "seed_vc": _vendor_integrity(repo_root, "seed-vc", SEED_VC_REVISION),
    }
    worker_health: dict[str, dict] = {}
    if deep:
        for name in active_workers:
            if not workers[name]:
                worker_health[name] = {"ok": False, "error": "worker .venv is missing"}
                continue
            try:
                health = worker(repo_root, name).call(
                    repo_root,
                    "health",
                    {"deep": True, "model": "large-v3", "compute_type": "auto"},
                )
                expected = _expected_worker_backend(name, setup)
                health["expected_backend"] = expected
                if _requires_cuda(expected) and not bool(health.get("cuda")):
                    health["ok"] = False
                    health["error"] = (
                        f"{name} was installed for {expected}, but its runtime cannot see CUDA. "
                        "Re-run `persona setup` and verify the NVIDIA driver."
                    )
                worker_health[name] = health
            except Exception as exc:
                worker_health[name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "expected_backend": _expected_worker_backend(name, setup),
                }
        try:
            worker_health["irodori"] = _irodori_health(repo_root, setup)
        except Exception as exc:
            worker_health["irodori"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    lockfiles = {
        "root": (repo_root / "uv.lock").exists(),
        **{
            name: (repo_root / "workers" / name / "uv.lock").exists()
            for name in WORKER_NAMES
        },
        "irodori_managed": (repo_root / "locks" / "Irodori-TTS.uv.lock").exists(),
    }
    required_model_keys = {
        "irodori",
        "irodori_dacvae",
        "lfm",
        "asr",
        "pyannote",
        "sense",
        "irodori_vendor",
    }
    required_vendor_keys = {"irodori"}
    required_worker_keys = set(active_workers)
    required_lock_keys = {"root", "asr", "diarization", "sense", "lfm", "irodori_managed"}
    if require_seed_vc:
        required_model_keys.update({"seed_vc_models", "seed_vc_vendor"})
        required_vendor_keys.add("seed_vc")
        required_lock_keys.add("seed_vc")

    locks_ready = all(lockfiles[key] for key in required_lock_keys)
    vendors_ready = all(vendor_integrity[key].get("ok") for key in required_vendor_keys)
    base_ready = (
        all(required.values())
        and bool(setup)
        and bool(model_assets.get("ok"))
        and all(models[key] for key in required_model_keys)
        and all(workers[key] for key in required_worker_keys)
        and locks_ready
        and vendors_ready
    )
    deep_ready = all(bool(value.get("ok")) for value in worker_health.values()) if deep else True
    return {
        "python": sys.version.split()[0],
        "commands": required,
        "commands_ok": all(required.values()),
        "hardware": hardware_report(),
        "setup": setup,
        "models": models,
        "model_asset_integrity": model_assets,
        "workers": workers,
        "worker_health": worker_health if deep else None,
        "vendor_integrity": vendor_integrity,
        "lockfiles": lockfiles,
        "reproducible_environment": locks_ready and bool(model_assets.get("ok")),
        "ready_offline": base_ready and deep_ready,
        "seed_vc_required": require_seed_vc,
    }
