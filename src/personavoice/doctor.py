from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from personavoice.hardware import hardware_report
from personavoice.process import run
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


def _irodori_health(repo_root: Path, setup: dict) -> dict:
    vendor = repo_root / "vendor" / "Irodori-TTS"
    checkpoint = repo_root / "models" / "irodori" / "v4.1-small" / "model.safetensors"
    if not (vendor / "infer.py").exists():
        return {"ok": False, "error": "Irodori vendor checkout is missing"}
    if not checkpoint.exists():
        return {"ok": False, "error": "Irodori base checkpoint is missing"}

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
        "lfm",
        "asr",
        "pyannote",
        "sense",
        "irodori_vendor",
    }
    required_worker_keys = set(active_workers)
    required_lock_keys = {"root", "asr", "diarization", "sense", "lfm", "irodori_managed"}
    if require_seed_vc:
        required_model_keys.update({"seed_vc_models", "seed_vc_vendor"})
        required_lock_keys.add("seed_vc")

    base_ready = (
        all(required.values())
        and all(models[key] for key in required_model_keys)
        and all(workers[key] for key in required_worker_keys)
    )
    deep_ready = all(bool(value.get("ok")) for value in worker_health.values()) if deep else True
    return {
        "python": sys.version.split()[0],
        "commands": required,
        "commands_ok": all(required.values()),
        "hardware": hardware_report(),
        "setup": setup,
        "models": models,
        "workers": workers,
        "worker_health": worker_health if deep else None,
        "lockfiles": lockfiles,
        "reproducible_environment": all(lockfiles[key] for key in required_lock_keys),
        "ready_offline": base_ready and deep_ready,
        "seed_vc_required": require_seed_vc,
    }
