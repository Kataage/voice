from __future__ import annotations

import shutil
import sys
from pathlib import Path

from personavoice.hardware import hardware_report
from personavoice.workers import worker

WORKER_NAMES = ("asr", "diarization", "sense", "lfm", "seed_vc")


def report(repo_root: Path, *, deep: bool = False) -> dict:
    required = {name: shutil.which(name) for name in ("uv", "git", "ffmpeg", "ffprobe")}
    models = {
        "irodori": (repo_root / "models/irodori/v4.1-small/model.safetensors").exists(),
        "lfm": (repo_root / "models/lfm/base/config.json").exists(),
        "asr": (repo_root / "models/asr/large-v3").exists(),
        "pyannote": (repo_root / "models/pyannote/community-1/config.yaml").exists(),
        "sense": (repo_root / "models/sense/SenseVoiceSmall").exists(),
        "seed_vc_vendor": (repo_root / "vendor/seed-vc/inference_v2.py").exists(),
        "irodori_vendor": (repo_root / "vendor/Irodori-TTS/infer.py").exists(),
    }
    workers = {
        name: (repo_root / "workers" / name / ".venv").exists()
        for name in WORKER_NAMES
    }
    worker_health = {}
    if deep:
        for name in WORKER_NAMES:
            if not workers[name]:
                worker_health[name] = {"ok": False, "error": "worker .venv is missing"}
                continue
            try:
                worker_health[name] = worker(repo_root, name).call(
                    repo_root,
                    "health",
                    {"deep": True, "model": "large-v3", "compute_type": "auto"},
                )
            except Exception as exc:
                worker_health[name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    base_ready = all(required.values()) and all(models.values()) and all(workers.values())
    deep_ready = all(bool(value.get("ok")) for value in worker_health.values()) if deep else True
    return {
        "python": sys.version.split()[0],
        "commands": required,
        "commands_ok": all(required.values()),
        "hardware": hardware_report(),
        "models": models,
        "workers": workers,
        "worker_health": worker_health if deep else None,
        "ready_offline": base_ready and deep_ready,
    }
