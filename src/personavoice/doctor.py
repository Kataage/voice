from __future__ import annotations

import shutil
import sys
from pathlib import Path

from personavoice.hardware import hardware_report


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
    workers = {}
    for name in ("asr", "diarization", "sense", "lfm", "seed_vc"):
        workers[name] = (repo_root / "workers" / name / ".venv").exists()
    return {
        "python": sys.version.split()[0],
        "commands": required,
        "commands_ok": all(required.values()),
        "hardware": hardware_report(),
        "models": models,
        "workers": workers,
        "ready_offline": all(required.values()) and all(models.values()) and all(workers.values()),
    }
