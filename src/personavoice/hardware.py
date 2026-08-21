from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int


def nvidia_gpus() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        return []
    out = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            out.append(GpuInfo(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3]))))
        except ValueError:
            continue
    return out


def detect_irodori_backend() -> str:
    if nvidia_gpus():
        return "cu128"
    if platform.system() == "Darwin":
        return "cpu"
    return "cpu"


def hardware_report() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nvidia_gpus": [asdict(gpu) for gpu in nvidia_gpus()],
        "irodori_backend": detect_irodori_backend(),
    }


def safe_batch_profile() -> dict[str, int | bool]:
    gpus = nvidia_gpus()
    if not gpus:
        return {"batch_size": 1, "gradient_accumulation_steps": 8, "num_workers": 2, "gradient_checkpointing": True}
    vram = max(gpu.total_mib for gpu in gpus)
    if vram >= 48000:
        return {"batch_size": 12, "gradient_accumulation_steps": 1, "num_workers": 8, "gradient_checkpointing": False}
    if vram >= 24000:
        return {"batch_size": 6, "gradient_accumulation_steps": 2, "num_workers": 6, "gradient_checkpointing": True}
    if vram >= 16000:
        return {"batch_size": 3, "gradient_accumulation_steps": 4, "num_workers": 4, "gradient_checkpointing": True}
    if vram >= 12000:
        return {"batch_size": 2, "gradient_accumulation_steps": 6, "num_workers": 3, "gradient_checkpointing": True}
    return {"batch_size": 1, "gradient_accumulation_steps": 12, "num_workers": 2, "gradient_checkpointing": True}
