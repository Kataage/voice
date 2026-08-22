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
    compute_capability: str | None = None


def _run_nvidia_query(fields: str) -> subprocess.CompletedProcess[str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    return subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _parse_compute_capability(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        major, minor = value.strip().split(".", 1)
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def nvidia_gpus() -> list[GpuInfo]:
    completed = _run_nvidia_query("index,name,memory.total,memory.free,compute_cap")
    include_compute_cap = completed is not None and completed.returncode == 0
    if not include_compute_cap:
        completed = _run_nvidia_query("index,name,memory.total,memory.free")
    if completed is None or completed.returncode != 0:
        return []

    out = []
    expected_parts = 5 if include_compute_cap else 4
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != expected_parts:
            continue
        try:
            out.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mib=int(float(parts[2])),
                    free_mib=int(float(parts[3])),
                    compute_capability=parts[4] if include_compute_cap else None,
                )
            )
        except ValueError:
            continue
    return out


def cuda_backend_for_gpu(gpu: GpuInfo) -> str:
    """Return the audited PyTorch CUDA wheel family for one NVIDIA GPU.

    PyTorch 2.10 CUDA 12.8 wheels used by PersonaVoice require compute
    capability 7.0 or newer. Pascal-class 6.x GPUs instead use the CUDA 12.6
    wheel family, which keeps those architectures available. Unknown or older
    capabilities fail closed to CPU rather than selecting a CUDA wheel that can
    install successfully but later raise ``cudaErrorNoKernelImageForDevice``.
    """

    capability = _parse_compute_capability(gpu.compute_capability)
    if capability is None:
        return "cpu"
    if capability >= (7, 0):
        return "cu128"
    if capability >= (6, 0):
        return "cu126"
    return "cpu"


def detect_irodori_backend() -> str:
    gpus = nvidia_gpus()
    if gpus:
        # Model workers use CUDA device 0 unless a future explicit device
        # selection contract says otherwise, so auto-selection must describe
        # the same device rather than an arbitrary second GPU.
        gpu0 = min(gpus, key=lambda gpu: gpu.index)
        return cuda_backend_for_gpu(gpu0)
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


def safe_batch_profile(*, backend: str | None = None) -> dict[str, int | bool]:
    """Return a conservative Irodori training profile for the selected backend.

    NVIDIA VRAM is only relevant when the configured Irodori backend is one of
    the audited CUDA wheel families. This prevents an explicitly selected
    CPU/ROCm/XPU backend from accidentally receiving an NVIDIA-sized batch merely
    because nvidia-smi is installed.
    """

    gpus = nvidia_gpus() if backend in {None, "cu126", "cu128"} else []
    if not gpus:
        return {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_workers": 2,
            "gradient_checkpointing": True,
        }
    vram = max(gpu.total_mib for gpu in gpus)
    if vram >= 48000:
        return {
            "batch_size": 12,
            "gradient_accumulation_steps": 1,
            "num_workers": 8,
            "gradient_checkpointing": False,
        }
    if vram >= 24000:
        return {
            "batch_size": 6,
            "gradient_accumulation_steps": 2,
            "num_workers": 6,
            "gradient_checkpointing": True,
        }
    if vram >= 16000:
        return {
            "batch_size": 3,
            "gradient_accumulation_steps": 4,
            "num_workers": 4,
            "gradient_checkpointing": True,
        }
    if vram >= 12000:
        return {
            "batch_size": 2,
            "gradient_accumulation_steps": 6,
            "num_workers": 3,
            "gradient_checkpointing": True,
        }
    return {
        "batch_size": 1,
        "gradient_accumulation_steps": 12,
        "num_workers": 2,
        "gradient_checkpointing": True,
    }
