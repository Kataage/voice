from __future__ import annotations

import os
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
    uuid: str | None = None
    pci_bus_id: str | None = None
    driver_version: str | None = None


def _run_nvidia_query(fields: str) -> subprocess.CompletedProcess[str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
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
    except OSError:
        # PATH can change between discovery and process creation (and tests may
        # deliberately stub command discovery). Treat an unavailable NVIDIA
        # utility as "no detectable GPU" rather than aborting setup/doctor.
        return None


def _parse_compute_capability(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        major, minor = value.strip().split(".", 1)
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def nvidia_gpus() -> list[GpuInfo]:
    fields = "index,uuid,pci.bus_id,name,memory.total,memory.free,driver_version"
    completed = _run_nvidia_query(fields + ",compute_cap")
    include_compute_cap = completed is not None and completed.returncode == 0
    if not include_compute_cap:
        completed = _run_nvidia_query(fields)
    if completed is None or completed.returncode != 0:
        return []

    out = []
    expected_parts = 8 if include_compute_cap else 7
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != expected_parts:
            continue
        try:
            out.append(
                GpuInfo(
                    index=int(parts[0]),
                    uuid=parts[1] or None,
                    pci_bus_id=parts[2] or None,
                    name=parts[3],
                    total_mib=int(float(parts[4])),
                    free_mib=int(float(parts[5])),
                    driver_version=parts[6] or None,
                    compute_capability=parts[7] if include_compute_cap else None,
                )
            )
        except ValueError:
            continue
    return out


def _pci_ordered(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """Mirror CUDA_DEVICE_ORDER=PCI_BUS_ID with a safe index fallback."""

    return sorted(
        gpus,
        key=lambda gpu: (
            gpu.pci_bus_id is None,
            (gpu.pci_bus_id or "").casefold(),
            gpu.index,
        ),
    )


def selected_nvidia_gpu(gpus: list[GpuInfo] | None = None) -> GpuInfo | None:
    """Return the physical GPU exposed as PersonaVoice logical CUDA device 0.

    PersonaVoice CUDA subprocesses force ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so
    setup/runtime hardware detection can deterministically mirror the same
    ordering without importing a CUDA framework in the root environment. When
    ``CUDA_VISIBLE_DEVICES`` remaps devices, numeric ordinals are interpreted in
    that PCI order and GPU UUIDs are resolved directly. MIG IDs and malformed or
    unresolvable selectors fail closed rather than guessing.
    """

    candidates = _pci_ordered(nvidia_gpus() if gpus is None else gpus)
    if not candidates:
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return candidates[0]
    visible = visible.strip()
    if not visible or visible == "-1":
        return None

    first = visible.split(",", 1)[0].strip()
    if not first or first == "-1" or first.upper().startswith("MIG-"):
        return None
    try:
        logical_index = int(first)
    except ValueError:
        token = first.casefold()
        matches = [
            gpu
            for gpu in candidates
            if gpu.uuid
            and (
                gpu.uuid.casefold() == token
                or gpu.uuid.casefold().startswith(token)
            )
        ]
        return matches[0] if len(matches) == 1 else None
    if logical_index < 0 or logical_index >= len(candidates):
        return None
    return candidates[logical_index]


def gpu_record(gpu: GpuInfo | None) -> dict | None:
    return asdict(gpu) if gpu is not None else None


def _host_arch() -> str:
    value = platform.machine().lower()
    if value in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "aarch64"
    return value


def _known_pytorch_210_architecture(capability: tuple[int, int], *, backend: str) -> bool:
    """Match the PersonaVoice-audited PyTorch 2.10 binary architecture matrix.

    The published PyTorch wheel matrix is broader than PersonaVoice's complete
    model stack. CUDA is therefore enabled only on host architectures exercised
    by PersonaVoice CI and only for GPU architectures explicitly covered below.
    Unknown future GPUs and unaudited hosts fail closed instead of installing a
    wheel that might expose CUDA but fail on the first real model kernel.
    """

    major, minor = capability
    if _host_arch() != "x86_64":
        return False

    # Linux x86_64 and Windows official wheels. Ada sm_89 executes the Ampere
    # sm_86 compatibility path. Maxwell is deliberately excluded from the
    # whole-stack CUDA policy because the prebuilt CTranslate2 ASR runtime is
    # not reliably executable on sm_5x. sm_103 and other post-2.10 variants are
    # intentionally excluded until the complete PersonaVoice stack is audited.
    if backend == "cu126":
        return (
            (major == 6 and minor in {0, 1})
            or (major == 7 and minor in {0, 5})
            or (major == 8 and minor in {0, 6, 9})
            or (major == 9 and minor == 0)
        )
    if backend == "cu128":
        return (
            (major == 7 and minor == 5)
            or (major == 8 and minor in {0, 6, 9})
            or (major == 9 and minor == 0)
            or (major == 10 and minor == 0)
            or (major == 12 and minor == 0)
        )
    return False


def backend_supports_gpu(backend: str, gpu: GpuInfo) -> bool:
    if backend == "cpu":
        return True
    if backend not in {"cu126", "cu128"}:
        return False
    capability = _parse_compute_capability(gpu.compute_capability)
    return bool(capability and _known_pytorch_210_architecture(capability, backend=backend))


def seed_vc_cuda_supported(gpu: GpuInfo) -> bool:
    """Whether Seed-VC's audited PyTorch 2.4/cu124 stack can use this GPU.

    That older stack is audited on x86_64 for Pascal through Hopper but predates
    Blackwell. Other host architectures and newer GPUs therefore use Seed-VC's
    CPU environment rather than assuming an unverified CUDA wheel/runtime pair.
    """

    if _host_arch() != "x86_64":
        return False
    capability = _parse_compute_capability(gpu.compute_capability)
    if capability is None:
        return False
    major, minor = capability
    return (
        (major == 6 and minor in {0, 1})
        or (major == 7 and minor in {0, 5})
        or (major == 8 and minor in {0, 6, 9})
        or (major == 9 and minor == 0)
    )


def cuda_backend_for_gpu(gpu: GpuInfo) -> str:
    """Return the safest audited PyTorch CUDA wheel family for one NVIDIA GPU."""

    if backend_supports_gpu("cu128", gpu):
        return "cu128"
    if backend_supports_gpu("cu126", gpu):
        return "cu126"
    return "cpu"


def irodori_training_precision(
    backend: str,
    *,
    gpu: GpuInfo | None = None,
) -> dict[str, str | bool]:
    """Return an audited Irodori training precision policy for the active GPU.

    cu126 is deliberately fp32 for the pre-Turing compatibility path. Turing is
    also kept on fp32 even when using cu128 because it has neither BF16 tensor
    cores nor TF32. Ampere and newer audited cu128 GPUs may use Irodori's native
    BF16 + TF32 path. Unknown capabilities fail closed to fp32/no-TF32.
    """

    if backend not in {"cu126", "cu128"}:
        return {"precision": "fp32", "allow_tf32": False}
    selected = selected_nvidia_gpu() if gpu is None else gpu
    capability = _parse_compute_capability(
        selected.compute_capability if selected is not None else None
    )
    if (
        backend == "cu128"
        and selected is not None
        and capability is not None
        and capability >= (8, 0)
        and backend_supports_gpu("cu128", selected)
    ):
        return {"precision": "bf16", "allow_tf32": True}
    return {"precision": "fp32", "allow_tf32": False}


def detect_irodori_backend() -> str:
    gpu = selected_nvidia_gpu()
    if gpu is not None:
        return cuda_backend_for_gpu(gpu)
    if platform.system() == "Darwin":
        return "cpu"
    return "cpu"


def hardware_report() -> dict:
    gpus = nvidia_gpus()
    selected = selected_nvidia_gpu(gpus)
    backend = cuda_backend_for_gpu(selected) if selected is not None else "cpu"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_gpus": [asdict(gpu) for gpu in gpus],
        "selected_nvidia_gpu": asdict(selected) if selected is not None else None,
        "irodori_backend": backend,
    }


def safe_batch_profile(*, backend: str | None = None) -> dict[str, int | bool]:
    """Return a conservative Irodori training profile for the selected backend.

    NVIDIA VRAM is only relevant when the configured Irodori backend is one of
    the audited CUDA wheel families. Use the GPU exposed as logical CUDA device
    0 rather than the largest physical GPU in a multi-GPU host.
    """

    gpu = selected_nvidia_gpu() if backend in {None, "cu126", "cu128"} else None
    if gpu is None:
        return {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_workers": 2,
            "gradient_checkpointing": True,
        }
    vram = gpu.total_mib
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
