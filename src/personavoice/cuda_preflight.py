from __future__ import annotations

from pathlib import Path
from typing import Any

from personavoice.hardware import GpuInfo
from personavoice.process import CommandError, run_json
from personavoice.workers import local_model_env

_TORCH_PROBE = r'''
import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false in the synced CUDA environment")
capability = torch.cuda.get_device_capability(0)
values = torch.arange(1, 1025, dtype=torch.float32, device="cuda")
result = (values.square() + 1.0).sum()
if not torch.isfinite(result):
    raise RuntimeError("CUDA smoke kernel produced a non-finite result")
torch.cuda.synchronize()
print(json.dumps({
    "torch": str(torch.__version__),
    "device_name": str(torch.cuda.get_device_name(0)),
    "compute_capability": [int(capability[0]), int(capability[1])],
    "arch_list": list(torch.cuda.get_arch_list()),
    "smoke_value": float(result.item()),
}))
'''

_ASR_PROBE = r'''
import json
import ctranslate2
from runtime_policy import choose_compute_type

cuda_count = int(ctranslate2.get_cuda_device_count())
selected_device = "cpu"
selected_compute_type = None
cuda_types = []
cuda_error = None
if cuda_count > 0:
    try:
        cuda_types = sorted(set(ctranslate2.get_supported_compute_types("cuda", 0)))
        selected_compute_type = choose_compute_type("cuda", set(cuda_types), "auto")
        selected_device = "cuda"
    except (RuntimeError, OSError, ValueError) as exc:
        cuda_error = f"{type(exc).__name__}: {exc}"
if selected_compute_type is None:
    cpu_types = sorted(set(ctranslate2.get_supported_compute_types("cpu")))
    selected_compute_type = choose_compute_type("cpu", set(cpu_types), "auto")
else:
    cpu_types = []
print(json.dumps({
    "cuda_device_count": cuda_count,
    "cuda_supported_compute_types": cuda_types,
    "cuda_error": cuda_error,
    "selected_device": selected_device,
    "selected_compute_type": selected_compute_type,
    "cpu_supported_compute_types": cpu_types,
}))
'''


def _capability_tuple(value: str | None) -> tuple[int, int]:
    if not value:
        raise RuntimeError("Selected NVIDIA GPU has no reported compute capability")
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except ValueError as exc:
        raise RuntimeError(f"Invalid NVIDIA compute capability reported by nvidia-smi: {value!r}") from exc


def _torch_project_probe(
    repo_root: Path,
    project: Path,
    *,
    label: str,
    expected_capability: tuple[int, int],
) -> dict[str, Any]:
    try:
        result = run_json(
            [
                "uv",
                "run",
                "--project",
                project,
                "--no-sync",
                "python",
                "-c",
                _TORCH_PROBE,
            ],
            cwd=repo_root,
            env=local_model_env(repo_root),
        )
    except (CommandError, OSError) as exc:
        raise RuntimeError(
            f"CUDA runtime preflight failed for {label}. The locked environment was synced, "
            "but a real CUDA tensor/kernel could not execute on logical device 0. Check the "
            "NVIDIA driver, CUDA_VISIBLE_DEVICES, and GPU/backend compatibility before model downloads."
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"CUDA runtime preflight returned an invalid response for {label}")
    raw_capability = result.get("compute_capability")
    if not (
        isinstance(raw_capability, list)
        and len(raw_capability) == 2
        and all(isinstance(value, int) for value in raw_capability)
    ):
        raise RuntimeError(f"CUDA runtime preflight did not report a valid capability for {label}")
    actual_capability = (raw_capability[0], raw_capability[1])
    if actual_capability != expected_capability:
        raise RuntimeError(
            f"CUDA device mapping mismatch for {label}: nvidia-smi/setup selected "
            f"sm_{expected_capability[0]}{expected_capability[1]}, but the worker sees "
            f"sm_{actual_capability[0]}{actual_capability[1]} as CUDA device 0. "
            "Check CUDA_VISIBLE_DEVICES/CUDA_DEVICE_ORDER and rerun `persona setup --backend auto`."
        )
    return result


def _asr_preflight(repo_root: Path) -> dict[str, Any]:
    project = repo_root / "workers" / "asr"
    try:
        result = run_json(
            [
                "uv",
                "run",
                "--project",
                project,
                "--no-sync",
                "python",
                "-c",
                _ASR_PROBE,
            ],
            cwd=project,
            env=local_model_env(repo_root),
        )
    except (CommandError, OSError) as exc:
        raise RuntimeError(
            "ASR runtime preflight failed before model download. The locked CTranslate2 "
            "environment could not determine a safe CUDA/CPU compute type."
        ) from exc
    if not isinstance(result, dict) or result.get("selected_device") not in {"cuda", "cpu"}:
        raise RuntimeError("ASR runtime preflight returned an invalid device policy")
    if not isinstance(result.get("selected_compute_type"), str):
        raise RuntimeError("ASR runtime preflight returned an invalid compute type policy")
    return result


def run_cuda_preflight(
    repo_root: Path,
    *,
    irodori_project: Path,
    gpu: GpuInfo,
    worker_backends: dict[str, str | None],
) -> dict[str, Any]:
    """Execute real CUDA kernels in every synced CUDA PyTorch environment.

    This runs after dependency sync but before setup.json is finalized or large
    model assets are downloaded. Static compute-capability policy prevents known
    incompatible wheels; this dynamic gate catches stale drivers, missing CUDA
    DLLs, device-order mismatches, or a wheel that still cannot launch a kernel
    on the actual machine.

    ASR is intentionally different: CTranslate2 may safely fall back to CPU on a
    GPU that cannot execute its preferred CUDA compute type, so its probe records
    the resolved runtime policy instead of requiring CUDA.
    """

    expected = _capability_tuple(gpu.compute_capability)
    projects: list[tuple[str, Path]] = [("irodori", irodori_project)]
    for name in ("diarization", "sense", "lfm", "seed_vc"):
        backend = worker_backends.get(name)
        if isinstance(backend, str) and backend.startswith("cu"):
            projects.append((name, repo_root / "workers" / name))

    torch_results: dict[str, Any] = {}
    for label, project in projects:
        torch_results[label] = _torch_project_probe(
            repo_root,
            project,
            label=label,
            expected_capability=expected,
        )

    return {
        "ok": True,
        "expected_compute_capability": f"{expected[0]}.{expected[1]}",
        "torch_projects": torch_results,
        "asr": _asr_preflight(repo_root),
    }
