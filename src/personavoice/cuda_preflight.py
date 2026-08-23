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

def finite(value, label):
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{label} CUDA smoke kernel produced a non-finite result")

values = torch.arange(1, 1025, dtype=torch.float32, device="cuda")
fp32_result = (values.square() + 1.0).sum()
finite(fp32_result, "float32")

fp16_left = torch.randn((16, 16), dtype=torch.float16, device="cuda")
fp16_right = torch.randn((16, 16), dtype=torch.float16, device="cuda")
fp16_result = (fp16_left @ fp16_right).float().abs().mean()
finite(fp16_result, "float16")

bf16_value = None
if capability >= (8, 0):
    bf16_left = torch.randn((16, 16), dtype=torch.bfloat16, device="cuda")
    bf16_right = torch.randn((16, 16), dtype=torch.bfloat16, device="cuda")
    bf16_result = (bf16_left @ bf16_right).float().abs().mean()
    finite(bf16_result, "bfloat16")
    bf16_value = float(bf16_result.item())

torch.cuda.synchronize()
print(json.dumps({
    "torch": str(torch.__version__),
    "device_name": str(torch.cuda.get_device_name(0)),
    "compute_capability": [int(capability[0]), int(capability[1])],
    "arch_list": list(torch.cuda.get_arch_list()),
    "fp32_smoke": float(fp32_result.item()),
    "fp16_smoke": float(fp16_result.item()),
    "bf16_smoke": bf16_value,
}))
'''

_ASR_PROBE = r'''
import ctypes
import json
import sys

import ctranslate2
from runtime_policy import choose_compute_type


def require_status(status, label):
    value = int(status)
    if value != 0:
        raise RuntimeError(f"{label} failed with native status {value}")


def native_cuda_smoke():
    if sys.platform == "win32":
        names = [
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudnn_adv64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn64_9.dll",
        ]
        libraries = {name: ctypes.WinDLL(name) for name in names}
        cublas = libraries["cublas64_12.dll"]
        cudnn = libraries["cudnn64_9.dll"]
    elif sys.platform.startswith("linux"):
        names = ["libcublas.so.12", "libcudnn.so.9"]
        libraries = {
            name: ctypes.CDLL(name, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            for name in names
        }
        cublas = libraries["libcublas.so.12"]
        cudnn = libraries["libcudnn.so.9"]
    else:
        raise RuntimeError(f"CTranslate2 CUDA preflight is unsupported on {sys.platform!r}")

    cublas_handle = ctypes.c_void_p()
    cublas_create = cublas.cublasCreate_v2
    cublas_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublas_create.restype = ctypes.c_int
    cublas_destroy = cublas.cublasDestroy_v2
    cublas_destroy.argtypes = [ctypes.c_void_p]
    cublas_destroy.restype = ctypes.c_int
    require_status(cublas_create(ctypes.byref(cublas_handle)), "cublasCreate_v2")
    try:
        if not cublas_handle.value:
            raise RuntimeError("cublasCreate_v2 returned a null handle")
    finally:
        if cublas_handle.value:
            require_status(cublas_destroy(cublas_handle), "cublasDestroy_v2")

    cudnn_handle = ctypes.c_void_p()
    cudnn_create = cudnn.cudnnCreate
    cudnn_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudnn_create.restype = ctypes.c_int
    cudnn_destroy = cudnn.cudnnDestroy
    cudnn_destroy.argtypes = [ctypes.c_void_p]
    cudnn_destroy.restype = ctypes.c_int
    require_status(cudnn_create(ctypes.byref(cudnn_handle)), "cudnnCreate")
    try:
        if not cudnn_handle.value:
            raise RuntimeError("cudnnCreate returned a null handle")
    finally:
        if cudnn_handle.value:
            require_status(cudnn_destroy(cudnn_handle), "cudnnDestroy")

    return {"ok": True, "libraries": names}


cuda_count = int(ctranslate2.get_cuda_device_count())
selected_device = "cpu"
selected_compute_type = None
cuda_types = []
cuda_error = None
native_runtime = {"ok": None, "skipped": True}
if cuda_count > 0:
    try:
        native_runtime = native_cuda_smoke()
        cuda_types = sorted(set(ctranslate2.get_supported_compute_types("cuda", 0)))
        selected_compute_type = choose_compute_type("cuda", set(cuda_types), "auto")
        selected_device = "cuda"
    except (RuntimeError, OSError, ValueError) as exc:
        cuda_error = f"{type(exc).__name__}: {exc}"
        raise
if selected_compute_type is None:
    cpu_types = sorted(set(ctranslate2.get_supported_compute_types("cpu")))
    selected_compute_type = choose_compute_type("cpu", set(cpu_types), "auto")
else:
    cpu_types = []
print(json.dumps({
    "cuda_device_count": cuda_count,
    "cuda_supported_compute_types": cuda_types,
    "cuda_error": cuda_error,
    "native_cuda_runtime": native_runtime,
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
            "environment could not load and initialize its audited CUDA 12 cuBLAS/cuDNN "
            "runtime or determine a safe CUDA/CPU compute type."
        ) from exc
    if not isinstance(result, dict) or result.get("selected_device") not in {"cuda", "cpu"}:
        raise RuntimeError("ASR runtime preflight returned an invalid device policy")
    if not isinstance(result.get("selected_compute_type"), str):
        raise RuntimeError("ASR runtime preflight returned an invalid compute type policy")
    if result.get("selected_device") == "cuda":
        native = result.get("native_cuda_runtime")
        if not isinstance(native, dict) or native.get("ok") is not True:
            raise RuntimeError("ASR CUDA preflight did not prove the native cuBLAS/cuDNN runtime")
    return result


def run_cuda_preflight(
    repo_root: Path,
    *,
    irodori_project: Path,
    gpu: GpuInfo,
    worker_backends: dict[str, str | None],
) -> dict[str, Any]:
    """Execute real native CUDA work in every synced CUDA environment.

    This runs after dependency sync but before setup.json is finalized or large
    model assets are downloaded. Static compute-capability policy prevents known
    incompatible wheels; this dynamic gate catches stale drivers, missing CUDA
    DLLs/shared objects, device-order mismatches, or a runtime that still cannot
    initialize on the actual machine.

    PyTorch workers execute tensor/matmul kernels. ASR uses CTranslate2 directly,
    so it separately opens and initializes the audited CUDA 12 cuBLAS/cuDNN
    runtime before accepting a CUDA compute type. A GPU that CTranslate2 cannot
    use at all may still select its intentional CPU fallback.
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
