from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found < count:
        raise RuntimeError(
            f"Expected at least {count} patch anchors in {path}, found {found}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


patch(
    "src/personavoice/setup_env.py",
    "from personavoice.hardware import cuda_backend_for_gpu, detect_irodori_backend, nvidia_gpus\n",
    "from personavoice.hardware import (\n"
    "    backend_supports_gpu,\n"
    "    cuda_backend_for_gpu,\n"
    "    detect_irodori_backend,\n"
    "    seed_vc_cuda_supported,\n"
    "    selected_nvidia_gpu,\n"
    ")\n",
)

patch(
    "src/personavoice/setup_env.py",
    '''def _worker_extras(selected_backend: str) -> dict[str, str | None]:\n    """Map the Irodori backend to compatible isolated worker backends."""\n\n    if selected_backend in {"cu126", "cu128"}:\n        return {\n            "asr": None,\n            "diarization": selected_backend,\n            "sense": selected_backend,\n            "lfm": selected_backend,\n            # Seed-VC intentionally remains on its audited Torch 2.4/CUDA 12.4 stack.\n            "seed_vc": "cu124",\n        }\n    return {\n        "asr": None,\n        "diarization": "cpu",\n        "sense": "cpu",\n        "lfm": "cpu",\n        "seed_vc": "cpu",\n    }\n''',
    '''def _worker_extras(selected_backend: str, *, gpu=None) -> dict[str, str | None]:\n    """Map the selected backend to safe isolated worker environments."""\n\n    if selected_backend in {"cu126", "cu128"}:\n        seed_backend = "cu124" if gpu is None or seed_vc_cuda_supported(gpu) else "cpu"\n        return {\n            "asr": None,\n            "diarization": selected_backend,\n            "sense": selected_backend,\n            "lfm": selected_backend,\n            # Seed-VC stays on its audited Torch 2.4 stack. Blackwell and newer\n            # GPUs predate that wheel's cubins, so only this worker falls back.\n            "seed_vc": seed_backend,\n        }\n    return {\n        "asr": None,\n        "diarization": "cpu",\n        "sense": "cpu",\n        "lfm": "cpu",\n        "seed_vc": "cpu",\n    }\n''',
)

patch(
    "src/personavoice/setup_env.py",
    '''def _validate_explicit_backend(backend: str | None) -> None:\n    if backend != "cu128":\n        return\n    gpus = nvidia_gpus()\n    if not gpus:\n        return\n    gpu0 = min(gpus, key=lambda gpu: gpu.index)\n    compatible = cuda_backend_for_gpu(gpu0)\n    if compatible == "cu126":\n        capability = gpu0.compute_capability or "unknown"\n        raise ValueError(\n            f"The selected NVIDIA GPU {gpu0.name} has compute capability {capability}; "\n            "the audited PyTorch CUDA 12.8 stack requires sm_70 or newer. "\n            "Use `--backend auto` or `--backend cu126` for this GPU."\n        )\n''',
    '''def _validate_cuda_backend(backend: str | None):\n    """Return the selected GPU or reject an unsafe explicit/automatic CUDA stack."""\n\n    if backend not in {"cu126", "cu128"}:\n        return None\n    gpu = selected_nvidia_gpu()\n    if gpu is None:\n        raise ValueError(\n            f"The selected backend {backend} requires an NVIDIA GPU exposed as CUDA device 0, "\n            "but no NVIDIA GPU could be selected. Check the driver and CUDA_VISIBLE_DEVICES, "\n            "or use `--backend auto`."\n        )\n    if backend_supports_gpu(backend, gpu):\n        return gpu\n\n    preferred = cuda_backend_for_gpu(gpu)\n    capability = gpu.compute_capability or "unknown"\n    fallback = f"--backend {preferred}" if preferred in {"cu126", "cu128"} else "--backend cpu"\n    raise ValueError(\n        f"The selected NVIDIA GPU {gpu.name} has compute capability {capability}; "\n        f"the audited {backend} PyTorch stack does not contain a compatible kernel image. "\n        f"Use `--backend auto` (recommended) or `{fallback}`."\n    )\n''',
)

patch(
    "src/personavoice/setup_env.py",
    '''    require_ffmpeg_runtime()\n    _validate_explicit_backend(backend)\n    selected_backend = backend or detect_irodori_backend()\n    if selected_backend not in SUPPORTED_IRODORI_BACKENDS:\n        expected = ", ".join(sorted(SUPPORTED_IRODORI_BACKENDS))\n        raise ValueError(f"Unsupported Irodori backend {selected_backend!r}; choose one of: {expected}")\n\n    worker_extras = _worker_extras(selected_backend)\n''',
    '''    require_ffmpeg_runtime()\n    selected_backend = backend or detect_irodori_backend()\n    if selected_backend not in SUPPORTED_IRODORI_BACKENDS:\n        expected = ", ".join(sorted(SUPPORTED_IRODORI_BACKENDS))\n        raise ValueError(f"Unsupported Irodori backend {selected_backend!r}; choose one of: {expected}")\n\n    selected_gpu = _validate_cuda_backend(selected_backend)\n    worker_extras = _worker_extras(selected_backend, gpu=selected_gpu)\n''',
)

# Doctor reports GPU swaps/incompatible wheels as a first-class readiness error
# and never runs the direct Irodori smoke test against a known incompatible GPU.
patch(
    "src/personavoice/doctor.py",
    "from personavoice.environment_contract import environment_contract_status\n",
    "from personavoice.environment_contract import environment_contract_status, runtime_hardware_status\n",
)
patch(
    "src/personavoice/doctor.py",
    '    environment = environment_contract_status(repo_root, setup.get("environment_contract"))\n',
    '    environment = environment_contract_status(repo_root, setup.get("environment_contract"))\n'
    '    runtime_hardware = runtime_hardware_status(setup)\n',
)
patch(
    "src/personavoice/doctor.py",
    '''        try:\n            worker_health["irodori"] = _irodori_health(repo_root, setup)\n        except Exception as exc:\n            worker_health["irodori"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}\n''',
    '''        if not runtime_hardware.get("ok"):\n            worker_health["irodori"] = {\n                "ok": False,\n                "error": str(runtime_hardware.get("error")),\n            }\n        else:\n            try:\n                worker_health["irodori"] = _irodori_health(repo_root, setup)\n            except Exception as exc:\n                worker_health["irodori"] = {\n                    "ok": False,\n                    "error": f"{type(exc).__name__}: {exc}",\n                }\n''',
)
patch(
    "src/personavoice/doctor.py",
    '    reproducible = locks_ready and bool(model_assets.get("ok")) and bool(environment.get("ok"))\n',
    '    reproducible = (\n'
    '        locks_ready\n'
    '        and bool(model_assets.get("ok"))\n'
    '        and bool(environment.get("ok"))\n'
    '        and bool(runtime_hardware.get("ok"))\n'
    '    )\n',
)
patch(
    "src/personavoice/doctor.py",
    '        "environment_contract": environment,\n',
    '        "environment_contract": environment,\n        "runtime_hardware": runtime_hardware,\n',
)

# This old private inference hook disappeared during the runtime-integrity
# refactor. The test only verifies stale environment generation rejection, which
# occurs before model runtime, so retaining this monkeypatch is both unnecessary
# and brittle.
patch(
    "tests/test_environment_generation.py",
    '    monkeypatch.setattr(inference, "_verify_irodori_runtime", lambda *_args, **_kwargs: None)\n',
    "",
)

print("GPU-generalization integration patches applied")
