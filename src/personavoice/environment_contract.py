from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from personavoice.hardware import (
    backend_supports_gpu,
    cuda_backend_for_gpu,
    seed_vc_cuda_supported,
    selected_nvidia_gpu,
)

WORKER_NAMES = ("asr", "diarization", "sense", "lfm", "seed_vc")
ENVIRONMENT_CONTRACT_SCHEMA = 4
SETUP_TRANSACTION_MARKER = "setup-in-progress.json"


def _sha256(path: Path) -> str:
    try:
        if not path.is_file():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "unreadable"


def environment_contract(repo_root: Path) -> dict[str, Any]:
    """Fingerprint every dependency and immutable asset declaration used by setup.

    `persona setup` records this value only after all isolated environments sync
    successfully. Consumers recompute it from the current checkout. A mismatch
    means the installed environments or pinned transitive model contract belong
    to a different repository generation and must be resynced before use.
    """

    workers = {}
    for name in WORKER_NAMES:
        project = repo_root / "workers" / name
        workers[name] = {
            "pyproject_sha256": _sha256(project / "pyproject.toml"),
            "lock_sha256": _sha256(project / "uv.lock"),
        }
    return {
        "schema": ENVIRONMENT_CONTRACT_SCHEMA,
        "root": {
            "pyproject_sha256": _sha256(repo_root / "pyproject.toml"),
            "lock_sha256": _sha256(repo_root / "uv.lock"),
        },
        "irodori": {
            "managed_project_sha256": _sha256(
                repo_root / "locks" / "Irodori-TTS.pyproject.toml"
            ),
            "managed_lock_sha256": _sha256(repo_root / "locks" / "Irodori-TTS.uv.lock"),
        },
        "seed_vc": {
            "asset_contract_sha256": _sha256(repo_root / "config" / "seed_vc_assets.json"),
        },
        "runtime_policy": {
            "environment_contract_sha256": _sha256(
                repo_root / "src" / "personavoice" / "environment_contract.py"
            ),
            "hardware_sha256": _sha256(repo_root / "src" / "personavoice" / "hardware.py"),
            "irodori_sha256": _sha256(repo_root / "src" / "personavoice" / "irodori.py"),
            "inference_sha256": _sha256(repo_root / "src" / "personavoice" / "inference.py"),
            "setup_sha256": _sha256(repo_root / "src" / "personavoice" / "setup_env.py"),
            "runtime_dependencies_sha256": _sha256(
                repo_root / "src" / "personavoice" / "runtime_dependencies.py"
            ),
            "cuda_preflight_sha256": _sha256(
                repo_root / "src" / "personavoice" / "cuda_preflight.py"
            ),
            "workers_sha256": _sha256(repo_root / "src" / "personavoice" / "workers.py"),
            "asr_runtime_policy_sha256": _sha256(
                repo_root / "workers" / "asr" / "runtime_policy.py"
            ),
        },
        "workers": workers,
    }


def environment_contract_status(repo_root: Path, recorded: Any) -> dict[str, Any]:
    current = environment_contract(repo_root)
    valid_recorded = recorded if isinstance(recorded, dict) else {}
    transaction = repo_root / ".runtime" / SETUP_TRANSACTION_MARKER
    in_progress = transaction.is_file()
    contract_matches = valid_recorded == current
    if in_progress:
        error = (
            "an environment setup transaction is incomplete; rerun `persona setup` to finish "
            "synchronizing all audited environments"
        )
    elif not contract_matches:
        error = (
            "installed environments were created for a different dependency contract "
            "or model-asset contract; run `persona setup` to resync the audited local environments"
        )
    else:
        error = None
    return {
        "ok": contract_matches and not in_progress,
        "recorded": valid_recorded,
        "current": current,
        "setup_in_progress": in_progress,
        "error": error,
    }


def _selected_gpu_dict(gpu) -> dict[str, Any]:
    return {
        "index": gpu.index,
        "uuid": gpu.uuid,
        "name": gpu.name,
        "compute_capability": gpu.compute_capability,
        "total_mib": gpu.total_mib,
        "free_mib": gpu.free_mib,
        "driver_version": gpu.driver_version,
    }


def runtime_hardware_status(
    setup: Any,
    *,
    worker_name: str | None = None,
) -> dict[str, Any]:
    """Verify recorded CUDA environments against the current preflighted GPU.

    Hardware identity is deliberately not part of the dependency hash because a
    compatible replacement GPU can reuse the same locked virtualenvs. CUDA
    runtime authorization is stricter: setup records the physical device UUID
    and NVIDIA driver version only after real kernel preflight succeeds. If the
    selected physical GPU or driver changes, direct model work fails closed until
    `persona setup --backend auto` reuses/rebuilds the appropriate environments
    and repeats that preflight.

    Seed-VC is checked separately because its audited Torch 2.4/cu124 stack has
    a different architecture envelope from the Torch 2.10 cu126/cu128 workers.
    This matters on hardware such as Blackwell: the main cu128 workers can remain
    valid while Seed-VC is intentionally synchronized as CPU.
    """

    value = setup if isinstance(setup, dict) else {}
    backend = value.get("irodori_backend")
    worker_backends = value.get("worker_backends")
    if not isinstance(worker_backends, dict):
        worker_backends = {}
    seed_vc_backend = worker_backends.get("seed_vc")

    main_cuda = backend in {"cu126", "cu128"}
    seed_cuda = worker_name == "seed_vc" and seed_vc_backend == "cu124"
    if not main_cuda and not seed_cuda:
        return {
            "ok": True,
            "backend": backend,
            "seed_vc_backend": seed_vc_backend,
            "selected_gpu": None,
            "preferred_backend": None,
            "preferred_seed_vc_backend": "cpu",
            "error": None,
        }

    gpu = selected_nvidia_gpu()
    if gpu is None:
        requested = [
            str(item)
            for item in (
                backend if main_cuda else None,
                seed_vc_backend if seed_cuda else None,
            )
            if item
        ]
        return {
            "ok": False,
            "backend": backend,
            "seed_vc_backend": seed_vc_backend,
            "selected_gpu": None,
            "preferred_backend": "cpu",
            "preferred_seed_vc_backend": "cpu",
            "error": (
                "PersonaVoice has CUDA environments recorded "
                f"({', '.join(requested)}), but no NVIDIA GPU is currently exposed as CUDA "
                "device 0. The GPU, driver, or CUDA_VISIBLE_DEVICES setting may have changed; "
                "run `persona setup --backend auto` before model work."
            ),
        }

    selected = _selected_gpu_dict(gpu)

    recorded_contract = value.get("environment_contract")
    strict_gpu_provenance = (
        isinstance(recorded_contract, dict)
        and isinstance(recorded_contract.get("schema"), int)
        and recorded_contract["schema"] >= ENVIRONMENT_CONTRACT_SCHEMA
    )
    if strict_gpu_provenance:
        recorded_gpu = value.get("selected_gpu")
        if not isinstance(recorded_gpu, dict):
            return {
                "ok": False,
                "backend": backend,
                "seed_vc_backend": seed_vc_backend,
                "selected_gpu": selected,
                "preferred_backend": cuda_backend_for_gpu(gpu),
                "preferred_seed_vc_backend": (
                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"
                ),
                "error": (
                    "The current CUDA setup state has no audited GPU provenance. "
                    "Run `persona setup --backend auto` before model work."
                ),
            }
        recorded_uuid = recorded_gpu.get("uuid")
        current_uuid = gpu.uuid
        if (
            not isinstance(recorded_uuid, str)
            or not recorded_uuid
            or not isinstance(current_uuid, str)
            or not current_uuid
            or recorded_uuid != current_uuid
        ):
            return {
                "ok": False,
                "backend": backend,
                "seed_vc_backend": seed_vc_backend,
                "selected_gpu": selected,
                "preferred_backend": cuda_backend_for_gpu(gpu),
                "preferred_seed_vc_backend": (
                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"
                ),
                "error": (
                    "The physical CUDA GPU selected as device 0 changed after setup. "
                    "Run `persona setup --backend auto` to rebuild/reuse the appropriate locked "
                    "environments and rerun the real CUDA kernel preflight before model work."
                ),
            }
        recorded_capability = recorded_gpu.get("compute_capability")
        current_capability = gpu.compute_capability
        if (
            not isinstance(recorded_capability, str)
            or not recorded_capability
            or not isinstance(current_capability, str)
            or not current_capability
            or recorded_capability != current_capability
        ):
            return {
                "ok": False,
                "backend": backend,
                "seed_vc_backend": seed_vc_backend,
                "selected_gpu": selected,
                "preferred_backend": cuda_backend_for_gpu(gpu),
                "preferred_seed_vc_backend": (
                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"
                ),
                "error": (
                    "The selected NVIDIA GPU compute capability changed after the CUDA "
                    "environments were preflighted. Run `persona setup --backend auto` to "
                    "rerun the real CUDA kernel preflight before model work."
                ),
            }
        recorded_driver = recorded_gpu.get("driver_version")
        current_driver = gpu.driver_version
        if (
            not isinstance(recorded_driver, str)
            or not recorded_driver
            or not isinstance(current_driver, str)
            or not current_driver
            or recorded_driver != current_driver
        ):
            return {
                "ok": False,
                "backend": backend,
                "seed_vc_backend": seed_vc_backend,
                "selected_gpu": selected,
                "preferred_backend": cuda_backend_for_gpu(gpu),
                "preferred_seed_vc_backend": (
                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"
                ),
                "error": (
                    "The NVIDIA driver version changed after the CUDA environments were "
                    "preflighted. Run `persona setup --backend auto` to rerun the real CUDA "
                    "kernel preflight before model work."
                ),
            }

    preferred = cuda_backend_for_gpu(gpu)
    preferred_seed = "cu124" if seed_vc_cuda_supported(gpu) else "cpu"

    if main_cuda and not backend_supports_gpu(str(backend), gpu):
        return {
            "ok": False,
            "backend": backend,
            "seed_vc_backend": seed_vc_backend,
            "selected_gpu": selected,
            "preferred_backend": preferred,
            "preferred_seed_vc_backend": preferred_seed,
            "error": (
                f"PersonaVoice was set up for {backend}, but the current CUDA-visible GPU "
                f"{gpu.name} (compute capability {gpu.compute_capability or 'unknown'}) is not "
                "supported by that audited PyTorch wheel. The GPU selection appears to have "
                "changed; run `persona setup --backend auto` before model work."
            ),
        }

    if seed_cuda and not seed_vc_cuda_supported(gpu):
        return {
            "ok": False,
            "backend": backend,
            "seed_vc_backend": seed_vc_backend,
            "selected_gpu": selected,
            "preferred_backend": preferred,
            "preferred_seed_vc_backend": preferred_seed,
            "error": (
                f"Seed-VC was set up for {seed_vc_backend}, but the current CUDA-visible GPU "
                f"{gpu.name} (compute capability {gpu.compute_capability or 'unknown'}) is not "
                "supported by the audited Torch 2.4/cu124 Seed-VC stack. The GPU selection "
                "appears to have changed; run `persona setup --backend auto` to rebuild only "
                "the required worker environments safely."
            ),
        }

    return {
        "ok": True,
        "backend": backend,
        "seed_vc_backend": seed_vc_backend,
        "selected_gpu": selected,
        "preferred_backend": preferred,
        "preferred_seed_vc_backend": preferred_seed,
        "error": None,
    }


def require_current_environment(
    repo_root: Path,
    *,
    worker_name: str | None = None,
) -> dict[str, Any]:
    """Return setup state only when dependency and current-hardware contracts are valid."""

    setup_path = repo_root / ".runtime" / "setup.json"
    if not setup_path.is_file():
        raise FileNotFoundError(
            "PersonaVoice setup state is missing. Run `persona setup` before model work."
        )
    try:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"PersonaVoice setup state is unreadable: {setup_path}. Re-run `persona setup`."
        ) from exc
    if not isinstance(setup, dict):
        raise RuntimeError(
            f"PersonaVoice setup state has an invalid format: {setup_path}. Re-run `persona setup`."
        )

    status = environment_contract_status(repo_root, setup.get("environment_contract"))
    if not status["ok"]:
        raise RuntimeError(str(status["error"]))
    hardware = runtime_hardware_status(setup, worker_name=worker_name)
    if not hardware["ok"]:
        raise RuntimeError(str(hardware["error"]))
    return setup
