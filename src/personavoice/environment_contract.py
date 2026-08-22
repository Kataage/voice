from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from personavoice.hardware import backend_supports_gpu, cuda_backend_for_gpu, selected_nvidia_gpu

WORKER_NAMES = ("asr", "diarization", "sense", "lfm", "seed_vc")
ENVIRONMENT_CONTRACT_SCHEMA = 3
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


def runtime_hardware_status(setup: Any) -> dict[str, Any]:
    """Verify that a recorded CUDA environment is safe for the current visible GPU.

    GPU hardware is intentionally not part of the dependency hash: a compatible
    GPU swap should keep working without reinstalling identical environments.
    Incompatible swaps, GPU removal, or CUDA visibility changes fail closed at
    every direct worker entry point and instruct the user to rerun auto setup.
    """

    value = setup if isinstance(setup, dict) else {}
    backend = value.get("irodori_backend")
    if backend not in {"cu126", "cu128"}:
        return {"ok": True, "backend": backend, "selected_gpu": None, "preferred_backend": None}

    gpu = selected_nvidia_gpu()
    if gpu is None:
        return {
            "ok": False,
            "backend": backend,
            "selected_gpu": None,
            "preferred_backend": "cpu",
            "error": (
                f"PersonaVoice was set up for {backend}, but no NVIDIA GPU is currently exposed "
                "as CUDA device 0. The GPU, driver, or CUDA_VISIBLE_DEVICES setting may have "
                "changed; run `persona setup --backend auto` before model work."
            ),
        }

    selected = {
        "index": gpu.index,
        "uuid": gpu.uuid,
        "name": gpu.name,
        "compute_capability": gpu.compute_capability,
        "total_mib": gpu.total_mib,
        "free_mib": gpu.free_mib,
    }
    preferred = cuda_backend_for_gpu(gpu)
    if not backend_supports_gpu(str(backend), gpu):
        return {
            "ok": False,
            "backend": backend,
            "selected_gpu": selected,
            "preferred_backend": preferred,
            "error": (
                f"PersonaVoice was set up for {backend}, but the current CUDA-visible GPU "
                f"{gpu.name} (compute capability {gpu.compute_capability or 'unknown'}) is not "
                "supported by that audited PyTorch wheel. The GPU selection appears to have "
                "changed; run `persona setup --backend auto` before model work."
            ),
        }
    return {
        "ok": True,
        "backend": backend,
        "selected_gpu": selected,
        "preferred_backend": preferred,
        "error": None,
    }


def require_current_environment(repo_root: Path) -> dict[str, Any]:
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
    hardware = runtime_hardware_status(setup)
    if not hardware["ok"]:
        raise RuntimeError(str(hardware["error"]))
    return setup
