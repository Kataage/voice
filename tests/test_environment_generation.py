from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import inference, setup_env
from personavoice.environment_contract import (
    environment_contract,
    environment_contract_status,
    require_current_environment,
)
from personavoice.project import PersonaPaths


def _write(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _dependency_tree(root: Path) -> None:
    _write(root / "pyproject.toml", b"root")
    _write(root / "uv.lock", b"root-lock")
    _write(root / "locks" / "Irodori-TTS.uv.lock", b"irodori-lock")
    for name in ("asr", "diarization", "sense", "lfm", "seed_vc"):
        _write(root / "workers" / name / "pyproject.toml", name.encode())
        _write(root / "workers" / name / "uv.lock", f"{name}-lock".encode())


def _record_setup(root: Path, *, backend: str = "cpu") -> None:
    runtime = root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "setup.json").write_text(
        json.dumps(
            {
                "irodori_backend": backend,
                "environment_contract": environment_contract(root),
            }
        ),
        encoding="utf-8",
    )


def test_environment_contract_detects_dependency_generation_change(tmp_path: Path):
    _dependency_tree(tmp_path)
    recorded = environment_contract(tmp_path)
    assert environment_contract_status(tmp_path, recorded)["ok"]

    (tmp_path / "workers" / "lfm" / "uv.lock").write_bytes(b"new-lock")
    status = environment_contract_status(tmp_path, recorded)
    assert not status["ok"]
    assert "different dependency contract" in status["error"]


def test_runtime_environment_contract_accepts_current_rejects_stale_and_recovers(
    tmp_path: Path,
):
    _dependency_tree(tmp_path)
    _record_setup(tmp_path)
    assert require_current_environment(tmp_path)["irodori_backend"] == "cpu"

    # A dependency declaration changing after setup must invalidate every --no-sync runtime.
    (tmp_path / "workers" / "lfm" / "uv.lock").write_bytes(b"new-lock")
    with pytest.raises(RuntimeError, match="environments are stale"):
        require_current_environment(tmp_path)

    # A successful setup records the new exact generation and restores runtime readiness.
    _record_setup(tmp_path)
    assert require_current_environment(tmp_path)["irodori_backend"] == "cpu"


def test_irodori_install_refuses_missing_managed_lock(tmp_path: Path):
    vendor = tmp_path / "vendor" / "Irodori-TTS"
    vendor.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Audited Irodori lockfile is missing"):
        setup_env._install_irodori(tmp_path, vendor, "cpu")


def test_interrupted_irodori_lock_swap_rejects_vendor_head_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vendor = tmp_path / "vendor" / "Irodori-TTS"
    vendor.mkdir(parents=True)
    _write(vendor / "uv.lock", b"managed")
    marker = tmp_path / ".runtime" / setup_env.IRODORI_LOCK_SWAP_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "vendor_head": "old-head",
                "original_exists": True,
                "original_sha256": "unused",
                "managed_sha256": setup_env.sha256_file(vendor / "uv.lock"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(setup_env, "_git_head", lambda _path: "new-head")

    with pytest.raises(RuntimeError, match="different vendor HEAD"):
        setup_env._recover_irodori_lock_swap(tmp_path, vendor)
    assert marker.exists()


def test_interrupted_irodori_lock_swap_restores_known_managed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vendor = tmp_path / "vendor" / "Irodori-TTS"
    vendor.mkdir(parents=True)
    lock = _write(vendor / "uv.lock", b"managed")
    marker = tmp_path / ".runtime" / setup_env.IRODORI_LOCK_SWAP_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "vendor_head": "head",
                "original_exists": True,
                "original_sha256": "original-hash",
                "managed_sha256": setup_env.sha256_file(lock),
            }
        ),
        encoding="utf-8",
    )
    restored: list[Path] = []
    monkeypatch.setattr(setup_env, "_git_head", lambda _path: "head")
    monkeypatch.setattr(setup_env, "_restore_vendor_lock", lambda path: restored.append(path))

    setup_env._recover_irodori_lock_swap(tmp_path, vendor)
    assert restored == [vendor]
    assert not marker.exists()


def test_best_irodori_adapter_ignores_partial_directories(tmp_path: Path):
    paths = PersonaPaths(tmp_path / "personas" / "alice")
    root = paths.models / "irodori" / "lora"
    partial_best = root / "checkpoint_best_val_loss_0.01"
    _write(partial_best / "adapter_config.json", b"{}")
    final = root / "checkpoint_final"
    _write(final / "adapter_config.json", b"{}")
    _write(final / "adapter_model.safetensors", b"weights")

    assert inference._best_lora_adapter(paths) == final
