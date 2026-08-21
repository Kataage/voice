from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import inference, setup_env
from personavoice.environment_contract import (
    SETUP_TRANSACTION_MARKER,
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
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "seed_vc_assets.json").write_text(
        json.dumps({"schema_version": 1, "snapshots": {}}),
        encoding="utf-8",
    )
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
    with pytest.raises(RuntimeError, match="different dependency contract"):
        require_current_environment(tmp_path)

    # A successful setup records the new exact generation and restores runtime readiness.
    _record_setup(tmp_path)
    assert require_current_environment(tmp_path)["irodori_backend"] == "cpu"


def test_setup_transaction_marker_blocks_old_setup_state(tmp_path: Path):
    _dependency_tree(tmp_path)
    _record_setup(tmp_path, backend="cpu")
    marker = tmp_path / ".runtime" / SETUP_TRANSACTION_MARKER
    marker.write_text(json.dumps({"irodori_backend": "cu128"}), encoding="utf-8")

    status = environment_contract_status(
        tmp_path,
        json.loads((tmp_path / ".runtime" / "setup.json").read_text(encoding="utf-8"))[
            "environment_contract"
        ],
    )
    assert not status["ok"]
    assert status["setup_in_progress"] is True
    with pytest.raises(RuntimeError, match="setup transaction is incomplete"):
        require_current_environment(tmp_path)


def test_failed_setup_keeps_transaction_marker_and_blocks_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _dependency_tree(tmp_path)
    _record_setup(tmp_path, backend="cpu")
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: "/tool")

    def fail_clone(*_args, **_kwargs):
        raise RuntimeError("simulated setup failure")

    monkeypatch.setattr(setup_env, "_clone_pinned", fail_clone)
    with pytest.raises(RuntimeError, match="simulated setup failure"):
        setup_env.install_environments(tmp_path, backend="cu128")

    marker = tmp_path / ".runtime" / SETUP_TRANSACTION_MARKER
    assert marker.is_file()
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_value["irodori_backend"] == "cu128"
    # The previous committed setup stays on disk for provenance, but the marker
    # prevents it from authorizing the now potentially mixed environments.
    assert json.loads((tmp_path / ".runtime" / "setup.json").read_text(encoding="utf-8"))[
        "irodori_backend"
    ] == "cpu"
    with pytest.raises(RuntimeError, match="setup transaction is incomplete"):
        require_current_environment(tmp_path)


def test_successful_setup_commits_state_then_clears_transaction_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _dependency_tree(tmp_path)
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: "/tool")
    irodori = tmp_path / "vendor" / "Irodori-TTS"
    seed = tmp_path / "vendor" / "seed-vc"
    irodori.mkdir(parents=True)
    seed.mkdir(parents=True)

    def fake_clone(_root, name, _url, _revision):
        return irodori if name == "Irodori-TTS" else seed

    synced: list[tuple[str, str | None]] = []

    class FakeWorker:
        def __init__(self, name: str):
            self.name = name

        def sync(self, _root: Path, *, extra: str | None = None) -> None:
            synced.append((self.name, extra))

    monkeypatch.setattr(setup_env, "_clone_pinned", fake_clone)
    monkeypatch.setattr(setup_env, "_install_irodori", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(setup_env, "worker", lambda _root, name: FakeWorker(name))

    result = setup_env.install_environments(tmp_path, backend="cu128")

    assert result["irodori_backend"] == "cu128"
    assert len(synced) == 5
    assert ("lfm", "cu128") in synced
    assert ("seed_vc", "cu124") in synced
    assert not (tmp_path / ".runtime" / SETUP_TRANSACTION_MARKER).exists()
    assert require_current_environment(tmp_path)["irodori_backend"] == "cu128"


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
