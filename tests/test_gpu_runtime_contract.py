from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import environment_contract as env_contract
from personavoice import hardware, setup_env
from personavoice.workers import local_model_env


def _gpu(
    capability: str,
    *,
    index: int = 0,
    uuid: str = "GPU-a",
    pci: str = "00000000:01:00.0",
    memory: int = 16384,
) -> hardware.GpuInfo:
    return hardware.GpuInfo(
        index=index,
        name=f"GPU-{capability}",
        total_mib=memory,
        free_mib=memory - 512,
        compute_capability=capability,
        uuid=uuid,
        pci_bus_id=pci,
    )


def test_cuda_visible_devices_numeric_order_matches_forced_pci_order(monkeypatch):
    first_pci = _gpu("8.6", index=7, uuid="GPU-first", pci="00000000:01:00.0")
    second_pci = _gpu("6.1", index=2, uuid="GPU-second", pci="00000000:09:00.0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert hardware.selected_nvidia_gpu([second_pci, first_pci]) == second_pci


def test_cuda_visible_devices_uuid_selects_exact_physical_gpu(monkeypatch):
    first = _gpu("8.6", index=0, uuid="GPU-aaaaaaaa", pci="00000000:01:00.0")
    second = _gpu("6.1", index=1, uuid="GPU-bbbbbbbb", pci="00000000:02:00.0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
    assert hardware.selected_nvidia_gpu([first, second]) == second


def test_unknown_or_mig_cuda_selector_fails_closed(monkeypatch):
    gpu = _gpu("8.6")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-unknown")
    assert hardware.selected_nvidia_gpu([gpu]) is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-a-device")
    assert hardware.selected_nvidia_gpu([gpu]) is None


def test_gpu_generation_policy_is_conservative_and_blackwell_aware(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware.cuda_backend_for_gpu(_gpu("6.1")) == "cu126"
    assert hardware.cuda_backend_for_gpu(_gpu("7.0")) == "cu126"
    assert hardware.cuda_backend_for_gpu(_gpu("7.5")) == "cu128"
    assert hardware.cuda_backend_for_gpu(_gpu("9.0")) == "cu128"
    assert hardware.cuda_backend_for_gpu(_gpu("10.0")) == "cu128"
    assert hardware.cuda_backend_for_gpu(_gpu("12.0")) == "cu128"
    assert hardware.cuda_backend_for_gpu(_gpu("13.0")) == "cpu"


def test_batch_profile_uses_logical_cuda_device_zero_not_largest_gpu(monkeypatch):
    small = _gpu("8.6", index=0, uuid="GPU-small", pci="00000000:01:00.0", memory=11000)
    huge = _gpu("8.6", index=1, uuid="GPU-huge", pci="00000000:02:00.0", memory=65536)
    monkeypatch.setattr(hardware, "nvidia_gpus", lambda: [small, huge])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert hardware.safe_batch_profile(backend="cu128")["batch_size"] == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert hardware.safe_batch_profile(backend="cu128")["batch_size"] == 12


def test_compatible_gpu_swap_keeps_main_runtime_available(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: _gpu("8.6", uuid="GPU-new"))
    setup = {
        "irodori_backend": "cu128",
        "worker_backends": {"seed_vc": "cu124"},
    }
    assert env_contract.runtime_hardware_status(setup)["ok"] is True


def test_blackwell_swap_only_blocks_seed_vc_worker(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: _gpu("12.0", uuid="GPU-new"))
    setup = {
        "irodori_backend": "cu128",
        "worker_backends": {"seed_vc": "cu124"},
    }
    assert env_contract.runtime_hardware_status(setup)["ok"] is True
    seed = env_contract.runtime_hardware_status(setup, worker_name="seed_vc")
    assert seed["ok"] is False
    assert seed["preferred_seed_vc_backend"] == "cpu"


def test_pascal_swap_from_cu128_requires_resync(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: _gpu("6.1"))
    setup = {"irodori_backend": "cu128", "worker_backends": {"seed_vc": "cu124"}}
    status = env_contract.runtime_hardware_status(setup)
    assert status["ok"] is False
    assert status["preferred_backend"] == "cu126"
    assert "persona setup --backend auto" in status["error"]


def test_blackwell_setup_keeps_main_cuda_and_falls_seed_vc_back_to_cpu(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    gpu = _gpu("12.0")
    extras = setup_env._worker_extras("cu128", gpu=gpu)
    assert extras["diarization"] == "cu128"
    assert extras["sense"] == "cu128"
    assert extras["lfm"] == "cu128"
    assert extras["seed_vc"] == "cpu"


def test_local_model_env_forces_deterministic_cuda_order(tmp_path: Path):
    env = local_model_env(tmp_path)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_environment_contract_includes_runtime_policy_sources(tmp_path: Path):
    for relative in (
        "src/personavoice/hardware.py",
        "src/personavoice/setup_env.py",
        "src/personavoice/runtime_dependencies.py",
        "src/personavoice/workers.py",
        "workers/asr/runtime_policy.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("v1", encoding="utf-8")
    recorded = env_contract.environment_contract(tmp_path)
    assert recorded["schema"] == 4
    (tmp_path / "src/personavoice/hardware.py").write_text("v2", encoding="utf-8")
    status = env_contract.environment_contract_status(tmp_path, recorded)
    assert status["ok"] is False
    assert "different dependency contract" in str(status["error"])


def test_setup_state_records_gpu_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # This is a focused publication test; environment mutation is stubbed.
    gpu = _gpu("8.6", uuid="GPU-recorded")
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: "/tool")
    monkeypatch.setattr(setup_env, "require_ffmpeg_runtime", lambda: None)
    monkeypatch.setattr(setup_env, "_validate_cuda_backend", lambda _backend: gpu)
    monkeypatch.setattr(setup_env, "seed_vc_contract_digest", lambda _root: "0" * 64)
    monkeypatch.setattr(
        setup_env,
        "run_cuda_preflight",
        lambda *_args, **_kwargs: {"ok": True, "fixture": True},
    )
    irodori = tmp_path / "vendor/Irodori-TTS"
    seed = tmp_path / "vendor/seed-vc"
    irodori.mkdir(parents=True)
    seed.mkdir(parents=True)
    monkeypatch.setattr(
        setup_env,
        "_clone_pinned",
        lambda _root, name, _url, _revision: irodori if name == "Irodori-TTS" else seed,
    )
    monkeypatch.setattr(setup_env, "_install_irodori", lambda *_args, **_kwargs: None)

    class FakeWorker:
        def sync(self, _root: Path, *, extra: str | None = None) -> None:
            pass

    monkeypatch.setattr(setup_env, "worker", lambda *_args: FakeWorker())
    result = setup_env.install_environments(tmp_path, backend="cu128")
    assert result["selected_gpu"]["uuid"] == "GPU-recorded"
    recorded = json.loads((tmp_path / ".runtime/setup.json").read_text(encoding="utf-8"))
    assert recorded["selected_gpu"]["compute_capability"] == "8.6"
