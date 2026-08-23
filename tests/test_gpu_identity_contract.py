from __future__ import annotations

from types import SimpleNamespace

from personavoice import environment_contract as env_contract
from personavoice import hardware


def _gpu(*, uuid="GPU-a", driver="600.01", capability="8.6"):
    return hardware.GpuInfo(
        index=0,
        name="NVIDIA test GPU",
        total_mib=24576,
        free_mib=22000,
        compute_capability=capability,
        uuid=uuid,
        pci_bus_id="00000000:01:00.0",
        driver_version=driver,
    )


def _setup(gpu):
    return {
        "irodori_backend": "cu128",
        "worker_backends": {"seed_vc": "cu124"},
        "selected_gpu": hardware.gpu_record(gpu),
        "environment_contract": {"schema": env_contract.ENVIRONMENT_CONTRACT_SCHEMA},
    }


def test_runtime_accepts_same_preflighted_gpu_and_driver(monkeypatch):
    gpu = _gpu()
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: gpu)
    assert env_contract.runtime_hardware_status(_setup(gpu))["ok"] is True


def test_runtime_rejects_physical_gpu_replacement_until_preflight(monkeypatch):
    setup_gpu = _gpu(uuid="GPU-old")
    current = _gpu(uuid="GPU-new")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: current)
    status = env_contract.runtime_hardware_status(_setup(setup_gpu))
    assert status["ok"] is False
    assert "physical CUDA GPU" in status["error"]
    assert "persona setup --backend auto" in status["error"]


def test_runtime_rejects_driver_change_until_kernel_preflight(monkeypatch):
    setup_gpu = _gpu(driver="600.01")
    current = _gpu(driver="601.02")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: current)
    status = env_contract.runtime_hardware_status(_setup(setup_gpu))
    assert status["ok"] is False
    assert "driver version changed" in status["error"]
    assert "persona setup --backend auto" in status["error"]


def test_gpu_query_records_driver_version(monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            "0, GPU-abc, 00000000:01:00.0, NVIDIA Test, 24576, 22000, "
            "600.01, 8.6\n"
        ),
    )
    monkeypatch.setattr(hardware, "_run_nvidia_query", lambda _fields: completed)
    found = hardware.nvidia_gpus()
    assert len(found) == 1
    assert found[0].driver_version == "600.01"
    assert found[0].compute_capability == "8.6"


def test_gpu_record_persists_driver_and_uuid():
    record = hardware.gpu_record(_gpu(uuid="GPU-x", driver="602.03"))
    assert record["uuid"] == "GPU-x"
    assert record["driver_version"] == "602.03"
