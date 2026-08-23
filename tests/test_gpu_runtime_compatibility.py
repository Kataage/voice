from __future__ import annotations

import types
from pathlib import Path

import pytest

from personavoice import doctor, hardware, setup_env, workers
from personavoice import environment_contract as env_contract


def _gpu(
    index: int,
    capability: str,
    *,
    uuid: str | None = None,
    pci_bus_id: str | None = None,
    name: str = "GPU",
):
    return hardware.GpuInfo(
        index=index,
        name=name,
        total_mib=16384,
        free_mib=12000,
        compute_capability=capability,
        uuid=uuid,
        pci_bus_id=pci_bus_id,
    )


def test_nvidia_query_parses_uuid_pci_bus_and_compute_capability(monkeypatch):
    calls = []

    def fake_query(fields: str):
        calls.append(fields)
        return types.SimpleNamespace(
            returncode=0,
            stdout="7, GPU-abcdef, 00000000:65:00.0, NVIDIA RTX, 24576, 23000, 8.6\n",
        )

    monkeypatch.setattr(hardware, "_run_nvidia_query", fake_query)
    assert hardware.nvidia_gpus() == [
        hardware.GpuInfo(
            index=7,
            uuid="GPU-abcdef",
            pci_bus_id="00000000:65:00.0",
            name="NVIDIA RTX",
            total_mib=24576,
            free_mib=23000,
            compute_capability="8.6",
        )
    ]
    assert calls == ["index,uuid,pci.bus_id,name,memory.total,memory.free,compute_cap"]


def test_selected_gpu_honors_cuda_visible_devices_numeric_pci_order(monkeypatch):
    pascal = _gpu(7, "6.1", pci_bus_id="00000000:01:00.0")
    modern = _gpu(2, "8.6", pci_bus_id="00000000:65:00.0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert hardware.selected_nvidia_gpu([modern, pascal]) == modern


def test_selected_gpu_honors_cuda_visible_devices_uuid_prefix(monkeypatch):
    gpus = [
        _gpu(0, "6.1", uuid="GPU-aaaaaaaa-bbbb"),
        _gpu(1, "8.6", uuid="GPU-cccccccc-dddd"),
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cccc")
    assert hardware.selected_nvidia_gpu(gpus) == gpus[1]


@pytest.mark.parametrize("selector", ["", "-1", "MIG-deadbeef", "GPU-not-present", "999"])
def test_selected_gpu_fails_closed_for_hidden_or_unresolvable_device(monkeypatch, selector):
    gpus = [_gpu(0, "8.6", uuid="GPU-aaaaaaaa-bbbb")]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selector)
    assert hardware.selected_nvidia_gpu(gpus) is None


def test_selected_gpu_defaults_to_pci_bus_order_not_nvidia_index(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    first_pci = _gpu(9, "6.1", pci_bus_id="00000000:01:00.0")
    lower_nvidia_index = _gpu(1, "8.6", pci_bus_id="00000000:65:00.0")
    assert hardware.selected_nvidia_gpu([lower_nvidia_index, first_pci]) == first_pci


def test_worker_environment_forces_same_cuda_pci_order(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(workers, "ffmpeg_environment", lambda: {})
    env = workers.local_model_env(tmp_path)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_asr_honors_recorded_cpu_policy_and_cuda_runtime_fallback_policy():
    assert workers._asr_device_environment({"irodori_backend": "cpu"}) == {
        "CUDA_VISIBLE_DEVICES": ""
    }
    assert workers._asr_device_environment({"irodori_backend": "rocm"}) == {
        "CUDA_VISIBLE_DEVICES": ""
    }
    assert workers._asr_device_environment({"irodori_backend": "cu126"}) == {}
    assert workers._asr_device_environment({"irodori_backend": "cu128"}) == {}
    assert doctor._expected_worker_backend("asr", {"irodori_backend": "cpu"}) == "cpu"
    assert (
        doctor._expected_worker_backend("asr", {"irodori_backend": "cu126"})
        == "runtime-auto"
    )


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ("5.2", "cpu"),
        ("6.1", "cu126"),
        ("7.0", "cu126"),
        ("7.5", "cu128"),
        ("8.6", "cu128"),
        ("8.9", "cu128"),
        ("9.0", "cu128"),
        ("10.0", "cu128"),
        ("12.0", "cu128"),
        ("13.0", "cpu"),
    ],
)
def test_x86_gpu_generation_selects_only_audited_backend(monkeypatch, capability, expected):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware.cuda_backend_for_gpu(_gpu(0, capability)) == expected


def test_seed_vc_falls_back_on_blackwell_but_not_hopper(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware.seed_vc_cuda_supported(_gpu(0, "9.0"))
    assert not hardware.seed_vc_cuda_supported(_gpu(0, "12.0"))


def test_runtime_hardware_allows_legacy_wheel_on_newer_compatible_gpu(monkeypatch):
    ampere = _gpu(0, "8.6", name="RTX-class GPU")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: ampere)
    status = env_contract.runtime_hardware_status(
        {
            "irodori_backend": "cu126",
            "worker_backends": {"seed_vc": "cu124"},
        }
    )
    assert status["ok"] is True
    assert status["preferred_backend"] == "cu128"


def test_runtime_hardware_rejects_modern_wheel_after_swap_to_pascal(monkeypatch):
    pascal = _gpu(0, "6.1", name="GTX 1080 Ti")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: pascal)
    status = env_contract.runtime_hardware_status(
        {
            "irodori_backend": "cu128",
            "worker_backends": {"seed_vc": "cu124"},
        }
    )
    assert status["ok"] is False
    assert status["preferred_backend"] == "cu126"
    assert "persona setup --backend auto" in status["error"]


def test_runtime_hardware_rejects_cuda_state_when_gpu_disappears(monkeypatch):
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: None)
    status = env_contract.runtime_hardware_status(
        {
            "irodori_backend": "cu126",
            "worker_backends": {"seed_vc": "cu124"},
        }
    )
    assert status["ok"] is False
    assert status["preferred_backend"] == "cpu"


def test_runtime_hardware_blocks_only_stale_seed_vc_stack_on_blackwell(monkeypatch):
    blackwell = _gpu(0, "12.0", name="Blackwell GPU")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: blackwell)
    setup = {
        "irodori_backend": "cu128",
        "worker_backends": {"seed_vc": "cu124"},
    }
    assert env_contract.runtime_hardware_status(setup)["ok"] is True
    status = env_contract.runtime_hardware_status(setup, worker_name="seed_vc")
    assert status["ok"] is False
    assert status["preferred_backend"] == "cu128"
    assert status["preferred_seed_vc_backend"] == "cpu"
    assert "Seed-VC" in status["error"]


def test_runtime_hardware_cpu_setup_is_portable_across_gpu_changes(monkeypatch):
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: _gpu(0, "12.0"))
    status = env_contract.runtime_hardware_status(
        {
            "irodori_backend": "cpu",
            "worker_backends": {"seed_vc": "cpu"},
        }
    )
    assert status["ok"] is True


def test_explicit_cuda_backend_requires_current_visible_gpu(monkeypatch):
    monkeypatch.setattr(setup_env, "selected_nvidia_gpu", lambda: None)
    with pytest.raises(ValueError, match="requires an NVIDIA GPU"):
        setup_env._validate_cuda_backend("cu126")
    with pytest.raises(ValueError, match="requires an NVIDIA GPU"):
        setup_env._validate_cuda_backend("cu128")


def test_explicit_cu128_rejects_volta_but_cu126_accepts_it(monkeypatch):
    volta = _gpu(0, "7.0", name="V100")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(setup_env, "selected_nvidia_gpu", lambda: volta)
    with pytest.raises(ValueError, match="--backend auto"):
        setup_env._validate_cuda_backend("cu128")
    assert setup_env._validate_cuda_backend("cu126") == volta
