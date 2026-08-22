from __future__ import annotations

import pytest

from personavoice import environment_contract as env_contract
from personavoice import hardware, setup_env


def _gpu(index: int, capability: str, *, uuid: str | None = None, name: str = "GPU"):
    return hardware.GpuInfo(
        index=index,
        name=name,
        total_mib=16384,
        free_mib=12000,
        compute_capability=capability,
        uuid=uuid,
    )


def test_selected_gpu_honors_cuda_visible_devices_numeric_order(monkeypatch):
    gpus = [_gpu(0, "6.1"), _gpu(1, "8.6")]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert hardware.selected_nvidia_gpu(gpus) == gpus[1]


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


def test_selected_gpu_defaults_to_lowest_physical_index(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    high = _gpu(4, "8.6")
    low = _gpu(1, "6.1")
    assert hardware.selected_nvidia_gpu([high, low]) == low


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


def test_runtime_hardware_blocks_stale_seed_vc_gpu_stack_on_blackwell(monkeypatch):
    blackwell = _gpu(0, "12.0", name="Blackwell GPU")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: blackwell)
    status = env_contract.runtime_hardware_status(
        {
            "irodori_backend": "cu128",
            "worker_backends": {"seed_vc": "cu124"},
        }
    )
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
