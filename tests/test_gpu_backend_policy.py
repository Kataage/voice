from __future__ import annotations

import json

import pytest

from personavoice import environment_contract as environment
from personavoice import hardware, setup_env


def gpu(capability: str, *, index: int = 0, mib: int = 16384, uuid: str | None = None):
    return hardware.GpuInfo(
        index=index,
        uuid=uuid or f"GPU-{index:04d}",
        name=f"NVIDIA test sm_{capability.replace('.', '')}",
        total_mib=mib,
        free_mib=max(1, mib - 1024),
        compute_capability=capability,
    )


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ("5.2", "cpu"),
        ("6.0", "cu126"),
        ("6.1", "cu126"),
        ("7.0", "cu126"),
        ("7.5", "cu128"),
        ("8.0", "cu128"),
        ("8.6", "cu128"),
        ("8.9", "cu128"),
        ("9.0", "cu128"),
        ("10.0", "cu128"),
        ("10.3", "cpu"),
        ("12.0", "cu128"),
        ("12.1", "cpu"),
        ("13.0", "cpu"),
    ],
)
def test_auto_cuda_backend_matches_audited_x86_wheels(monkeypatch, capability: str, expected: str):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "AMD64")
    assert hardware.cuda_backend_for_gpu(gpu(capability)) == expected


def test_volta_is_not_sent_to_cu128(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    volta = gpu("7.0")
    assert hardware.backend_supports_gpu("cu126", volta)
    assert not hardware.backend_supports_gpu("cu128", volta)


def test_blackwell_is_not_sent_to_legacy_cu126_or_seed_vc_cu124(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    blackwell = gpu("12.0")
    assert not hardware.backend_supports_gpu("cu126", blackwell)
    assert hardware.backend_supports_gpu("cu128", blackwell)
    assert not hardware.seed_vc_cuda_supported(blackwell)
    assert hardware.seed_vc_cuda_supported(gpu("9.0"))


def test_non_x86_cuda_fails_closed_until_full_stack_is_ci_audited(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "aarch64")
    assert not hardware.backend_supports_gpu("cu126", gpu("8.0"))
    assert not hardware.backend_supports_gpu("cu128", gpu("10.0"))
    assert hardware.cuda_backend_for_gpu(gpu("9.0")) == "cpu"
    assert not hardware.seed_vc_cuda_supported(gpu("9.0"))


def test_cuda_visible_devices_numeric_mapping_selects_logical_device_zero(monkeypatch):
    gpus = [gpu("9.0", index=0), gpu("6.1", index=1)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert hardware.selected_nvidia_gpu(gpus) == gpus[1]


def test_cuda_visible_devices_uuid_mapping_selects_logical_device_zero(monkeypatch):
    gpus = [
        gpu("9.0", index=0, uuid="GPU-aaaaaaaa-bbbb"),
        gpu("6.1", index=1, uuid="GPU-cccccccc-dddd"),
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cccccccc, GPU-aaaaaaaa")
    assert hardware.selected_nvidia_gpu(gpus) == gpus[1]


def test_cuda_visible_devices_hidden_or_invalid_fails_closed(monkeypatch):
    gpus = [gpu("8.6")]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert hardware.selected_nvidia_gpu(gpus) is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-unsupported")
    assert hardware.selected_nvidia_gpu(gpus) is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    assert hardware.selected_nvidia_gpu(gpus) is None


def test_safe_batch_profile_uses_selected_gpu_not_largest_physical_gpu(monkeypatch):
    gpus = [gpu("9.0", index=0, mib=49152), gpu("8.6", index=1, mib=8192)]
    monkeypatch.setattr(hardware, "nvidia_gpus", lambda: gpus)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    profile = hardware.safe_batch_profile(backend="cu128")
    assert profile["batch_size"] == 1
    assert profile["gradient_accumulation_steps"] == 12


def test_explicit_cuda_backend_validates_current_selected_gpu(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    volta = gpu("7.0")
    monkeypatch.setattr(setup_env, "selected_nvidia_gpu", lambda: volta)
    assert setup_env._validate_cuda_backend("cu126") == volta
    with pytest.raises(ValueError, match="--backend auto"):
        setup_env._validate_cuda_backend("cu128")

    blackwell = gpu("12.0")
    monkeypatch.setattr(setup_env, "selected_nvidia_gpu", lambda: blackwell)
    assert setup_env._validate_cuda_backend("cu128") == blackwell
    with pytest.raises(ValueError, match="--backend auto"):
        setup_env._validate_cuda_backend("cu126")


def test_explicit_cuda_backend_requires_visible_gpu(monkeypatch):
    monkeypatch.setattr(setup_env, "selected_nvidia_gpu", lambda: None)
    with pytest.raises(ValueError, match="no NVIDIA GPU"):
        setup_env._validate_cuda_backend("cu126")
    with pytest.raises(ValueError, match="no NVIDIA GPU"):
        setup_env._validate_cuda_backend("cu128")
    assert setup_env._validate_cuda_backend("cpu") is None


def test_seed_vc_falls_back_to_cpu_only_when_legacy_stack_cannot_run(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    hopper = setup_env._worker_extras("cu128", gpu=gpu("9.0"))
    blackwell = setup_env._worker_extras("cu128", gpu=gpu("12.0"))
    assert hopper["seed_vc"] == "cu124"
    assert blackwell["seed_vc"] == "cpu"
    assert blackwell["diarization"] == "cu128"
    assert blackwell["sense"] == "cu128"
    assert blackwell["lfm"] == "cu128"


def _write_setup(tmp_path, backend: str, *, seed_vc_backend: str | None = None) -> dict:
    runtime = tmp_path / ".runtime"
    runtime.mkdir(parents=True)
    setup = {
        "irodori_backend": backend,
        "environment_contract": environment.environment_contract(tmp_path),
    }
    if seed_vc_backend is not None:
        setup["worker_backends"] = {"seed_vc": seed_vc_backend}
    (runtime / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    return setup


def test_direct_runtime_rejects_incompatible_gpu_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    _write_setup(tmp_path, "cu128")
    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("7.0"))
    with pytest.raises(RuntimeError, match="GPU selection appears to have changed"):
        environment.require_current_environment(tmp_path)


def test_direct_runtime_accepts_compatible_gpu_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    setup = _write_setup(tmp_path, "cu126")
    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("8.6"))
    assert environment.require_current_environment(tmp_path) == setup


def test_direct_runtime_rejects_legacy_seed_vc_after_blackwell_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    _write_setup(tmp_path, "cu128", seed_vc_backend="cu124")
    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("12.0"))
    with pytest.raises(RuntimeError, match="Seed-VC was set up for cu124"):
        environment.require_current_environment(tmp_path)


def test_direct_runtime_accepts_blackwell_when_seed_vc_is_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    setup = _write_setup(tmp_path, "cu128", seed_vc_backend="cpu")
    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("12.0"))
    assert environment.require_current_environment(tmp_path) == setup


def test_cpu_setup_remains_safe_when_gpu_changes(tmp_path, monkeypatch):
    setup = _write_setup(tmp_path, "cpu")
    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("12.0"))
    assert environment.require_current_environment(tmp_path) == setup
