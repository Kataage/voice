from __future__ import annotations

import pytest

from personavoice import hardware, setup_env


def test_explicit_cu128_is_rejected_on_pascal(monkeypatch):
    pascal = hardware.GpuInfo(
        index=0,
        name="NVIDIA GeForce GTX 1080 Ti",
        total_mib=11264,
        free_mib=10000,
        compute_capability="6.1",
    )
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(setup_env, "selected_nvidia_gpu", lambda: pascal)

    with pytest.raises(ValueError, match="--backend auto"):
        setup_env._validate_cuda_backend("cu128")
    assert setup_env._validate_cuda_backend("cu126") == pascal


def test_explicit_cu128_is_allowed_on_modern_cuda_gpu(monkeypatch):
    modern = hardware.GpuInfo(
        index=0,
        name="NVIDIA GPU",
        total_mib=16384,
        free_mib=15000,
        compute_capability="8.6",
    )
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(setup_env, "selected_nvidia_gpu", lambda: modern)

    assert setup_env._validate_cuda_backend("cu128") == modern
