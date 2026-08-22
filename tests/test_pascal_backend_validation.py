from __future__ import annotations

import pytest

from personavoice import hardware, setup_env


def test_explicit_cu128_is_rejected_on_pascal(monkeypatch):
    monkeypatch.setattr(
        setup_env,
        "nvidia_gpus",
        lambda: [
            hardware.GpuInfo(
                index=0,
                name="NVIDIA GeForce GTX 1080 Ti",
                total_mib=11264,
                free_mib=10000,
                compute_capability="6.1",
            )
        ],
    )

    with pytest.raises(ValueError, match="--backend cu126"):
        setup_env._validate_explicit_backend("cu128")


def test_explicit_cu128_is_allowed_on_modern_cuda_gpu(monkeypatch):
    monkeypatch.setattr(
        setup_env,
        "nvidia_gpus",
        lambda: [
            hardware.GpuInfo(
                index=0,
                name="NVIDIA GPU",
                total_mib=16384,
                free_mib=15000,
                compute_capability="8.6",
            )
        ],
    )

    setup_env._validate_explicit_backend("cu128")
