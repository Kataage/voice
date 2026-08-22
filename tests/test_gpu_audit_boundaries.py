from __future__ import annotations

import pytest

from personavoice import hardware


def _gpu(capability: str) -> hardware.GpuInfo:
    return hardware.GpuInfo(
        index=0,
        name=f"NVIDIA test sm_{capability.replace('.', '')}",
        total_mib=24576,
        free_mib=22000,
        compute_capability=capability,
    )


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ("6.0", "cu126"),
        ("6.1", "cu126"),
        ("7.0", "cu126"),
        ("7.5", "cu128"),
        ("8.0", "cu128"),
        ("8.6", "cu128"),
        ("8.9", "cu128"),
        ("9.0", "cu128"),
        ("10.0", "cu128"),
        ("12.0", "cu128"),
    ],
)
def test_known_x86_64_gpu_generations_use_audited_stack(monkeypatch, capability, expected):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware.cuda_backend_for_gpu(_gpu(capability)) == expected


@pytest.mark.parametrize("capability", ["5.2", "10.3", "12.1", "13.0", "99.0"])
def test_unverified_gpu_architectures_fail_closed(monkeypatch, capability):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware.cuda_backend_for_gpu(_gpu(capability)) == "cpu"
    assert not hardware.backend_supports_gpu("cu126", _gpu(capability))
    assert not hardware.backend_supports_gpu("cu128", _gpu(capability))
