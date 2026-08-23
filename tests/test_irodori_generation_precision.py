from __future__ import annotations

from personavoice import hardware


def _gpu(capability: str) -> hardware.GpuInfo:
    return hardware.GpuInfo(
        index=0,
        name="NVIDIA test GPU",
        total_mib=16384,
        free_mib=12000,
        compute_capability=capability,
        uuid="GPU-test",
        pci_bus_id="00000000:01:00.0",
        driver_version="600.00",
    )


def test_irodori_precision_is_fp32_on_pascal_volta_and_turing(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    for capability, backend in (("6.1", "cu126"), ("7.0", "cu126"), ("7.5", "cu128")):
        policy = hardware.irodori_training_precision(backend, gpu=_gpu(capability))
        assert policy == {"precision": "fp32", "allow_tf32": False}


def test_irodori_precision_uses_bf16_tf32_only_on_audited_ampere_or_newer(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    for capability in ("8.0", "8.6", "8.9", "9.0", "10.0", "12.0"):
        policy = hardware.irodori_training_precision("cu128", gpu=_gpu(capability))
        assert policy == {"precision": "bf16", "allow_tf32": True}


def test_irodori_precision_fails_closed_for_unknown_or_mismatched_gpu(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware.irodori_training_precision("cu128", gpu=_gpu("13.0")) == {
        "precision": "fp32",
        "allow_tf32": False,
    }
    assert hardware.irodori_training_precision("cpu", gpu=_gpu("8.6")) == {
        "precision": "fp32",
        "allow_tf32": False,
    }
