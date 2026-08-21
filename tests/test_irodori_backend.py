from __future__ import annotations

import json
from pathlib import Path

from personavoice import hardware, irodori


def test_recorded_irodori_backend_overrides_hardware_auto_detection(tmp_path: Path, monkeypatch):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "setup.json").write_text(
        json.dumps({"irodori_backend": "cpu"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(irodori, "detect_irodori_backend", lambda: "cu128")
    assert irodori.configured_backend(tmp_path) == "cpu"
    assert irodori.backend_device("cpu") == "cpu"


def test_backend_device_maps_rocm_to_torch_cuda_name():
    assert irodori.backend_device("cu128") == "cuda"
    assert irodori.backend_device("rocm") == "cuda"
    assert irodori.backend_device("xpu") == "xpu"


def test_cpu_batch_profile_does_not_use_visible_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "nvidia_gpus",
        lambda: [hardware.GpuInfo(index=0, name="GPU", total_mib=65536, free_mib=60000)],
    )
    cpu = hardware.safe_batch_profile(backend="cpu")
    cuda = hardware.safe_batch_profile(backend="cu128")
    assert cpu["batch_size"] == 1
    assert cuda["batch_size"] == 12
