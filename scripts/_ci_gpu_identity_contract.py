from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found < count:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:120]!r} ({found=})")
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


patch(
    "src/personavoice/hardware.py",
    "    uuid: str | None = None\n    pci_bus_id: str | None = None\n",
    "    uuid: str | None = None\n    pci_bus_id: str | None = None\n    driver_version: str | None = None\n",
)
patch(
    "src/personavoice/hardware.py",
    '    fields = "index,uuid,pci.bus_id,name,memory.total,memory.free"\n',
    '    fields = "index,uuid,pci.bus_id,name,memory.total,memory.free,driver_version"\n',
)
patch(
    "src/personavoice/hardware.py",
    "    expected_parts = 7 if include_compute_cap else 6\n",
    "    expected_parts = 8 if include_compute_cap else 7\n",
)
patch(
    "src/personavoice/hardware.py",
    '                    free_mib=int(float(parts[5])),\n'
    '                    compute_capability=parts[6] if include_compute_cap else None,\n',
    '                    free_mib=int(float(parts[5])),\n'
    '                    driver_version=parts[6] or None,\n'
    '                    compute_capability=parts[7] if include_compute_cap else None,\n',
)

patch(
    "src/personavoice/environment_contract.py",
    '        "free_mib": gpu.free_mib,\n'
    '    }\n',
    '        "free_mib": gpu.free_mib,\n'
    '        "driver_version": gpu.driver_version,\n'
    '    }\n',
)

patch(
    "src/personavoice/environment_contract.py",
    '    selected = _selected_gpu_dict(gpu)\n'
    '    preferred = cuda_backend_for_gpu(gpu)\n',
    '''    selected = _selected_gpu_dict(gpu)\n\n    recorded_contract = value.get("environment_contract")\n    strict_gpu_provenance = (\n        isinstance(recorded_contract, dict)\n        and isinstance(recorded_contract.get("schema"), int)\n        and recorded_contract["schema"] >= ENVIRONMENT_CONTRACT_SCHEMA\n    )\n    if strict_gpu_provenance:\n        recorded_gpu = value.get("selected_gpu")\n        if not isinstance(recorded_gpu, dict):\n            return {\n                "ok": False,\n                "backend": backend,\n                "seed_vc_backend": seed_vc_backend,\n                "selected_gpu": selected,\n                "preferred_backend": cuda_backend_for_gpu(gpu),\n                "preferred_seed_vc_backend": (\n                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"\n                ),\n                "error": (\n                    "The current CUDA setup state has no audited GPU provenance. "\n                    "Run `persona setup --backend auto` before model work."\n                ),\n            }\n        recorded_uuid = recorded_gpu.get("uuid")\n        current_uuid = gpu.uuid\n        if (\n            not isinstance(recorded_uuid, str)\n            or not recorded_uuid\n            or not isinstance(current_uuid, str)\n            or not current_uuid\n            or recorded_uuid != current_uuid\n        ):\n            return {\n                "ok": False,\n                "backend": backend,\n                "seed_vc_backend": seed_vc_backend,\n                "selected_gpu": selected,\n                "preferred_backend": cuda_backend_for_gpu(gpu),\n                "preferred_seed_vc_backend": (\n                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"\n                ),\n                "error": (\n                    "The physical CUDA GPU selected as device 0 changed after setup. "\n                    "Run `persona setup --backend auto` to rebuild/reuse the appropriate locked "\n                    "environments and rerun the real CUDA kernel preflight before model work."\n                ),\n            }\n        recorded_driver = recorded_gpu.get("driver_version")\n        current_driver = gpu.driver_version\n        if (\n            not isinstance(recorded_driver, str)\n            or not recorded_driver\n            or not isinstance(current_driver, str)\n            or not current_driver\n            or recorded_driver != current_driver\n        ):\n            return {\n                "ok": False,\n                "backend": backend,\n                "seed_vc_backend": seed_vc_backend,\n                "selected_gpu": selected,\n                "preferred_backend": cuda_backend_for_gpu(gpu),\n                "preferred_seed_vc_backend": (\n                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"\n                ),\n                "error": (\n                    "The NVIDIA driver version changed after the CUDA environments were "\n                    "preflighted. Run `persona setup --backend auto` to rerun the real CUDA "\n                    "kernel preflight before model work."\n                ),\n            }\n\n    preferred = cuda_backend_for_gpu(gpu)\n''',
)

(ROOT / "tests" / "test_gpu_identity_contract.py").write_text(
    '''from __future__ import annotations\n\nfrom types import SimpleNamespace\n\nfrom personavoice import environment_contract as env_contract\nfrom personavoice import hardware\n\n\ndef _gpu(*, uuid="GPU-a", driver="600.01", capability="8.6"):\n    return hardware.GpuInfo(\n        index=0,\n        name="NVIDIA test GPU",\n        total_mib=24576,\n        free_mib=22000,\n        compute_capability=capability,\n        uuid=uuid,\n        pci_bus_id="00000000:01:00.0",\n        driver_version=driver,\n    )\n\n\ndef _setup(gpu):\n    return {\n        "irodori_backend": "cu128",\n        "worker_backends": {"seed_vc": "cu124"},\n        "selected_gpu": hardware.gpu_record(gpu),\n        "environment_contract": {"schema": env_contract.ENVIRONMENT_CONTRACT_SCHEMA},\n    }\n\n\ndef test_runtime_accepts_same_preflighted_gpu_and_driver(monkeypatch):\n    gpu = _gpu()\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: gpu)\n    assert env_contract.runtime_hardware_status(_setup(gpu))["ok"] is True\n\n\ndef test_runtime_rejects_physical_gpu_replacement_until_preflight(monkeypatch):\n    setup_gpu = _gpu(uuid="GPU-old")\n    current = _gpu(uuid="GPU-new")\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: current)\n    status = env_contract.runtime_hardware_status(_setup(setup_gpu))\n    assert status["ok"] is False\n    assert "physical CUDA GPU" in status["error"]\n    assert "persona setup --backend auto" in status["error"]\n\n\ndef test_runtime_rejects_driver_change_until_kernel_preflight(monkeypatch):\n    setup_gpu = _gpu(driver="600.01")\n    current = _gpu(driver="601.02")\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: current)\n    status = env_contract.runtime_hardware_status(_setup(setup_gpu))\n    assert status["ok"] is False\n    assert "driver version changed" in status["error"]\n    assert "persona setup --backend auto" in status["error"]\n\n\ndef test_gpu_query_records_driver_version(monkeypatch):\n    completed = SimpleNamespace(\n        returncode=0,\n        stdout=(\n            "0, GPU-abc, 00000000:01:00.0, NVIDIA Test, 24576, 22000, "\n            "600.01, 8.6\\n"\n        ),\n    )\n    monkeypatch.setattr(hardware, "_run_nvidia_query", lambda _fields: completed)\n    found = hardware.nvidia_gpus()\n    assert len(found) == 1\n    assert found[0].driver_version == "600.01"\n    assert found[0].compute_capability == "8.6"\n\n\ndef test_gpu_record_persists_driver_and_uuid():\n    record = hardware.gpu_record(_gpu(uuid="GPU-x", driver="602.03"))\n    assert record["uuid"] == "GPU-x"\n    assert record["driver_version"] == "602.03"\n''',
    encoding="utf-8",
    newline="\n",
)

print("GPU identity/driver contract applied")
