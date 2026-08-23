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
    "tests/test_gpu_backend_policy.py",
    '''def gpu(capability: str, *, index: int = 0, mib: int = 16384, uuid: str | None = None):\n    return hardware.GpuInfo(\n        index=index,\n        uuid=uuid or f"GPU-{index:04d}",\n        name=f"NVIDIA test sm_{capability.replace('.', '')}",\n        total_mib=mib,\n        free_mib=max(1, mib - 1024),\n        compute_capability=capability,\n    )\n''',
    '''def gpu(\n    capability: str,\n    *,\n    index: int = 0,\n    mib: int = 16384,\n    uuid: str | None = None,\n    driver: str = "600.01",\n):\n    return hardware.GpuInfo(\n        index=index,\n        uuid=uuid or f"GPU-{index:04d}",\n        name=f"NVIDIA test sm_{capability.replace('.', '')}",\n        total_mib=mib,\n        free_mib=max(1, mib - 1024),\n        compute_capability=capability,\n        driver_version=driver,\n    )\n''',
)
patch(
    "tests/test_gpu_backend_policy.py",
    '''def _write_setup(tmp_path, backend: str, *, seed_vc_backend: str | None = None) -> dict:\n    runtime = tmp_path / ".runtime"\n    runtime.mkdir(parents=True)\n    setup = {\n        "irodori_backend": backend,\n        "environment_contract": environment.environment_contract(tmp_path),\n    }\n    if seed_vc_backend is not None:\n        setup["worker_backends"] = {"seed_vc": seed_vc_backend}\n    (runtime / "setup.json").write_text(json.dumps(setup), encoding="utf-8")\n    return setup\n''',
    '''def _write_setup(\n    tmp_path,\n    backend: str,\n    *,\n    seed_vc_backend: str | None = None,\n    selected_gpu=None,\n) -> dict:\n    runtime = tmp_path / ".runtime"\n    runtime.mkdir(parents=True)\n    if selected_gpu is None and backend in {"cu126", "cu128"}:\n        selected_gpu = gpu("6.1" if backend == "cu126" else "8.6", uuid="GPU-setup")\n    setup = {\n        "irodori_backend": backend,\n        "environment_contract": environment.environment_contract(tmp_path),\n        "selected_gpu": hardware.gpu_record(selected_gpu),\n    }\n    if seed_vc_backend is not None:\n        setup["worker_backends"] = {"seed_vc": seed_vc_backend}\n    (runtime / "setup.json").write_text(json.dumps(setup), encoding="utf-8")\n    return setup\n''',
)
patch(
    "tests/test_gpu_backend_policy.py",
    '''def test_direct_runtime_rejects_incompatible_gpu_swap(tmp_path, monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    _write_setup(tmp_path, "cu128")\n    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("7.0"))\n    with pytest.raises(RuntimeError, match="GPU selection appears to have changed"):\n        environment.require_current_environment(tmp_path)\n\n\ndef test_direct_runtime_accepts_compatible_gpu_swap(tmp_path, monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    setup = _write_setup(tmp_path, "cu126")\n    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("8.6"))\n    assert environment.require_current_environment(tmp_path) == setup\n''',
    '''def test_direct_runtime_rejects_physical_gpu_swap_before_model_start(tmp_path, monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    _write_setup(tmp_path, "cu128")\n    monkeypatch.setattr(\n        environment,\n        "selected_nvidia_gpu",\n        lambda: gpu("7.0", uuid="GPU-replacement"),\n    )\n    with pytest.raises(RuntimeError, match="physical CUDA GPU"):\n        environment.require_current_environment(tmp_path)\n\n\ndef test_direct_runtime_rejects_even_compatible_gpu_swap_until_repreflight(\n    tmp_path, monkeypatch\n):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    _write_setup(tmp_path, "cu126")\n    monkeypatch.setattr(\n        environment,\n        "selected_nvidia_gpu",\n        lambda: gpu("8.6", uuid="GPU-compatible-replacement"),\n    )\n    with pytest.raises(RuntimeError, match="physical CUDA GPU"):\n        environment.require_current_environment(tmp_path)\n''',
)
patch(
    "tests/test_gpu_backend_policy.py",
    '''def test_direct_runtime_only_blocks_seed_vc_after_blackwell_swap(tmp_path, monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    setup = _write_setup(tmp_path, "cu128", seed_vc_backend="cu124")\n    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("12.0"))\n    assert environment.require_current_environment(tmp_path) == setup\n    with pytest.raises(RuntimeError, match="Seed-VC was set up for cu124"):\n        environment.require_current_environment(tmp_path, worker_name="seed_vc")\n\n\ndef test_direct_runtime_accepts_blackwell_when_seed_vc_is_cpu(tmp_path, monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    setup = _write_setup(tmp_path, "cu128", seed_vc_backend="cpu")\n    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: gpu("12.0"))\n    assert environment.require_current_environment(tmp_path) == setup\n''',
    '''def test_direct_runtime_only_blocks_legacy_seed_vc_on_same_blackwell(tmp_path, monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    blackwell = gpu("12.0", uuid="GPU-blackwell")\n    setup = _write_setup(\n        tmp_path,\n        "cu128",\n        seed_vc_backend="cu124",\n        selected_gpu=blackwell,\n    )\n    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: blackwell)\n    assert environment.require_current_environment(tmp_path) == setup\n    with pytest.raises(RuntimeError, match="Seed-VC was set up for cu124"):\n        environment.require_current_environment(tmp_path, worker_name="seed_vc")\n\n\ndef test_direct_runtime_accepts_same_blackwell_when_seed_vc_is_cpu(tmp_path, monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    blackwell = gpu("12.0", uuid="GPU-blackwell")\n    setup = _write_setup(\n        tmp_path,\n        "cu128",\n        seed_vc_backend="cpu",\n        selected_gpu=blackwell,\n    )\n    monkeypatch.setattr(environment, "selected_nvidia_gpu", lambda: blackwell)\n    assert environment.require_current_environment(tmp_path) == setup\n''',
)

patch(
    "tests/test_gpu_runtime_compatibility.py",
    '            stdout="7, GPU-abcdef, 00000000:65:00.0, NVIDIA RTX, 24576, 23000, 8.6\\n",\n',
    '            stdout=(\n'
    '                "7, GPU-abcdef, 00000000:65:00.0, NVIDIA RTX, 24576, 23000, "\n'
    '                "600.01, 8.6\\n"\n'
    '            ),\n',
)
patch(
    "tests/test_gpu_runtime_compatibility.py",
    '            compute_capability="8.6",\n'
    '        )\n',
    '            compute_capability="8.6",\n'
    '            driver_version="600.01",\n'
    '        )\n',
)
patch(
    "tests/test_gpu_runtime_compatibility.py",
    '    assert calls == ["index,uuid,pci.bus_id,name,memory.total,memory.free,compute_cap"]\n',
    '    assert calls == [\n'
    '        "index,uuid,pci.bus_id,name,memory.total,memory.free,driver_version,compute_cap"\n'
    '    ]\n',
)

print("GPU identity fixtures synchronized")
