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


# Persist a non-secret setup-time GPU identity for provenance while runtime
# compatibility remains architecture-based, allowing compatible GPU swaps.
patch(
    "src/personavoice/hardware.py",
    "\ndef _host_arch() -> str:\n",
    "\ndef gpu_record(gpu: GpuInfo | None) -> dict | None:\n"
    "    return asdict(gpu) if gpu is not None else None\n\n\n"
    "def _host_arch() -> str:\n",
)

# Dependency generations must also change when backend-selection/runtime policy
# code changes, not only when pyproject/lock bytes change.
patch(
    "src/personavoice/environment_contract.py",
    "ENVIRONMENT_CONTRACT_SCHEMA = 3\n",
    "ENVIRONMENT_CONTRACT_SCHEMA = 4\n",
)
patch(
    "src/personavoice/environment_contract.py",
    '        "seed_vc": {\n'
    '            "asset_contract_sha256": _sha256(repo_root / "config" / "seed_vc_assets.json"),\n'
    '        },\n'
    '        "workers": workers,\n',
    '        "seed_vc": {\n'
    '            "asset_contract_sha256": _sha256(repo_root / "config" / "seed_vc_assets.json"),\n'
    '        },\n'
    '        "runtime_policy": {\n'
    '            "hardware_sha256": _sha256(repo_root / "src" / "personavoice" / "hardware.py"),\n'
    '            "setup_sha256": _sha256(repo_root / "src" / "personavoice" / "setup_env.py"),\n'
    '            "runtime_dependencies_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "runtime_dependencies.py"\n'
    '            ),\n'
    '            "workers_sha256": _sha256(repo_root / "src" / "personavoice" / "workers.py"),\n'
    '            "asr_runtime_policy_sha256": _sha256(\n'
    '                repo_root / "workers" / "asr" / "runtime_policy.py"\n'
    '            ),\n'
    '        },\n'
    '        "workers": workers,\n',
)

# A Blackwell swap can leave the main cu128 stack valid while only Seed-VC's
# older cu124 environment becomes stale. Validate the worker-specific backend
# only when that worker is actually invoked instead of blocking unrelated TTS.
patch(
    "src/personavoice/environment_contract.py",
    "def runtime_hardware_status(setup: Any) -> dict[str, Any]:\n",
    "def runtime_hardware_status(\n"
    "    setup: Any,\n"
    "    *,\n"
    "    worker_name: str | None = None,\n"
    ") -> dict[str, Any]:\n",
)
patch(
    "src/personavoice/environment_contract.py",
    '    main_cuda = backend in {"cu126", "cu128"}\n'
    '    seed_cuda = seed_vc_backend == "cu124"\n',
    '    main_cuda = backend in {"cu126", "cu128"}\n'
    '    seed_cuda = worker_name == "seed_vc" and seed_vc_backend == "cu124"\n',
)
patch(
    "src/personavoice/environment_contract.py",
    "def require_current_environment(repo_root: Path) -> dict[str, Any]:\n",
    "def require_current_environment(\n"
    "    repo_root: Path,\n"
    "    *,\n"
    "    worker_name: str | None = None,\n"
    ") -> dict[str, Any]:\n",
)
patch(
    "src/personavoice/environment_contract.py",
    '    hardware = runtime_hardware_status(setup)\n',
    '    hardware = runtime_hardware_status(setup, worker_name=worker_name)\n',
)

# Workers validate exactly the hardware contract relevant to that worker.
patch(
    "src/personavoice/workers.py",
    "        setup = require_current_environment(repo_root)\n",
    "        setup = require_current_environment(repo_root, worker_name=self.name)\n",
)

# Record the setup-time GPU as provenance. Compatible replacements remain valid;
# runtime_hardware_status is deliberately architecture-based rather than UUID-bound.
patch(
    "src/personavoice/setup_env.py",
    "    detect_irodori_backend,\n",
    "    detect_irodori_backend,\n    gpu_record,\n",
)
patch(
    "src/personavoice/setup_env.py",
    "    selected_gpu = _validate_cuda_backend(selected_backend)\n"
    "    worker_extras = _worker_extras(selected_backend, gpu=selected_gpu)\n",
    "    selected_gpu = _validate_cuda_backend(selected_backend)\n"
    "    selected_gpu_state = gpu_record(selected_gpu)\n"
    "    worker_extras = _worker_extras(selected_backend, gpu=selected_gpu)\n",
)
patch(
    "src/personavoice/setup_env.py",
    '        "worker_backends": worker_extras,\n'
    '        "irodori_revision": IRODORI_REVISION,\n',
    '        "worker_backends": worker_extras,\n'
    '        "selected_gpu": selected_gpu_state,\n'
    '        "irodori_revision": IRODORI_REVISION,\n',
    count=2,
)

# Add targeted tests for multi-GPU selection, hardware swaps and runtime-policy
# fingerprint invalidation.
new_test = ROOT / "tests" / "test_gpu_runtime_contract.py"
new_test.write_text(
    '''from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport pytest\n\nfrom personavoice import environment_contract as env_contract\nfrom personavoice import hardware, setup_env\nfrom personavoice.workers import local_model_env\n\n\ndef _gpu(\n    capability: str,\n    *,\n    index: int = 0,\n    uuid: str = "GPU-a",\n    pci: str = "00000000:01:00.0",\n    memory: int = 16384,\n) -> hardware.GpuInfo:\n    return hardware.GpuInfo(\n        index=index,\n        name=f"GPU-{capability}",\n        total_mib=memory,\n        free_mib=memory - 512,\n        compute_capability=capability,\n        uuid=uuid,\n        pci_bus_id=pci,\n    )\n\n\ndef test_cuda_visible_devices_numeric_order_matches_forced_pci_order(monkeypatch):\n    first_pci = _gpu("8.6", index=7, uuid="GPU-first", pci="00000000:01:00.0")\n    second_pci = _gpu("6.1", index=2, uuid="GPU-second", pci="00000000:09:00.0")\n    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")\n    assert hardware.selected_nvidia_gpu([second_pci, first_pci]) == second_pci\n\n\ndef test_cuda_visible_devices_uuid_selects_exact_physical_gpu(monkeypatch):\n    first = _gpu("8.6", index=0, uuid="GPU-aaaaaaaa", pci="00000000:01:00.0")\n    second = _gpu("6.1", index=1, uuid="GPU-bbbbbbbb", pci="00000000:02:00.0")\n    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")\n    assert hardware.selected_nvidia_gpu([first, second]) == second\n\n\ndef test_unknown_or_mig_cuda_selector_fails_closed(monkeypatch):\n    gpu = _gpu("8.6")\n    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-unknown")\n    assert hardware.selected_nvidia_gpu([gpu]) is None\n    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-a-device")\n    assert hardware.selected_nvidia_gpu([gpu]) is None\n\n\ndef test_gpu_generation_policy_is_conservative_and_blackwell_aware(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    assert hardware.cuda_backend_for_gpu(_gpu("6.1")) == "cu126"\n    assert hardware.cuda_backend_for_gpu(_gpu("7.0")) == "cu126"\n    assert hardware.cuda_backend_for_gpu(_gpu("7.5")) == "cu128"\n    assert hardware.cuda_backend_for_gpu(_gpu("9.0")) == "cu128"\n    assert hardware.cuda_backend_for_gpu(_gpu("10.0")) == "cu128"\n    assert hardware.cuda_backend_for_gpu(_gpu("12.0")) == "cu128"\n    assert hardware.cuda_backend_for_gpu(_gpu("13.0")) == "cpu"\n\n\ndef test_batch_profile_uses_logical_cuda_device_zero_not_largest_gpu(monkeypatch):\n    small = _gpu("8.6", index=0, uuid="GPU-small", pci="00000000:01:00.0", memory=11000)\n    huge = _gpu("8.6", index=1, uuid="GPU-huge", pci="00000000:02:00.0", memory=65536)\n    monkeypatch.setattr(hardware, "nvidia_gpus", lambda: [small, huge])\n    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)\n    assert hardware.safe_batch_profile(backend="cu128")["batch_size"] == 1\n    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")\n    assert hardware.safe_batch_profile(backend="cu128")["batch_size"] == 12\n\n\ndef test_compatible_gpu_swap_keeps_main_runtime_available(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: _gpu("8.6", uuid="GPU-new"))\n    setup = {\n        "irodori_backend": "cu128",\n        "worker_backends": {"seed_vc": "cu124"},\n    }\n    assert env_contract.runtime_hardware_status(setup)["ok"] is True\n\n\ndef test_blackwell_swap_only_blocks_seed_vc_worker(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: _gpu("12.0", uuid="GPU-new"))\n    setup = {\n        "irodori_backend": "cu128",\n        "worker_backends": {"seed_vc": "cu124"},\n    }\n    assert env_contract.runtime_hardware_status(setup)["ok"] is True\n    seed = env_contract.runtime_hardware_status(setup, worker_name="seed_vc")\n    assert seed["ok"] is False\n    assert seed["preferred_seed_vc_backend"] == "cpu"\n\n\ndef test_pascal_swap_from_cu128_requires_resync(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: _gpu("6.1"))\n    setup = {"irodori_backend": "cu128", "worker_backends": {"seed_vc": "cu124"}}\n    status = env_contract.runtime_hardware_status(setup)\n    assert status["ok"] is False\n    assert status["preferred_backend"] == "cu126"\n    assert "persona setup --backend auto" in status["error"]\n\n\ndef test_blackwell_setup_keeps_main_cuda_and_falls_seed_vc_back_to_cpu(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    gpu = _gpu("12.0")\n    extras = setup_env._worker_extras("cu128", gpu=gpu)\n    assert extras["diarization"] == "cu128"\n    assert extras["sense"] == "cu128"\n    assert extras["lfm"] == "cu128"\n    assert extras["seed_vc"] == "cpu"\n\n\ndef test_local_model_env_forces_deterministic_cuda_order(tmp_path: Path):\n    env = local_model_env(tmp_path)\n    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"\n\n\ndef test_environment_contract_includes_runtime_policy_sources(tmp_path: Path):\n    for relative in (\n        "src/personavoice/hardware.py",\n        "src/personavoice/setup_env.py",\n        "src/personavoice/runtime_dependencies.py",\n        "src/personavoice/workers.py",\n        "workers/asr/runtime_policy.py",\n    ):\n        path = tmp_path / relative\n        path.parent.mkdir(parents=True, exist_ok=True)\n        path.write_text("v1", encoding="utf-8")\n    recorded = env_contract.environment_contract(tmp_path)\n    assert recorded["schema"] == 4\n    (tmp_path / "src/personavoice/hardware.py").write_text("v2", encoding="utf-8")\n    status = env_contract.environment_contract_status(tmp_path, recorded)\n    assert status["ok"] is False\n    assert "different dependency contract" in str(status["error"])\n\n\ndef test_setup_state_records_gpu_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):\n    # This is a focused publication test; environment mutation is stubbed.\n    gpu = _gpu("8.6", uuid="GPU-recorded")\n    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: "/tool")\n    monkeypatch.setattr(setup_env, "require_ffmpeg_runtime", lambda: None)\n    monkeypatch.setattr(setup_env, "_validate_cuda_backend", lambda _backend: gpu)\n    irodori = tmp_path / "vendor/Irodori-TTS"\n    seed = tmp_path / "vendor/seed-vc"\n    irodori.mkdir(parents=True)\n    seed.mkdir(parents=True)\n    monkeypatch.setattr(\n        setup_env,\n        "_clone_pinned",\n        lambda _root, name, _url, _revision: irodori if name == "Irodori-TTS" else seed,\n    )\n    monkeypatch.setattr(setup_env, "_install_irodori", lambda *_args, **_kwargs: None)\n\n    class FakeWorker:\n        def sync(self, _root: Path, *, extra: str | None = None) -> None:\n            pass\n\n    monkeypatch.setattr(setup_env, "worker", lambda *_args: FakeWorker())\n    result = setup_env.install_environments(tmp_path, backend="cu128")\n    assert result["selected_gpu"]["uuid"] == "GPU-recorded"\n    recorded = json.loads((tmp_path / ".runtime/setup.json").read_text(encoding="utf-8"))\n    assert recorded["selected_gpu"]["compute_capability"] == "8.6"\n''',
    encoding="utf-8",
    newline="\n",
)

print("GPU policy hardening applied")
