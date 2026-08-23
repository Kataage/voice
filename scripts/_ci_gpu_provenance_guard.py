from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if text.count(old) < count:
        raise RuntimeError(f"Patch anchor missing in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


patch(
    "src/personavoice/environment_contract.py",
    '        "runtime_policy": {\n            "hardware_sha256": _sha256(repo_root / "src" / "personavoice" / "hardware.py"),\n',
    '        "runtime_policy": {\n'
    '            "environment_contract_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "environment_contract.py"\n'
    '            ),\n'
    '            "hardware_sha256": _sha256(repo_root / "src" / "personavoice" / "hardware.py"),\n',
)

anchor = '''        recorded_driver = recorded_gpu.get("driver_version")\n        current_driver = gpu.driver_version\n'''
insert = '''        recorded_capability = recorded_gpu.get("compute_capability")\n        current_capability = gpu.compute_capability\n        if (\n            not isinstance(recorded_capability, str)\n            or not recorded_capability\n            or not isinstance(current_capability, str)\n            or not current_capability\n            or recorded_capability != current_capability\n        ):\n            return {\n                "ok": False,\n                "backend": backend,\n                "seed_vc_backend": seed_vc_backend,\n                "selected_gpu": selected,\n                "preferred_backend": cuda_backend_for_gpu(gpu),\n                "preferred_seed_vc_backend": (\n                    "cu124" if seed_vc_cuda_supported(gpu) else "cpu"\n                ),\n                "error": (\n                    "The selected NVIDIA GPU compute capability changed after the CUDA "\n                    "environments were preflighted. Run `persona setup --backend auto` to "\n                    "rerun the real CUDA kernel preflight before model work."\n                ),\n            }\n        recorded_driver = recorded_gpu.get("driver_version")\n        current_driver = gpu.driver_version\n'''
patch("src/personavoice/environment_contract.py", anchor, insert)

patch(
    "tests/test_gpu_runtime_compatibility.py",
    "import types\n",
    "import hashlib\nimport types\n",
)

append = ROOT / "tests/test_gpu_runtime_compatibility.py"
text = append.read_text(encoding="utf-8")
extra = r'''
def test_runtime_hardware_rejects_capability_change_for_same_gpu_uuid(monkeypatch):
    current = hardware.GpuInfo(
        index=0,
        name="Stable UUID GPU",
        total_mib=16384,
        free_mib=12000,
        compute_capability="8.9",
        uuid="GPU-stable",
        driver_version="600.01",
    )
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_contract, "selected_nvidia_gpu", lambda: current)
    status = env_contract.runtime_hardware_status(
        {
            "irodori_backend": "cu128",
            "worker_backends": {"seed_vc": "cpu"},
            "environment_contract": {"schema": env_contract.ENVIRONMENT_CONTRACT_SCHEMA},
            "selected_gpu": {
                "uuid": "GPU-stable",
                "compute_capability": "8.6",
                "driver_version": "600.01",
            },
        }
    )
    assert status["ok"] is False
    assert "compute capability changed" in status["error"]
    assert "persona setup --backend auto" in status["error"]


def test_environment_contract_hashes_its_own_runtime_guard():
    root = Path(__file__).resolve().parents[1]
    contract = env_contract.environment_contract(root)
    expected = hashlib.sha256(
        (root / "src" / "personavoice" / "environment_contract.py").read_bytes()
    ).hexdigest()
    assert contract["runtime_policy"]["environment_contract_sha256"] == expected
'''
if "test_runtime_hardware_rejects_capability_change_for_same_gpu_uuid" in text:
    raise RuntimeError("GPU provenance tests already present")
append.write_text(
    text.rstrip() + "\n\n\n" + extra.strip() + "\n",
    encoding="utf-8",
    newline="\n",
)

print("GPU provenance guard hardening applied")
