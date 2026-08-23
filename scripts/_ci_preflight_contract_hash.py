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
    "src/personavoice/environment_contract.py",
    '            "runtime_dependencies_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "runtime_dependencies.py"\n'
    '            ),\n'
    '            "workers_sha256": _sha256(repo_root / "src" / "personavoice" / "workers.py"),\n',
    '            "runtime_dependencies_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "runtime_dependencies.py"\n'
    '            ),\n'
    '            "cuda_preflight_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "cuda_preflight.py"\n'
    '            ),\n'
    '            "workers_sha256": _sha256(repo_root / "src" / "personavoice" / "workers.py"),\n',
)

patch(
    "tests/test_gpu_runtime_contract.py",
    '        "src/personavoice/runtime_dependencies.py",\n'
    '        "src/personavoice/workers.py",\n',
    '        "src/personavoice/runtime_dependencies.py",\n'
    '        "src/personavoice/cuda_preflight.py",\n'
    '        "src/personavoice/workers.py",\n',
)

# Prove a preflight-only policy change invalidates the setup generation.
patch(
    "tests/test_gpu_runtime_contract.py",
    '    assert "different dependency contract" in str(status["error"])\n\n\n'
    'def test_setup_state_records_gpu_provenance',
    '    assert "different dependency contract" in str(status["error"])\n\n'
    '    refreshed = env_contract.environment_contract(tmp_path)\n'
    '    (tmp_path / "src/personavoice/cuda_preflight.py").write_text("v2", encoding="utf-8")\n'
    '    preflight_status = env_contract.environment_contract_status(tmp_path, refreshed)\n'
    '    assert preflight_status["ok"] is False\n'
    '    assert "different dependency contract" in str(preflight_status["error"])\n\n\n'
    'def test_setup_state_records_gpu_provenance',
)

print("CUDA preflight contract fingerprinting applied")
