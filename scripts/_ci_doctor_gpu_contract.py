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
    "src/personavoice/doctor.py",
    "    runtime_hardware = runtime_hardware_status(setup)\n"
    "    model_assets = _model_asset_integrity(\n",
    "    runtime_hardware = runtime_hardware_status(setup)\n"
    "    seed_vc_runtime_hardware = (\n"
    "        runtime_hardware_status(setup, worker_name=\"seed_vc\")\n"
    "        if require_seed_vc\n"
    "        else {\"ok\": True, \"skipped\": True, \"reason\": \"Seed-VC is not required\"}\n"
    "    )\n"
    "    model_assets = _model_asset_integrity(\n",
)
patch(
    "src/personavoice/doctor.py",
    "        and bool(environment.get(\"ok\"))\n"
    "        and bool(runtime_hardware.get(\"ok\"))\n"
    "    )\n",
    "        and bool(environment.get(\"ok\"))\n"
    "        and bool(runtime_hardware.get(\"ok\"))\n"
    "        and bool(seed_vc_runtime_hardware.get(\"ok\"))\n"
    "    )\n",
)
patch(
    "src/personavoice/doctor.py",
    '        "runtime_hardware": runtime_hardware,\n'
    '        "models": models,\n',
    '        "runtime_hardware": runtime_hardware,\n'
    '        "seed_vc_runtime_hardware": seed_vc_runtime_hardware,\n'
    '        "models": models,\n',
)

(ROOT / "tests" / "test_doctor_gpu_contract.py").write_text(
    '''from __future__ import annotations\n\nfrom personavoice import doctor\n\n\ndef _seed_status(*_args, **_kwargs):\n    return {"ok": True, "contract_sha256": None}\n\n\ndef test_doctor_checks_seed_vc_runtime_hardware_when_required(tmp_path, monkeypatch):\n    calls: list[str | None] = []\n\n    def fake_runtime(_setup, *, worker_name=None):\n        calls.append(worker_name)\n        return {"ok": worker_name != "seed_vc", "worker_name": worker_name}\n\n    monkeypatch.setattr(doctor, "runtime_hardware_status", fake_runtime)\n    monkeypatch.setattr(doctor, "seed_vc_materialization_status", _seed_status)\n    result = doctor.report(tmp_path, require_seed_vc=True)\n\n    assert calls == [None, "seed_vc"]\n    assert result["runtime_hardware"]["ok"] is True\n    assert result["seed_vc_runtime_hardware"]["ok"] is False\n    assert result["reproducible_environment"] is False\n    assert result["ready_offline"] is False\n\n\ndef test_doctor_skips_seed_vc_runtime_hardware_when_not_required(tmp_path, monkeypatch):\n    calls: list[str | None] = []\n\n    def fake_runtime(_setup, *, worker_name=None):\n        calls.append(worker_name)\n        return {"ok": True, "worker_name": worker_name}\n\n    monkeypatch.setattr(doctor, "runtime_hardware_status", fake_runtime)\n    monkeypatch.setattr(doctor, "seed_vc_materialization_status", _seed_status)\n    result = doctor.report(tmp_path, require_seed_vc=False)\n\n    assert calls == [None]\n    assert result["seed_vc_runtime_hardware"]["ok"] is True\n    assert result["seed_vc_runtime_hardware"]["skipped"] is True\n''',
    encoding="utf-8",
    newline="\n",
)

print("Doctor GPU contract hardening applied")
