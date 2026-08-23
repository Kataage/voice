from __future__ import annotations

from personavoice import doctor


def _seed_status(*_args, **_kwargs):
    return {"ok": True, "contract_sha256": None}


def test_doctor_checks_seed_vc_runtime_hardware_when_required(tmp_path, monkeypatch):
    calls: list[str | None] = []

    def fake_runtime(_setup, *, worker_name=None):
        calls.append(worker_name)
        return {"ok": worker_name != "seed_vc", "worker_name": worker_name}

    monkeypatch.setattr(doctor, "runtime_hardware_status", fake_runtime)
    monkeypatch.setattr(doctor, "seed_vc_materialization_status", _seed_status)
    result = doctor.report(tmp_path, require_seed_vc=True)

    assert calls == [None, "seed_vc"]
    assert result["runtime_hardware"]["ok"] is True
    assert result["seed_vc_runtime_hardware"]["ok"] is False
    assert result["reproducible_environment"] is False
    assert result["ready_offline"] is False


def test_doctor_skips_seed_vc_runtime_hardware_when_not_required(tmp_path, monkeypatch):
    calls: list[str | None] = []

    def fake_runtime(_setup, *, worker_name=None):
        calls.append(worker_name)
        return {"ok": True, "worker_name": worker_name}

    monkeypatch.setattr(doctor, "runtime_hardware_status", fake_runtime)
    monkeypatch.setattr(doctor, "seed_vc_materialization_status", _seed_status)
    result = doctor.report(tmp_path, require_seed_vc=False)

    assert calls == [None]
    assert result["seed_vc_runtime_hardware"]["ok"] is True
    assert result["seed_vc_runtime_hardware"]["skipped"] is True
