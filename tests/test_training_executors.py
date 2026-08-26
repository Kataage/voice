from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from personavoice.executors import (
    LocalExecutor,
    LocalPreflightError,
    LocalResources,
    ModalExecutor,
    RemoteConsent,
    RemoteConsentError,
    dispatch_training,
    preflight_local_full,
    select_executor,
)
from personavoice.training_bundle import BundleInventory, TrainingBundle, canonical_plan_bytes
from personavoice.training_plan import FamilyPlan, TrainingPlan

GIB = 1024**3


def _family(name: str, *, method: str = "full", enabled: bool = True) -> FamilyPlan:
    return FamilyPlan(
        family=name,
        enabled=enabled,
        method=method,
        dataset_fingerprint="d" * 64,
        training={"seed": 42},
        model_contract={"revision": "pinned"},
        implementation_contract={"train.py": "a" * 64},
        checkpoint_policy={"resume_complete_only": True},
        evaluation_policy={"max_cer": 0.2},
    )


def _plan(*, irodori_method: str = "full", lfm_method: str = "full") -> TrainingPlan:
    return TrainingPlan(
        persona="alice",
        files=(),
        families=(
            _family("irodori", method=irodori_method),
            _family("lfm", method=lfm_method),
        ),
    )


def _resources(*, enough: bool) -> LocalResources:
    if enough:
        return LocalResources(
            backend="cu128",
            setup_current=True,
            gpu_total_mib=48 * 1024,
            gpu_free_mib=44 * 1024,
            ram_available_bytes=64 * GIB,
            disk_free_bytes=100 * GIB,
        )
    return LocalResources(
        backend="cpu",
        setup_current=False,
        gpu_total_mib=None,
        gpu_free_mib=None,
        ram_available_bytes=None,
        disk_free_bytes=None,
    )


class RecordingLocalRunner:
    def __init__(self) -> None:
        self.plan_bytes: bytes | None = None

    def run(self, *, plan, plan_bytes, status_callback):
        self.plan_bytes = plan_bytes
        return {"executor": "local", "fingerprint": plan.fingerprint}


class RecordingRemoteRunner:
    def __init__(self) -> None:
        self.plan_bytes: bytes | None = None

    def run(self, *, plan, plan_bytes, bundle, status_callback):
        self.plan_bytes = plan_bytes
        assert bundle.inventory.plan_fingerprint == plan.fingerprint
        return {"executor": "modal", "fingerprint": plan.fingerprint}


def _empty_bundle(tmp_path: Path, plan: TrainingPlan) -> TrainingBundle:
    root = tmp_path / "bundle-placeholder"
    root.mkdir(exist_ok=True)
    return TrainingBundle(
        root=root,
        inventory=BundleInventory.create(plan_fingerprint=plan.fingerprint, files=()),
    )


def test_local_full_preflight_fails_closed_without_known_headroom() -> None:
    plan = _plan()
    before = plan.as_dict()

    preflight = preflight_local_full(plan, _resources(enough=False))

    assert not preflight.ok
    assert {failure.code for failure in preflight.failures} == {
        "backend-not-audited",
        "disk-space",
        "gpu-free-vram",
        "gpu-total-vram",
        "host-ram",
        "setup-not-current",
    }
    assert preflight.required_disk_free_bytes == 64 * GIB
    assert plan.as_dict() == before
    assert all(family.method == "full" for family in plan.families)


def test_explicit_local_never_silently_downgrades_full_to_lora() -> None:
    plan = _plan()
    fingerprint = plan.fingerprint

    with pytest.raises(LocalPreflightError, match="method was not changed"):
        select_executor("local", plan, _resources(enough=False))

    assert plan.fingerprint == fingerprint
    assert [family.method for family in plan.families] == ["full", "full"]


def test_unauthorized_auto_stops_before_auth_and_bundle(tmp_path: Path) -> None:
    plan = _plan()
    events: list[str] = []

    def auth_probe():
        events.append("auth")
        return SimpleNamespace(configured=True)

    def bundle_factory():
        events.append("bundle")
        return _empty_bundle(tmp_path, plan)

    with pytest.raises(RemoteConsentError):
        dispatch_training(
            "auto",
            plan,
            _resources(enough=False),
            local_executor=LocalExecutor(RecordingLocalRunner()),
            modal_executor=ModalExecutor(RecordingRemoteRunner()),
            bundle_factory=bundle_factory,
            consent=RemoteConsent(scopes=frozenset({"remote-processing"})),
            modal_auth_probe=auth_probe,
        )

    assert events == []


def test_auth_failure_stops_before_bundle(tmp_path: Path) -> None:
    plan = _plan()
    events: list[str] = []

    def auth_probe():
        events.append("auth")
        return SimpleNamespace(configured=False)

    def bundle_factory():
        events.append("bundle")
        return _empty_bundle(tmp_path, plan)

    with pytest.raises(RuntimeError, match="authentication"):
        dispatch_training(
            "modal",
            plan,
            _resources(enough=False),
            local_executor=LocalExecutor(RecordingLocalRunner()),
            modal_executor=ModalExecutor(RecordingRemoteRunner()),
            bundle_factory=bundle_factory,
            consent=RemoteConsent(remote_data_authorized=True),
            modal_auth_probe=auth_probe,
        )

    assert events == ["auth"]


def test_local_and_modal_receive_byte_identical_training_plan(tmp_path: Path) -> None:
    plan = _plan()
    local_runner = RecordingLocalRunner()
    remote_runner = RecordingRemoteRunner()
    local = LocalExecutor(local_runner)
    modal = ModalExecutor(remote_runner)

    local_result = dispatch_training(
        "local",
        plan,
        _resources(enough=True),
        local_executor=local,
    )
    remote_result = dispatch_training(
        "modal",
        plan,
        _resources(enough=False),
        local_executor=local,
        modal_executor=modal,
        bundle_factory=lambda: _empty_bundle(tmp_path, plan),
        consent=RemoteConsent(
            remote_data_authorized=True,
            scopes=frozenset({"remote-processing"}),
        ),
        modal_auth_probe=lambda: SimpleNamespace(configured=True),
    )

    assert local_result.decision.plan_fingerprint == remote_result.decision.plan_fingerprint
    assert local_runner.plan_bytes == remote_runner.plan_bytes == canonical_plan_bytes(plan)
    assert [family.method for family in plan.families] == ["full", "full"]


def test_auto_uses_local_without_touching_remote_auth() -> None:
    plan = _plan()
    probes = 0

    def probe():
        nonlocal probes
        probes += 1
        return SimpleNamespace(configured=True)

    decision = select_executor(
        "auto",
        plan,
        _resources(enough=True),
        modal_auth_probe=probe,
    )

    assert decision.executor == "local"
    assert probes == 0


def test_non_full_methods_are_not_reinterpreted_by_preflight() -> None:
    plan = _plan(irodori_method="lora", lfm_method="lora")

    preflight = preflight_local_full(plan, _resources(enough=False))

    assert preflight.ok
    assert preflight.full_families == ()
    assert [family.method for family in plan.families] == ["lora", "lora"]
