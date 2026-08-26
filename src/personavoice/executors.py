from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from personavoice.hardware import selected_nvidia_gpu
from personavoice.training_bundle import TrainingBundle, canonical_plan_bytes
from personavoice.training_plan import TrainingPlan

ExecutorName = Literal["auto", "local", "modal"]
ResolvedExecutorName = Literal["local", "modal"]
StatusCallback = Callable[[dict[str, Any]], None]

_GIB = 1024**3


@dataclass(frozen=True)
class FullResourceRequirement:
    gpu_total_mib: int
    gpu_free_mib: int
    ram_available_bytes: int
    disk_free_bytes: int


# These are admission thresholds, not optimistic peak estimates. A full run is
# refused when the audited backend or headroom cannot be established.
FULL_RESOURCE_REQUIREMENTS: dict[str, FullResourceRequirement] = {
    "irodori": FullResourceRequirement(
        gpu_total_mib=24 * 1024,
        gpu_free_mib=20 * 1024,
        ram_available_bytes=32 * _GIB,
        disk_free_bytes=40 * _GIB,
    ),
    "lfm": FullResourceRequirement(
        gpu_total_mib=24 * 1024,
        gpu_free_mib=18 * 1024,
        ram_available_bytes=32 * _GIB,
        disk_free_bytes=24 * _GIB,
    ),
}


@dataclass(frozen=True)
class LocalResources:
    backend: str
    setup_current: bool
    gpu_total_mib: int | None
    gpu_free_mib: int | None
    ram_available_bytes: int | None
    disk_free_bytes: int | None


@dataclass(frozen=True)
class PreflightFailure:
    code: str
    message: str


@dataclass(frozen=True)
class LocalPreflight:
    ok: bool
    full_families: tuple[str, ...]
    failures: tuple[PreflightFailure, ...]
    required_gpu_total_mib: int
    required_gpu_free_mib: int
    required_ram_available_bytes: int
    required_disk_free_bytes: int

    def require(self) -> None:
        if not self.ok:
            raise LocalPreflightError(self)


class LocalPreflightError(RuntimeError):
    def __init__(self, preflight: LocalPreflight) -> None:
        self.preflight = preflight
        details = "; ".join(item.message for item in preflight.failures)
        super().__init__(
            "Local full training preflight failed; the requested method was not changed: "
            f"{details}"
        )


class RemoteConsentError(PermissionError):
    pass


class ModalUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteConsent:
    remote_data_authorized: bool = False
    scopes: frozenset[str] = frozenset()

    @property
    def permits_remote_processing(self) -> bool:
        # A broad consent scope is not permission to transmit prepared data.
        # Issue #26 requires the training-specific opt-in to be explicit so an
        # automatic executor decision can never turn local consent into upload.
        return self.remote_data_authorized


@dataclass(frozen=True)
class ExecutorDecision:
    executor: ResolvedExecutorName
    plan_fingerprint: str
    reason: str
    local_preflight: LocalPreflight


@dataclass(frozen=True)
class DispatchResult:
    decision: ExecutorDecision
    result: Any


class AuthStatus(Protocol):
    @property
    def configured(self) -> bool: ...


class LocalRunner(Protocol):
    def run(
        self,
        *,
        plan: TrainingPlan,
        plan_bytes: bytes,
        status_callback: StatusCallback | None,
    ) -> Any: ...


class RemoteRunner(Protocol):
    def run(
        self,
        *,
        plan: TrainingPlan,
        plan_bytes: bytes,
        bundle: TrainingBundle,
        status_callback: StatusCallback | None,
    ) -> Any: ...


def _available_ram_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError, ValueError):
            return None
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * available_pages)


def inspect_local_resources(
    storage_root: Path,
    *,
    backend: str,
    setup_current: bool,
) -> LocalResources:
    """Inspect only cheap host signals; no model import or allocation occurs."""

    gpu = selected_nvidia_gpu()
    try:
        disk_free = shutil.disk_usage(storage_root.resolve()).free
    except OSError:
        disk_free = None
    return LocalResources(
        backend=backend,
        setup_current=setup_current,
        gpu_total_mib=gpu.total_mib if gpu is not None else None,
        gpu_free_mib=gpu.free_mib if gpu is not None else None,
        ram_available_bytes=_available_ram_bytes(),
        disk_free_bytes=disk_free,
    )


def _plan_method_snapshot(plan: TrainingPlan) -> tuple[tuple[str, bool, str], ...]:
    return tuple((family.family, family.enabled, family.method) for family in plan.families)


def _assert_plan_unchanged(
    plan: TrainingPlan,
    *,
    fingerprint: str,
    methods: tuple[tuple[str, bool, str], ...],
) -> None:
    if plan.fingerprint != fingerprint or _plan_method_snapshot(plan) != methods:
        raise RuntimeError("An executor attempted to mutate the immutable TrainingPlan")


def preflight_local_full(plan: TrainingPlan, resources: LocalResources) -> LocalPreflight:
    """Conservatively admit enabled full-model families on an audited local stack."""

    fingerprint = plan.fingerprint
    methods = _plan_method_snapshot(plan)
    full_families = tuple(
        sorted(
            family.family
            for family in plan.families
            if family.enabled and family.method == "full"
        )
    )
    requirements = [FULL_RESOURCE_REQUIREMENTS[name] for name in full_families if name in FULL_RESOURCE_REQUIREMENTS]
    failures: list[PreflightFailure] = []
    unknown = [name for name in full_families if name not in FULL_RESOURCE_REQUIREMENTS]
    if unknown:
        failures.append(
            PreflightFailure(
                "unknown-full-family",
                f"no audited local full-training resource contract exists for {', '.join(unknown)}",
            )
        )

    required_total = max((item.gpu_total_mib for item in requirements), default=0)
    required_free = max((item.gpu_free_mib for item in requirements), default=0)
    required_ram = max((item.ram_available_bytes for item in requirements), default=0)
    required_disk = sum(item.disk_free_bytes for item in requirements)

    if full_families:
        if not resources.setup_current:
            failures.append(
                PreflightFailure(
                    "setup-not-current",
                    "the pinned local training environment is missing or stale",
                )
            )
        if resources.backend not in {"cu126", "cu128"}:
            failures.append(
                PreflightFailure(
                    "backend-not-audited",
                    f"backend {resources.backend!r} is not an audited CUDA full-training backend",
                )
            )
        if resources.gpu_total_mib is None or resources.gpu_total_mib < required_total:
            failures.append(
                PreflightFailure(
                    "gpu-total-vram",
                    f"at least {required_total} MiB total GPU memory is required",
                )
            )
        if resources.gpu_free_mib is None or resources.gpu_free_mib < required_free:
            failures.append(
                PreflightFailure(
                    "gpu-free-vram",
                    f"at least {required_free} MiB free GPU memory is required",
                )
            )
        if resources.ram_available_bytes is None or resources.ram_available_bytes < required_ram:
            failures.append(
                PreflightFailure(
                    "host-ram",
                    f"at least {required_ram // _GIB} GiB available host memory is required",
                )
            )
        if resources.disk_free_bytes is None or resources.disk_free_bytes < required_disk:
            failures.append(
                PreflightFailure(
                    "disk-space",
                    f"at least {required_disk // _GIB} GiB free workspace storage is required",
                )
            )

    _assert_plan_unchanged(plan, fingerprint=fingerprint, methods=methods)
    return LocalPreflight(
        ok=not failures,
        full_families=full_families,
        failures=tuple(failures),
        required_gpu_total_mib=required_total,
        required_gpu_free_mib=required_free,
        required_ram_available_bytes=required_ram,
        required_disk_free_bytes=required_disk,
    )


def assert_remote_dispatch_authorized(consent: RemoteConsent) -> None:
    if not consent.permits_remote_processing:
        raise RemoteConsentError(
            "Remote training is disabled until remote_data_authorized=true is explicitly set"
        )


def _require_modal_auth(modal_auth_probe: Callable[[], AuthStatus] | None) -> AuthStatus:
    if modal_auth_probe is None:
        raise ModalUnavailableError("Modal authentication is not configured")
    status = modal_auth_probe()
    if not status.configured:
        raise ModalUnavailableError("Modal authentication is not configured")
    return status


def select_executor(
    requested: ExecutorName,
    plan: TrainingPlan,
    resources: LocalResources,
    *,
    consent: RemoteConsent = RemoteConsent(),
    modal_auth_probe: Callable[[], AuthStatus] | None = None,
) -> ExecutorDecision:
    """Resolve execution without changing any family method.

    Consent is intentionally checked before calling the authentication probe.
    This order ensures an unauthorized ``auto`` request never reaches any remote
    preparation or credential-dependent operation.
    """

    if requested not in {"auto", "local", "modal"}:
        raise ValueError(f"Unknown training executor: {requested}")
    fingerprint = plan.fingerprint
    methods = _plan_method_snapshot(plan)
    local = preflight_local_full(plan, resources)

    if requested == "local":
        local.require()
        decision = ExecutorDecision(
            executor="local",
            plan_fingerprint=fingerprint,
            reason="explicit local executor passed conservative full-training preflight",
            local_preflight=local,
        )
    elif requested == "modal":
        assert_remote_dispatch_authorized(consent)
        _require_modal_auth(modal_auth_probe)
        decision = ExecutorDecision(
            executor="modal",
            plan_fingerprint=fingerprint,
            reason="explicit remote executor is authorized and configured",
            local_preflight=local,
        )
    elif local.ok:
        decision = ExecutorDecision(
            executor="local",
            plan_fingerprint=fingerprint,
            reason="automatic selection passed conservative local full-training preflight",
            local_preflight=local,
        )
    else:
        assert_remote_dispatch_authorized(consent)
        _require_modal_auth(modal_auth_probe)
        decision = ExecutorDecision(
            executor="modal",
            plan_fingerprint=fingerprint,
            reason="local full-training preflight failed; the unchanged plan was routed remotely",
            local_preflight=local,
        )

    _assert_plan_unchanged(plan, fingerprint=fingerprint, methods=methods)
    return decision


class LocalExecutor:
    def __init__(self, runner: LocalRunner) -> None:
        self._runner = runner

    def run(
        self,
        *,
        plan: TrainingPlan,
        plan_bytes: bytes,
        status_callback: StatusCallback | None = None,
    ) -> Any:
        if plan_bytes != canonical_plan_bytes(plan):
            raise ValueError("Local executor received non-canonical TrainingPlan bytes")
        return self._runner.run(
            plan=plan,
            plan_bytes=plan_bytes,
            status_callback=status_callback,
        )


class ModalExecutor:
    def __init__(self, runner: RemoteRunner) -> None:
        self._runner = runner

    def run(
        self,
        *,
        plan: TrainingPlan,
        plan_bytes: bytes,
        bundle: TrainingBundle,
        status_callback: StatusCallback | None = None,
    ) -> Any:
        if plan_bytes != canonical_plan_bytes(plan):
            raise ValueError("Modal executor received non-canonical TrainingPlan bytes")
        if bundle.inventory.plan_fingerprint != plan.fingerprint:
            raise ValueError("Modal executor bundle does not match the TrainingPlan")
        return self._runner.run(
            plan=plan,
            plan_bytes=plan_bytes,
            bundle=bundle,
            status_callback=status_callback,
        )


def dispatch_training(
    requested: ExecutorName,
    plan: TrainingPlan,
    resources: LocalResources,
    *,
    local_executor: LocalExecutor,
    modal_executor: ModalExecutor | None = None,
    bundle_factory: Callable[[], TrainingBundle] | None = None,
    consent: RemoteConsent = RemoteConsent(),
    modal_auth_probe: Callable[[], AuthStatus] | None = None,
    status_callback: StatusCallback | None = None,
) -> DispatchResult:
    """Select and run one executor while preserving one byte-identical plan.

    Selection (including the remote consent and auth gates) finishes before the
    bundle factory can be invoked. This makes the privacy ordering testable and
    prevents accidental upload preparation on an unauthorized request.
    """

    fingerprint = plan.fingerprint
    methods = _plan_method_snapshot(plan)
    decision = select_executor(
        requested,
        plan,
        resources,
        consent=consent,
        modal_auth_probe=modal_auth_probe,
    )
    plan_bytes = canonical_plan_bytes(plan)
    if decision.executor == "local":
        result = local_executor.run(
            plan=plan,
            plan_bytes=plan_bytes,
            status_callback=status_callback,
        )
    else:
        if modal_executor is None or bundle_factory is None:
            raise ModalUnavailableError("Modal transport is not available")
        bundle = bundle_factory()
        if bundle.inventory.plan_fingerprint != fingerprint:
            raise ValueError("Remote bundle was built for a different TrainingPlan")
        result = modal_executor.run(
            plan=plan,
            plan_bytes=plan_bytes,
            bundle=bundle,
            status_callback=status_callback,
        )
    _assert_plan_unchanged(plan, fingerprint=fingerprint, methods=methods)
    return DispatchResult(decision=decision, result=result)
