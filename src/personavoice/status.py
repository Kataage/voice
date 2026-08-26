from __future__ import annotations

from pathlib import Path
from typing import Any

from personavoice.config import PersonaConfig
from personavoice.doctor import (
    _without_secret_values,
    modal_readiness_status,
    training_preflight_status,
)
from personavoice.pipeline import _prepare_fingerprint
from personavoice.prepare_checkpoints import prepare_batch_progress
from personavoice.project import PersonaPaths
from personavoice.stage_lock import stage_lock_held
from personavoice.state import StateStore
from personavoice.training import _fingerprint as _training_fingerprint


def _stage_audit(
    store: StateStore,
    state: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    stages = state.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    stage = stages.get(name)
    stage = stage if isinstance(stage, dict) else {}
    recorded_status = stage.get("status")
    recorded_fingerprint = stage.get("fingerprint")
    artifact_complete = False
    if isinstance(recorded_fingerprint, str) and recorded_fingerprint:
        try:
            artifact_complete = store.is_complete(name, recorded_fingerprint)
        except (OSError, ValueError, TypeError):
            artifact_complete = False

    runner = stage.get("runner") if isinstance(stage.get("runner"), dict) else None
    protocol_known = bool(runner and runner.get("lock_protocol") == 1)
    lock_held: bool | None = None
    activity_error: str | None = None
    if recorded_status == "running" and protocol_known:
        try:
            lock_held = stage_lock_held(store.path.parent, name)
        except (OSError, RuntimeError, ValueError) as exc:
            activity_error = f"{type(exc).__name__}: {exc}"

    activity_known = recorded_status == "running" and protocol_known and activity_error is None
    running_active = lock_held is True if activity_known else None
    stale_running = lock_held is False if activity_known else None
    recovery_hint = None
    if stale_running is True:
        recovery_hint = (
            f"The previous {name} process no longer owns its OS lock. Rerun the same command "
            "without --force to resume from valid caches/artifacts."
        )

    return {
        "recorded_status": recorded_status,
        "recorded_fingerprint": recorded_fingerprint,
        "artifact_complete": artifact_complete,
        "runner": runner,
        "activity_known": activity_known,
        "run_lock_held": lock_held,
        "running_active": running_active,
        "stale_running": stale_running,
        "activity_error": activity_error,
        "recovery_hint": recovery_hint,
        "current_fingerprint": None,
        "fingerprint_current": None,
        "current_complete": None,
    }


def _training_operation_status(
    store: StateStore,
    state: dict[str, Any],
) -> dict[str, Any]:
    stages = state.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    stage = stages.get("train")
    stage = stage if isinstance(stage, dict) else {}
    progress = stage.get("progress")
    progress = progress if isinstance(progress, dict) else {}
    submission = progress.get("submission")
    submission = submission if isinstance(submission, dict) else {}
    bundle_audit = submission.get("bundle_audit")
    if not isinstance(bundle_audit, dict):
        bundle_audit = progress.get("bundle_audit")
    bundle_audit = bundle_audit if isinstance(bundle_audit, dict) else None
    result = stage.get("result")
    result = result if isinstance(result, dict) else {}
    executor_result = result.get("executor")
    executor_result = executor_result if isinstance(executor_result, dict) else {}
    quality = result.get("quality_gate")
    quality = quality if isinstance(quality, dict) else {}

    recorded_fingerprint = stage.get("fingerprint")
    candidate_complete = False
    published_complete = False
    if isinstance(recorded_fingerprint, str) and recorded_fingerprint:
        try:
            candidate_complete = store.is_trained(recorded_fingerprint)
        except (OSError, RuntimeError, TypeError, ValueError):
            candidate_complete = False
        try:
            published_complete = store.is_complete("train", recorded_fingerprint)
        except (OSError, RuntimeError, TypeError, ValueError):
            published_complete = False

    plan_fingerprint = result.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or not plan_fingerprint:
        plan_fingerprint = submission.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or not plan_fingerprint:
        plan_fingerprint = progress.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or not plan_fingerprint:
        plan_fingerprint = None

    families_value = result.get("families")
    families_value = families_value if isinstance(families_value, dict) else {}
    families: dict[str, dict[str, Any]] = {}
    for name in ("irodori", "lfm", "seed-vc"):
        family = families_value.get(name)
        if not isinstance(family, dict):
            continue
        enabled = family.get("enabled") is True
        artifact_recorded = isinstance(family.get("artifact"), str) and bool(family.get("artifact"))
        families[name] = {
            "enabled": enabled,
            "method": family.get("method") if isinstance(family.get("method"), str) else None,
            "candidate_recorded": enabled and artifact_recorded,
            "candidate_ready": enabled and artifact_recorded and candidate_complete,
            "published": enabled and artifact_recorded and published_complete,
        }

    executor = progress.get("executor")
    if not isinstance(executor, str) or not executor:
        executor = executor_result.get("kind")
    if not isinstance(executor, str) or not executor:
        executor = None
    call_id = submission.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        call_id = None
    checkpoint = progress.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        checkpoint = None
    step = progress.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        step = None

    return {
        "executor": executor,
        "remote_call_id": call_id,
        "remote_state": (
            progress.get("remote_state") if isinstance(progress.get("remote_state"), str) else None
        ),
        "model": progress.get("model") if isinstance(progress.get("model"), str) else None,
        "step": step,
        "checkpoint": checkpoint,
        "bundle_audit": _without_secret_values(bundle_audit),
        "plan_fingerprint": plan_fingerprint,
        "local_preflight": _without_secret_values(executor_result.get("local_preflight")),
        "candidate_complete": candidate_complete,
        "published_complete": published_complete,
        "quality_gate": _without_secret_values(quality),
        "families": families,
    }


def persona_status(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    """Return raw persona state plus inexpensive artifact and optional input freshness audits."""

    try:
        local_training_preflight = training_preflight_status(repo_root, cfg.training)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        local_training_preflight = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        modal = modal_readiness_status()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        modal = {
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "network_probe_performed": False,
        }
    store = StateStore(paths.state)
    try:
        state = store.load()
    except (OSError, ValueError, TypeError) as exc:
        return {
            "config": _without_secret_values(cfg.model_dump(mode="json")),
            "state": None,
            "audit": {
                "state_readable": False,
                "state_error": f"{type(exc).__name__}: {exc}",
                "prepare": None,
                "train": None,
                "inputs_verified": False,
                "local_training_preflight": local_training_preflight,
                "modal": modal,
            },
        }
    if not isinstance(state, dict):
        return {
            "config": _without_secret_values(cfg.model_dump(mode="json")),
            "state": _without_secret_values(state),
            "audit": {
                "state_readable": False,
                "state_error": "state.json root is not an object",
                "prepare": None,
                "train": None,
                "inputs_verified": False,
                "local_training_preflight": local_training_preflight,
                "modal": modal,
            },
        }

    prepare = _stage_audit(store, state, "prepare")
    prepare["batch_progress"] = prepare_batch_progress(paths.root)
    train = _stage_audit(store, state, "train")
    train["blocked_by_prepare"] = None
    train["operation"] = _training_operation_status(store, state)
    if verify_inputs:
        try:
            current_prepare = _prepare_fingerprint(paths, cfg)
        except (OSError, ValueError, TypeError) as exc:
            prepare["verification_error"] = f"{type(exc).__name__}: {exc}"
        else:
            prepare["current_fingerprint"] = current_prepare
            prepare["fingerprint_current"] = prepare["recorded_fingerprint"] == current_prepare
            try:
                prepare["current_complete"] = store.is_complete("prepare", current_prepare)
            except (OSError, ValueError, TypeError):
                prepare["current_complete"] = False

        try:
            current_train = _training_fingerprint(paths, cfg)
        except (OSError, ValueError, TypeError) as exc:
            train["verification_error"] = f"{type(exc).__name__}: {exc}"
        else:
            train["current_fingerprint"] = current_train
            train["fingerprint_current"] = train["recorded_fingerprint"] == current_train
            try:
                train["current_complete"] = store.is_complete("train", current_train)
            except (OSError, ValueError, TypeError):
                train["current_complete"] = False

        # Training has the same hard dependency in train_persona: a current,
        # complete prepare stage is required before a trained artifact can be
        # considered valid for the current raw/identity inputs. A still-matching
        # dataset fingerprint must not mask stale source media in status output.
        prepare_ready = (
            prepare.get("fingerprint_current") is True and prepare.get("current_complete") is True
        )
        train["blocked_by_prepare"] = not prepare_ready
        if not prepare_ready:
            train["current_complete"] = False

    return {
        "config": _without_secret_values(cfg.model_dump(mode="json")),
        "state": _without_secret_values(state),
        "audit": {
            "state_readable": True,
            "state_error": None,
            "prepare": prepare,
            "train": train,
            "inputs_verified": verify_inputs,
            "local_training_preflight": local_training_preflight,
            "modal": modal,
        },
    }
