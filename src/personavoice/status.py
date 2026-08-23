from __future__ import annotations

from pathlib import Path
from typing import Any

from personavoice.config import PersonaConfig
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


def persona_status(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    """Return raw persona state plus inexpensive artifact and optional input freshness audits."""

    store = StateStore(paths.state)
    try:
        state = store.load()
    except (OSError, ValueError, TypeError) as exc:
        return {
            "config": cfg.model_dump(mode="json"),
            "state": None,
            "audit": {
                "state_readable": False,
                "state_error": f"{type(exc).__name__}: {exc}",
                "prepare": None,
                "train": None,
                "inputs_verified": False,
            },
        }
    if not isinstance(state, dict):
        return {
            "config": cfg.model_dump(mode="json"),
            "state": state,
            "audit": {
                "state_readable": False,
                "state_error": "state.json root is not an object",
                "prepare": None,
                "train": None,
                "inputs_verified": False,
            },
        }

    prepare = _stage_audit(store, state, "prepare")
    prepare["batch_progress"] = prepare_batch_progress(paths.root)
    train = _stage_audit(store, state, "train")
    train["blocked_by_prepare"] = None
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
            prepare.get("fingerprint_current") is True
            and prepare.get("current_complete") is True
        )
        train["blocked_by_prepare"] = not prepare_ready
        if not prepare_ready:
            train["current_complete"] = False

    return {
        "config": cfg.model_dump(mode="json"),
        "state": state,
        "audit": {
            "state_readable": True,
            "state_error": None,
            "prepare": prepare,
            "train": train,
            "inputs_verified": verify_inputs,
        },
    }
