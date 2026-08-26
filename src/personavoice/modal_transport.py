from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import uuid4

from personavoice.environment import SECRET_ENV_KEYS
from personavoice.training_bundle import COMPLETION_PATH as BUNDLE_COMPLETION_PATH
from personavoice.training_bundle import (
    TrainingBundle,
    canonical_plan_bytes,
    verify_training_bundle,
)
from personavoice.training_plan import FamilyPlan, TrainingPlan

REMOTE_CONTRACT_SCHEMA_VERSION = 1
REMOTE_REDIRECT_SCHEMA_VERSION = 1
REMOTE_RECOVERY_SCHEMA_VERSION = 1
REMOTE_RECOVERY_FUNCTION_NAME = "recover_terminal_claim"
BUNDLE_TRANSFER_AUDIT_SCHEMA_VERSION = 1
CHECKPOINT_COMPLETION_NAME = "checkpoint-complete.json"
CHECKPOINT_FAMILY_NAME = "checkpoint-family.json"
RESULT_COMPLETION_NAME = "result-complete.json"
TRAINING_RESULT_NAME = "training-result.json"

_HASH_CHUNK_BYTES = 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CALL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_DIRECTORY = re.compile(r"^checkpoint-(\d+)$")
_SECRET_KEYS = frozenset(
    {
        "access_token",
        "api_token",
        "credential",
        "credentials",
        "hf_token",
        "modal_token_id",
        "modal_token_secret",
        "password",
        "secret",
        "token",
        "token_id",
        "token_secret",
    }
)


class ModalFunctionTerminalError(RuntimeError):
    """Internal signal that a durable Modal FunctionCall is terminally failed."""

    def __init__(self, call_id: str) -> None:
        super().__init__("The durable Modal FunctionCall reached a terminal failure")
        self.call_id = call_id


class ModalTerminalCallRecoveredError(RuntimeError):
    """A terminal call's persistent claims were safely released for a rerun."""

    def __init__(self, call_id: str) -> None:
        super().__init__(
            "Modal exhausted its retries, and the failed call's durable claims were "
            "released. Rerun `persona train --executor modal`; the replacement call "
            "will resume only from fully verified method-native checkpoints."
        )
        self.call_id = call_id


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker is not None and checker())


def _portable_relative(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} is not a portable path: {value!r}")
    if value.startswith(("/", "~")) or ":" in value:
        raise ValueError(f"{label} is absolute or platform-specific: {value!r}")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{label} contains traversal: {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"{label} is not canonical: {value!r}")
    return path


def _remote_absolute(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} must be an absolute Volume path")
    if "\x00" in value or ":" in value:
        raise ValueError(f"{label} is invalid")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value[1:].split("/")):
        raise ValueError(f"{label} contains traversal")
    if path.as_posix() != value:
        raise ValueError(f"{label} is not canonical")
    return path


def _configured_secret_values() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                secret
                for key in SECRET_ENV_KEYS
                if isinstance((secret := os.environ.get(key)), str) and secret
            },
            key=len,
            reverse=True,
        )
    )


def _reject_secret_mapping(
    value: Any,
    *,
    label: str,
    secret_values: tuple[str, ...] | None = None,
) -> None:
    if secret_values is None:
        secret_values = _configured_secret_values()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            folded = key_text.casefold().replace("-", "_")
            if any(secret in key_text for secret in secret_values):
                raise ValueError(f"{label} contains a configured secret value")
            if (
                folded in _SECRET_KEYS
                or "credential" in folded
                or "password" in folded
                or "secret" in folded
            ):
                raise ValueError(f"{label} contains secret-like data: {key}")
            _reject_secret_mapping(child, label=label, secret_values=secret_values)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_secret_mapping(child, label=label, secret_values=secret_values)
    elif isinstance(value, str) and any(secret in value for secret in secret_values):
        raise ValueError(f"{label} contains a configured secret value")


def _reject_absolute_strings(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_absolute_strings(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_absolute_strings(child, label=label)
    elif isinstance(value, str) and (
        value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError(f"{label} contains an absolute local path")


@dataclass(frozen=True)
class ModalAuthStatus:
    configured: bool
    source: str

    def as_dict(self) -> dict[str, str | bool]:
        return {"configured": self.configured, "source": self.source}


def _profile_contains_token_pair(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    token_id = value.get("token_id")
    token_secret = value.get("token_secret")
    if isinstance(token_id, str) and token_id.strip() and isinstance(token_secret, str) and token_secret.strip():
        return True
    return any(_profile_contains_token_pair(child) for child in value.values())


def detect_modal_auth(
    *,
    env: Mapping[str, str] | None = None,
    profile_path: Path | None = None,
) -> ModalAuthStatus:
    """Report only whether credentials exist and which non-secret source provides them."""

    values = os.environ if env is None else env
    token_id = values.get("MODAL_TOKEN_ID", "").strip()
    token_secret = values.get("MODAL_TOKEN_SECRET", "").strip()
    if token_id and token_secret:
        return ModalAuthStatus(configured=True, source="environment")
    if token_id or token_secret:
        return ModalAuthStatus(configured=False, source="environment-incomplete")

    path = Path.home() / ".modal.toml" if profile_path is None else profile_path
    if not path.is_file():
        return ModalAuthStatus(configured=False, source="none")
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return ModalAuthStatus(configured=False, source="profile-invalid")
    if _profile_contains_token_pair(value):
        return ModalAuthStatus(configured=True, source="profile")
    return ModalAuthStatus(configured=False, source="profile-incomplete")


def _setting_name(value: str, *, label: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"Invalid Modal {label}: {value!r}")
    return value


@dataclass(frozen=True)
class ModalSettings:
    app_name: str
    function_name: str
    volume_name: str
    environment_name: str | None = None

    def __post_init__(self) -> None:
        _setting_name(self.app_name, label="app name")
        _setting_name(self.function_name, label="function name")
        _setting_name(self.volume_name, label="volume name")
        if self.environment_name is not None:
            _setting_name(self.environment_name, label="environment name")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModalSettings:
        values = os.environ if env is None else env
        return cls(
            app_name=values.get("PERSONAVOICE_MODAL_APP", "personavoice-training"),
            function_name=values.get("PERSONAVOICE_MODAL_FUNCTION", "train"),
            volume_name=values.get("PERSONAVOICE_MODAL_VOLUME", "personavoice-training"),
            environment_name=values.get("MODAL_ENVIRONMENT") or None,
        )


@dataclass(frozen=True)
class CompletionFile:
    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class CompletionManifest:
    kind: str
    plan_fingerprint: str
    model: str
    step: int
    checkpoint: str | None
    quality_gate_passed: bool
    files: tuple[CompletionFile, ...]
    fingerprint: str
    status: str = "complete"
    schema_version: int = REMOTE_CONTRACT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        plan_fingerprint: str,
        model: str,
        step: int,
        checkpoint: str | None,
        quality_gate_passed: bool,
        files: Iterable[CompletionFile],
    ) -> CompletionManifest:
        ordered = tuple(sorted(files, key=lambda item: item.path))
        payload = {
            "schema_version": REMOTE_CONTRACT_SCHEMA_VERSION,
            "kind": kind,
            "plan_fingerprint": plan_fingerprint,
            "model": model,
            "step": step,
            "checkpoint": checkpoint,
            "status": "complete",
            "quality_gate_passed": quality_gate_passed,
            "files": [item.as_dict() for item in ordered],
        }
        return cls(
            kind=kind,
            plan_fingerprint=plan_fingerprint,
            model=model,
            step=step,
            checkpoint=checkpoint,
            quality_gate_passed=quality_gate_passed,
            files=ordered,
            fingerprint=_sha256_bytes(_canonical_json_bytes(payload)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "plan_fingerprint": self.plan_fingerprint,
            "model": self.model,
            "step": self.step,
            "checkpoint": self.checkpoint,
            "status": self.status,
            "quality_gate_passed": self.quality_gate_passed,
            "files": [item.as_dict() for item in self.files],
            "fingerprint": self.fingerprint,
        }


def _completion_from_value(value: Any) -> CompletionManifest:
    if not isinstance(value, Mapping):
        raise ValueError("Completion manifest must be an object")
    _reject_secret_mapping(value, label="Completion manifest")
    expected_keys = {
        "schema_version",
        "kind",
        "plan_fingerprint",
        "model",
        "step",
        "checkpoint",
        "status",
        "quality_gate_passed",
        "files",
        "fingerprint",
    }
    if set(value) != expected_keys or value.get("schema_version") != REMOTE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Completion manifest schema is invalid")
    if value.get("status") != "complete" or value.get("kind") not in {"checkpoint", "result"}:
        raise ValueError("Completion manifest is not complete")
    plan_fingerprint = value.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or len(plan_fingerprint) != 64:
        raise ValueError("Completion manifest plan fingerprint is invalid")
    model = value.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Completion manifest model is invalid")
    step = value.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Completion manifest step is invalid")
    checkpoint = value.get("checkpoint")
    if checkpoint is not None:
        _portable_relative(checkpoint, label="Completion checkpoint")
    quality = value.get("quality_gate_passed")
    if not isinstance(quality, bool):
        raise ValueError("Completion manifest quality status is invalid")
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Completion manifest files are invalid")
    files: list[CompletionFile] = []
    for raw in raw_files:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "size"}:
            raise ValueError("Completion manifest file record is invalid")
        _portable_relative(raw.get("path"), label="Completion file")
        digest = raw.get("sha256")
        size = raw.get("size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Completion manifest file checksum is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Completion manifest file size is invalid")
        files.append(CompletionFile(path=raw["path"], sha256=digest, size=size))
    rebuilt = CompletionManifest.create(
        kind=value["kind"],
        plan_fingerprint=plan_fingerprint,
        model=model,
        step=step,
        checkpoint=checkpoint,
        quality_gate_passed=quality,
        files=files,
    )
    if tuple(files) != rebuilt.files or value.get("fingerprint") != rebuilt.fingerprint:
        raise ValueError("Completion manifest fingerprint or ordering is invalid")
    return rebuilt


@dataclass(frozen=True)
class ResultCandidate:
    artifact_path: str
    validation: Mapping[str, Any]
    files: tuple[CompletionFile, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "validation": dict(self.validation),
            "files": [item.as_dict() for item in self.files],
        }


@dataclass(frozen=True)
class ResultFamily:
    family: str
    method: str
    family_fingerprint: str
    selected_artifact_path: str
    candidates: tuple[ResultCandidate, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "method": self.method,
            "family_fingerprint": self.family_fingerprint,
            "selected_artifact_path": self.selected_artifact_path,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class TrainingResultContract:
    plan_fingerprint: str
    families: tuple[ResultFamily, ...]
    schema_version: int = REMOTE_CONTRACT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_fingerprint": self.plan_fingerprint,
            "families": [family.as_dict() for family in self.families],
        }


def _validate_result_candidate(
    value: Any,
    *,
    family: str,
    method: str,
) -> ResultCandidate:
    if not isinstance(value, Mapping) or set(value) != {"artifact_path", "validation", "files"}:
        raise ValueError(f"Training result candidate for {family} is invalid")
    artifact = _portable_relative(value.get("artifact_path"), label="Result artifact path")
    validation = value.get("validation")
    if not isinstance(validation, Mapping) or not isinstance(validation.get("passed"), bool):
        raise ValueError(f"Training result candidate for {family} has no gate decision")
    validation_loss = validation.get("validation_loss")
    finite_loss = (
        isinstance(validation_loss, (int, float))
        and not isinstance(validation_loss, bool)
        and math.isfinite(float(validation_loss))
    )
    if not finite_loss:
        raise ValueError(f"Training result candidate for {family} has no finite validation loss")
    _reject_secret_mapping(validation, label="Training result validation")
    _reject_absolute_strings(validation, label="Training result validation")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"Training result candidate for {family} has no file checksums")
    files: list[CompletionFile] = []
    for raw in raw_files:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "size"}:
            raise ValueError(f"Training result candidate file for {family} is invalid")
        relative = _portable_relative(raw.get("path"), label="Result candidate file")
        digest = raw.get("sha256")
        size = raw.get("size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Training result candidate checksum for {family} is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"Training result candidate size for {family} is invalid")
        artifact_prefix = artifact.as_posix().rstrip("/") + "/"
        if not relative.as_posix().startswith(artifact_prefix):
            raise ValueError(f"Training result candidate file for {family} escapes its artifact")
        files.append(CompletionFile(path=relative.as_posix(), sha256=digest, size=size))
    if tuple(sorted(files, key=lambda item: item.path)) != tuple(files) or len(
        {item.path for item in files}
    ) != len(files):
        raise ValueError(f"Training result candidate files for {family} are not canonical")
    return ResultCandidate(
        artifact_path=artifact.as_posix(),
        validation=dict(validation),
        files=tuple(files),
    )


def training_result_from_value(
    value: Any,
    *,
    expected_plan_fingerprint: str,
    expected_families: Mapping[str, tuple[str, str]],
) -> TrainingResultContract:
    """Parse the portable handoff from remote candidates to local publication."""

    _reject_secret_mapping(value, label="Training result")
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "plan_fingerprint", "families"}
        or value.get("schema_version") != REMOTE_CONTRACT_SCHEMA_VERSION
        or value.get("plan_fingerprint") != expected_plan_fingerprint
    ):
        raise ValueError("Training result root contract is invalid")
    raw_families = value.get("families")
    if not isinstance(raw_families, list):
        raise ValueError("Training result families must be a list")
    raw_family_names = [
        raw.get("family") if isinstance(raw, Mapping) else None for raw in raw_families
    ]
    if raw_family_names != sorted(raw_family_names, key=lambda item: str(item)):
        raise ValueError("Training result families are not canonically ordered")
    families: list[ResultFamily] = []
    seen: set[str] = set()
    for raw in raw_families:
        expected_keys = {
            "family",
            "method",
            "family_fingerprint",
            "selected_artifact_path",
            "candidates",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise ValueError("Training result family contract is invalid")
        family = raw.get("family")
        if not isinstance(family, str) or family in seen or family not in expected_families:
            raise ValueError(f"Training result family is unexpected: {family!r}")
        seen.add(family)
        expected_fingerprint, expected_method = expected_families[family]
        if (
            raw.get("family_fingerprint") != expected_fingerprint
            or raw.get("method") != expected_method
        ):
            raise ValueError(f"Training result contract changed the {family} plan")
        raw_candidates = raw.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"Training result has no {family} candidates")
        candidates = tuple(
            _validate_result_candidate(candidate, family=family, method=expected_method)
            for candidate in raw_candidates
        )
        paths = [candidate.artifact_path for candidate in candidates]
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise ValueError(f"Training result repeats a {family} candidate path")
        selected = _portable_relative(
            raw.get("selected_artifact_path"),
            label="Selected result artifact path",
        ).as_posix()
        selected_candidates = [candidate for candidate in candidates if candidate.artifact_path == selected]
        if len(selected_candidates) != 1 or selected_candidates[0].validation.get("passed") is not True:
            raise ValueError(f"Training result selected {family} candidate did not pass")
        families.append(
            ResultFamily(
                family=family,
                method=expected_method,
                family_fingerprint=expected_fingerprint,
                selected_artifact_path=selected,
                candidates=candidates,
            )
        )
    if seen != set(expected_families):
        raise ValueError("Training result is missing an enabled family")
    families.sort(key=lambda item: item.family)
    candidate_paths = [
        candidate.artifact_path
        for family in families
        for candidate in family.candidates
    ]
    for index, path in enumerate(candidate_paths):
        for other in candidate_paths[index + 1 :]:
            if path == other or path.startswith(other + "/") or other.startswith(path + "/"):
                raise ValueError("Training result candidate artifact paths overlap")
    return TrainingResultContract(
        plan_fingerprint=expected_plan_fingerprint,
        families=tuple(families),
    )


def write_training_result_contract(
    directory: Path,
    *,
    plan: TrainingPlan,
    families: Sequence[ResultFamily],
) -> TrainingResultContract:
    expected = {
        family.family: (family.fingerprint, family.method)
        for family in plan.families
        if family.enabled
    }
    materialized_families: list[ResultFamily] = []
    for family in families:
        candidates: list[ResultCandidate] = []
        for candidate in family.candidates:
            path = directory.joinpath(*PurePosixPath(candidate.artifact_path).parts)
            if path.is_symlink() or _is_junction(path) or not path.is_dir():
                raise ValueError(f"Training result candidate directory is missing: {candidate.artifact_path}")
            relative_files = _iter_payload_files(path, completion_name="__no_completion_marker__")
            if not relative_files:
                raise ValueError(f"Training result candidate directory is empty: {candidate.artifact_path}")
            files = tuple(
                CompletionFile(
                    path=PurePosixPath(candidate.artifact_path, item.path).as_posix(),
                    sha256=item.sha256,
                    size=item.size,
                )
                for item in relative_files
            )
            candidates.append(
                ResultCandidate(
                    artifact_path=candidate.artifact_path,
                    validation=candidate.validation,
                    files=files,
                )
            )
        materialized_families.append(
            ResultFamily(
                family=family.family,
                method=family.method,
                family_fingerprint=family.family_fingerprint,
                selected_artifact_path=family.selected_artifact_path,
                candidates=tuple(sorted(candidates, key=lambda item: item.artifact_path)),
            )
        )
    materialized_families.sort(key=lambda item: item.family)
    raw = {
        "schema_version": REMOTE_CONTRACT_SCHEMA_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "families": [family.as_dict() for family in materialized_families],
    }
    contract = training_result_from_value(
        raw,
        expected_plan_fingerprint=plan.fingerprint,
        expected_families=expected,
    )
    _write_atomic_bytes(directory / TRAINING_RESULT_NAME, _canonical_json_bytes(contract.as_dict()))
    return contract


def _iter_payload_files(root: Path, *, completion_name: str) -> tuple[CompletionFile, ...]:
    files: list[CompletionFile] = []
    for path in root.rglob("*"):
        relative = PurePosixPath(path.relative_to(root)).as_posix()
        if path.is_symlink() or _is_junction(path):
            raise ValueError(f"Completed directory contains a link or junction: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Completed directory contains a non-regular entry: {relative}")
        if relative == completion_name:
            continue
        portable = _portable_relative(relative, label="Completed payload file")
        files.append(
            CompletionFile(
                path=portable.as_posix(),
                sha256=_sha256_file(path),
                size=path.stat().st_size,
            )
        )
    return tuple(sorted(files, key=lambda item: item.path))


def write_completion_manifest(
    directory: Path,
    *,
    kind: str,
    plan_fingerprint: str,
    model: str,
    step: int,
    checkpoint: str | None = None,
    quality_gate_passed: bool,
    completion_name: str | None = None,
) -> CompletionManifest:
    """Hash a finished directory and atomically write its completion marker last."""

    if kind not in {"checkpoint", "result"}:
        raise ValueError(f"Unknown completion kind: {kind}")
    name = completion_name or (
        CHECKPOINT_COMPLETION_NAME if kind == "checkpoint" else RESULT_COMPLETION_NAME
    )
    _portable_relative(name, label="Completion manifest name")
    if directory.is_symlink() or _is_junction(directory) or not directory.is_dir():
        raise ValueError("Completed directory must be a real directory")
    manifest = CompletionManifest.create(
        kind=kind,
        plan_fingerprint=plan_fingerprint,
        model=model,
        step=step,
        checkpoint=checkpoint,
        quality_gate_passed=quality_gate_passed,
        files=_iter_payload_files(directory, completion_name=name),
    )
    _reject_secret_mapping(manifest.as_dict(), label="Completion manifest")
    _write_atomic_bytes(directory / name, _canonical_json_bytes(manifest.as_dict()))
    return manifest


def verify_completed_directory(
    directory: Path,
    *,
    expected_plan_fingerprint: str,
    completion_name: str,
    expected_kind: str | None = None,
    expected_model: str | None = None,
    require_quality_gate: bool = False,
) -> CompletionManifest:
    if directory.is_symlink() or _is_junction(directory) or not directory.is_dir():
        raise ValueError("Completed directory must be a real directory")
    completion_relative = _portable_relative(completion_name, label="Completion manifest name")
    completion = directory.joinpath(*completion_relative.parts)
    if completion.is_symlink() or _is_junction(completion) or not completion.is_file():
        raise ValueError("Completion marker is missing")
    try:
        value = json.loads(completion.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Completion marker is unreadable") from exc
    manifest = _completion_from_value(value)
    if manifest.plan_fingerprint != expected_plan_fingerprint:
        raise ValueError("Completed directory belongs to a different TrainingPlan")
    if expected_kind is not None and manifest.kind != expected_kind:
        raise ValueError("Completion manifest kind is incorrect")
    if expected_model is not None and manifest.model != expected_model:
        raise ValueError("Completion manifest model is incorrect")
    if require_quality_gate and not manifest.quality_gate_passed:
        raise ValueError("Result cannot be published because its quality gate did not pass")

    actual = _iter_payload_files(directory, completion_name=completion_name)
    if actual != manifest.files:
        raise ValueError("Completed directory file inventory or checksum is invalid")
    return manifest


def latest_verified_checkpoint(
    checkpoint_root: Path,
    *,
    plan_fingerprint: str,
    model: str,
) -> tuple[Path, CompletionManifest] | None:
    """Return the newest fully checksummed checkpoint, ignoring partial writes."""

    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink() or _is_junction(checkpoint_root):
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        match = _CHECKPOINT_DIRECTORY.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink() and not _is_junction(path):
            candidates.append((int(match.group(1)), path))
    for directory_step, path in sorted(candidates, reverse=True):
        try:
            manifest = verify_completed_directory(
                path,
                expected_plan_fingerprint=plan_fingerprint,
                completion_name=CHECKPOINT_COMPLETION_NAME,
                expected_kind="checkpoint",
                expected_model=model,
            )
        except ValueError:
            continue
        if manifest.step == directory_step:
            return path, manifest
    return None


def write_checkpoint_family_contract(directory: Path, family: FamilyPlan) -> Path:
    """Bind a resumable checkpoint to the training-compatible family contract.

    The overall plan also contains publication/evaluation policy.  Those values
    may change without making optimizer, scheduler, or dataloader state
    incompatible, so the durable checkpoint namespace uses ``FamilyPlan``'s
    fingerprint and records that digest plus the family, method, and dataset
    identity inside the checksummed payload.
    """

    if directory.is_symlink() or _is_junction(directory) or not directory.is_dir():
        raise ValueError("Checkpoint family contract requires a real directory")
    value = {
        "schema_version": REMOTE_CONTRACT_SCHEMA_VERSION,
        "family": family.family,
        "family_fingerprint": family.fingerprint,
        "method": family.method,
        "dataset_fingerprint": family.dataset_fingerprint,
    }
    path = directory / CHECKPOINT_FAMILY_NAME
    _write_atomic_bytes(path, _canonical_json_bytes(value))
    return path


def _verify_checkpoint_family_contract(directory: Path, family: FamilyPlan) -> None:
    path = directory / CHECKPOINT_FAMILY_NAME
    if path.is_symlink() or _is_junction(path) or not path.is_file():
        raise ValueError("Checkpoint family contract is missing")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Checkpoint family contract is unreadable") from exc
    expected = {
        "schema_version": REMOTE_CONTRACT_SCHEMA_VERSION,
        "family": family.family,
        "family_fingerprint": family.fingerprint,
        "method": family.method,
        "dataset_fingerprint": family.dataset_fingerprint,
    }
    if value != expected:
        raise ValueError("Checkpoint belongs to a different family training contract")


def latest_verified_family_checkpoint(
    checkpoint_root: Path,
    *,
    plan_fingerprint: str,
    family: FamilyPlan,
    rebind_plan_marker: bool = True,
) -> tuple[Path, CompletionManifest] | None:
    """Return the latest family-compatible checkpoint.

    The completion marker is still tied to the invoking overall TrainingPlan so
    every remote run has an auditable provenance chain.  When only plan-level
    policy changes, a fully checksummed checkpoint with the exact embedded family
    fingerprint can be reused. A writer that owns the family claim may atomically
    re-issue its marker for the new plan after verification. Pure status/resume-
    index readers pass ``rebind_plan_marker=False`` so inspecting a shared
    namespace can never race or mutate another compatible plan's run.
    """

    if (
        checkpoint_root.name != family.fingerprint
        or checkpoint_root.parent.name != family.family
        or not checkpoint_root.is_dir()
        or checkpoint_root.is_symlink()
        or _is_junction(checkpoint_root)
    ):
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        match = _CHECKPOINT_DIRECTORY.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink() and not _is_junction(path):
            candidates.append((int(match.group(1)), path))
    for directory_step, path in sorted(candidates, reverse=True):
        completion = path / CHECKPOINT_COMPLETION_NAME
        try:
            raw = json.loads(completion.read_bytes())
            advertised = _completion_from_value(raw)
            manifest = verify_completed_directory(
                path,
                expected_plan_fingerprint=advertised.plan_fingerprint,
                completion_name=CHECKPOINT_COMPLETION_NAME,
                expected_kind="checkpoint",
                expected_model=family.family,
            )
            _verify_checkpoint_family_contract(path, family)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if manifest.step != directory_step:
            continue
        if manifest.plan_fingerprint != plan_fingerprint and rebind_plan_marker:
            manifest = write_completion_manifest(
                path,
                kind="checkpoint",
                plan_fingerprint=plan_fingerprint,
                model=family.family,
                step=manifest.step,
                checkpoint=manifest.checkpoint,
                quality_gate_passed=manifest.quality_gate_passed,
            )
        return path, manifest
    return None


@dataclass(frozen=True)
class UploadMember:
    local_path: Path
    remote_path: str
    sha256: str
    size: int


class ModalBackend(Protocol):
    def upload_payload(self, members: Sequence[UploadMember]) -> None: ...

    def upload_completion(self, member: UploadMember) -> None: ...

    def spawn(self, payload: Mapping[str, Any]) -> str: ...

    def poll(self, call_id: str, *, timeout: float) -> Mapping[str, Any] | None: ...

    def recover_terminal_call(
        self,
        submission: RemoteSubmission,
        *,
        call_id: str,
    ) -> str: ...

    def read_file(self, remote_path: str) -> bytes | Iterable[bytes]: ...


def _modal_poll_exception_state(modal: Any, exc: Exception) -> str:
    """Classify an official SDK poll exception without mistaking wait for failure.

    Modal 1.5.x raises ``modal.exception.TimeoutError`` (not Python's built-in
    ``TimeoutError``) when ``FunctionCall.get(timeout=...)`` is merely still
    running.  Hard function timeouts and returned remote exceptions are
    terminal outputs.  Authentication, transport and service failures are kept
    indeterminate so they can never authorize destructive claim recovery.
    """

    exceptions = getattr(modal, "exception", None)
    if exceptions is None:
        # SDK-shaped test doubles and incompatible SDKs must fail closed. A
        # remote user exception can be arbitrary, so only the official module
        # supplies enough type information to classify it safely.
        return "indeterminate"
    timeout_type = getattr(exceptions, "TimeoutError", None)
    function_timeout_type = getattr(exceptions, "FunctionTimeoutError", None)
    output_expired_type = getattr(exceptions, "OutputExpiredError", None)
    remote_error_type = getattr(exceptions, "RemoteError", None)
    execution_error_type = getattr(exceptions, "ExecutionError", None)
    internal_failure_type = getattr(exceptions, "InternalFailure", None)
    modal_error_type = getattr(exceptions, "Error", None)
    if isinstance(function_timeout_type, type) and isinstance(exc, function_timeout_type):
        return "terminal"
    if isinstance(output_expired_type, type) and isinstance(exc, output_expired_type):
        # An expired output does not prove whether the old invocation completed
        # successfully, so it cannot authorize claim removal.
        return "indeterminate"
    if isinstance(timeout_type, type) and type(exc) is timeout_type:
        return "running"
    terminal_sdk_types = tuple(
        kind
        for kind in (remote_error_type, execution_error_type, internal_failure_type)
        if isinstance(kind, type)
    )
    if terminal_sdk_types and isinstance(exc, terminal_sdk_types):
        return "terminal"
    if isinstance(modal_error_type, type) and isinstance(exc, modal_error_type):
        return "indeterminate"
    # Modal deserializes ordinary user exceptions and raises their original
    # Python type only after a terminal output exists.
    return "terminal"


class ModalSDKBackend:
    """Thin lazy adapter around Modal's official deployed Function/Volume APIs."""

    def __init__(self, settings: ModalSettings, *, modal_module: Any | None = None) -> None:
        self.settings = settings
        self._modal_module = modal_module

    def _modal(self) -> Any:
        if self._modal_module is None:
            try:
                self._modal_module = importlib.import_module("modal")
            except ImportError as exc:
                raise RuntimeError(
                    "Modal support is optional; install the Modal SDK to use the modal executor"
                ) from exc
        return self._modal_module

    def _lookup_kwargs(self) -> dict[str, str]:
        return (
            {"environment_name": self.settings.environment_name}
            if self.settings.environment_name is not None
            else {}
        )

    def _volume(self) -> Any:
        modal = self._modal()
        return modal.Volume.from_name(
            self.settings.volume_name,
            create_if_missing=True,
            **self._lookup_kwargs(),
        )

    def upload_payload(self, members: Sequence[UploadMember]) -> None:
        volume = self._volume()
        with volume.batch_upload(force=True) as batch:
            for member in sorted(members, key=lambda item: item.remote_path):
                batch.put_file(str(member.local_path), member.remote_path)

    def upload_completion(self, member: UploadMember) -> None:
        volume = self._volume()
        # A separate committed batch makes the marker the final remote write.
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(member.local_path), member.remote_path)

    def spawn(self, payload: Mapping[str, Any]) -> str:
        modal = self._modal()
        function = modal.Function.from_name(
            self.settings.app_name,
            self.settings.function_name,
            **self._lookup_kwargs(),
        )
        call = function.spawn(dict(payload))
        call_id = getattr(call, "object_id", None)
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError("Modal Function.spawn did not return a durable call object ID")
        return call_id

    def poll(self, call_id: str, *, timeout: float) -> Mapping[str, Any] | None:
        modal = self._modal()
        current_call_id = call_id
        visited: set[str] = set()
        # Duplicate dispatches can occur only when the local process is killed
        # after Modal accepted spawn() but before its FunctionCall ID was saved.
        # The remote app elects one canonical call with an atomic Modal Dict
        # claim.  Follow its strict redirect here so even a saved losing call is
        # a durable handle to the winner; no second trainer is allowed to race.
        while True:
            if current_call_id in visited or len(visited) >= 8:
                raise RuntimeError("Modal duplicate-call redirect cycle is invalid")
            visited.add(current_call_id)
            call = modal.FunctionCall.from_id(current_call_id)
            try:
                result = call.get(timeout=timeout)
            except Exception as exc:
                state = _modal_poll_exception_state(modal, exc)
                if state == "running":
                    return None
                if state == "terminal":
                    raise ModalFunctionTerminalError(current_call_id) from exc
                raise
            if not isinstance(result, Mapping):
                raise RuntimeError("Modal training function returned an invalid result contract")
            if result.get("remote_state") != "redirect":
                return result
            _reject_secret_mapping(result, label="Modal duplicate-call redirect")
            if set(result) != {
                "schema_version",
                "remote_state",
                "plan_fingerprint",
                "canonical_call_id",
            } or result.get("schema_version") != REMOTE_REDIRECT_SCHEMA_VERSION:
                raise RuntimeError("Modal duplicate-call redirect contract is invalid")
            plan_fingerprint = result.get("plan_fingerprint")
            canonical_call_id = result.get("canonical_call_id")
            if (
                not isinstance(plan_fingerprint, str)
                or _SHA256.fullmatch(plan_fingerprint) is None
                or not isinstance(canonical_call_id, str)
                or _CALL_ID.fullmatch(canonical_call_id) is None
            ):
                raise RuntimeError("Modal duplicate-call redirect values are invalid")
            current_call_id = canonical_call_id

    def recover_terminal_call(
        self,
        submission: RemoteSubmission,
        *,
        call_id: str,
    ) -> str:
        """Ask the serialized deployed recovery function to release stale claims.

        The recovery function independently polls the old FunctionCall before
        touching the Dict.  A local network error or an untrusted client claim
        therefore cannot remove a running trainer's plan/family ownership.
        """

        if not isinstance(call_id, str) or _CALL_ID.fullmatch(call_id) is None:
            raise ValueError("Terminal Modal call ID is invalid")
        modal = self._modal()
        payload = {
            "schema_version": REMOTE_RECOVERY_SCHEMA_VERSION,
            "plan_fingerprint": submission.plan_fingerprint,
            "bundle_fingerprint": submission.bundle_audit.bundle_fingerprint,
            "call_id": call_id,
            "family_contracts": [
                {
                    "family": family,
                    "fingerprint": fingerprint,
                    "method": method,
                }
                for family, fingerprint, method in submission.family_contracts
            ],
        }
        _reject_secret_mapping(payload, label="Modal terminal-call recovery payload")
        function = modal.Function.from_name(
            self.settings.app_name,
            REMOTE_RECOVERY_FUNCTION_NAME,
            **self._lookup_kwargs(),
        )
        result = function.remote(payload)
        expected = {
            "schema_version",
            "recovery_state",
            "plan_fingerprint",
            "call_id",
        }
        if (
            not isinstance(result, Mapping)
            or set(result) != expected
            or result.get("schema_version") != REMOTE_RECOVERY_SCHEMA_VERSION
            or result.get("plan_fingerprint") != submission.plan_fingerprint
            or result.get("call_id") != call_id
            or result.get("recovery_state")
            not in {"released", "running", "complete", "expired", "superseded"}
        ):
            raise RuntimeError("Modal terminal-call recovery returned an invalid contract")
        _reject_secret_mapping(result, label="Modal terminal-call recovery result")
        return str(result["recovery_state"])

    def read_file(self, remote_path: str) -> bytes | Iterable[bytes]:
        _remote_absolute(remote_path, label="Modal Volume read path")
        return self._volume().read_file(remote_path)


@dataclass(frozen=True)
class RemoteStatus:
    remote_state: str
    model: str
    step: int
    checkpoint: str | None
    executor: str = "modal"

    def as_dict(self) -> dict[str, Any]:
        return {
            "executor": self.executor,
            "remote_state": self.remote_state,
            "model": self.model,
            "step": self.step,
            "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True)
class BundleTransferFile:
    path: str
    role: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class BundleTransferAudit:
    """Complete, portable record of bytes authorized for one Modal upload."""

    bundle_fingerprint: str
    files: tuple[BundleTransferFile, ...]
    file_count: int
    total_bytes: int
    fingerprint: str
    schema_version: int = BUNDLE_TRANSFER_AUDIT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        bundle_fingerprint: str,
        files: Iterable[BundleTransferFile],
    ) -> BundleTransferAudit:
        if (
            not isinstance(bundle_fingerprint, str)
            or _SHA256.fullmatch(bundle_fingerprint) is None
        ):
            raise ValueError("Bundle transfer source fingerprint is invalid")
        ordered = tuple(sorted(files, key=lambda item: (item.path, item.role)))
        if not ordered or len({item.path for item in ordered}) != len(ordered):
            raise ValueError("Bundle transfer file list is empty or duplicated")
        for item in ordered:
            _portable_relative(item.path, label="Bundle transfer path")
            if not isinstance(item.role, str) or _SAFE_NAME.fullmatch(item.role) is None:
                raise ValueError("Bundle transfer role is invalid")
            if not isinstance(item.sha256, str) or _SHA256.fullmatch(item.sha256) is None:
                raise ValueError("Bundle transfer checksum is invalid")
            if not isinstance(item.size, int) or isinstance(item.size, bool) or item.size < 0:
                raise ValueError("Bundle transfer size is invalid")
        payload = {
            "schema_version": BUNDLE_TRANSFER_AUDIT_SCHEMA_VERSION,
            "bundle_fingerprint": bundle_fingerprint,
            "files": [item.as_dict() for item in ordered],
            "file_count": len(ordered),
            "total_bytes": sum(item.size for item in ordered),
        }
        return cls(
            bundle_fingerprint=bundle_fingerprint,
            files=ordered,
            file_count=payload["file_count"],
            total_bytes=payload["total_bytes"],
            fingerprint=_sha256_bytes(_canonical_json_bytes(payload)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_fingerprint": self.bundle_fingerprint,
            "files": [item.as_dict() for item in self.files],
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BundleTransferAudit:
        expected = {
            "schema_version",
            "bundle_fingerprint",
            "files",
            "file_count",
            "total_bytes",
            "fingerprint",
        }
        if set(value) != expected or value.get("schema_version") != (
            BUNDLE_TRANSFER_AUDIT_SCHEMA_VERSION
        ):
            raise ValueError("Saved bundle transfer audit schema is invalid")
        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("Saved bundle transfer file list is invalid")
        files: list[BundleTransferFile] = []
        for raw in raw_files:
            if not isinstance(raw, Mapping) or set(raw) != {
                "path",
                "role",
                "sha256",
                "size",
            }:
                raise ValueError("Saved bundle transfer file is invalid")
            files.append(
                BundleTransferFile(
                    path=raw["path"],
                    role=raw["role"],
                    sha256=raw["sha256"],
                    size=raw["size"],
                )
            )
        rebuilt = cls.create(
            bundle_fingerprint=value.get("bundle_fingerprint"),
            files=files,
        )
        if rebuilt.as_dict() != dict(value):
            raise ValueError("Saved bundle transfer audit totals or fingerprint are invalid")
        return rebuilt


@dataclass(frozen=True)
class RemoteSubmission:
    call_id: str
    plan_fingerprint: str
    bundle_namespace: str
    result_namespace: str
    model: str
    family_contracts: tuple[tuple[str, str, str], ...]
    bundle_audit: BundleTransferAudit

    def resume_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "plan_fingerprint": self.plan_fingerprint,
            "bundle_namespace": self.bundle_namespace,
            "result_namespace": self.result_namespace,
            "model": self.model,
            "bundle_audit": self.bundle_audit.as_dict(),
            "family_contracts": [
                {"family": family, "fingerprint": fingerprint, "method": method}
                for family, fingerprint, method in self.family_contracts
            ],
        }

    @classmethod
    def from_resume_dict(cls, value: Mapping[str, Any]) -> RemoteSubmission:
        _reject_secret_mapping(value, label="Saved Modal submission state")
        expected = {
            "call_id",
            "plan_fingerprint",
            "bundle_namespace",
            "result_namespace",
            "model",
            "bundle_audit",
            "family_contracts",
        }
        if set(value) != expected:
            raise ValueError("Saved Modal submission state is invalid")
        scalar_keys = expected - {"bundle_audit", "family_contracts"}
        if any(not isinstance(value[key], str) or not value[key] for key in scalar_keys):
            raise ValueError("Saved Modal submission state contains an invalid identifier")
        if not _CALL_ID.fullmatch(value["call_id"]):
            raise ValueError("Saved Modal call object ID is invalid")
        if len(value["plan_fingerprint"]) != 64:
            raise ValueError("Saved Modal submission plan fingerprint is invalid")
        _portable_relative(value["bundle_namespace"], label="Saved bundle namespace")
        _portable_relative(value["result_namespace"], label="Saved result namespace")
        raw_families = value["family_contracts"]
        if not isinstance(raw_families, list):
            raise ValueError("Saved Modal family contracts are invalid")
        families: list[tuple[str, str, str]] = []
        for raw in raw_families:
            if (
                not isinstance(raw, Mapping)
                or set(raw) != {"family", "fingerprint", "method"}
                or not isinstance(raw["family"], str)
                or not isinstance(raw["fingerprint"], str)
                or len(raw["fingerprint"]) != 64
                or not isinstance(raw["method"], str)
            ):
                raise ValueError("Saved Modal family contract is invalid")
            families.append((raw["family"], raw["fingerprint"], raw["method"]))
        if len({family for family, _, _ in families}) != len(families):
            raise ValueError("Saved Modal family contracts are duplicated")
        raw_audit = value["bundle_audit"]
        if not isinstance(raw_audit, Mapping):
            raise ValueError("Saved Modal bundle audit is invalid")
        return cls(
            call_id=value["call_id"],
            plan_fingerprint=value["plan_fingerprint"],
            bundle_namespace=value["bundle_namespace"],
            result_namespace=value["result_namespace"],
            model=value["model"],
            family_contracts=tuple(sorted(families)),
            bundle_audit=BundleTransferAudit.from_dict(raw_audit),
        )


@dataclass(frozen=True)
class RemoteResult:
    plan_fingerprint: str
    remote_state: str
    model: str
    step: int
    checkpoint: str | None
    completion_manifest_path: str
    completion_manifest_sha256: str
    family_contracts: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class DownloadedTrainingResult:
    completion: CompletionManifest
    contract: TrainingResultContract


RemoteStatusCallback = Callable[[dict[str, Any]], None]


def _remote_chunks(value: bytes | Iterable[bytes]) -> Iterable[bytes]:
    if isinstance(value, bytes):
        yield value
        return
    try:
        iterator = iter(value)
    except Exception:
        raise RuntimeError("Modal Volume read failed") from None
    while True:
        try:
            chunk = next(iterator)
        except StopIteration:
            return
        except (FileNotFoundError, KeyError):
            raise
        except Exception:
            raise RuntimeError("Modal Volume read failed") from None
        if not isinstance(chunk, bytes):
            raise ValueError("Modal Volume returned a non-byte file chunk")
        if chunk:
            yield chunk


def _read_remote_file(backend: ModalBackend, remote_path: str) -> bytes | Iterable[bytes]:
    try:
        return backend.read_file(remote_path)
    except (FileNotFoundError, KeyError):
        raise
    except Exception:
        raise RuntimeError("Modal Volume read failed") from None


def _read_small_remote_file(
    backend: ModalBackend,
    remote_path: str,
    *,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in _remote_chunks(_read_remote_file(backend, remote_path)):
        total += len(chunk)
        if total > maximum_bytes:
            raise ValueError("Remote completion contract exceeds its safe size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _download_verified_member(
    backend: ModalBackend,
    remote_path: str,
    local_path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(f".{local_path.name}.{uuid4().hex}.tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in _remote_chunks(_read_remote_file(backend, remote_path)):
                size += len(chunk)
                if size > expected_size:
                    raise ValueError(f"Downloaded result is larger than declared: {remote_path}")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError(f"Downloaded result checksum is invalid: {remote_path}")
        temporary.replace(local_path)
    finally:
        temporary.unlink(missing_ok=True)


def _volume_path(namespace: str, relative: str) -> str:
    namespace_path = _portable_relative(namespace, label="Modal namespace")
    relative_path = _portable_relative(relative, label="Modal payload path")
    return "/" + PurePosixPath(namespace_path, relative_path).as_posix()


def _model_label(plan: TrainingPlan) -> str:
    enabled = [family.family for family in plan.families if family.enabled]
    return ",".join(enabled) if enabled else "none"


def _result_from_value(
    value: Mapping[str, Any],
    *,
    submission: RemoteSubmission,
) -> RemoteResult:
    _reject_secret_mapping(value, label="Modal result")
    required = {
        "plan_fingerprint",
        "remote_state",
        "model",
        "step",
        "checkpoint",
        "completion_manifest_path",
        "completion_manifest_sha256",
    }
    if set(value) != required or value.get("remote_state") != "complete":
        raise RuntimeError("Modal result is not a complete remote result contract")
    if value.get("plan_fingerprint") != submission.plan_fingerprint:
        raise RuntimeError("Modal result belongs to a different TrainingPlan")
    model = value.get("model")
    step = value.get("step")
    checkpoint = value.get("checkpoint")
    if not isinstance(model, str) or model != submission.model:
        raise RuntimeError("Modal result model is invalid")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise RuntimeError("Modal result step is invalid")
    if checkpoint is not None:
        _portable_relative(checkpoint, label="Modal result checkpoint")
    manifest_path = value.get("completion_manifest_path")
    remote_manifest = _remote_absolute(manifest_path, label="Modal result completion path")
    result_root = PurePosixPath("/", submission.result_namespace)
    try:
        remote_manifest.relative_to(result_root)
    except ValueError as exc:
        raise RuntimeError("Modal result completion path escapes its per-plan namespace") from exc
    digest = value.get("completion_manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("Modal result completion checksum is invalid")
    return RemoteResult(
        plan_fingerprint=submission.plan_fingerprint,
        remote_state="complete",
        model=model,
        step=step,
        checkpoint=checkpoint,
        completion_manifest_path=remote_manifest.as_posix(),
        completion_manifest_sha256=digest,
        family_contracts=submission.family_contracts,
    )


def _remote_status_path(submission: RemoteSubmission) -> str:
    namespace = _portable_relative(
        submission.bundle_namespace,
        label="Saved bundle namespace",
    )
    parts = namespace.parts
    if (
        len(parts) != 4
        or parts[0] != "plans"
        or parts[1] != submission.plan_fingerprint[:24]
        or parts[2] != "bundles"
    ):
        raise ValueError("Saved bundle namespace is not tied to its TrainingPlan")
    return PurePosixPath("/", parts[0], parts[1], "remote-status.json").as_posix()


def _read_persistent_remote_status(
    backend: ModalBackend,
    submission: RemoteSubmission,
) -> RemoteStatus | None:
    try:
        raw = _read_small_remote_file(
            backend,
            _remote_status_path(submission),
            maximum_bytes=64 * 1024,
        )
    except (FileNotFoundError, KeyError):
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Persistent Modal status is unreadable") from exc
    _reject_secret_mapping(value, label="Persistent Modal status")
    expected = {"executor", "remote_state", "model", "step", "checkpoint"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Persistent Modal status contract is invalid")
    model = value.get("model")
    permitted_models = {family for family, _, _ in submission.family_contracts}
    permitted_models.add(submission.model)
    step = value.get("step")
    checkpoint = value.get("checkpoint")
    if (
        value.get("executor") != "modal"
        or value.get("remote_state") not in {"running", "complete"}
        or not isinstance(model, str)
        or model not in permitted_models
        or not isinstance(step, int)
        or isinstance(step, bool)
        or step < 0
    ):
        raise ValueError("Persistent Modal status values are invalid")
    if checkpoint is not None:
        _portable_relative(checkpoint, label="Persistent Modal checkpoint")
    return RemoteStatus(
        remote_state=value["remote_state"],
        model=model,
        step=step,
        checkpoint=checkpoint,
    )


class ModalTransport:
    """Network boundary with deterministic fakes and no credential-bearing payloads."""

    def __init__(
        self,
        settings: ModalSettings,
        *,
        backend: ModalBackend | None = None,
    ) -> None:
        self.settings = settings
        self.backend = ModalSDKBackend(settings) if backend is None else backend

    def run(
        self,
        *,
        plan: TrainingPlan,
        plan_bytes: bytes,
        bundle: TrainingBundle,
        status_callback: RemoteStatusCallback | None,
    ) -> RemoteSubmission:
        return self.submit(
            plan=plan,
            plan_bytes=plan_bytes,
            bundle=bundle,
            status_callback=status_callback,
        )

    def submit(
        self,
        *,
        plan: TrainingPlan,
        plan_bytes: bytes,
        bundle: TrainingBundle,
        status_callback: RemoteStatusCallback | None = None,
    ) -> RemoteSubmission:
        if plan_bytes != canonical_plan_bytes(plan):
            raise ValueError("Modal submission received non-canonical TrainingPlan bytes")
        verified = verify_training_bundle(
            bundle.root,
            expected_plan_fingerprint=plan.fingerprint,
        )
        bundle_namespace = f"plans/{plan.plan_id}/bundles/{verified.inventory.fingerprint}"
        result_namespace = f"plans/{plan.plan_id}/results"
        payload_members = [
            UploadMember(
                local_path=verified.root.joinpath(*PurePosixPath(item.path).parts),
                remote_path=_volume_path(bundle_namespace, item.path),
                sha256=item.sha256,
                size=item.size,
            )
            for item in verified.inventory.files
        ]
        completion = UploadMember(
            local_path=verified.completion_path,
            remote_path=_volume_path(bundle_namespace, BUNDLE_COMPLETION_PATH),
            sha256=_sha256_file(verified.completion_path),
            size=verified.completion_path.stat().st_size,
        )
        bundle_audit = BundleTransferAudit.create(
            bundle_fingerprint=verified.inventory.fingerprint,
            files=(
                *(
                    BundleTransferFile(
                        path=item.path,
                        role=item.role,
                        sha256=item.sha256,
                        size=item.size,
                    )
                    for item in verified.inventory.files
                ),
                BundleTransferFile(
                    path=BUNDLE_COMPLETION_PATH,
                    role="bundle-completion",
                    sha256=completion.sha256,
                    size=completion.size,
                ),
            ),
        )
        model = _model_label(plan)
        if status_callback is not None:
            uploading = {
                "executor": "modal",
                "remote_state": "uploading",
                "model": model,
                "step": 0,
                "checkpoint": None,
                "bundle_audit": bundle_audit.as_dict(),
            }
            _reject_secret_mapping(uploading, label="Modal pre-dispatch audit")
            # This callback is deliberately synchronous and precedes the first
            # external write, making the exact authorized file list durable in
            # local state before any byte leaves the machine.
            status_callback(uploading)
        try:
            self.backend.upload_payload(payload_members)
            self.backend.upload_completion(completion)
        except Exception:
            raise RuntimeError("Modal bundle upload failed") from None

        payload = {
            "schema_version": REMOTE_CONTRACT_SCHEMA_VERSION,
            "plan_fingerprint": plan.fingerprint,
            "plan_path": _volume_path(bundle_namespace, "contracts/training-plan.json"),
            "bundle_namespace": bundle_namespace,
            "bundle_fingerprint": verified.inventory.fingerprint,
            "result_namespace": result_namespace,
        }
        _reject_secret_mapping(payload, label="Modal spawn payload")
        try:
            call_id = self.backend.spawn(payload)
        except Exception:
            raise RuntimeError("Modal training dispatch failed") from None
        if not isinstance(call_id, str) or not _CALL_ID.fullmatch(call_id):
            raise RuntimeError("Modal backend returned an invalid call object ID")
        submission = RemoteSubmission(
            call_id=call_id,
            plan_fingerprint=plan.fingerprint,
            bundle_namespace=bundle_namespace,
            result_namespace=result_namespace,
            model=model,
            family_contracts=tuple(
                sorted(
                    (
                        family.family,
                        family.fingerprint,
                        family.method,
                    )
                    for family in plan.families
                    if family.enabled
                )
            ),
            bundle_audit=bundle_audit,
        )
        if status_callback is not None:
            running = RemoteStatus(
                remote_state="running",
                model=model,
                step=0,
                checkpoint=None,
            ).as_dict()
            running["bundle_audit"] = bundle_audit.as_dict()
            # Save the durable call ID before returning control to the caller.
            # A CLI interruption after this callback can reconnect and poll;
            # it must not create a second remote job for the same plan.
            running["submission"] = submission.resume_dict()
            _reject_secret_mapping(running, label="Modal running submission state")
            status_callback(running)
        return submission

    def poll(
        self,
        submission: RemoteSubmission,
        *,
        timeout: float = 0.0,
        status_callback: RemoteStatusCallback | None = None,
    ) -> RemoteResult | None:
        try:
            value = self.backend.poll(submission.call_id, timeout=timeout)
        except ModalFunctionTerminalError as exc:
            recovery = getattr(self.backend, "recover_terminal_call", None)
            if not callable(recovery):
                raise RuntimeError(
                    "Modal reached a terminal failure, but this backend cannot safely "
                    "recover its durable claims"
                ) from None
            try:
                recovery_state = recovery(submission, call_id=exc.call_id)
            except Exception:
                # Keep the saved submission intact when terminal state or Dict
                # ownership could not be re-verified. A transient local/API
                # error must never make a running remote writer reclaimable.
                raise RuntimeError(
                    "Modal terminal-call claim recovery could not be verified"
                ) from None
            if recovery_state != "released":
                raise RuntimeError(
                    "Modal terminal-call claim recovery was not authorized: "
                    f"{recovery_state}"
                ) from None
            raise ModalTerminalCallRecoveredError(exc.call_id) from None
        except Exception:
            raise RuntimeError("Modal training status polling failed") from None
        if value is None:
            if status_callback is not None:
                persisted = _read_persistent_remote_status(self.backend, submission)
                status_callback(
                    (
                        persisted
                        if persisted is not None
                        else RemoteStatus(
                            remote_state="running",
                            model=submission.model,
                            step=0,
                            checkpoint=None,
                        )
                    ).as_dict()
                )
            return None
        result = _result_from_value(value, submission=submission)
        if status_callback is not None:
            status_callback(
                RemoteStatus(
                    remote_state="complete",
                    model=result.model,
                    step=result.step,
                    checkpoint=result.checkpoint,
                ).as_dict()
            )
        return result

    def download_result(
        self,
        result: RemoteResult,
        destination: Path,
        *,
        expected_plan_fingerprint: str,
    ) -> DownloadedTrainingResult:
        if result.plan_fingerprint != expected_plan_fingerprint:
            raise ValueError("Modal result does not match the expected TrainingPlan")
        destination = destination.absolute()
        if destination.exists():
            raise FileExistsError(f"Result destination already exists: {destination}")

        manifest_bytes = _read_small_remote_file(
            self.backend,
            result.completion_manifest_path,
        )
        if _sha256_bytes(manifest_bytes) != result.completion_manifest_sha256:
            raise ValueError("Downloaded completion marker checksum is invalid")
        try:
            manifest = _completion_from_value(json.loads(manifest_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Downloaded completion marker is unreadable") from exc
        if manifest.kind != "result" or manifest.plan_fingerprint != expected_plan_fingerprint:
            raise ValueError("Downloaded result completion contract is invalid")
        manifest_remote = _remote_absolute(
            result.completion_manifest_path,
            label="Modal result completion path",
        )
        remote_root = manifest_remote.parent
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary.mkdir(parents=True)
            for member in manifest.files:
                relative = _portable_relative(member.path, label="Modal result member")
                remote_path = PurePosixPath(remote_root, relative).as_posix()
                local_path = temporary.joinpath(*relative.parts)
                _download_verified_member(
                    self.backend,
                    remote_path,
                    local_path,
                    expected_sha256=member.sha256,
                    expected_size=member.size,
                )
            # The checksummed remote completion marker is reproduced last locally.
            _write_atomic_bytes(temporary / RESULT_COMPLETION_NAME, manifest_bytes)
            verified = verify_completed_directory(
                temporary,
                expected_plan_fingerprint=expected_plan_fingerprint,
                completion_name=RESULT_COMPLETION_NAME,
                expected_kind="result",
                expected_model=result.model,
                # This is a candidate transfer contract, not a publication
                # contract. Held-out quality evaluation runs locally after the
                # checksummed download and alone may authorize publication.
                require_quality_gate=False,
            )
            result_contract_path = temporary / TRAINING_RESULT_NAME
            if not result_contract_path.is_file():
                raise ValueError("Downloaded result has no training-result.json contract")
            try:
                result_contract_value = json.loads(result_contract_path.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Downloaded training-result.json is unreadable") from exc
            expected_families = {
                family: (fingerprint, method)
                for family, fingerprint, method in result.family_contracts
            }
            result_contract = training_result_from_value(
                result_contract_value,
                expected_plan_fingerprint=expected_plan_fingerprint,
                expected_families=expected_families,
            )
            completion_files = {item.path: item for item in verified.files}
            for family in result_contract.families:
                for candidate in family.candidates:
                    prefix = candidate.artifact_path.rstrip("/") + "/"
                    actual_candidate = {
                        path: (item.sha256, item.size)
                        for path, item in completion_files.items()
                        if path.startswith(prefix)
                    }
                    expected_candidate = {
                        item.path: (item.sha256, item.size) for item in candidate.files
                    }
                    if actual_candidate != expected_candidate:
                        raise ValueError(
                            "Downloaded result candidate checksums do not match training-result.json: "
                            f"{candidate.artifact_path}"
                        )
            temporary.replace(destination)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return DownloadedTrainingResult(completion=verified, contract=result_contract)
