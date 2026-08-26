from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import uuid4

from personavoice.atomic import atomic_write_json, atomic_write_text
from personavoice.environment import load_root_environment
from personavoice.modal_transport import (
    REMOTE_RECOVERY_FUNCTION_NAME,
    REMOTE_RECOVERY_SCHEMA_VERSION,
    RESULT_COMPLETION_NAME,
    CompletionFile,
    ResultFamily,
    _modal_poll_exception_state,
    _reject_secret_mapping,
    latest_verified_family_checkpoint,
    write_completion_manifest,
    write_training_result_contract,
)
from personavoice.model_assets import (
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_ID,
    IRODORI_DACVAE_REVISION,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_ID,
    IRODORI_MODEL_REVISION,
    IRODORI_MODEL_SHA256,
    IRODORI_SOURCE_REVISION,
    IRODORI_TEXT_ENCODER_ID,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_ASSET_SHA256,
    LFM_MODEL_ID,
    LFM_MODEL_REQUIRED_FILES,
    LFM_MODEL_REVISION,
    LFM_MODEL_WEIGHT_SHA256,
)
from personavoice.training_bundle import PLAN_PATH, canonical_plan_bytes, verify_training_bundle
from personavoice.training_plan import FamilyPlan, TrainingPlan

MODAL_VOLUME_MOUNT = "/personavoice-volume"
MODAL_IMAGE_ROOT = "/opt/personavoice"
IRODORI_UPSTREAM_URL = "https://github.com/Aratako/Irodori-TTS.git"
UV_VERSION = "0.12.5"
ASSET_MARKER_NAME = "asset-complete.json"
ASSET_INDEX_NAME = "asset-index.json"
RESUME_INDEX_NAME = "resume-contract.json"
REMOTE_STATUS_NAME = "remote-status.json"
REMOTE_TRAIN_FUNCTION_NAME = "train"
MODAL_APP_CONTRACT_SCHEMA = 1
REMOTE_CLAIM_SCHEMA = 1

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MODAL_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MODAL_CALL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,255}$")
_IRODORI_TEXT_ENCODER_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


class _RetainPlanClaimError(RuntimeError):
    """A claim-store failure for which releasing the plan would permit a race."""


# Audited at the exact Irodori-pinned ModernBERT revision.  Keeping the weight
# digest here makes the remote materializer at least as strict as the local
# model assets that already carry explicit weight hashes.
_IRODORI_TEXT_ENCODER_MODEL_SHA256 = (
    "96e31693fe9bc8b2900e47ddbf7a389e940bb20b165b44e182a490ff71fb17d3"
)
_REVISION_MARKER = ".personavoice-revision"
_IMAGE_COPY_IGNORE = (
    "**/.venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
)

TrainingStatusCallback = Callable[[str, float, str], None]
CommonTrainingRunner = Callable[
    [bytes, Path, Path, Path, Path],
    Sequence[ResultFamily],
]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker is not None and checker())


def _portable(value: str, *, label: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(("/", "~"))
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} is not a portable relative path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value.split("/")) or path.as_posix() != value:
        raise ValueError(f"{label} contains traversal or non-canonical segments")
    return path


def _volume_relative(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} must be an absolute Volume path")
    return _portable(value[1:], label=label)


def _real_volume_directory(
    volume_root: Path,
    relative: PurePosixPath,
    *,
    create: bool,
) -> Path:
    """Resolve one Volume directory without following user-controlled links."""

    current = volume_root
    for part in relative.parts:
        candidate = current / part
        if candidate.is_symlink() or _is_junction(candidate):
            raise ValueError(f"Modal Volume namespace traverses a link or junction: {relative}")
        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError(f"Modal Volume namespace contains a non-directory: {relative}")
        elif create:
            candidate.mkdir()
        else:
            raise ValueError(f"Modal Volume namespace is missing: {relative}")
        current = candidate
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(volume_root)
    except ValueError as exc:
        raise ValueError(f"Modal Volume namespace escapes its mount: {relative}") from exc
    return resolved


def _safe_name(value: str, *, label: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _safe_modal_name(value: str, *, label: str) -> str:
    if not _MODAL_RESOURCE_NAME.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


@dataclass(frozen=True)
class ModalAppContract:
    app_name: str = "personavoice-training"
    volume_name: str = "personavoice-training"
    hf_secret_name: str = "personavoice-huggingface"
    gpu: str = "A100-40GB"
    timeout_seconds: int = 86_100
    max_retries: int = 2
    retry_initial_delay_seconds: float = 1.0
    retry_backoff_coefficient: float = 2.0

    def __post_init__(self) -> None:
        _safe_modal_name(self.app_name, label="Modal app name")
        _safe_modal_name(self.volume_name, label="Modal Volume name")
        _safe_modal_name(self.hf_secret_name, label="Modal Secret name")
        _safe_modal_name(self.claim_dict_name, label="Modal claim Dict name")
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?::1)?", self.gpu) or "," in self.gpu:
            raise ValueError("Modal training requires exactly one explicitly named GPU")
        if not 60 <= self.timeout_seconds <= 86_400:
            raise ValueError("Modal timeout must be between one minute and 24 hours")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("Modal retry count is invalid")
        if self.retry_initial_delay_seconds <= 0 or self.retry_backoff_coefficient < 1:
            raise ValueError("Modal retry backoff is invalid")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModalAppContract:
        values = os.environ if env is None else env
        return cls(
            app_name=values.get("PERSONAVOICE_MODAL_APP", "personavoice-training"),
            volume_name=values.get("PERSONAVOICE_MODAL_VOLUME", "personavoice-training"),
            hf_secret_name=values.get(
                "PERSONAVOICE_MODAL_HF_SECRET",
                "personavoice-huggingface",
            ),
            gpu=values.get("PERSONAVOICE_MODAL_GPU", "A100-40GB"),
            timeout_seconds=int(
                values.get("PERSONAVOICE_MODAL_TIMEOUT_SECONDS", "86400")
            ),
            max_retries=int(values.get("PERSONAVOICE_MODAL_RETRIES", "2")),
        )

    @property
    def claim_dict_name(self) -> str:
        """Stable distributed-lock namespace shared by every app deployment."""

        return f"{self.app_name}-claims"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODAL_APP_CONTRACT_SCHEMA,
            "app_name": self.app_name,
            "volume_name": self.volume_name,
            "hf_secret_name": self.hf_secret_name,
            "gpu": self.gpu,
            "gpu_count": 1,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "volume_mount": MODAL_VOLUME_MOUNT,
            "claim_dict_name": self.claim_dict_name,
            "required_secret_keys": ["HF_TOKEN"],
        }


@dataclass(frozen=True)
class AssetSpec:
    family: str
    name: str
    repo_id: str
    revision: str
    required_files: tuple[str, ...]
    expected_sha256: Mapping[str, str]
    runtime_path: str | None = None
    generated_files: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _safe_name(self.family, label="asset family")
        _safe_name(self.name, label="asset name")
        if (
            not isinstance(self.repo_id, str)
            or self.repo_id.count("/") != 1
            or any(not part or not _SAFE_NAME.fullmatch(part) for part in self.repo_id.split("/"))
        ):
            raise ValueError("Pinned asset repository id is invalid")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("Pinned asset revision must be a full lowercase Git commit")
        if not self.required_files or len(set(self.required_files)) != len(self.required_files):
            raise ValueError("Pinned asset required files must be non-empty and unique")
        required = {
            _portable(relative, label="Pinned asset file").as_posix()
            for relative in self.required_files
        }
        if set(self.expected_sha256) - required:
            raise ValueError("Pinned asset checksums contain an undeclared file")
        if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in self.expected_sha256.values()):
            raise ValueError("Pinned asset checksum is invalid")
        generated = dict(self.generated_files)
        if len(generated) != len(self.generated_files) or set(generated) - required:
            raise ValueError("Pinned generated asset files are invalid")
        for relative, value in self.generated_files:
            _portable(relative, label="Generated pinned asset file")
            if not isinstance(value, str) or not value:
                raise ValueError("Generated pinned asset content is invalid")
        if self.runtime_path is not None:
            runtime = _portable(self.runtime_path, label="Pinned asset runtime path")
            if not runtime.parts or runtime.parts[0] != "models":
                raise ValueError("Pinned assets may only project below models/")

    @property
    def cache_key(self) -> str:
        value = {
            "family": self.family,
            "name": self.name,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "required_files": list(self.required_files),
            "expected_sha256": dict(sorted(self.expected_sha256.items())),
            "generated_files": [list(item) for item in sorted(self.generated_files)],
        }
        return _sha256_bytes(_canonical_json_bytes(value))


class AssetDownloader(Protocol):
    def download(self, spec: AssetSpec, destination: Path, *, token: str | None) -> None: ...


class HuggingFaceAssetDownloader:
    def download(self, spec: AssetSpec, destination: Path, *, token: str | None) -> None:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise RuntimeError("Remote asset materialization requires huggingface-hub") from exc
        hub.snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=destination,
            cache_dir=destination.parents[1] / "hub-cache",
            allow_patterns=list(spec.required_files),
            token=token,
        )
        # ``local_dir`` metadata is an implementation cache, not a model input.
        # It may contain host-local paths and must never enter the verified
        # portable asset contract.
        shutil.rmtree(destination / ".cache", ignore_errors=True)


def _require_contract(actual: Mapping[str, Any], expected: Mapping[str, Any], *, family: str) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(f"Remote {family} asset pin disagrees with TrainingPlan field {key}")


def asset_specs_for_plan(plan: TrainingPlan) -> tuple[AssetSpec, ...]:
    specs: list[AssetSpec] = []
    for family in plan.families:
        if not family.enabled:
            continue
        contract = family.model_contract
        if family.family == "irodori":
            _require_contract(
                contract,
                {
                    "source_revision": IRODORI_SOURCE_REVISION,
                    "base_revision": IRODORI_MODEL_REVISION,
                    "base_sha256": IRODORI_MODEL_SHA256,
                    "dacvae_revision": IRODORI_DACVAE_REVISION,
                    "dacvae_sha256": IRODORI_DACVAE_SHA256,
                    "text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
                },
                family="Irodori",
            )
            specs.extend(
                (
                    AssetSpec(
                        family="irodori",
                        name="base",
                        repo_id=IRODORI_MODEL_ID,
                        revision=IRODORI_MODEL_REVISION,
                        required_files=(IRODORI_MODEL_FILENAME,),
                        expected_sha256={IRODORI_MODEL_FILENAME: IRODORI_MODEL_SHA256},
                        runtime_path="models/irodori/v4.1-small",
                    ),
                    AssetSpec(
                        family="irodori",
                        name="dacvae",
                        repo_id=IRODORI_DACVAE_ID,
                        revision=IRODORI_DACVAE_REVISION,
                        required_files=(IRODORI_DACVAE_FILENAME,),
                        expected_sha256={IRODORI_DACVAE_FILENAME: IRODORI_DACVAE_SHA256},
                        runtime_path="models/irodori/dacvae",
                    ),
                    AssetSpec(
                        family="irodori",
                        name="text-encoder",
                        repo_id=IRODORI_TEXT_ENCODER_ID,
                        revision=IRODORI_TEXT_ENCODER_REVISION,
                        required_files=_IRODORI_TEXT_ENCODER_REQUIRED_FILES,
                        expected_sha256={
                            "model.safetensors": _IRODORI_TEXT_ENCODER_MODEL_SHA256,
                        },
                        runtime_path=(
                            "models/hf-cache/hub/"
                            "models--sbintuitions--modernbert-ja-310m/snapshots/"
                            f"{IRODORI_TEXT_ENCODER_REVISION}"
                        ),
                    ),
                )
            )
        elif family.family == "lfm":
            _require_contract(
                contract,
                {
                    "base_revision": LFM_MODEL_REVISION,
                    "base_sha256": LFM_MODEL_WEIGHT_SHA256,
                    "base_assets_sha256": LFM_MODEL_ASSET_SHA256,
                },
                family="LFM",
            )
            specs.append(
                AssetSpec(
                    family="lfm",
                    name="base",
                    repo_id=LFM_MODEL_ID,
                    revision=LFM_MODEL_REVISION,
                    required_files=(*LFM_MODEL_REQUIRED_FILES, _REVISION_MARKER),
                    expected_sha256={
                        **LFM_MODEL_ASSET_SHA256,
                        _REVISION_MARKER: _sha256_bytes(
                            f"{LFM_MODEL_REVISION}\n".encode()
                        ),
                    },
                    runtime_path="models/lfm/base",
                    generated_files=((_REVISION_MARKER, f"{LFM_MODEL_REVISION}\n"),),
                )
            )
        elif family.family == "seed-vc":
            raise ValueError("Seed-VC remote fine-tuning is outside the Modal executor contract")
        else:
            raise ValueError(f"Unsupported remote training family: {family.family}")
    return tuple(specs)


def _asset_files(root: Path) -> tuple[CompletionFile, ...]:
    files: list[CompletionFile] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or _is_junction(path):
            raise ValueError(f"Pinned asset cache contains a link or junction: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Pinned asset cache contains a non-regular entry: {relative}")
        if relative == ASSET_MARKER_NAME:
            continue
        if path.stat().st_size <= 0:
            raise ValueError(f"Pinned asset cache contains an empty file: {relative}")
        files.append(
            CompletionFile(
                path=_portable(relative, label="Pinned asset file").as_posix(),
                sha256=_sha256_file(path),
                size=path.stat().st_size,
            )
        )
    return tuple(sorted(files, key=lambda item: item.path))


def _asset_marker_value(spec: AssetSpec, files: Sequence[CompletionFile]) -> dict[str, Any]:
    payload = {
        "schema_version": MODAL_APP_CONTRACT_SCHEMA,
        "status": "complete",
        "cache_key": spec.cache_key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "files": [item.as_dict() for item in files],
    }
    return {**payload, "fingerprint": _sha256_bytes(_canonical_json_bytes(payload))}


def verify_asset_cache(path: Path, spec: AssetSpec) -> tuple[CompletionFile, ...]:
    marker = path / ASSET_MARKER_NAME
    if path.is_symlink() or _is_junction(path) or not marker.is_file():
        raise ValueError("Pinned asset completion marker is missing")
    try:
        value = json.loads(marker.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Pinned asset completion marker is unreadable") from exc
    files = _asset_files(path)
    if value != _asset_marker_value(spec, files):
        raise ValueError("Pinned asset completion contract is invalid")
    by_path = {item.path: item for item in files}
    expected_files = set(spec.required_files)
    if set(by_path) != expected_files:
        raise ValueError("Pinned asset cache does not match its minimal required-file allowlist")
    for relative, expected in spec.expected_sha256.items():
        if by_path[relative].sha256 != expected:
            raise ValueError(f"Pinned asset checksum failed: {relative}")
    return files


def _write_generated_asset_files(path: Path, spec: AssetSpec) -> None:
    for relative, value in spec.generated_files:
        destination = path.joinpath(*PurePosixPath(relative).parts)
        if destination.exists():
            raise ValueError(f"Downloaded asset tried to shadow generated file: {relative}")
        atomic_write_text(destination, value)


def _project_verified_asset(asset_root: Path, spec: AssetSpec, cache: Path) -> Path:
    if spec.runtime_path is None:
        return cache
    relative = _portable(spec.runtime_path, label="Pinned asset runtime path")
    destination = asset_root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = cache.resolve(strict=True)
    if destination.is_symlink():
        try:
            actual = destination.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Pinned asset runtime link is broken") from exc
        if actual != expected:
            raise ValueError("Pinned asset runtime path points at a different revision")
        return destination
    if destination.exists() or _is_junction(destination):
        raise ValueError("Pinned asset runtime path is not a managed symbolic link")

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        target = os.path.relpath(cache, start=destination.parent)
        temporary.symlink_to(target, target_is_directory=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if not destination.is_symlink() or destination.resolve(strict=True) != expected:
        raise RuntimeError("Pinned asset runtime link failed post-write verification")
    return destination


def materialize_verified_assets(
    plan: TrainingPlan,
    asset_root: Path,
    *,
    downloader: AssetDownloader | None = None,
) -> Mapping[str, Path]:
    """Materialize content-addressed assets once, then write plan/family indexes."""

    asset_root = asset_root.absolute()
    if asset_root.is_symlink() or _is_junction(asset_root):
        raise ValueError("Pinned asset root must not be a link or junction")
    asset_root.mkdir(parents=True, exist_ok=True)
    if not asset_root.is_dir():
        raise ValueError("Pinned asset root must be a real directory")
    asset_root = asset_root.resolve(strict=True)
    materializer = HuggingFaceAssetDownloader() if downloader is None else downloader
    token = os.environ.get("HF_TOKEN") or None
    resolved: dict[str, Path] = {}
    specs = asset_specs_for_plan(plan)
    for spec in specs:
        destination = asset_root / "cache" / spec.cache_key
        if destination.exists():
            verify_asset_cache(destination, spec)
        else:
            if token is None:
                raise RuntimeError(
                    "Remote pinned asset download requires HF_TOKEN from the configured Modal Secret"
                )
            staging = destination.with_name(f".{spec.cache_key}.{uuid4().hex}.tmp")
            try:
                staging.mkdir(parents=True)
                try:
                    materializer.download(spec, staging, token=token)
                except Exception:
                    # Third-party HTTP exceptions are not a stable or secret-safe
                    # logging contract.  In particular, never let an auth value
                    # supplied via the Modal Secret appear in a remote traceback.
                    raise RuntimeError("Pinned remote asset download failed") from None
                _write_generated_asset_files(staging, spec)
                files = _asset_files(staging)
                marker = _asset_marker_value(spec, files)
                atomic_write_text(
                    staging / ASSET_MARKER_NAME,
                    _canonical_json_bytes(marker).decode("utf-8"),
                )
                verify_asset_cache(staging, spec)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    staging.replace(destination)
                except OSError:
                    # Another identical plan may have won the content-addressed
                    # cache race. Adopt it only after full marker/hash verification.
                    if not destination.exists():
                        raise
                    verify_asset_cache(destination, spec)
                    shutil.rmtree(staging)
            except BaseException:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
        resolved[f"{spec.family}/{spec.name}"] = _project_verified_asset(
            asset_root,
            spec,
            destination,
        )

    for family in plan.families:
        if not family.enabled:
            continue
        family_specs = [spec for spec in specs if spec.family == family.family]
        namespace = (
            asset_root
            / "plans"
            / plan.plan_id
            / family.family
            / family.fingerprint
        )
        index = {
            "schema_version": MODAL_APP_CONTRACT_SCHEMA,
            "status": "complete",
            "plan_fingerprint": plan.fingerprint,
            "family": family.family,
            "family_fingerprint": family.fingerprint,
            "assets": [
                {
                    "name": spec.name,
                    "repo_id": spec.repo_id,
                    "cache_key": spec.cache_key,
                    "revision": spec.revision,
                    "runtime_path": spec.runtime_path,
                }
                for spec in sorted(family_specs, key=lambda item: item.name)
            ],
        }
        atomic_write_json(
            namespace / ASSET_INDEX_NAME,
            {**index, "fingerprint": _sha256_bytes(_canonical_json_bytes(index))},
        )
    (asset_root / ".runtime").mkdir(parents=True, exist_ok=True)
    return resolved


def training_plan_from_bytes(value: bytes, *, expected_fingerprint: str) -> TrainingPlan:
    try:
        raw = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Remote TrainingPlan JSON is unreadable") from exc
    try:
        plan = TrainingPlan.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Remote TrainingPlan contract is invalid") from exc
    if plan.fingerprint != expected_fingerprint or canonical_plan_bytes(plan) != value:
        raise ValueError("Remote TrainingPlan canonical bytes or fingerprint are invalid")
    return plan


def _validated_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        "schema_version",
        "plan_fingerprint",
        "plan_path",
        "bundle_namespace",
        "bundle_fingerprint",
        "result_namespace",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError("Modal invocation payload schema is invalid")
    values = {key: payload[key] for key in expected - {"schema_version"}}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("Modal invocation payload contains an invalid identifier")
    if len(values["plan_fingerprint"]) != 64 or len(values["bundle_fingerprint"]) != 64:
        raise ValueError("Modal invocation fingerprint is invalid")
    _volume_relative(values["plan_path"], label="Modal plan path")
    _portable(values["bundle_namespace"], label="Modal bundle namespace")
    _portable(values["result_namespace"], label="Modal result namespace")
    return values


def acquire_remote_training_claim(
    payload: Mapping[str, Any],
    *,
    claim_store: Any,
    call_id: str,
) -> dict[str, Any] | None:
    """Atomically select one canonical Modal call for a TrainingPlan.

    ``Function.spawn`` returns its durable ID only after the external dispatch
    has been accepted.  A local process can therefore be killed in the narrow
    interval before that ID reaches ``state.json``.  Modal Dict's
    ``put(..., skip_if_exists=True)`` is a server-side compare-and-create
    primitive: duplicate calls may be spawned, but exactly one call is allowed
    to enter the expensive shared trainer.  Losers return a strict redirect to
    the canonical FunctionCall, which the client follows when polling.

    A retry of the *same* Modal FunctionCall retains its call ID and is allowed
    through so platform preemption and configured retries still resume from the
    method-native checkpoint contract.
    """

    values = _validated_payload(payload)
    if not isinstance(call_id, str) or _MODAL_CALL_ID.fullmatch(call_id) is None:
        raise RuntimeError("Modal did not expose a valid current FunctionCall ID")
    claim = {
        "schema_version": REMOTE_CLAIM_SCHEMA,
        "plan_fingerprint": values["plan_fingerprint"],
        "bundle_fingerprint": values["bundle_fingerprint"],
        "call_id": call_id,
    }
    _reject_secret_mapping(claim, label="Modal execution claim")
    try:
        created = claim_store.put(
            values["plan_fingerprint"],
            claim,
            skip_if_exists=True,
        )
        existing = claim if created is True else claim_store.get(values["plan_fingerprint"])
    except Exception:
        raise RuntimeError("Modal execution claim store is unavailable") from None
    if not isinstance(created, bool):
        raise RuntimeError("Modal execution claim store returned an invalid result")
    expected_keys = {
        "schema_version",
        "plan_fingerprint",
        "bundle_fingerprint",
        "call_id",
    }
    if (
        not isinstance(existing, Mapping)
        or set(existing) != expected_keys
        or existing.get("schema_version") != REMOTE_CLAIM_SCHEMA
        or existing.get("plan_fingerprint") != values["plan_fingerprint"]
        or existing.get("bundle_fingerprint") != values["bundle_fingerprint"]
        or not isinstance(existing.get("call_id"), str)
        or _MODAL_CALL_ID.fullmatch(existing["call_id"]) is None
    ):
        raise RuntimeError("Modal execution claim contract is invalid")
    _reject_secret_mapping(existing, label="Modal execution claim")
    canonical_call_id = existing["call_id"]
    if canonical_call_id == call_id:
        return None
    redirect = {
        "schema_version": REMOTE_CLAIM_SCHEMA,
        "remote_state": "redirect",
        "plan_fingerprint": values["plan_fingerprint"],
        "canonical_call_id": canonical_call_id,
    }
    _reject_secret_mapping(redirect, label="Modal duplicate-call redirect")
    return redirect


def release_remote_training_claim(
    payload: Mapping[str, Any],
    *,
    claim_store: Any,
    call_id: str,
) -> None:
    """Release a failed canonical call without deleting another call's claim."""

    values = _validated_payload(payload)
    if not isinstance(call_id, str) or _MODAL_CALL_ID.fullmatch(call_id) is None:
        raise RuntimeError("Modal did not expose a valid current FunctionCall ID")
    key = values["plan_fingerprint"]
    try:
        existing = claim_store.get(key)
    except Exception:
        raise RuntimeError("Modal execution claim store is unavailable") from None
    if not isinstance(existing, Mapping):
        raise RuntimeError("Modal execution claim contract is invalid")
    # Only the elected call can reach this path. A duplicate returns before the
    # trainer starts, so checking the immutable owner immediately before pop is
    # sufficient and prevents one call from releasing another call's lease.
    if (
        existing.get("schema_version") != REMOTE_CLAIM_SCHEMA
        or existing.get("plan_fingerprint") != values["plan_fingerprint"]
        or existing.get("bundle_fingerprint") != values["bundle_fingerprint"]
        or existing.get("call_id") != call_id
    ):
        raise RuntimeError("Modal execution claim ownership is invalid")
    _reject_secret_mapping(existing, label="Modal execution claim")
    try:
        removed = claim_store.pop(key)
    except Exception:
        raise RuntimeError("Modal execution claim release failed") from None
    if removed != existing:
        raise RuntimeError("Modal execution claim changed during release")


def _validated_recovery_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, str], tuple[tuple[str, str, str], ...]]:
    expected = {
        "schema_version",
        "plan_fingerprint",
        "bundle_fingerprint",
        "call_id",
        "family_contracts",
    }
    if set(payload) != expected or payload.get("schema_version") != (
        REMOTE_RECOVERY_SCHEMA_VERSION
    ):
        raise ValueError("Modal terminal-call recovery payload schema is invalid")
    scalar_names = ("plan_fingerprint", "bundle_fingerprint", "call_id")
    values = {name: payload.get(name) for name in scalar_names}
    if (
        not isinstance(values["plan_fingerprint"], str)
        or re.fullmatch(r"[0-9a-f]{64}", values["plan_fingerprint"]) is None
        or not isinstance(values["bundle_fingerprint"], str)
        or re.fullmatch(r"[0-9a-f]{64}", values["bundle_fingerprint"]) is None
        or not isinstance(values["call_id"], str)
        or _MODAL_CALL_ID.fullmatch(values["call_id"]) is None
    ):
        raise ValueError("Modal terminal-call recovery identifiers are invalid")
    raw_families = payload.get("family_contracts")
    if not isinstance(raw_families, list):
        raise ValueError("Modal terminal-call recovery family contracts are invalid")
    allowed = {
        "irodori": {"full", "lora", "speaker-inversion"},
        "lfm": {"full", "lora"},
        "seed-vc": {"finetune"},
    }
    families: list[tuple[str, str, str]] = []
    for raw in raw_families:
        if not isinstance(raw, Mapping) or set(raw) != {
            "family",
            "fingerprint",
            "method",
        }:
            raise ValueError("Modal terminal-call recovery family contract is invalid")
        family = raw.get("family")
        fingerprint = raw.get("fingerprint")
        method = raw.get("method")
        if (
            not isinstance(family, str)
            or family not in allowed
            or not isinstance(method, str)
            or method not in allowed[family]
            or not isinstance(fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        ):
            raise ValueError("Modal terminal-call recovery family values are invalid")
        families.append((family, fingerprint, method))
    canonical = sorted(families)
    if families != canonical or len({family for family, _, _ in families}) != len(families):
        raise ValueError("Modal terminal-call recovery families are not canonical")
    _reject_secret_mapping(payload, label="Modal terminal-call recovery payload")
    return ({name: str(values[name]) for name in scalar_names}, tuple(families))


def recover_remote_terminal_claim(
    payload: Mapping[str, Any],
    *,
    claim_store: Any,
    terminal_probe: Callable[[str], str],
) -> dict[str, Any]:
    """Idempotently remove claims only after the old call is independently terminal.

    The deployed recovery Function is limited to one container, serializing
    concurrent cleanup requests. Family claims are removed before the plan
    claim, so a replacement call cannot enter a shared checkpoint namespace
    until all ownership held by the terminal writer is gone.
    """

    values, families = _validated_recovery_payload(payload)
    state = terminal_probe(values["call_id"])
    if state not in {"failed", "running", "complete", "expired"}:
        raise RuntimeError("Modal terminal-call probe returned an invalid state")

    def result(recovery_state: str) -> dict[str, Any]:
        value = {
            "schema_version": REMOTE_RECOVERY_SCHEMA_VERSION,
            "recovery_state": recovery_state,
            "plan_fingerprint": values["plan_fingerprint"],
            "call_id": values["call_id"],
        }
        _reject_secret_mapping(value, label="Modal terminal-call recovery result")
        return value

    if state != "failed":
        return result(state)

    plan_key = values["plan_fingerprint"]
    try:
        existing_plan = claim_store.get(plan_key)
    except Exception:
        raise RuntimeError("Modal terminal-call recovery claim store is unavailable") from None
    if existing_plan is not None:
        expected_plan_keys = {
            "schema_version",
            "plan_fingerprint",
            "bundle_fingerprint",
            "call_id",
        }
        if (
            not isinstance(existing_plan, Mapping)
            or set(existing_plan) != expected_plan_keys
            or existing_plan.get("schema_version") != REMOTE_CLAIM_SCHEMA
            or existing_plan.get("plan_fingerprint") != values["plan_fingerprint"]
            or existing_plan.get("bundle_fingerprint") != values["bundle_fingerprint"]
            or not isinstance(existing_plan.get("call_id"), str)
            or _MODAL_CALL_ID.fullmatch(existing_plan["call_id"]) is None
        ):
            raise RuntimeError("Modal terminal-call recovery plan claim is invalid")
        _reject_secret_mapping(existing_plan, label="Modal terminal-call recovery plan claim")
        if existing_plan.get("call_id") != values["call_id"]:
            return result("superseded")

    # When the plan claim is already absent this is an idempotent retry of a
    # cleanup that may have completed. A concurrent replacement cannot have its
    # family entry removed: a different owner is detected before every pop.
    for family, fingerprint, _method in families:
        key = f"family:{family}:{fingerprint}"
        try:
            existing_family = claim_store.get(key)
        except Exception:
            raise RuntimeError(
                "Modal terminal-call recovery family claim store is unavailable"
            ) from None
        if existing_family is None:
            continue
        expected_family_keys = {
            "schema_version",
            "family",
            "family_fingerprint",
            "plan_fingerprint",
            "call_id",
        }
        if (
            not isinstance(existing_family, Mapping)
            or set(existing_family) != expected_family_keys
            or existing_family.get("schema_version") != REMOTE_CLAIM_SCHEMA
            or existing_family.get("family") != family
            or existing_family.get("family_fingerprint") != fingerprint
            or not isinstance(existing_family.get("plan_fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", existing_family["plan_fingerprint"]) is None
            or not isinstance(existing_family.get("call_id"), str)
            or _MODAL_CALL_ID.fullmatch(existing_family["call_id"]) is None
        ):
            raise RuntimeError("Modal terminal-call recovery family claim is invalid")
        _reject_secret_mapping(
            existing_family,
            label="Modal terminal-call recovery family claim",
        )
        if existing_family.get("call_id") != values["call_id"]:
            return result("superseded")
        try:
            removed = claim_store.pop(key)
        except Exception:
            raise RuntimeError("Modal terminal-call recovery family release failed") from None
        if removed != existing_family:
            raise RuntimeError("Modal terminal-call recovery family claim changed")

    if existing_plan is not None:
        try:
            removed_plan = claim_store.pop(plan_key)
        except Exception:
            raise RuntimeError("Modal terminal-call recovery plan release failed") from None
        if removed_plan != existing_plan:
            raise RuntimeError("Modal terminal-call recovery plan claim changed")
    return result("released")


def execute_claimed_remote_training(
    payload: Mapping[str, Any],
    *,
    claim_store: Any,
    call_id: str,
    executor: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one elected call and relinquish the claim on a normal failure."""

    redirect = acquire_remote_training_claim(
        payload,
        claim_store=claim_store,
        call_id=call_id,
    )
    if redirect is not None:
        return redirect
    try:
        return executor()
    except BaseException as exc:
        # A normal trainer exception releases ownership before Modal applies
        # its retry policy. The same call can reclaim, while a concurrently
        # submitted recovery call can win only after the failed attempt stops.
        # Hard container preemption retains the claim and Modal retries the same
        # FunctionCall ID, preserving exactly-one training.
        if not isinstance(exc, _RetainPlanClaimError):
            release_remote_training_claim(
                payload,
                claim_store=claim_store,
                call_id=call_id,
            )
        raise


def _family_claim_value(
    family: FamilyPlan,
    *,
    plan_fingerprint: str,
    call_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": REMOTE_CLAIM_SCHEMA,
        "family": family.family,
        "family_fingerprint": family.fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "call_id": call_id,
    }


def _family_claim_key(family: FamilyPlan) -> str:
    return f"family:{family.family}:{family.fingerprint}"


def _release_owned_family_claims(
    claim_store: Any,
    owned: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    for key, expected in reversed(tuple(owned)):
        try:
            existing = claim_store.get(key)
        except Exception:
            raise _RetainPlanClaimError("Modal family claim store is unavailable") from None
        if existing != expected:
            raise _RetainPlanClaimError("Modal family claim ownership is invalid")
        try:
            removed = claim_store.pop(key)
        except Exception:
            raise _RetainPlanClaimError("Modal family claim release failed") from None
        if removed != expected:
            raise _RetainPlanClaimError("Modal family claim changed during release")


def acquire_remote_family_claims(
    plan: TrainingPlan,
    *,
    claim_store: Any,
    call_id: str,
    wait: Callable[[], None] | None = None,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Serialize writers to every shared family-fingerprint namespace.

    Plan fingerprints intentionally change for evaluation and executor policy
    while family fingerprints—and therefore native checkpoint directories—can
    remain reusable.  These sorted locks prevent two such plans from writing a
    shared family namespace concurrently.  A call preempted after acquiring a
    prefix recognizes its own locks on retry; newly acquired prefixes are
    rolled back before waiting, so mixed-family plans cannot deadlock.
    """

    if not isinstance(call_id, str) or _MODAL_CALL_ID.fullmatch(call_id) is None:
        raise RuntimeError("Modal did not expose a valid current FunctionCall ID")
    families = tuple(
        sorted(
            (family for family in plan.families if family.enabled),
            key=lambda family: _family_claim_key(family),
        )
    )
    sleeper = (lambda: time.sleep(5.0)) if wait is None else wait
    while True:
        owned: list[tuple[str, Mapping[str, Any]]] = []
        newly_owned: list[tuple[str, Mapping[str, Any]]] = []
        conflict = False
        for family in families:
            key = _family_claim_key(family)
            expected = _family_claim_value(
                family,
                plan_fingerprint=plan.fingerprint,
                call_id=call_id,
            )
            _reject_secret_mapping(expected, label="Modal family execution claim")
            try:
                created = claim_store.put(key, expected, skip_if_exists=True)
                existing = expected if created is True else claim_store.get(key)
            except Exception:
                if newly_owned:
                    _release_owned_family_claims(claim_store, newly_owned)
                raise _RetainPlanClaimError(
                    "Modal family claim store is unavailable"
                ) from None
            if not isinstance(created, bool) or not isinstance(existing, Mapping):
                if newly_owned:
                    _release_owned_family_claims(claim_store, newly_owned)
                raise _RetainPlanClaimError("Modal family claim contract is invalid")
            expected_keys = {
                "schema_version",
                "family",
                "family_fingerprint",
                "plan_fingerprint",
                "call_id",
            }
            if (
                set(existing) != expected_keys
                or existing.get("schema_version") != REMOTE_CLAIM_SCHEMA
                or existing.get("family") != family.family
                or existing.get("family_fingerprint") != family.fingerprint
                or not isinstance(existing.get("plan_fingerprint"), str)
                or re.fullmatch(r"[0-9a-f]{64}", existing["plan_fingerprint"]) is None
                or not isinstance(existing.get("call_id"), str)
                or _MODAL_CALL_ID.fullmatch(existing["call_id"]) is None
            ):
                if newly_owned:
                    _release_owned_family_claims(claim_store, newly_owned)
                raise _RetainPlanClaimError("Modal family claim contract is invalid")
            _reject_secret_mapping(existing, label="Modal family execution claim")
            if existing.get("call_id") != call_id:
                conflict = True
                break
            item = (key, dict(existing))
            owned.append(item)
            if created:
                newly_owned.append(item)
        if not conflict:
            return tuple(owned)
        if newly_owned:
            _release_owned_family_claims(claim_store, newly_owned)
        sleeper()


def release_remote_family_claims(
    *,
    claim_store: Any,
    owned: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    _release_owned_family_claims(claim_store, owned)


def _common_runner() -> Any:
    from personavoice.training import run_training_plan

    return run_training_plan


def _checkpoint_namespace(checkpoint_root: Path, family: FamilyPlan) -> Path:
    return checkpoint_root / family.family / family.fingerprint


def _write_resume_contract(
    plan: TrainingPlan,
    checkpoint_root: Path,
    resume_contract_path: Path,
) -> None:
    families: list[dict[str, Any]] = []
    for family in plan.families:
        if not family.enabled:
            continue
        namespace = _checkpoint_namespace(checkpoint_root, family)
        latest = latest_verified_family_checkpoint(
            namespace,
            plan_fingerprint=plan.fingerprint,
            family=family,
            rebind_plan_marker=False,
        )
        families.append(
            {
                "family": family.family,
                "family_fingerprint": family.fingerprint,
                "latest_complete": (
                    None
                    if latest is None
                    else {
                        "path": latest[0].relative_to(checkpoint_root).as_posix(),
                        "step": latest[1].step,
                        "fingerprint": latest[1].fingerprint,
                    }
                ),
            }
        )
    atomic_write_json(
        resume_contract_path,
        {
            "schema_version": MODAL_APP_CONTRACT_SCHEMA,
            "plan_fingerprint": plan.fingerprint,
            "families": families,
        },
    )


def _latest_checkpoint_after_run(
    plan: TrainingPlan,
    checkpoint_root: Path,
) -> tuple[int, str | None]:
    latest_items: list[tuple[int, str]] = []
    for family in plan.families:
        if not family.enabled:
            continue
        latest = latest_verified_family_checkpoint(
            _checkpoint_namespace(checkpoint_root, family),
            plan_fingerprint=plan.fingerprint,
            family=family,
            rebind_plan_marker=False,
        )
        if latest is not None:
            latest_items.append(
                (latest[1].step, latest[0].relative_to(checkpoint_root).as_posix())
            )
    return max(latest_items, default=(0, None), key=lambda item: item[0])


def execute_remote_training(
    payload: Mapping[str, Any],
    *,
    volume_root: Path,
    runner: Any | None = None,
    asset_materializer: Callable[[TrainingPlan, Path], Mapping[str, Path]] | None = None,
    volume_commit: Callable[[], None] | None = None,
    family_claim_store: Any | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Verify one uploaded plan, invoke only the shared runner, and finalize atomically."""

    values = _validated_payload(payload)
    volume_root = volume_root.absolute()
    if volume_root.is_symlink() or _is_junction(volume_root) or not volume_root.is_dir():
        raise ValueError("Modal Volume mount must be a real directory")
    volume_root = volume_root.resolve(strict=True)
    bundle_namespace = PurePosixPath(values["bundle_namespace"])
    plan_relative = _volume_relative(values["plan_path"], label="Modal plan path")
    expected_plan_relative = bundle_namespace / PurePosixPath(PLAN_PATH)
    if plan_relative != expected_plan_relative:
        raise ValueError("Modal plan path does not belong to the uploaded bundle")
    bundle_root = _real_volume_directory(
        volume_root,
        bundle_namespace,
        create=False,
    )
    verified_bundle = verify_training_bundle(
        bundle_root,
        expected_plan_fingerprint=values["plan_fingerprint"],
    )
    if verified_bundle.inventory.fingerprint != values["bundle_fingerprint"]:
        raise ValueError("Modal bundle fingerprint is invalid")
    plan_bytes = volume_root.joinpath(*plan_relative.parts).read_bytes()
    plan = training_plan_from_bytes(
        plan_bytes,
        expected_fingerprint=values["plan_fingerprint"],
    )
    expected_prefix = PurePosixPath("plans", plan.plan_id)
    if bundle_namespace.parts[:2] != expected_prefix.parts:
        raise ValueError("Modal bundle namespace does not match its TrainingPlan")
    result_namespace = PurePosixPath(values["result_namespace"])
    if result_namespace != expected_prefix / "results":
        raise ValueError("Modal result namespace does not match its TrainingPlan")

    plan_root = _real_volume_directory(volume_root, expected_prefix, create=True)
    result_root = _real_volume_directory(volume_root, result_namespace, create=True)
    asset_root = _real_volume_directory(volume_root, PurePosixPath("assets"), create=True)
    materialize = materialize_verified_assets if asset_materializer is None else asset_materializer
    materialize(plan, asset_root)
    if volume_commit is not None:
        volume_commit()

    # Training-compatible checkpoints are keyed only by the family fingerprint.
    # Plan-level changes such as publication/evaluation thresholds must not
    # strand a fully resumable optimizer/dataloader state.
    checkpoint_root = _real_volume_directory(
        volume_root,
        PurePosixPath("checkpoints"),
        create=True,
    )
    for family in plan.families:
        if family.enabled:
            _real_volume_directory(
                volume_root,
                PurePosixPath("checkpoints", family.family, family.fingerprint),
                create=True,
            )
    _write_resume_contract(
        plan,
        checkpoint_root,
        plan_root / RESUME_INDEX_NAME,
    )
    if volume_commit is not None:
        volume_commit()
    runs_root = _real_volume_directory(
        volume_root,
        result_namespace / "runs",
        create=True,
    )
    run_root = runs_root / uuid4().hex
    run_root.mkdir(parents=True)
    if run_root.is_symlink() or _is_junction(run_root) or run_root.parent != result_root / "runs":
        raise RuntimeError("Remote run directory failed containment verification")
    status_path = plan_root / REMOTE_STATUS_NAME
    enabled_models = {family.family for family in plan.families if family.enabled}

    def persist_status(model: str, progress: float, checkpoint: str) -> None:
        if (
            model not in enabled_models
            or not isinstance(progress, (int, float))
            or isinstance(progress, bool)
            or not math.isfinite(float(progress))
            or progress < 0
            or not float(progress).is_integer()
        ):
            raise ValueError("Remote runner reported invalid progress")
        checkpoint_value: str | None = checkpoint or None
        if checkpoint_value is not None:
            _portable(checkpoint_value, label="Remote checkpoint status")
        status_value = {
            "executor": "modal",
            "remote_state": "running",
            "model": str(model),
            "step": int(progress),
            "checkpoint": checkpoint_value,
        }
        _reject_secret_mapping(status_value, label="Remote runner status")
        atomic_write_json(status_path, status_value)
        if volume_commit is not None:
            # The shared runner writes a checkpoint completion marker before
            # reporting it. Committing at this boundary makes that complete
            # checkpoint survive a retry/preemption; partial directories remain
            # harmless because resume accepts only fully checksummed markers.
            volume_commit()

    if (family_claim_store is None) != (call_id is None):
        raise ValueError("Remote family claim store and FunctionCall ID must be provided together")
    owned_family_claims = (
        ()
        if family_claim_store is None or call_id is None
        else acquire_remote_family_claims(
            plan,
            claim_store=family_claim_store,
            call_id=call_id,
        )
    )
    common_runner = _common_runner() if runner is None else runner
    try:
        try:
            families = tuple(
                common_runner(
                    plan_bytes,
                    bundle_root,
                    run_root,
                    checkpoint_root,
                    asset_root,
                    status_callback=persist_status,
                )
            )
        except Exception:
            # The common runner can call multiple third-party frameworks.  Keep
            # its raw diagnostics out of Modal logs because they are not
            # guaranteed to redact credentials inherited through the function
            # environment.
            raise RuntimeError("Remote common training runner failed") from None
    finally:
        if family_claim_store is not None:
            release_remote_family_claims(
                claim_store=family_claim_store,
                owned=owned_family_claims,
            )
    if not families:
        raise RuntimeError("Common training runner returned no family results")
    write_training_result_contract(run_root, plan=plan, families=families)
    checkpoint_step, checkpoint = _latest_checkpoint_after_run(plan, checkpoint_root)
    candidate_steps: list[int] = []
    for family in families:
        for candidate in family.candidates:
            candidate_step = candidate.validation.get("step", 0)
            if (
                isinstance(candidate_step, (int, float))
                and not isinstance(candidate_step, bool)
                and math.isfinite(float(candidate_step))
                and candidate_step >= 0
            ):
                candidate_steps.append(int(candidate_step))
    step = max([checkpoint_step, *candidate_steps], default=0)
    model = ",".join(sorted(family.family for family in plan.families if family.enabled))
    write_completion_manifest(
        run_root,
        kind="result",
        plan_fingerprint=plan.fingerprint,
        model=model,
        step=step,
        checkpoint=checkpoint,
        # Remote validation chooses a complete candidate.  Publication remains
        # a local-only decision after held-out evaluation on the downloaded
        # artifact, so the remote result must never claim that gate has passed.
        quality_gate_passed=False,
    )
    marker = run_root / RESULT_COMPLETION_NAME
    marker_relative = marker.relative_to(volume_root).as_posix()
    final_status = {
        "executor": "modal",
        "remote_state": "complete",
        "model": model,
        "step": step,
        "checkpoint": checkpoint,
    }
    _reject_secret_mapping(final_status, label="Remote runner status")
    atomic_write_json(status_path, final_status)
    if volume_commit is not None:
        volume_commit()
    result = {
        "plan_fingerprint": plan.fingerprint,
        "remote_state": "complete",
        "model": model,
        "step": step,
        "checkpoint": checkpoint,
        "completion_manifest_path": f"/{marker_relative}",
        "completion_manifest_sha256": _sha256_file(marker),
    }
    _reject_secret_mapping(result, label="Remote result")
    return result


def _probe_modal_function_call(modal: Any, call_id: str) -> str:
    """Return a recovery-safe state for one official durable FunctionCall."""

    if not isinstance(call_id, str) or _MODAL_CALL_ID.fullmatch(call_id) is None:
        raise ValueError("Modal terminal-call probe received an invalid call ID")
    call = modal.FunctionCall.from_id(call_id)
    try:
        call.get(timeout=0)
    except Exception as exc:
        state = _modal_poll_exception_state(modal, exc)
        if state == "terminal":
            return "failed"
        if state == "running":
            return "running"
        output_expired = getattr(getattr(modal, "exception", None), "OutputExpiredError", None)
        if isinstance(output_expired, type) and isinstance(exc, output_expired):
            return "expired"
        raise RuntimeError(
            "Modal FunctionCall terminal state could not be independently verified"
        ) from None
    return "complete"


def create_modal_app(
    *,
    modal_module: Any | None = None,
    contract: ModalAppContract | None = None,
    repository_root: Path | None = None,
) -> Any:
    """Create the deployable Modal App without making Modal a local dependency."""

    modal = importlib.import_module("modal") if modal_module is None else modal_module
    local_root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    if contract is None:
        # ``modal deploy -m ...`` bypasses PersonaVoice's CLI callback. Load the
        # same strict allowlist here so non-secret app/Volume/GPU overrides and
        # local Modal authentication behave identically without ever copying
        # the root .env into the image.
        load_root_environment(local_root)
        cfg = ModalAppContract.from_env()
    else:
        cfg = contract
    required_image_inputs = (
        local_root / "src",
        local_root / "workers",
        local_root / "locks",
        local_root / "config",
        local_root / "pyproject.toml",
        local_root / "uv.lock",
    )
    missing = [str(path) for path in required_image_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Modal image inputs are incomplete: " + ", ".join(missing)
        )
    app = modal.App(cfg.app_name)
    volume = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)
    claim_store = modal.Dict.from_name(cfg.claim_dict_name, create_if_missing=True)
    hf_secret = modal.Secret.from_name(cfg.hf_secret_name, required_keys=["HF_TOKEN"])
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "git")
        .uv_sync(
            uv_project_dir=str(local_root),
            frozen=True,
            extra_options="--no-dev",
            uv_version=UV_VERSION,
        )
        .add_local_dir(
            str(local_root / "src"),
            f"{MODAL_IMAGE_ROOT}/src",
            copy=True,
            ignore=_IMAGE_COPY_IGNORE,
        )
        .add_local_dir(
            str(local_root / "workers"),
            f"{MODAL_IMAGE_ROOT}/workers",
            copy=True,
            ignore=_IMAGE_COPY_IGNORE,
        )
        .add_local_dir(
            str(local_root / "locks"),
            f"{MODAL_IMAGE_ROOT}/locks",
            copy=True,
            ignore=_IMAGE_COPY_IGNORE,
        )
        .add_local_dir(
            str(local_root / "config"),
            f"{MODAL_IMAGE_ROOT}/config",
            copy=True,
            ignore=_IMAGE_COPY_IGNORE,
        )
        .add_local_file(
            str(local_root / "pyproject.toml"),
            f"{MODAL_IMAGE_ROOT}/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            str(local_root / "uv.lock"),
            f"{MODAL_IMAGE_ROOT}/uv.lock",
            copy=True,
        )
        .run_commands(
            f"mkdir -p {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS",
            f"git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS init",
            (
                f"git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS remote add origin "
                f"{IRODORI_UPSTREAM_URL}"
            ),
            (
                f"git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS fetch --depth=1 origin "
                f"{IRODORI_SOURCE_REVISION}"
            ),
            f"git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS checkout --detach FETCH_HEAD",
            (
                f"git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS rev-parse HEAD "
                f"| grep -Fx {IRODORI_SOURCE_REVISION}"
            ),
            (
                f"uv sync --project {MODAL_IMAGE_ROOT} --frozen --no-dev "
                "--no-install-project"
            ),
            (
                f"uv sync --project {MODAL_IMAGE_ROOT}/workers/lfm --frozen "
                "--extra cu128"
            ),
            (
                f"cp {MODAL_IMAGE_ROOT}/locks/Irodori-TTS.pyproject.toml "
                f"{MODAL_IMAGE_ROOT}/vendor/Irodori-TTS/pyproject.toml"
            ),
            (
                f"cp {MODAL_IMAGE_ROOT}/locks/Irodori-TTS.uv.lock "
                f"{MODAL_IMAGE_ROOT}/vendor/Irodori-TTS/uv.lock"
            ),
            (
                f"uv sync --project {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS --frozen "
                "--extra cu128"
            ),
            (
                f"git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS restore "
                "--source=HEAD --worktree --staged -- ."
            ),
            (
                f"git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS clean -f -- "
                "pyproject.toml uv.lock"
            ),
            (
                f"test -z \"$(git -C {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS "
                "status --porcelain --untracked-files=all)\""
            ),
        )
        .env(
            {
                "PYTHONPATH": f"{MODAL_IMAGE_ROOT}/src",
                "PERSONAVOICE_IMAGE_ROOT": MODAL_IMAGE_ROOT,
            }
        )
        .workdir(MODAL_IMAGE_ROOT)
    )
    retries = modal.Retries(
        max_retries=cfg.max_retries,
        initial_delay=cfg.retry_initial_delay_seconds,
        backoff_coefficient=cfg.retry_backoff_coefficient,
    )

    @app.function(
        image=image,
        gpu=cfg.gpu,
        volumes={MODAL_VOLUME_MOUNT: volume},
        secrets=[hf_secret],
        timeout=cfg.timeout_seconds,
        retries=retries,
        # The deployable app is built by a factory so tests can inject a fake
        # SDK without credentials. Modal requires nested function definitions
        # to be explicitly serialized.
        name=REMOTE_TRAIN_FUNCTION_NAME,
        serialized=True,
    )
    def train(payload: Mapping[str, Any]) -> dict[str, Any]:
        call_id = modal.current_function_call_id()
        return execute_claimed_remote_training(
            payload,
            claim_store=claim_store,
            call_id=call_id,
            executor=lambda: execute_remote_training(
                payload,
                volume_root=Path(MODAL_VOLUME_MOUNT),
                volume_commit=volume.commit,
                family_claim_store=claim_store,
                call_id=call_id,
            ),
        )

    @app.function(
        image=image,
        timeout=120,
        retries=2,
        max_containers=1,
        name=REMOTE_RECOVERY_FUNCTION_NAME,
        # Exactly one recovery container serializes cleanup requests. The
        # helper itself is idempotent, but this prevents two clients from
        # racing a plan-claim pop with a replacement FunctionCall.
        serialized=True,
    )
    def recover_terminal_claim(payload: Mapping[str, Any]) -> dict[str, Any]:
        return recover_remote_terminal_claim(
            payload,
            claim_store=claim_store,
            terminal_probe=lambda call_id: _probe_modal_function_call(modal, call_id),
        )

    return app


def _optional_default_app() -> Any | None:
    try:
        modal = importlib.import_module("modal")
    except ImportError:
        return None
    return create_modal_app(modal_module=modal)


# ``modal deploy -m personavoice.modal_app`` discovers this when the optional SDK
# is installed. Ordinary local imports remain usable without Modal.
app = _optional_default_app()
