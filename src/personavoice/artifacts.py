from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from personavoice.atomic import atomic_write_json
from personavoice.irodori import (
    irodori_full_artifact_complete,
    irodori_lora_candidate_complete,
    lora_adapter_complete,
    speaker_embedding_complete,
)
from personavoice.model_assets import LFM_MODEL_REVISION
from personavoice.training_plan import TrainingPlan, sha256_file

ARTIFACT_SCHEMA_VERSION = 1
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_MANIFEST_NAME = "manifest.json"
_PROVENANCE_NAME = "provenance.json"
PUBLICATION_SCHEMA_VERSION = 1
PUBLICATION_NAME = "publication.json"
_LFM_REVISION_MARKER = ".personavoice-base-revision"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker is not None and checker())


def _portable_path(value: str) -> bool:
    if not value or value.startswith(("/", "\\")) or _DRIVE_PATH_RE.match(value):
        return False
    path = PurePosixPath(value.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _assert_no_absolute_paths(value: Any, *, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_no_absolute_paths(child, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_absolute_paths(child, location=f"{location}[{index}]")
        return
    if not isinstance(value, str):
        return
    if value.startswith(("/", "\\\\")) or _DRIVE_PATH_RE.match(value):
        raise ValueError(f"Portable artifact metadata contains an absolute path at {location}")


@dataclass(frozen=True)
class ArtifactVerification:
    family: str
    method: str
    plan_fingerprint: str
    published: bool
    files: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PublicationItem:
    family: str
    method: str
    family_fingerprint: str
    candidate: Path
    destination: Path
    component: str = "primary"


@dataclass(frozen=True)
class CandidateVerification:
    family: str
    method: str
    family_fingerprint: str
    files: tuple[dict[str, Any], ...]
    digest: str


def _inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or _is_junction(root):
        raise ValueError(f"Portable artifact root may not be a link or junction: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink() or _is_junction(path):
            raise ValueError(f"Portable artifacts may not contain links or junctions: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Portable artifacts may contain regular files only: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in {_MANIFEST_NAME, _PROVENANCE_NAME}:
            continue
        if not _portable_path(relative):
            raise ValueError(f"Portable artifact path is unsafe: {relative}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Portable artifact contains an empty file: {relative}")
        files.append({"path": relative, "size": size, "sha256": sha256_file(path)})
    return files


def _all_file_inventory(path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or _is_junction(path):
        raise ValueError(f"Training candidates may not be links or junctions: {path}")
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    files: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.is_symlink() or _is_junction(candidate):
            raise ValueError(
                f"Training candidates may not contain links or junctions: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"Training candidates may contain regular files only: {candidate}")
        relative = candidate.name if path.is_file() else candidate.relative_to(path).as_posix()
        if not _portable_path(relative):
            raise ValueError(f"Training candidate path is unsafe: {relative}")
        size = candidate.stat().st_size
        if size <= 0:
            raise ValueError(f"Training candidate contains an empty file: {relative}")
        files.append({"path": relative, "size": size, "sha256": sha256_file(candidate)})
    if not files:
        raise RuntimeError(f"Training candidate contains no files: {path}")
    return tuple(files)


def _inventory_digest(files: tuple[dict[str, Any], ...]) -> str:
    payload = json.dumps(
        list(files),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_lfm_full(path: Path, *, family_fingerprint: str) -> None:
    provenance_path = path / _PROVENANCE_NAME
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("LFM full-model provenance is unreadable") from exc
    if not isinstance(provenance, dict) or (
        provenance.get("schema_version") != 1
        or provenance.get("family") != "lfm"
        or provenance.get("method") != "full"
        or provenance.get("training_plan_fingerprint") != family_fingerprint
    ):
        raise RuntimeError("LFM full-model provenance does not match the family plan")
    _assert_no_absolute_paths(provenance)
    raw_files = provenance.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise RuntimeError("LFM full-model provenance has no file inventory")
    expected: dict[str, tuple[int, str]] = {}
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise RuntimeError("LFM full-model file inventory is invalid")
        relative = item.get("path")
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not _portable_path(relative)
            or relative in expected
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise RuntimeError("LFM full-model file inventory entry is invalid")
        expected[relative] = (size, digest)
    actual = {
        item["path"]: (item["size"], item["sha256"])
        for item in _all_file_inventory(path)
        if item["path"] != _PROVENANCE_NAME
    }
    if actual != expected:
        raise RuntimeError("LFM full-model file inventory or checksum is invalid")
    required = {"config.json", "model.safetensors", "tokenizer_config.json"}
    if not required.issubset(actual):
        raise RuntimeError("LFM full-model artifact is not independently loadable")
    if "adapter_config.json" in actual or any(
        name in actual for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError("LFM full-model artifact unexpectedly contains a PEFT adapter")
    try:
        revision = (path / _LFM_REVISION_MARKER).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("LFM full-model revision marker is missing") from exc
    if revision != LFM_MODEL_REVISION:
        raise RuntimeError("LFM full-model revision marker is incorrect")


def _verify_lfm_lora(path: Path, *, family_fingerprint: str) -> None:
    weights = [
        candidate
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (candidate := path / name).is_file() and candidate.stat().st_size > 0
    ]
    if not (path / "adapter_config.json").is_file() or len(weights) != 1:
        raise RuntimeError("LFM LoRA adapter is incomplete")
    try:
        revision = (path / _LFM_REVISION_MARKER).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("LFM LoRA base revision marker is missing") from exc
    if revision != LFM_MODEL_REVISION:
        raise RuntimeError("LFM LoRA base revision marker is incorrect")
    provenance_path = path / _PROVENANCE_NAME
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("LFM LoRA provenance is unreadable") from exc
    if not isinstance(provenance, dict) or (
        provenance.get("schema_version") != 1
        or provenance.get("family") != "lfm"
        or provenance.get("method") != "lora"
        or provenance.get("training_plan_fingerprint") != family_fingerprint
    ):
        raise RuntimeError("LFM LoRA provenance does not match the family plan")
    best_validation_loss = provenance.get("best_validation_loss")
    if (
        not isinstance(best_validation_loss, (int, float))
        or isinstance(best_validation_loss, bool)
        or not math.isfinite(float(best_validation_loss))
    ):
        raise RuntimeError("LFM LoRA provenance has no finite best validation loss")
    _assert_no_absolute_paths(provenance)


def verify_training_candidate(
    path: Path,
    *,
    family: str,
    method: str,
    family_fingerprint: str,
) -> CandidateVerification:
    """Verify one method-specific candidate without relying on its source machine."""

    if not re.fullmatch(r"[0-9a-f]{64}", family_fingerprint):
        raise ValueError("Family fingerprint must be a lowercase SHA-256 digest")
    if family == "irodori" and method == "full":
        if not irodori_full_artifact_complete(
            path,
            plan_fingerprint=family_fingerprint,
        ):
            raise RuntimeError("Irodori full-model artifact failed its portable contract")
    elif family == "irodori" and method == "lora":
        if not lora_adapter_complete(path):
            raise RuntimeError("Irodori LoRA adapter is incomplete")
        if not irodori_lora_candidate_complete(
            path,
            plan_fingerprint=family_fingerprint,
        ):
            raise RuntimeError("Irodori LoRA provenance does not match the family plan")
    elif family == "irodori" and method == "speaker-inversion":
        if not speaker_embedding_complete(path):
            raise RuntimeError("Irodori speaker embedding is incomplete")
    elif family == "lfm" and method == "full":
        _verify_lfm_full(path, family_fingerprint=family_fingerprint)
    elif family == "lfm" and method == "lora":
        _verify_lfm_lora(path, family_fingerprint=family_fingerprint)
    elif family == "seed-vc" and method == "finetune":
        if not _nonempty_file(path):
            raise RuntimeError("Seed-VC fine-tuned checkpoint is missing or empty")
    else:
        raise ValueError(f"Unsupported training candidate: {family}/{method}")
    files = _all_file_inventory(path)
    return CandidateVerification(
        family=family,
        method=method,
        family_fingerprint=family_fingerprint,
        files=files,
        digest=_inventory_digest(files),
    )


def _critical_files(family: str) -> tuple[str, ...]:
    if family == "irodori":
        return ("model.safetensors", "tokenizer/tokenizer_config.json")
    if family == "lfm":
        return ("config.json", "model.safetensors", "tokenizer_config.json")
    raise ValueError(f"Unsupported portable artifact family: {family!r}")


def write_artifact_contract(
    artifact_dir: Path,
    *,
    family: str,
    method: str,
    plan: TrainingPlan,
    family_fingerprint: str,
    training: dict[str, Any],
    runtime: dict[str, Any],
    source_checkpoint: str,
    quality: dict[str, Any] | None = None,
    published: bool = False,
) -> None:
    """Write portable inventory/provenance without embedding machine-local paths."""

    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
    if not _portable_path(source_checkpoint):
        raise ValueError("source_checkpoint must be relative to the training run directory")
    files = _inventory(artifact_dir)
    present = {item["path"] for item in files}
    missing = [relative for relative in _critical_files(family) if relative not in present]
    if missing:
        raise RuntimeError(
            f"Portable {family} artifact is missing required files: {', '.join(missing)}"
        )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "family": family,
        "files": files,
    }
    family_contract = plan.family(family).as_dict()
    provenance = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "family": family,
        "method": method,
        "plan_fingerprint": plan.fingerprint,
        "family_fingerprint": family_fingerprint,
        "model_contract": family_contract["model_contract"],
        "dataset_fingerprint": family_contract["dataset_fingerprint"],
        "training": training,
        "checkpoint_policy": family_contract["checkpoint_policy"],
        "evaluation_policy": family_contract["evaluation_policy"],
        "source_checkpoint": source_checkpoint,
        "runtime": runtime,
        "quality": quality,
        "published": bool(published),
        "created_at": _utc_now(),
    }
    _assert_no_absolute_paths(manifest)
    _assert_no_absolute_paths(provenance)
    atomic_write_json(artifact_dir / _MANIFEST_NAME, manifest)
    atomic_write_json(artifact_dir / _PROVENANCE_NAME, provenance)


def verify_portable_artifact(
    artifact_dir: Path,
    *,
    expected_family: str | None = None,
    expected_plan_fingerprint: str | None = None,
    require_published: bool = False,
) -> ArtifactVerification:
    if artifact_dir.is_symlink() or _is_junction(artifact_dir) or not artifact_dir.is_dir():
        raise RuntimeError("Portable artifact root must be a regular directory")
    try:
        manifest = json.loads((artifact_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
        provenance = json.loads((artifact_dir / _PROVENANCE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Portable artifact metadata is unreadable: {artifact_dir}") from exc
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        raise RuntimeError("Portable artifact manifest/provenance roots must be objects")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported portable artifact manifest schema")
    if provenance.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported portable artifact provenance schema")
    family = manifest.get("family")
    if family not in {"irodori", "lfm"} or provenance.get("family") != family:
        raise RuntimeError("Portable artifact family contract is inconsistent")
    if expected_family is not None and family != expected_family:
        raise RuntimeError(f"Expected {expected_family} artifact, got {family}")
    method = provenance.get("method")
    if method not in {"full", "lora"}:
        raise RuntimeError(f"Unsupported portable artifact method: {method!r}")
    plan_fingerprint = provenance.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or len(plan_fingerprint) != 64:
        raise RuntimeError("Portable artifact plan fingerprint is invalid")
    if expected_plan_fingerprint is not None and plan_fingerprint != expected_plan_fingerprint:
        raise RuntimeError("Portable artifact belongs to a different TrainingPlan")
    published = provenance.get("published") is True
    if require_published and not published:
        raise RuntimeError("Portable artifact has not passed the publication quality gate")
    _assert_no_absolute_paths(manifest)
    _assert_no_absolute_paths(provenance)

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise RuntimeError("Portable artifact file inventory is missing")
    seen: set[str] = set()
    files: list[dict[str, Any]] = []
    actual_files: set[str] = set()
    for entry in sorted(artifact_dir.rglob("*"), key=lambda value: value.as_posix()):
        if entry.is_symlink() or _is_junction(entry):
            raise RuntimeError(f"Portable artifact contains a link or junction: {entry}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise RuntimeError(f"Portable artifact contains a non-regular entry: {entry}")
        relative = entry.relative_to(artifact_dir).as_posix()
        if not _portable_path(relative):
            raise RuntimeError(f"Portable artifact contains an unsafe path: {relative}")
        actual_files.add(relative)
    root = artifact_dir.resolve()
    for item in raw_files:
        if not isinstance(item, dict):
            raise RuntimeError("Portable artifact file inventory contains a non-object")
        relative = item.get("path")
        if not isinstance(relative, str) or not _portable_path(relative) or relative in seen:
            raise RuntimeError(f"Portable artifact file path is invalid: {relative!r}")
        seen.add(relative)
        path = root.joinpath(*PurePosixPath(relative).parts).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Portable artifact file escapes its root: {relative}") from exc
        if path.is_symlink() or not _nonempty_file(path):
            raise RuntimeError(f"Portable artifact file is missing/empty: {relative}")
        size = item.get("size")
        digest = item.get("sha256")
        if not isinstance(size, int) or size <= 0 or path.stat().st_size != size:
            raise RuntimeError(f"Portable artifact size mismatch: {relative}")
        actual = sha256_file(path)
        if not isinstance(digest, str) or actual != digest:
            raise RuntimeError(f"Portable artifact checksum mismatch: {relative}")
        files.append(item)
    expected_files = seen | {_MANIFEST_NAME, _PROVENANCE_NAME}
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        absent = sorted(expected_files - actual_files)
        details: list[str] = []
        if extra:
            details.append(f"unlisted files: {', '.join(extra)}")
        if absent:
            details.append(f"missing files: {', '.join(absent)}")
        raise RuntimeError(f"Portable artifact inventory is not exact ({'; '.join(details)})")
    missing = [relative for relative in _critical_files(family) if relative not in seen]
    if missing:
        raise RuntimeError(
            f"Portable {family} artifact is missing required inventory: {', '.join(missing)}"
        )
    return ArtifactVerification(
        family=family,
        method=method,
        plan_fingerprint=plan_fingerprint,
        published=published,
        files=tuple(files),
    )


def publish_artifact(
    candidate_dir: Path,
    final_dir: Path,
    *,
    quality: dict[str, Any],
) -> Path:
    """Publish a verified candidate transactionally after a passing quality gate."""

    if quality.get("passed") is not True:
        raise RuntimeError("Quality gate did not pass; refusing to publish the candidate")
    candidate = verify_portable_artifact(candidate_dir, require_published=False)
    staging = final_dir.with_name(f".{final_dir.name}.{uuid4().hex}.staging")
    archive: Path | None = None
    try:
        shutil.copytree(candidate_dir, staging)
        provenance_path = staging / _PROVENANCE_NAME
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["quality"] = quality
        provenance["published"] = True
        provenance["published_at"] = _utc_now()
        _assert_no_absolute_paths(provenance)
        atomic_write_json(provenance_path, provenance)
        verify_portable_artifact(
            staging,
            expected_family=candidate.family,
            expected_plan_fingerprint=candidate.plan_fingerprint,
            require_published=True,
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            archive_root = final_dir.parent / "history"
            archive_root.mkdir(parents=True, exist_ok=True)
            archive = archive_root / (
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{candidate.plan_fingerprint[:16]}-{uuid4().hex[:8]}"
            )
            os.replace(final_dir, archive)
        os.replace(staging, final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if archive is not None and archive.exists() and not final_dir.exists():
            os.replace(archive, final_dir)
        raise
    return final_dir


def _relative_to_models(models_root: Path, path: Path) -> str:
    root = models_root.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Publication destination escapes the persona models root: {path}"
        ) from exc
    if not _portable_path(relative):
        raise ValueError(f"Publication destination is not portable: {relative}")
    return relative


def _copy_candidate(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        shutil.copytree(source, destination)


def _remove_materialized(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publication_payload(
    *,
    plan: TrainingPlan,
    models_root: Path,
    items: tuple[PublicationItem, ...],
    quality: dict[str, Any],
) -> dict[str, Any]:
    artifacts = []
    for item in items:
        verified = verify_training_candidate(
            item.destination,
            family=item.family,
            method=item.method,
            family_fingerprint=item.family_fingerprint,
        )
        artifacts.append(
            {
                "family": item.family,
                "component": item.component,
                "method": item.method,
                "family_fingerprint": item.family_fingerprint,
                "path": _relative_to_models(models_root, item.destination),
                "digest": verified.digest,
            }
        )
    artifacts.sort(key=lambda value: value["family"])
    payload = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "published_at": _utc_now(),
        "quality_gate": quality,
        "artifacts": artifacts,
    }
    _assert_no_absolute_paths(payload)
    return payload


def verify_publication(
    models_root: Path,
    *,
    expected_plan_fingerprint: str,
    expected_families: dict[str, tuple[str, str, str]] | None = None,
    expected_components: dict[tuple[str, str], tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Verify the local publication marker and every selected model byte."""

    try:
        value = json.loads((models_root / PUBLICATION_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Model publication contract is unreadable") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "plan_fingerprint",
        "published_at",
        "quality_gate",
        "artifacts",
    }:
        raise RuntimeError("Model publication contract schema is invalid")
    if (
        value.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        or value.get("plan_fingerprint") != expected_plan_fingerprint
    ):
        raise RuntimeError("Model publication belongs to a different TrainingPlan")
    quality = value.get("quality_gate")
    if not isinstance(quality, dict) or quality.get("passed") is not True:
        raise RuntimeError("Model publication has no passing quality gate")
    _assert_no_absolute_paths(value)
    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise RuntimeError("Model publication artifact list is invalid")
    found: dict[str, tuple[str, str, str]] = {}
    found_components: dict[tuple[str, str], tuple[str, str, str]] = {}
    root = models_root.resolve()
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or set(raw) != {
            "family",
            "component",
            "method",
            "family_fingerprint",
            "path",
            "digest",
        }:
            raise RuntimeError("Model publication artifact entry is invalid")
        family = raw.get("family")
        component = raw.get("component")
        method = raw.get("method")
        fingerprint = raw.get("family_fingerprint")
        relative = raw.get("path")
        if (
            not isinstance(family, str)
            or not isinstance(component, str)
            or not component
            or (family, component) in found_components
            or not isinstance(method, str)
            or not isinstance(fingerprint, str)
            or not isinstance(relative, str)
            or not _portable_path(relative)
        ):
            raise RuntimeError("Model publication artifact identity is invalid")
        artifact = root.joinpath(*PurePosixPath(relative).parts).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Model publication artifact escapes its root") from exc
        verified = verify_training_candidate(
            artifact,
            family=family,
            method=method,
            family_fingerprint=fingerprint,
        )
        if raw.get("digest") != verified.digest:
            raise RuntimeError("Published model digest has changed")
        found_components[(family, component)] = (method, fingerprint, relative)
        if component == "primary":
            found[family] = (method, fingerprint, relative)
    if expected_families is not None and found != expected_families:
        raise RuntimeError("Published model set does not match the training result")
    if expected_components is not None and found_components != expected_components:
        raise RuntimeError("Published model components do not match the training result")
    return value


def publish_training_candidates(
    models_root: Path,
    *,
    plan: TrainingPlan,
    items: list[PublicationItem],
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Atomically switch a complete multi-family candidate set after local gating.

    Existing published artifacts are moved to a recoverable history directory.
    A failure at any point restores every old destination and publication marker.
    """

    if quality.get("passed") is not True:
        raise RuntimeError("Quality gate did not pass; refusing to publish candidates")
    ordered = tuple(
        sorted(items, key=lambda item: (item.family, item.component, str(item.destination)))
    )
    if not ordered:
        raise ValueError("Publication requires at least one enabled candidate")
    if len({(item.family, item.component) for item in ordered}) != len(ordered):
        raise ValueError("Publication contains duplicate model components")
    destinations = [item.destination.resolve(strict=False) for item in ordered]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Publication destinations are duplicated")
    for index, destination in enumerate(destinations):
        _relative_to_models(models_root, destination)
        if any(
            destination != other and (destination in other.parents or other in destination.parents)
            for other in destinations[index + 1 :]
        ):
            raise ValueError("Publication destinations overlap")
    candidate_verifications: dict[tuple[str, str], CandidateVerification] = {}
    for item in ordered:
        candidate_verifications[(item.family, item.component)] = verify_training_candidate(
            item.candidate,
            family=item.family,
            method=item.method,
            family_fingerprint=item.family_fingerprint,
        )

    # A process may die after the atomic publication marker was committed but
    # before the stage state was updated.  Re-running evaluation must adopt the
    # already verified byte-identical publication, rather than archive and copy
    # the same large models again.
    expected = {
        item.family: (
            item.method,
            item.family_fingerprint,
            _relative_to_models(models_root, item.destination),
        )
        for item in ordered
        if item.component == "primary"
    }
    try:
        current = verify_publication(
            models_root,
            expected_plan_fingerprint=plan.fingerprint,
            expected_families=expected,
        )
    except RuntimeError:
        current = None
    if current is not None:
        recorded = {
            (str(item["family"]), str(item["component"])): (
                item["method"],
                item["family_fingerprint"],
                item["path"],
                item["digest"],
            )
            for item in current["artifacts"]
        }
        requested: dict[tuple[str, str], tuple[str, str, str, str]] = {}
        identical = True
        for item in ordered:
            destination_verification = verify_training_candidate(
                item.destination,
                family=item.family,
                method=item.method,
                family_fingerprint=item.family_fingerprint,
            )
            candidate_verification = candidate_verifications[(item.family, item.component)]
            if item.candidate.is_file() and item.destination.is_file():
                same_bytes = [
                    (entry["size"], entry["sha256"]) for entry in candidate_verification.files
                ] == [(entry["size"], entry["sha256"]) for entry in destination_verification.files]
            else:
                same_bytes = candidate_verification.files == destination_verification.files
            identical = identical and same_bytes
            requested[(item.family, item.component)] = (
                item.method,
                item.family_fingerprint,
                _relative_to_models(models_root, item.destination),
                destination_verification.digest,
            )
        if identical and recorded == requested:
            return current

    transaction_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex}"
    staging_root = models_root / ".publication-staging" / transaction_id
    history_root = models_root / ".publication-history" / transaction_id
    marker = models_root / PUBLICATION_NAME
    marker_backup = history_root / PUBLICATION_NAME
    staged: dict[Path, Path] = {}
    archived: dict[Path, Path] = {}
    installed: list[Path] = []
    models_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(ordered):
            if item.candidate.resolve() == item.destination.resolve(strict=False):
                continue
            temporary = staging_root / f"{index:03d}-{item.family}"
            _copy_candidate(item.candidate, temporary)
            verify_training_candidate(
                temporary,
                family=item.family,
                method=item.method,
                family_fingerprint=item.family_fingerprint,
            )
            staged[item.destination] = temporary

        history_root.mkdir(parents=True, exist_ok=False)
        if marker.is_file():
            shutil.copy2(marker, marker_backup)
        for index, item in enumerate(ordered):
            destination = item.destination
            if destination not in staged:
                continue
            if destination.exists():
                archive = history_root / f"{index:03d}-{item.family}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                archived[destination] = archive
                os.replace(destination, archive)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[destination], destination)
            installed.append(destination)

        payload = _publication_payload(
            plan=plan,
            models_root=models_root,
            items=ordered,
            quality=quality,
        )
        atomic_write_json(marker, payload)
        verify_publication(
            models_root,
            expected_plan_fingerprint=plan.fingerprint,
            expected_families=expected,
        )
    except BaseException:
        for destination in reversed(installed):
            _remove_materialized(destination)
        for destination, archive in reversed(tuple(archived.items())):
            if archive.exists():
                if destination.exists():
                    _remove_materialized(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(archive, destination)
        if marker_backup.is_file():
            os.replace(marker_backup, marker)
        elif marker.exists():
            marker.unlink(missing_ok=True)
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return verify_publication(
        models_root,
        expected_plan_fingerprint=plan.fingerprint,
    )
