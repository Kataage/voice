from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from personavoice.environment import SECRET_ENV_KEYS
from personavoice.training_plan import TrainingPlan

BUNDLE_SCHEMA_VERSION = 1
PLAN_PATH = "contracts/training-plan.json"
COMPLETION_PATH = "contracts/bundle-complete.json"
IRODORI_SOURCE_MANIFEST_PATH = "contracts/inputs/irodori-latent-manifest-source.jsonl"
IRODORI_SANITIZED_MANIFEST_PATH = "data/irodori/manifest.jsonl"
IRODORI_LATENT_ROOT = "data/irodori/latents"

_HASH_CHUNK_BYTES = 1024 * 1024
_TRANSFER_ROLES = frozenset({"irodori-latent-manifest", "lfm-conversations"})
_AUDIO_SUFFIXES = frozenset(
    {".aac", ".aiff", ".alac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
)
_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".cache",
        ".env",
        "identity",
        "identity_audio",
        "irodori_source.jsonl",
        "master.sqlite",
        "master.sqlite3",
        "catalog.sqlite",
        "catalog.sqlite3",
        "raw",
        "reference",
        "references",
    }
)
_PATHISH_LFM_KEYS = frozenset(
    {
        "audio",
        "audio_file",
        "audio_path",
        "file",
        "file_path",
        "identity",
        "identity_audio",
        "path",
        "raw",
        "reference",
        "source_audio",
        "source_path",
    }
)
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
_ROUTING_KEYS = frozenset({"consent", "executor", "remote_data_authorized"})


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


def canonical_plan_bytes(plan: TrainingPlan) -> bytes:
    """Return the one canonical plan representation shared by every executor."""

    value = _canonical_json_bytes(plan.as_dict())
    if _sha256_bytes(value) != plan.fingerprint:
        raise RuntimeError("TrainingPlan canonical serialization disagrees with its fingerprint")
    return value


def _portable_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} is not a portable POSIX path: {value!r}")
    if value.startswith(("/", "~")) or ":" in value:
        raise ValueError(f"{label} is absolute or platform-specific: {value!r}")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{label} contains traversal or non-canonical segments: {value!r}")
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError(f"{label} is not canonical: {value!r}")
    return path


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker is not None and checker())


def _reject_link_chain(root: Path, relative: PurePosixPath) -> Path:
    if root.is_symlink() or _is_junction(root):
        raise ValueError(f"Bundle source root must not be a link or junction: {root}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or _is_junction(current):
            raise ValueError(f"Bundle source must not traverse a link or junction: {relative}")
    resolved_root = root.resolve(strict=True)
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Bundle source escapes the persona root: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"Bundle source is not a regular file: {relative}")
    return resolved


def _reject_forbidden_payload_path(path: PurePosixPath, *, label: str) -> None:
    folded = tuple(part.casefold() for part in path.parts)
    if any(part in _FORBIDDEN_COMPONENTS or part.startswith(".env.") for part in folded):
        raise ValueError(f"{label} is privacy-sensitive and cannot be transferred: {path}")
    if path.suffix.casefold() in _AUDIO_SUFFIXES:
        raise ValueError(f"{label} is audio and cannot be transferred: {path}")


def _load_json_lines(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} line {line_number} is not valid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} line {line_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} must contain at least one record")
    return rows


def _reject_configured_secret_values(value: Any, *, location: str) -> None:
    """Reject configured credential values without copying them into diagnostics.

    Secret-key validation alone cannot catch a credential pasted into otherwise
    legitimate free text (for example, an Irodori caption or an LFM message).
    Read the current process environment at validation time so credentials loaded
    after module import are covered, and inspect every nested JSON string. The
    exception intentionally reports only the safe record location.
    """

    configured_values = tuple(
        configured for key in SECRET_ENV_KEYS if (configured := os.environ.get(key, ""))
    )
    if not configured_values:
        return

    def contains_configured_secret(candidate: Any) -> bool:
        if isinstance(candidate, Mapping):
            return any(
                contains_configured_secret(str(key)) or contains_configured_secret(child)
                for key, child in candidate.items()
            )
        if isinstance(candidate, (list, tuple)):
            return any(contains_configured_secret(child) for child in candidate)
        return isinstance(candidate, str) and any(
            secret in candidate for secret in configured_values
        )

    if contains_configured_secret(value):
        raise ValueError(f"{location} contains a configured secret value")


def _validate_irodori_manifest(path: Path) -> tuple[dict[str, Any], ...]:
    allowed = {"caption", "latent_path", "num_frames", "speaker_id", "text"}
    rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(
        _load_json_lines(path, label="Irodori latent manifest"),
        start=1,
    ):
        _reject_configured_secret_values(
            row,
            location=f"Irodori latent manifest line {line_number}",
        )
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(
                "Irodori latent manifest contains unapproved fields on line "
                f"{line_number}: {sorted(unknown)}"
            )
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"Irodori latent manifest line {line_number} has no text")
        latent_value = row.get("latent_path")
        if (
            not isinstance(latent_value, str)
            or not latent_value
            or "\\" in latent_value
            or "\x00" in latent_value
            or latent_value.startswith(("/", "~"))
            or ":" in latent_value
        ):
            raise ValueError(
                f"Irodori latent_path on line {line_number} is absolute or non-portable"
            )
        latent = PurePosixPath(latent_value)
        if any(part in {"", "."} for part in latent_value.split("/")):
            raise ValueError(f"Irodori latent_path on line {line_number} is not canonical")
        if latent.suffix.casefold() != ".pt":
            raise ValueError(f"Irodori latent_path on line {line_number} must end in .pt: {latent}")
        if "caption" in row and not isinstance(row["caption"], str):
            raise ValueError(f"Irodori caption on line {line_number} must be text")
        if "speaker_id" in row and (
            not isinstance(row["speaker_id"], (str, int)) or isinstance(row["speaker_id"], bool)
        ):
            raise ValueError(f"Irodori speaker_id on line {line_number} is invalid")
        if "num_frames" in row and (
            not isinstance(row["num_frames"], int) or row["num_frames"] <= 0
        ):
            raise ValueError(f"Irodori num_frames on line {line_number} is invalid")
        rows.append(dict(row))
    return tuple(rows)


def _resolve_manifest_latent(
    *,
    root: Path,
    manifest_relative: PurePosixPath,
    latent_value: str,
) -> tuple[PurePosixPath, Path]:
    """Resolve an upstream manifest reference without permitting root escape.

    Existing v0.3 manifests intentionally use ``../cache/irodori_latents`` from
    the dataset directory. Controlled parent segments are therefore accepted,
    then normalized into a bundle-owned path; an escape above the persona root
    still fails closed.
    """

    parts = list(manifest_relative.parent.parts)
    for part in PurePosixPath(latent_value).parts:
        if part == "..":
            if not parts:
                raise ValueError("Irodori latent_path escapes the persona root")
            parts.pop()
        else:
            parts.append(part)
    relative = PurePosixPath(*parts)
    source = _reject_link_chain(root, relative)
    return relative, source


def _sanitized_irodori_payload(
    *,
    root: Path,
    manifest_relative: PurePosixPath,
    manifest_source: Path,
    temporary: Path,
    members: dict[str, BundleFile],
) -> None:
    output: list[bytes] = []
    for row in _validate_irodori_manifest(manifest_source):
        _, latent_source = _resolve_manifest_latent(
            root=root,
            manifest_relative=manifest_relative,
            latent_value=str(row["latent_path"]),
        )
        latent_sha = _sha256_file(latent_source)
        latent_relative = PurePosixPath(IRODORI_LATENT_ROOT, f"{latent_sha}.pt")
        existing = members.get(latent_relative.as_posix())
        if existing is None:
            sha256, size = _copy_verified(
                source=latent_source,
                destination=temporary.joinpath(*latent_relative.parts),
                expected_sha256=latent_sha,
            )
            members[latent_relative.as_posix()] = BundleFile(
                path=latent_relative.as_posix(),
                role="irodori-latent",
                sha256=sha256,
                size=size,
            )
        elif existing.sha256 != latent_sha or existing.role != "irodori-latent":
            raise ValueError(f"Bundle path collision: {latent_relative}")
        sanitized = dict(row)
        sanitized["latent_path"] = f"latents/{latent_sha}.pt"
        output.append(_canonical_json_bytes(sanitized) + b"\n")

    sanitized_bytes = b"".join(output)
    sanitized_relative = PurePosixPath(IRODORI_SANITIZED_MANIFEST_PATH)
    sanitized_destination = temporary.joinpath(*sanitized_relative.parts)
    _write_atomic_bytes(sanitized_destination, sanitized_bytes)
    members[sanitized_relative.as_posix()] = BundleFile(
        path=sanitized_relative.as_posix(),
        role="irodori-latent-manifest",
        sha256=_sha256_bytes(sanitized_bytes),
        size=len(sanitized_bytes),
    )


def _contains_pathish_lfm_field(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold().replace("-", "_")
            if folded in _PATHISH_LFM_KEYS or any(
                part in folded for part in ("audio_path", "source_path", "identity_audio")
            ):
                return str(key)
            found = _contains_pathish_lfm_field(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _contains_pathish_lfm_field(child)
            if found is not None:
                return found
    return None


def _contains_sensitive_lfm_string(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_sensitive_lfm_string(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_sensitive_lfm_string(child) for child in value)
    if not isinstance(value, str):
        return False
    folded = value.casefold()
    return bool(
        re.search(r"(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\|file://)", value, re.IGNORECASE)
        or re.search(r"(?<![A-Za-z0-9._:/-])/(?!/)[^\s\"']+", value)
        or re.search(r"(?:audio_path|identity_audio|source_path)\s*[\"']?\s*:", folded)
        or re.search(r"[/\\][^\s\"']+\.(?:wav|flac|mp3|m4a|ogg|opus)(?:$|[\s\"'])", folded)
    )


def _validate_lfm_dataset(path: Path) -> None:
    for line_number, row in enumerate(_load_json_lines(path, label="LFM dataset"), start=1):
        _reject_configured_secret_values(row, location=f"LFM dataset line {line_number}")
        prompt = row.get("prompt")
        completion = row.get("completion")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(completion, list)
            or not completion
        ):
            raise ValueError(
                f"LFM dataset line {line_number} must preserve conversational prompt/completion"
            )
        for label, messages, allowed_roles in (
            ("prompt", prompt, {"system", "user"}),
            ("completion", completion, {"assistant"}),
        ):
            for message in messages:
                if not isinstance(message, Mapping):
                    raise ValueError(
                        f"LFM dataset line {line_number} contains an invalid {label} message"
                    )
                role = message.get("role")
                content = message.get("content")
                if (
                    not isinstance(role, str)
                    or role not in allowed_roles
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    expected = "/".join(sorted(allowed_roles))
                    raise ValueError(
                        f"LFM dataset line {line_number} {label} messages require "
                        f"non-empty content and role {expected}"
                    )
        forbidden = _contains_pathish_lfm_field(row)
        if forbidden is not None:
            raise ValueError(
                f"LFM dataset line {line_number} contains a transferable path field: {forbidden}"
            )
        if _contains_sensitive_lfm_string(row):
            raise ValueError(f"LFM dataset line {line_number} contains a local or audio path")


def _reject_sensitive_mapping(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold().replace("-", "_")
            if (
                folded in _ROUTING_KEYS
                or folded in _SECRET_KEYS
                or "credential" in folded
                or "password" in folded
                or "secret" in folded
            ):
                raise ValueError(f"{location} contains routing or secret material: {key}")
            _reject_sensitive_mapping(child, location=location)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_mapping(child, location=location)


def _reject_absolute_metadata(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and (
                key.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", key)
            ):
                raise ValueError(f"{location} contains an absolute machine-local path")
            _reject_absolute_metadata(child, location=location)
    elif isinstance(value, list):
        for child in value:
            _reject_absolute_metadata(child, location=location)
    elif isinstance(value, str) and (
        value.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError(f"{location} contains an absolute machine-local path")


@dataclass(frozen=True)
class BundleFile:
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
class BundleInventory:
    plan_fingerprint: str
    files: tuple[BundleFile, ...]
    file_count: int
    total_bytes: int
    fingerprint: str
    schema_version: int = BUNDLE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        plan_fingerprint: str,
        files: Iterable[BundleFile],
    ) -> BundleInventory:
        ordered = tuple(sorted(files, key=lambda item: (item.path, item.role)))
        payload = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "plan_fingerprint": plan_fingerprint,
            "files": [item.as_dict() for item in ordered],
            "file_count": len(ordered),
            "total_bytes": sum(item.size for item in ordered),
        }
        return cls(
            plan_fingerprint=plan_fingerprint,
            files=ordered,
            file_count=payload["file_count"],
            total_bytes=payload["total_bytes"],
            fingerprint=_sha256_bytes(_canonical_json_bytes(payload)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_fingerprint": self.plan_fingerprint,
            "files": [item.as_dict() for item in self.files],
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class TrainingBundle:
    root: Path
    inventory: BundleInventory

    @property
    def completion_path(self) -> Path:
        return self.root.joinpath(*PurePosixPath(COMPLETION_PATH).parts)


def _copy_verified(
    *,
    source: Path,
    destination: Path,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[str, int]:
    before_size = source.stat().st_size
    before_sha = _sha256_file(source)
    if expected_size is not None and before_size != expected_size:
        raise RuntimeError(f"Training input size changed before bundling: {source}")
    if expected_sha256 is not None and before_sha != expected_sha256:
        raise RuntimeError(f"Training input checksum changed before bundling: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after_size = source.stat().st_size
    after_sha = _sha256_file(source)
    copied_sha = _sha256_file(destination)
    if (after_size, after_sha) != (before_size, before_sha) or copied_sha != before_sha:
        raise RuntimeError(f"Training input changed while bundling: {source}")
    if destination.stat().st_size != before_size:
        raise RuntimeError(f"Bundle copy size verification failed: {destination}")
    return before_sha, before_size


def _write_atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _copy_bundle_member(
    *,
    root: Path,
    temporary: Path,
    relative: PurePosixPath,
    role: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> BundleFile:
    _reject_forbidden_payload_path(relative, label=role)
    source = _reject_link_chain(root, relative)
    destination = temporary.joinpath(*relative.parts)
    sha256, size = _copy_verified(
        source=source,
        destination=destination,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    return BundleFile(path=relative.as_posix(), role=role, sha256=sha256, size=size)


def build_training_bundle(
    plan: TrainingPlan,
    persona_root: Path,
    destination: Path,
) -> TrainingBundle:
    """Materialize the minimal, deterministic payload authorized for remote training.

    The persona tree is read-only. The destination appears only after every source
    member has been copied and re-hashed; the completion inventory is written last.
    """

    persona_root = persona_root.absolute()
    if persona_root.is_symlink() or _is_junction(persona_root):
        raise ValueError("Persona root must not be a link or junction")
    root = persona_root.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"Training bundle destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Training bundle temporary path already exists: {temporary}")

    members: dict[str, BundleFile] = {}
    try:
        temporary.mkdir(parents=True)
        plan_bytes = canonical_plan_bytes(plan)
        plan_value = plan.as_dict()
        _reject_sensitive_mapping(plan_value, location="TrainingPlan")
        _reject_configured_secret_values(plan_value, location="TrainingPlan")
        _reject_absolute_metadata(plan_value, location="TrainingPlan")
        plan_destination = temporary.joinpath(*PurePosixPath(PLAN_PATH).parts)
        _write_atomic_bytes(plan_destination, plan_bytes)
        members[PLAN_PATH] = BundleFile(
            path=PLAN_PATH,
            role="training-plan",
            sha256=plan.fingerprint,
            size=len(plan_bytes),
        )

        for contract in sorted(plan.files, key=lambda item: (item.path, item.role)):
            relative = _portable_path(contract.path, label=f"TrainingPlan {contract.role}")
            if not contract.transfer:
                continue
            if contract.role not in _TRANSFER_ROLES:
                raise ValueError(f"TrainingPlan role is not approved for transfer: {contract.role}")
            if contract.role == "irodori-latent-manifest":
                source = _reject_link_chain(root, relative)
                source_relative = PurePosixPath(IRODORI_SOURCE_MANIFEST_PATH)
                source_destination = temporary.joinpath(*source_relative.parts)
                source_sha, source_size = _copy_verified(
                    source=source,
                    destination=source_destination,
                    expected_sha256=contract.sha256,
                    expected_size=contract.size,
                )
                members[source_relative.as_posix()] = BundleFile(
                    path=source_relative.as_posix(),
                    role="irodori-latent-manifest-source",
                    sha256=source_sha,
                    size=source_size,
                )
                _sanitized_irodori_payload(
                    root=root,
                    manifest_relative=relative,
                    manifest_source=source_destination,
                    temporary=temporary,
                    members=members,
                )
                continue

            if relative.as_posix() in members:
                raise ValueError(f"Duplicate bundle path: {relative}")
            member = _copy_bundle_member(
                root=root,
                temporary=temporary,
                relative=relative,
                role=contract.role,
                expected_sha256=contract.sha256,
                expected_size=contract.size,
            )
            members[member.path] = member
            if contract.role == "lfm-conversations":
                source = root.joinpath(*relative.parts)
                _validate_lfm_dataset(source)

        inventory = BundleInventory.create(
            plan_fingerprint=plan.fingerprint,
            files=members.values(),
        )
        completion = temporary.joinpath(*PurePosixPath(COMPLETION_PATH).parts)
        _write_atomic_bytes(completion, _canonical_json_bytes(inventory.as_dict()))
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return verify_training_bundle(destination, expected_plan_fingerprint=plan.fingerprint)


def _inventory_from_value(value: Any) -> BundleInventory:
    if not isinstance(value, Mapping):
        raise ValueError("Bundle completion manifest must be an object")
    allowed = {
        "schema_version",
        "plan_fingerprint",
        "files",
        "file_count",
        "total_bytes",
        "fingerprint",
    }
    if set(value) != allowed or value.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Bundle completion manifest schema is invalid")
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Bundle completion manifest files must be a list")
    files: list[BundleFile] = []
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "role", "sha256", "size"}:
            raise ValueError("Bundle completion manifest contains an invalid file record")
        _portable_path(item.get("path"), label="Bundle inventory path")
        role = item.get("role")
        digest = item.get("sha256")
        size = item.get("size")
        if not isinstance(role, str) or not role:
            raise ValueError("Bundle inventory role is invalid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Bundle inventory checksum is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Bundle inventory size is invalid")
        files.append(BundleFile(path=item["path"], role=role, sha256=digest, size=size))
    plan_fingerprint = value.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or len(plan_fingerprint) != 64:
        raise ValueError("Bundle plan fingerprint is invalid")
    rebuilt = BundleInventory.create(plan_fingerprint=plan_fingerprint, files=files)
    if (
        value.get("file_count") != rebuilt.file_count
        or value.get("total_bytes") != rebuilt.total_bytes
        or value.get("fingerprint") != rebuilt.fingerprint
        or tuple(files) != rebuilt.files
    ):
        raise ValueError("Bundle completion manifest fingerprint or totals are invalid")
    return rebuilt


def load_bundle_inventory(root: Path) -> BundleInventory:
    completion = root.joinpath(*PurePosixPath(COMPLETION_PATH).parts)
    if completion.is_symlink() or _is_junction(completion) or not completion.is_file():
        raise ValueError("Bundle completion manifest is missing or is not a regular file")
    try:
        value = json.loads(completion.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Bundle completion manifest is unreadable") from exc
    return _inventory_from_value(value)


def _iter_regular_bundle_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = PurePosixPath(path.relative_to(root)).as_posix()
        if path.is_symlink() or _is_junction(path):
            raise ValueError(f"Bundle contains a link or junction: {relative}")
        if path.is_file():
            files.add(relative)
        elif not path.is_dir():
            raise ValueError(f"Bundle contains a non-regular entry: {relative}")
    return files


def verify_training_bundle(
    root: Path,
    *,
    expected_plan_fingerprint: str | None = None,
) -> TrainingBundle:
    root = root.absolute()
    if root.is_symlink() or _is_junction(root) or not root.is_dir():
        raise ValueError("Training bundle root must be a real directory")
    root = root.resolve(strict=True)
    inventory = load_bundle_inventory(root)
    if expected_plan_fingerprint is not None and (
        inventory.plan_fingerprint != expected_plan_fingerprint
    ):
        raise ValueError("Training bundle does not match the expected TrainingPlan")

    expected_paths = {item.path for item in inventory.files}
    if len(expected_paths) != len(inventory.files):
        raise ValueError("Training bundle inventory contains duplicate paths")
    actual_paths = _iter_regular_bundle_files(root)
    if actual_paths != expected_paths | {COMPLETION_PATH}:
        raise ValueError("Training bundle contains missing or unlisted files")

    allowed_roles = _TRANSFER_ROLES | {
        "irodori-latent",
        "irodori-latent-manifest-source",
        "training-plan",
    }
    by_path = {item.path: item for item in inventory.files}
    for item in inventory.files:
        relative = _portable_path(item.path, label="Bundle inventory path")
        _reject_forbidden_payload_path(relative, label=item.role)
        if item.role not in allowed_roles:
            raise ValueError(f"Training bundle contains an unapproved role: {item.role}")
        path = root.joinpath(*relative.parts)
        if path.stat().st_size != item.size or _sha256_file(path) != item.sha256:
            raise ValueError(f"Training bundle checksum contract failed: {item.path}")

    plan_item = by_path.get(PLAN_PATH)
    if plan_item is None or plan_item.role != "training-plan":
        raise ValueError("Training bundle does not contain its TrainingPlan contract")
    plan_bytes = root.joinpath(*PurePosixPath(PLAN_PATH).parts).read_bytes()
    if _sha256_bytes(plan_bytes) != inventory.plan_fingerprint:
        raise ValueError("Bundled TrainingPlan bytes do not match its fingerprint")
    try:
        plan_value = json.loads(plan_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Bundled TrainingPlan is not canonical JSON") from exc
    if _canonical_json_bytes(plan_value) != plan_bytes:
        raise ValueError("Bundled TrainingPlan is not canonically serialized")
    _reject_sensitive_mapping(plan_value, location="Bundled TrainingPlan")
    _reject_configured_secret_values(plan_value, location="Bundled TrainingPlan")
    _reject_absolute_metadata(plan_value, location="Bundled TrainingPlan")

    raw_contracts = plan_value.get("files") if isinstance(plan_value, Mapping) else None
    if not isinstance(raw_contracts, list):
        raise ValueError("Bundled TrainingPlan file contracts are invalid")
    for raw in raw_contracts:
        if not isinstance(raw, Mapping):
            raise ValueError("Bundled TrainingPlan file contract is invalid")
        relative = _portable_path(raw.get("path"), label="Bundled TrainingPlan path")
        role = raw.get("role")
        transfer = raw.get("transfer")
        if not isinstance(transfer, bool):
            raise ValueError("Bundled TrainingPlan transfer flag is invalid")
        member = by_path.get(relative.as_posix())
        if transfer:
            if role == "irodori-latent-manifest":
                source_member = by_path.get(IRODORI_SOURCE_MANIFEST_PATH)
                sanitized_member = by_path.get(IRODORI_SANITIZED_MANIFEST_PATH)
                if (
                    source_member is None
                    or source_member.role != "irodori-latent-manifest-source"
                    or source_member.sha256 != raw.get("sha256")
                    or source_member.size != raw.get("size")
                    or sanitized_member is None
                    or sanitized_member.role != "irodori-latent-manifest"
                ):
                    raise ValueError("Bundled Irodori manifest contract is incomplete")
            else:
                if role not in _TRANSFER_ROLES or member is None or member.role != role:
                    raise ValueError("Bundled TrainingPlan transfer contract is incomplete")
                if member.sha256 != raw.get("sha256") or member.size != raw.get("size"):
                    raise ValueError("Bundled TrainingPlan transfer checksum contract is invalid")
        elif member is not None:
            raise ValueError("Non-transferable TrainingPlan input was included in the bundle")

    irodori_contracts = [
        raw
        for raw in raw_contracts
        if isinstance(raw, Mapping) and raw.get("role") == "irodori-latent-manifest"
    ]
    referenced_latents: set[str] = set()
    if irodori_contracts:
        if len(irodori_contracts) != 1:
            raise ValueError("Bundle contains multiple Irodori manifest contracts")
        source_manifest = root.joinpath(*PurePosixPath(IRODORI_SOURCE_MANIFEST_PATH).parts)
        sanitized_manifest = root.joinpath(*PurePosixPath(IRODORI_SANITIZED_MANIFEST_PATH).parts)
        source_rows = _validate_irodori_manifest(source_manifest)
        sanitized_rows = _validate_irodori_manifest(sanitized_manifest)
        if len(source_rows) != len(sanitized_rows):
            raise ValueError("Sanitized Irodori manifest record count changed")
        for source_row, sanitized_row in zip(source_rows, sanitized_rows, strict=True):
            source_metadata = dict(source_row)
            source_metadata.pop("latent_path")
            sanitized_metadata = dict(sanitized_row)
            sanitized_path = str(sanitized_metadata.pop("latent_path"))
            if source_metadata != sanitized_metadata:
                raise ValueError("Sanitized Irodori manifest changed text or caption metadata")
            relative = _portable_path(sanitized_path, label="Sanitized Irodori latent path")
            if relative.parent.as_posix() != "latents" or relative.suffix != ".pt":
                raise ValueError("Sanitized Irodori manifest latent path is outside its payload")
            digest = relative.stem
            if len(digest) != 64:
                raise ValueError("Sanitized Irodori latent filename is not content-addressed")
            bundle_path = PurePosixPath(IRODORI_SANITIZED_MANIFEST_PATH).parent / relative
            latent_member = by_path.get(bundle_path.as_posix())
            if latent_member is None or latent_member.sha256 != digest:
                raise ValueError("Sanitized Irodori manifest latent checksum is invalid")
            referenced_latents.add(bundle_path.as_posix())
    elif any(
        item.role.startswith("irodori-latent")
        or item.path in {IRODORI_SOURCE_MANIFEST_PATH, IRODORI_SANITIZED_MANIFEST_PATH}
        for item in inventory.files
    ):
        raise ValueError("Bundle contains Irodori payload without a TrainingPlan contract")
    actual_latents = {item.path for item in inventory.files if item.role == "irodori-latent"}
    if referenced_latents != actual_latents:
        raise ValueError("Bundle Irodori latent inventory does not match its manifest")

    for raw in raw_contracts:
        if isinstance(raw, Mapping) and raw.get("role") == "lfm-conversations":
            relative = _portable_path(raw["path"], label="LFM dataset path")
            _validate_lfm_dataset(root.joinpath(*relative.parts))
    return TrainingBundle(root=root, inventory=inventory)
