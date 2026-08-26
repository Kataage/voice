"""Versioned Prepare/model-generation contracts.

This module intentionally contains no model loading.  It is the small,
deterministic boundary shared by setup, Prepare, training, publication and
runtime selection.  A lineage is a candidate until the activation pointer is
atomically replaced by the explicit ``persona activate`` command.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from personavoice.atomic import atomic_write_json
from personavoice.media import inventory_fingerprint
from personavoice.model_assets import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    QWEN_ASR_MODEL_ID,
    QWEN_ASR_MODEL_LICENSE,
    QWEN_ASR_MODEL_REVISION,
    QWEN_DOMAIN_ASR_CTC_HEAD_FILE,
    QWEN_DOMAIN_ASR_CTC_HEAD_LFS_OID,
    QWEN_DOMAIN_ASR_DATASET_ID,
    QWEN_DOMAIN_ASR_DATASET_LICENSE,
    QWEN_DOMAIN_ASR_DATASET_REVISION,
    QWEN_DOMAIN_ASR_EFFECTIVE_RESTRICTIONS,
    QWEN_DOMAIN_ASR_MODEL_ID,
    QWEN_DOMAIN_ASR_MODEL_PAGE_LICENSE,
    QWEN_DOMAIN_ASR_MODEL_REVISION,
    QWEN_FORCED_ALIGNER_MODEL_ID,
    QWEN_FORCED_ALIGNER_MODEL_LICENSE,
    QWEN_FORCED_ALIGNER_MODEL_REVISION,
    SEPARATOR_MODEL_FILENAME,
    SEPARATOR_MODEL_LICENSE,
    SEPARATOR_MODEL_LINEAGE,
    SEPARATOR_PACKAGE,
    SEPARATOR_SOURCE_LICENSE,
    SEPARATOR_SOURCE_REPOSITORY,
    SEPARATOR_SOURCE_REVISION,
    SEPARATOR_VERSION,
)
from personavoice.stage_lock import stage_lock

if TYPE_CHECKING:
    from personavoice.config import PersonaConfig
    from personavoice.project import PersonaPaths


LINEAGE_SCHEMA_VERSION = 1
PREPARE_LINEAGE_CONTRACT_VERSION = "prepare-lineage-v1"
ASR_CONTRACT_VERSION = "asr-normalized-v1"
ALIGNMENT_CONTRACT_VERSION = "alignment-v1"
SEPARATION_CONTRACT_VERSION = "separation-v1"
ACTIVE_GENERATION_SCHEMA_VERSION = 1
_LINEAGE_ID_RE = re.compile(r"^pl-[0-9a-f]{32}$")
_GENERATION_ID_RE = re.compile(r"^gen-[0-9a-f]{32}$")


class DomainBackendDisabledError(RuntimeError):
    """Raised when the restricted domain checkpoint is requested without proof."""


@dataclass(frozen=True)
class BackendSpec:
    key: str
    kind: str
    model_id: str
    revision: str
    license: str
    enabled: bool
    restrictions: tuple[str, ...] = ()
    runtime_model_id: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "enabled": self.enabled,
            "restrictions": list(self.restrictions),
        }
        if self.runtime_model_id is not None:
            value["runtime_model_id"] = self.runtime_model_id
        if self.reason is not None:
            value["reason"] = self.reason
        return value


LEGACY_WHISPER = BackendSpec(
    key="whisper-large-v3",
    kind="legacy-reference",
    model_id="openai/whisper-large-v3",
    revision=ASR_MODEL_REVISION,
    license="MIT",
    enabled=True,
    runtime_model_id=ASR_MODEL_ID,
)
GENERAL_QWEN = BackendSpec(
    key="qwen3-asr-1.7b",
    kind="general-modern",
    model_id=QWEN_ASR_MODEL_ID,
    revision=QWEN_ASR_MODEL_REVISION,
    license=QWEN_ASR_MODEL_LICENSE,
    enabled=True,
)
DOMAIN_QWEN = BackendSpec(
    key="qwen3-asr-1.7b-ja-anime-galgame-hf",
    kind="anime-domain",
    model_id=QWEN_DOMAIN_ASR_MODEL_ID,
    revision=QWEN_DOMAIN_ASR_MODEL_REVISION,
    license=QWEN_DOMAIN_ASR_MODEL_PAGE_LICENSE,
    enabled=False,
    restrictions=QWEN_DOMAIN_ASR_EFFECTIVE_RESTRICTIONS,
    reason=(
        "disabled: the converted checkpoint page displays Apache-2.0, but its "
        "fine-tune provenance points to GPL-3.0 Galgame_Speech_ASR_16kHz with "
        "commercial-use prohibition and an open-source-model requirement. "
        "The repository has no rights-holder evidence that makes the effective "
        "terms safe to enable or redistribute."
    ),
)

_BACKENDS = {
    LEGACY_WHISPER.key: LEGACY_WHISPER,
    GENERAL_QWEN.key: GENERAL_QWEN,
    DOMAIN_QWEN.key: DOMAIN_QWEN,
    "large-v3": LEGACY_WHISPER,
    "whisper": LEGACY_WHISPER,
    "qwen": GENERAL_QWEN,
    "domain-qwen": DOMAIN_QWEN,
}


def backend_registry() -> tuple[dict[str, Any], ...]:
    """Return auditable backend metadata, including disabled options."""

    return tuple(spec.as_dict() for spec in (LEGACY_WHISPER, GENERAL_QWEN, DOMAIN_QWEN))


def domain_backend_audit() -> dict[str, Any]:
    """Return the fail-closed legal/provenance decision for domain Qwen."""

    return {
        "enabled": False,
        "model_id": QWEN_DOMAIN_ASR_MODEL_ID,
        "model_revision": QWEN_DOMAIN_ASR_MODEL_REVISION,
        "page_license": QWEN_DOMAIN_ASR_MODEL_PAGE_LICENSE,
        "base_model": {
            "model_id": QWEN_ASR_MODEL_ID,
            "revision": QWEN_ASR_MODEL_REVISION,
            "license": QWEN_ASR_MODEL_LICENSE,
        },
        "training_dataset": {
            "dataset_id": QWEN_DOMAIN_ASR_DATASET_ID,
            "revision": QWEN_DOMAIN_ASR_DATASET_REVISION,
            "license": QWEN_DOMAIN_ASR_DATASET_LICENSE,
            "restrictions": list(QWEN_DOMAIN_ASR_EFFECTIVE_RESTRICTIONS),
        },
        "alignment_head": {
            "file": QWEN_DOMAIN_ASR_CTC_HEAD_FILE,
            "integrity_id": QWEN_DOMAIN_ASR_CTC_HEAD_LFS_OID,
            "coupled_encoder_model_id": QWEN_DOMAIN_ASR_MODEL_ID,
            "coupled_encoder_revision": QWEN_DOMAIN_ASR_MODEL_REVISION,
            "terms": "inherits the domain checkpoint provenance; not separately cleared",
        },
        "blocker": DOMAIN_QWEN.reason,
    }


def resolve_backend(value: str) -> BackendSpec:
    key = str(value).strip().lower()
    try:
        spec = _BACKENDS[key]
    except KeyError as exc:
        choices = ", ".join((LEGACY_WHISPER.key, GENERAL_QWEN.key, DOMAIN_QWEN.key))
        raise ValueError(f"Unsupported ASR backend {value!r}; choose one of {choices}") from exc
    if not spec.enabled:
        raise DomainBackendDisabledError(spec.reason or f"ASR backend {spec.key!r} is disabled")
    return spec


def backend_status(value: str) -> dict[str, Any]:
    key = str(value).strip().lower()
    spec = _BACKENDS.get(key)
    if spec is None:
        return {"key": key, "enabled": False, "reason": "unknown ASR backend"}
    return spec.as_dict()


@dataclass(frozen=True)
class AlignmentSpec:
    key: str
    model_id: str | None
    revision: str
    license: str | None
    coupled_encoder_model_id: str | None = None
    coupled_encoder_revision: str | None = None
    enabled: bool = True
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "contract_version": ALIGNMENT_CONTRACT_VERSION,
            "key": self.key,
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "enabled": self.enabled,
        }
        if self.coupled_encoder_model_id is not None:
            value["coupled_encoder_model_id"] = self.coupled_encoder_model_id
        if self.coupled_encoder_revision is not None:
            value["coupled_encoder_revision"] = self.coupled_encoder_revision
        if self.reason is not None:
            value["reason"] = self.reason
        return value


WHISPER_ALIGNMENT = AlignmentSpec(
    key="whisper-native-words",
    model_id=None,
    revision="whisper-word-timestamps-v1",
    license=None,
)
QWEN_FORCED_ALIGNMENT = AlignmentSpec(
    key="qwen3-forced-aligner-0.6b",
    model_id=QWEN_FORCED_ALIGNER_MODEL_ID,
    revision=QWEN_FORCED_ALIGNER_MODEL_REVISION,
    license=QWEN_FORCED_ALIGNER_MODEL_LICENSE,
)
DOMAIN_CTC_ALIGNMENT = AlignmentSpec(
    key="domain-ctc-aligner",
    model_id=QWEN_DOMAIN_ASR_MODEL_ID,
    revision=QWEN_DOMAIN_ASR_MODEL_REVISION,
    license=QWEN_DOMAIN_ASR_MODEL_PAGE_LICENSE,
    coupled_encoder_model_id=QWEN_DOMAIN_ASR_MODEL_ID,
    coupled_encoder_revision=QWEN_DOMAIN_ASR_MODEL_REVISION,
    enabled=False,
    reason=DOMAIN_QWEN.reason,
)


def resolve_alignment(asr_backend: str, requested: str = "auto") -> AlignmentSpec:
    backend = resolve_backend(asr_backend)
    selection = str(requested).strip().lower()
    if selection in {"", "auto"}:
        return WHISPER_ALIGNMENT if backend.key == LEGACY_WHISPER.key else QWEN_FORCED_ALIGNMENT
    aliases = {
        "native": WHISPER_ALIGNMENT,
        "whisper-native": WHISPER_ALIGNMENT,
        "whisper-native-words": WHISPER_ALIGNMENT,
        "qwen3-forced-aligner-0.6b": QWEN_FORCED_ALIGNMENT,
        "domain-ctc": DOMAIN_CTC_ALIGNMENT,
        "domain-ctc-aligner": DOMAIN_CTC_ALIGNMENT,
    }
    try:
        spec = aliases[selection]
    except KeyError as exc:
        raise ValueError(f"Unsupported alignment backend {requested!r}") from exc
    if spec is DOMAIN_CTC_ALIGNMENT:
        # This check is deliberately strict.  A CTC head is not a generic
        # phoneme/timestamp model and may only see its exact fine-tuned encoder.
        raise DomainBackendDisabledError(spec.reason or "Domain CTC alignment is disabled")
    if spec is WHISPER_ALIGNMENT and backend.key != LEGACY_WHISPER.key:
        raise ValueError("Whisper word alignment cannot be attached to a Qwen transcript backend")
    return spec


def separator_contract() -> dict[str, Any]:
    return {
        "contract_version": SEPARATION_CONTRACT_VERSION,
        "backend": SEPARATOR_PACKAGE,
        "version": SEPARATOR_VERSION,
        "source_repository": SEPARATOR_SOURCE_REPOSITORY,
        "source_revision": SEPARATOR_SOURCE_REVISION,
        "source_license": SEPARATOR_SOURCE_LICENSE,
        "model_filename": SEPARATOR_MODEL_FILENAME,
        "model_lineage": SEPARATOR_MODEL_LINEAGE,
        "model_license": SEPARATOR_MODEL_LICENSE,
        "analysis_only": True,
        "offline_requires_local_manifest": True,
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prepare_lineage_seed(
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    prepare_schema: int,
    cache_policy_version: str,
) -> dict[str, Any]:
    """Compute the upstream identity before any candidate files are written."""

    backend = resolve_backend(cfg.prepare.asr_model)
    alignment = resolve_alignment(backend.key, cfg.prepare.alignment_backend)
    # The separator weight is not downloaded implicitly.  Include the audited
    # local digest when present so replacing a model can never reuse a stem
    # under the old lineage identity.  Keep the record portable: source paths
    # and free-form terms stay in the local model manifest, not in TrainingPlan.
    from personavoice.separation import separator_model_audit

    separator_audit = separator_model_audit(paths.root.parents[1])
    return {
        "contract_version": PREPARE_LINEAGE_CONTRACT_VERSION,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "raw_inventory_fingerprint": inventory_fingerprint(paths.raw),
        "identity_inventory_fingerprint": inventory_fingerprint(paths.identity),
        "asr": backend.as_dict(),
        "alignment": alignment.as_dict(),
        "separation": {
            **separator_contract(),
            "policy": cfg.prepare.separation_policy,
            "model_audit": {
                "materialized": bool(separator_audit.get("materialized")),
                "sha256": separator_audit.get("sha256"),
                "size": separator_audit.get("size"),
                "reason": separator_audit.get("reason"),
            },
        },
        "prepare_schema": prepare_schema,
        "prepare_cache_policy": cache_policy_version,
    }


def lineage_identity(seed: dict[str, Any]) -> tuple[str, str]:
    encoded = _canonical(seed).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    return f"pl-{fingerprint[:32]}", fingerprint


def build_lineage_record(
    seed: dict[str, Any],
    *,
    lineage_id: str,
    lineage_fingerprint: str,
    master_fingerprint: str,
    source_count: int,
    utterance_count: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not _LINEAGE_ID_RE.fullmatch(lineage_id):
        raise ValueError(f"Invalid Prepare lineage id: {lineage_id!r}")
    return {
        **seed,
        "lineage_id": lineage_id,
        "lineage_fingerprint": lineage_fingerprint,
        "master_fingerprint": master_fingerprint,
        "sources": source_count,
        "utterances": utterance_count,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "active": False,
    }


def load_lineage(paths: PersonaPaths, lineage_id: str | None = None) -> dict[str, Any] | None:
    selected = lineage_id or paths.lineage_id
    if not selected or not _LINEAGE_ID_RE.fullmatch(selected):
        return None
    path = paths.for_lineage(selected).lineage_record
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def prepared_lineage_id(paths: PersonaPaths) -> str | None:
    """Return the latest lineage selected by the durable Prepare state.

    This is deliberately separate from :func:`active_lineage_id`: validation
    and publication may inspect a newly prepared candidate before the explicit
    activation pointer changes, while runtime inference must use only the
    active pointer.
    """

    if paths.lineage_id is not None:
        return paths.lineage_id
    try:
        state = json.loads(paths.state.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    stages = state.get("stages")
    prepare = stages.get("prepare") if isinstance(stages, dict) else None
    result = prepare.get("result") if isinstance(prepare, dict) else None
    selected = result.get("lineage_id") if isinstance(result, dict) else None
    return selected if isinstance(selected, str) and _LINEAGE_ID_RE.fullmatch(selected) else None


def prepared_paths(paths: PersonaPaths) -> PersonaPaths:
    """Resolve the current Prepare candidate without changing runtime activation."""

    selected = prepared_lineage_id(paths)
    return paths.for_lineage(selected) if selected is not None else paths


def active_generation_path(paths: PersonaPaths) -> Path:
    return paths.root / "generations" / "active.json"


def _active_pointer(paths: PersonaPaths) -> dict[str, Any] | None:
    path = active_generation_path(paths)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def active_lineage_id(paths: PersonaPaths) -> str | None:
    value = _active_pointer(paths)
    selected = value.get("active_lineage_id") if value is not None else None
    return selected if isinstance(selected, str) and _LINEAGE_ID_RE.fullmatch(selected) else None


def active_generation_id(paths: PersonaPaths) -> str | None:
    value = _active_pointer(paths)
    selected = value.get("active_generation_id") if value is not None else None
    return selected if isinstance(selected, str) and _GENERATION_ID_RE.fullmatch(selected) else None


def effective_paths(paths: PersonaPaths) -> PersonaPaths:
    """Resolve the explicit active model generation for runtime consumers."""

    if paths.lineage_id is not None:
        return paths
    lineage = active_lineage_id(paths)
    if lineage is None:
        return paths
    generation = active_generation_id(paths)
    return (
        paths.for_generation(lineage, generation)
        if generation is not None
        else paths.for_lineage(lineage)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_candidate_file(candidate: PersonaPaths, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("Generation artifact paths must be relative to the candidate generation")
    target = (candidate.generation_root / relative).resolve()
    try:
        target.relative_to(candidate.generation_root.resolve())
    except ValueError as exc:
        raise RuntimeError("Generation artifact escapes its candidate generation") from exc
    return target


def _verify_generation_manifest(
    candidate: PersonaPaths,
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_generation_fingerprint: str | None,
) -> str:
    if manifest.get("schema_version") != ACTIVE_GENERATION_SCHEMA_VERSION:
        raise RuntimeError("Candidate generation manifest has an unsupported schema")
    if manifest.get("kind") != "personavoice-v03-generation":
        raise RuntimeError("Candidate generation is not a v0.3 generation manifest")
    if candidate.generation_id is not None and manifest.get("generation_id") != candidate.generation_id:
        raise RuntimeError("Candidate generation id does not match its path")
    if candidate.generation_id is None and manifest.get("generation_id") is not None:
        raise RuntimeError("Lineage-root generation must not declare a separate generation id")
    for key in ("lineage_id", "lineage_fingerprint", "master_fingerprint"):
        if manifest.get(key) != record.get(key):
            raise RuntimeError(f"Candidate generation {key} does not match its Prepare lineage")
    if expected_generation_fingerprint is not None and manifest.get(
        "generation_fingerprint"
    ) != expected_generation_fingerprint:
        raise RuntimeError("Candidate generation fingerprint does not match the requested value")
    generation_fingerprint = manifest.get("generation_fingerprint")
    if not isinstance(generation_fingerprint, str) or re.fullmatch(
        r"[0-9a-f]{64}", generation_fingerprint
    ) is None:
        raise RuntimeError("Candidate generation has no valid fingerprint")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise RuntimeError("Candidate generation has not passed validation")
    families = manifest.get("families")
    if not isinstance(families, dict):
        raise RuntimeError("Candidate generation has no family manifest")
    required_families = {"irodori", "lfm", "seed_vc"}
    missing = sorted(required_families - set(families))
    if missing:
        raise RuntimeError("Candidate generation is missing family entries: " + ", ".join(missing))
    for family in sorted(required_families):
        value = families.get(family)
        if not isinstance(value, dict):
            raise RuntimeError(f"Candidate family {family} is invalid")
        status = value.get("status")
        if status not in {"complete", "not_requested"}:
            raise RuntimeError(f"Candidate family {family} is not complete")
        if status == "not_requested":
            continue
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError(f"Candidate family {family} has no artifacts")
        for item in artifacts:
            if not isinstance(item, dict):
                raise RuntimeError(f"Candidate family {family} contains an invalid artifact")
            path = _safe_candidate_file(candidate, item.get("path"))
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"Candidate artifact is missing or empty: {path}")
            recorded_size = item.get("size")
            recorded_sha = item.get("sha256")
            if type(recorded_size) is not int or recorded_size != path.stat().st_size:
                raise RuntimeError(f"Candidate artifact size mismatch: {path}")
            if not isinstance(recorded_sha, str) or recorded_sha != _sha256_file(path):
                raise RuntimeError(f"Candidate artifact digest mismatch: {path}")
    return generation_fingerprint


def activate_generation(
    paths: PersonaPaths,
    lineage_id: str,
    *,
    generation_id: str | None = None,
    generation_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Atomically activate a validated v0.3 candidate generation."""

    if not _LINEAGE_ID_RE.fullmatch(lineage_id):
        raise ValueError(f"Invalid Prepare lineage id: {lineage_id!r}")
    candidate = (
        paths.for_generation(lineage_id, generation_id)
        if generation_id is not None
        else paths.for_lineage(lineage_id)
    )
    record = load_lineage(paths, lineage_id)
    if record is None:
        raise RuntimeError(f"Prepare lineage record is missing or unreadable: {lineage_id}")
    manifest_path = candidate.generation_manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Candidate generation has no readable v0.3 generation manifest") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Candidate generation manifest is not an object")
    verified_fingerprint = _verify_generation_manifest(
        candidate,
        record,
        manifest,
        expected_generation_fingerprint=generation_fingerprint,
    )
    pointer = active_generation_path(paths)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with stage_lock(paths.root, "activate"):
        previous: dict[str, Any] | None = None
        try:
            raw_previous = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raw_previous = None
        if isinstance(raw_previous, dict):
            previous = raw_previous
        if previous is not None:
            history = paths.root / "generations" / "activation-history"
            history.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            previous_digest = hashlib.sha256(
                _canonical(previous).encode("utf-8")
            ).hexdigest()[:8]
            atomic_write_json(
                history / f"{stamp}-{lineage_id}-{previous_digest}.json",
                previous,
            )
        pointer_value = {
            "schema_version": ACTIVE_GENERATION_SCHEMA_VERSION,
            "kind": "personavoice-v03-active-generation",
            "persona": paths.root.name,
            "active_lineage_id": lineage_id,
            "active_generation_id": candidate.generation_id,
            "active_generation_fingerprint": verified_fingerprint,
            "activated_at": datetime.now(UTC).isoformat(),
            "previous_lineage_id": (
                previous.get("active_lineage_id") if previous is not None else None
            ),
            "previous_generation_id": (
                previous.get("active_generation_id") if previous is not None else None
            ),
        }
        # This is the single mutable pointer. All candidate artifacts were
        # verified before the replace, so an interrupted write leaves the old
        # active pointer intact.
        atomic_write_json(pointer, pointer_value)
    return pointer_value


def master_fingerprint(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical(rows).encode("utf-8")).hexdigest()
