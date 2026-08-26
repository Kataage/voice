from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from uuid import uuid4

from personavoice.atomic import atomic_write_json, atomic_write_text
from personavoice.irodori import prepare_manifest
from personavoice.lineage import load_lineage
from personavoice.model_assets import (
    IRODORI_DACVAE_REVISION,
    IRODORI_DACVAE_SHA256,
    IRODORI_SOURCE_REVISION,
)
from personavoice.project import PersonaPaths
from personavoice.media import sha256_file

IRODORI_INPUT_CONTRACT_SCHEMA = 1
IRODORI_LATENT_POLICY_VERSION = 1
_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolved_latent(manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or _DRIVE_RE.match(raw_path):
        raise ValueError("Irodori latent paths must be relative to the manifest")
    return (manifest.parent / path).resolve()


def read_valid_manifest(
    manifest: Path,
    *,
    allowed_root: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Validate a latent manifest without deserializing tensor payloads."""

    if not _nonempty_file(manifest):
        return None
    rows: list[dict[str, Any]] = []
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None
                text = value.get("text")
                latent_path = value.get("latent_path")
                frames = value.get("num_frames")
                if not isinstance(text, str) or not text.strip():
                    return None
                if not isinstance(latent_path, str) or not latent_path:
                    return None
                if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
                    return None
                try:
                    latent = _resolved_latent(manifest, latent_path)
                    if allowed_root is not None:
                        latent.relative_to(allowed_root.resolve())
                except ValueError:
                    return None
                if not _nonempty_file(latent):
                    return None
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return rows or None


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in normalized if char.isalnum())


def _source_rows(source: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None
                text = value.get("text")
                if not isinstance(text, str) or not text.strip():
                    return None
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return rows or None


def legacy_manifest_compatible(source: Path, manifest: Path) -> bool:
    """Conservatively adopt a v0.3 manifest when its lineage is unambiguous."""

    source_rows = _source_rows(source)
    manifest_rows = read_valid_manifest(manifest, allowed_root=source.parent.parent)
    if source_rows is None or manifest_rows is None or len(source_rows) != len(manifest_rows):
        return False
    try:
        if manifest.stat().st_mtime_ns < source.stat().st_mtime_ns:
            return False
    except OSError:
        return False
    for source_row, manifest_row in zip(source_rows, manifest_rows, strict=True):
        source_text = _normalized_text(str(source_row["text"]))
        manifest_text = _normalized_text(str(manifest_row["text"]))
        if not source_text or source_text != manifest_text:
            return False
        source_caption = source_row.get("caption")
        manifest_caption = manifest_row.get("caption")
        if source_caption is not None and source_caption != manifest_caption:
            return False
    return True


def irodori_input_contract(
    repo_root: Path,
    source: Path,
    *,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del repo_root  # Reserved for a future pinned local codec/preparation asset contract.
    value: dict[str, Any] = {
        "schema_version": IRODORI_INPUT_CONTRACT_SCHEMA,
        "latent_policy_version": IRODORI_LATENT_POLICY_VERSION,
        "source_sha256": sha256_file(source),
        "source_revision": IRODORI_SOURCE_REVISION,
        "dacvae_revision": IRODORI_DACVAE_REVISION,
        "dacvae_sha256": IRODORI_DACVAE_SHA256,
    }
    rows = _source_rows(source) or []
    value["source_evidence"] = [
        {
            "utterance_id": row.get("utterance_id"),
            "text_hash": row.get("text_hash") or hashlib.sha256(
                str(row.get("text") or "").encode("utf-8")
            ).hexdigest(),
            "asr_backend": (row.get("provenance") or {}).get("asr_backend")
            if isinstance(row.get("provenance"), dict)
            else None,
            "asr_model_revision": (row.get("provenance") or {}).get("asr_model_revision")
            if isinstance(row.get("provenance"), dict)
            else None,
            "alignment_backend": (row.get("provenance") or {}).get("alignment_backend")
            if isinstance(row.get("provenance"), dict)
            else None,
            "alignment_model_revision": (
                (row.get("provenance") or {}).get("alignment_model_revision")
                if isinstance(row.get("provenance"), dict)
                else None
            ),
            "boundary_evidence": row.get("boundary_evidence"),
        }
        for row in rows
    ]
    if lineage is not None:
        value["prepare_lineage"] = {
            "lineage_id": lineage.get("lineage_id"),
            "lineage_fingerprint": lineage.get("lineage_fingerprint"),
            "master_fingerprint": lineage.get("master_fingerprint"),
            "asr": lineage.get("asr"),
            "alignment": lineage.get("alignment"),
            "separation": lineage.get("separation"),
        }
    return value


def _contract_key(contract: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


def _read_contract(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_lineage(path: Path, contract: dict[str, Any], manifest: Path) -> None:
    atomic_write_json(
        path,
        {
            **contract,
            "manifest_sha256": sha256_file(manifest),
        },
    )


def _lineage_matches(path: Path, contract: dict[str, Any], manifest: Path) -> bool:
    recorded = _read_contract(path)
    if recorded is None or any(recorded.get(key) != value for key, value in contract.items()):
        return False
    expected_manifest = recorded.get("manifest_sha256")
    return (
        isinstance(expected_manifest, str)
        and len(expected_manifest) == 64
        and _nonempty_file(manifest)
        and sha256_file(manifest) == expected_manifest
    )


def _conditioned_view(
    manifest: Path,
    *,
    conditioning: str,
    persona_root: Path,
) -> Path:
    if conditioning == "speaker":
        return manifest
    if conditioning != "none":
        raise ValueError(f"Unsupported Irodori conditioning mode: {conditioning!r}")
    digest = sha256_file(manifest)
    destination = manifest.with_name(f"{manifest.stem}.no-speaker-{digest[:16]}.jsonl")
    rows = read_valid_manifest(destination, allowed_root=persona_root)
    if rows is not None:
        return destination

    source_rows = read_valid_manifest(manifest, allowed_root=persona_root)
    if source_rows is None:
        raise RuntimeError(f"Irodori latent manifest is invalid: {manifest}")
    output: list[str] = []
    for row in source_rows:
        value = dict(row)
        value.pop("speaker_id", None)
        latent = _resolved_latent(manifest, str(value["latent_path"]))
        value["latent_path"] = os.path.relpath(latent, start=destination.parent)
        output.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    atomic_write_text(destination, "\n".join(output) + "\n")
    if read_valid_manifest(destination, allowed_root=persona_root) is None:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Failed to construct a valid no-speaker Irodori manifest view")
    return destination


def ensure_irodori_manifest(
    repo_root: Path,
    paths: PersonaPaths,
    *,
    conditioning: str,
) -> Path:
    """Reuse or create content-addressed latents without training-driven deletion."""

    source = paths.dataset / "irodori_source.jsonl"
    if not _nonempty_file(source):
        raise FileNotFoundError("Prepared Irodori source dataset is missing or empty")
    contract = irodori_input_contract(repo_root, source, lineage=load_lineage(paths))
    key = _contract_key(contract)

    legacy = paths.dataset / "irodori_manifest.jsonl"
    legacy_lineage = paths.dataset / "irodori_manifest.contract.json"
    if read_valid_manifest(legacy, allowed_root=paths.root) is not None:
        if _lineage_matches(legacy_lineage, contract, legacy):
            return _conditioned_view(
                legacy,
                conditioning=conditioning,
                persona_root=paths.root,
            )
        if not legacy_lineage.exists() and legacy_manifest_compatible(source, legacy):
            _write_lineage(legacy_lineage, contract, legacy)
            return _conditioned_view(
                legacy,
                conditioning=conditioning,
                persona_root=paths.root,
            )

    cache_root = paths.cache / "irodori_prepared" / key
    manifest = cache_root / "manifest.jsonl"
    lineage = cache_root / "contract.json"
    if read_valid_manifest(manifest, allowed_root=paths.root) is not None and _lineage_matches(
        lineage, contract, manifest
    ):
        return _conditioned_view(
            manifest,
            conditioning=conditioning,
            persona_root=paths.root,
        )

    staging = cache_root.with_name(f".{key}.{uuid4().hex}.staging")
    staging.mkdir(parents=True, exist_ok=False)
    staged_manifest = staging / "manifest.jsonl"
    try:
        prepare_manifest(repo_root, source, staged_manifest, staging / "latents")
        if read_valid_manifest(staged_manifest, allowed_root=paths.root) is None:
            raise RuntimeError("Irodori preparation returned an invalid latent manifest")
        _write_lineage(staging / "contract.json", contract, staged_manifest)
        if cache_root.exists():
            invalid = cache_root.with_name(f".{key}.{uuid4().hex}.invalid")
            os.replace(cache_root, invalid)
        cache_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, cache_root)
    except Exception:
        # The staging directory is deliberately retained for forensic inspection;
        # it is never treated as a resumable or complete cache entry.
        raise
    return _conditioned_view(
        manifest,
        conditioning=conditioning,
        persona_root=paths.root,
    )
