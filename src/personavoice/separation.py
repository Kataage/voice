"""BGM-aware, analysis-only vocal separation.

The separator never replaces ``raw/`` or the lossless canonical extraction.
Its output is a content-addressed derived stem under the current Prepare
lineage and is suitable only for ASR/alignment diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from personavoice.atomic import atomic_write_json
from personavoice.lineage import SEPARATION_CONTRACT_VERSION, separator_contract
from personavoice.media import sha256_file
from personavoice.model_assets import SEPARATOR_MODEL_FILENAME, SEPARATOR_PACKAGE
from personavoice.project import PersonaPaths

_MUSIC_HINTS = frozenset(
    {
        "bgm",
        "music",
        "song",
        "ost",
        "karaoke",
        "game",
        "theme",
        "vocal",
        "実況",
        "歌",
        "音楽",
    }
)
SEPARATOR_MODEL_MANIFEST_SCHEMA = 1
SEPARATOR_MODEL_MANIFEST_FILENAME = "manifest.json"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _tokens(path: Path) -> set[str]:
    normalized = path.stem.casefold().replace("-", "_").replace(".", "_")
    return {part for part in normalized.split("_") if part}


def separation_decision(
    source: Path,
    *,
    policy: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a deterministic, inspectable ``off``/``auto``/``always`` choice."""

    normalized = str(policy).strip().lower()
    if normalized not in {"off", "auto", "always"}:
        raise ValueError(f"Unsupported separation policy: {policy!r}")
    metadata = metadata or {}
    hint_values = [str(source.name)]
    for key in ("path", "name", "filename", "source_path"):
        value = metadata.get(key)
        if isinstance(value, str):
            hint_values.append(value)
    hints: set[str] = set()
    for value in hint_values:
        hints.update(_tokens(Path(value)))
    explicit_music = bool(metadata.get("music_heavy"))
    if normalized == "off":
        selected = False
        reason = "policy_off"
    elif normalized == "always":
        selected = True
        reason = "policy_always"
    elif explicit_music:
        selected = True
        reason = "metadata_music_heavy"
    elif hints & _MUSIC_HINTS:
        selected = True
        reason = "deterministic_filename_hint:" + ",".join(sorted(hints & _MUSIC_HINTS))
    else:
        selected = False
        reason = "auto_no_music_heavy_evidence"
    return {
        "contract_version": SEPARATION_CONTRACT_VERSION,
        "policy": normalized,
        "selected": selected,
        "reason": reason,
        "source": str(source),
        "source_sha256": sha256_file(source) if source.is_file() else None,
        "music_hints": sorted(hints & _MUSIC_HINTS),
        "analysis_only": True,
        "separator": separator_contract(),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _model_root(repo_root: Path) -> Path:
    return repo_root / "models" / "separation"


def separator_model_audit(repo_root: Path) -> dict[str, Any]:
    """Return the offline admission decision for the model-specific UVR terms.

    ``audio-separator`` can fetch weights implicitly.  That is unsuitable for
    a reproducible, offline Prepare, so a local model is accepted only when a
    human has registered its exact digest, source and model-specific terms.
    """

    root = _model_root(repo_root)
    model = root / SEPARATOR_MODEL_FILENAME
    manifest_path = root / SEPARATOR_MODEL_MANIFEST_FILENAME
    base = {
        "schema_version": SEPARATOR_MODEL_MANIFEST_SCHEMA,
        "model_filename": SEPARATOR_MODEL_FILENAME,
        "model_lineage": separator_contract()["model_lineage"],
        "model_path": str(model),
        "manifest_path": str(manifest_path),
        "materialized": False,
        "sha256": None,
        "size": None,
        "reason": None,
    }
    if not model.is_file() or model.stat().st_size <= 0:
        base["reason"] = "model_missing"
        return base
    recorded = _read_manifest(manifest_path)
    expected_keys = {
        "schema_version",
        "contract_version",
        "backend",
        "version",
        "source_repository",
        "source_revision",
        "source_license",
        "source_url",
        "model_filename",
        "model_lineage",
        "model_terms",
        "sha256",
        "size",
    }
    if recorded is None or set(recorded) != expected_keys:
        base["reason"] = "missing_or_invalid_local_manifest"
        return base
    if (
        recorded.get("schema_version") != SEPARATOR_MODEL_MANIFEST_SCHEMA
        or recorded.get("contract_version") != SEPARATION_CONTRACT_VERSION
        or recorded.get("backend") != SEPARATOR_PACKAGE
        or recorded.get("version") != separator_contract()["version"]
        or recorded.get("source_repository") != separator_contract()["source_repository"]
        or recorded.get("source_revision") != separator_contract()["source_revision"]
        or recorded.get("source_license") != separator_contract()["source_license"]
        or recorded.get("model_filename") != SEPARATOR_MODEL_FILENAME
        or recorded.get("model_lineage") != separator_contract()["model_lineage"]
        or not isinstance(recorded.get("source_url"), str)
        or not recorded["source_url"].strip()
        or not isinstance(recorded.get("model_terms"), str)
        or not recorded["model_terms"].strip()
        or not isinstance(recorded.get("sha256"), str)
        or len(recorded["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in recorded["sha256"].lower())
        or type(recorded.get("size")) is not int
        or recorded["size"] <= 0
    ):
        base["reason"] = "manifest_contract_mismatch"
        return base
    actual = sha256_file(model)
    if actual != recorded["sha256"].lower() or model.stat().st_size != recorded["size"]:
        base["reason"] = "model_digest_or_size_mismatch"
        return base
    base.update(
        {
            "materialized": True,
            "sha256": actual,
            "size": model.stat().st_size,
            "source_url": recorded["source_url"],
            "model_terms": recorded["model_terms"],
        }
    )
    return base


def register_separator_model(
    repo_root: Path,
    source: Path,
    *,
    source_url: str,
    model_terms: str,
) -> dict[str, Any]:
    """Atomically register a user-materialized model after a terms audit."""

    source = source.expanduser().resolve()
    if source.name != SEPARATOR_MODEL_FILENAME:
        raise ValueError(
            f"Separator model must be named {SEPARATOR_MODEL_FILENAME!r}; got {source.name!r}"
        )
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Separator model is missing or empty: {source}")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("source_url is required so the model provenance is reviewable")
    if not isinstance(model_terms, str) or not model_terms.strip():
        raise ValueError(
            "model_terms is required; the wrapper's MIT license does not establish model-weight terms"
        )
    destination_root = _model_root(repo_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / SEPARATOR_MODEL_FILENAME
    digest = sha256_file(source)
    if destination.is_file() and destination.stat().st_size > 0:
        if sha256_file(destination) != digest:
            raise FileExistsError(
                f"A different separator model is already registered at {destination}; "
                "move it aside explicitly before registering a replacement."
            )
    else:
        incoming = destination.with_name(f".{destination.name}.{uuid4().hex}.incoming")
        try:
            shutil.copy2(source, incoming)
            os.replace(incoming, destination)
        finally:
            incoming.unlink(missing_ok=True)
    atomic_write_json(
        destination_root / SEPARATOR_MODEL_MANIFEST_FILENAME,
        {
            "schema_version": SEPARATOR_MODEL_MANIFEST_SCHEMA,
            "contract_version": SEPARATION_CONTRACT_VERSION,
            "backend": SEPARATOR_PACKAGE,
            "version": separator_contract()["version"],
            "source_repository": separator_contract()["source_repository"],
            "source_revision": separator_contract()["source_revision"],
            "source_license": separator_contract()["source_license"],
            "source_url": source_url.strip(),
            "model_filename": SEPARATOR_MODEL_FILENAME,
            "model_lineage": separator_contract()["model_lineage"],
            "model_terms": model_terms.strip(),
            "sha256": digest,
            "size": destination.stat().st_size,
        },
    )
    audit = separator_model_audit(repo_root)
    if not audit["materialized"]:
        raise RuntimeError("Registered separator model failed its local manifest audit")
    return audit


def _existing_stem(
    manifest_path: Path,
    source: Path,
    *,
    separator_model_sha256: str,
) -> Path | None:
    value = _read_manifest(manifest_path)
    if value is None or value.get("source_sha256") != sha256_file(source):
        return None
    if (
        value.get("contract_version") != SEPARATION_CONTRACT_VERSION
        or value.get("analysis_only") is not True
        or value.get("canonical_preserved") is not True
    ):
        return None
    model = value.get("separator_model")
    if not isinstance(model, dict) or model.get("sha256") != separator_model_sha256:
        return None
    derived = value.get("derived_audio")
    if not isinstance(derived, str):
        return None
    path = Path(derived)
    if not path.is_absolute():
        path = manifest_path.parent / path
    try:
        path = path.resolve()
        path.relative_to(manifest_path.parent.resolve())
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        recorded_digest = value.get("derived_sha256")
        return path if recorded_digest == sha256_file(path) else None
    except (OSError, ValueError):
        return None


def materialize_analysis_audio(
    source: Path,
    paths: PersonaPaths,
    source_id: str,
    *,
    policy: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Return canonical audio or a verified derived vocal stem."""

    source = source.resolve()
    decision = separation_decision(source, policy=policy, metadata=metadata)
    if not decision["selected"]:
        decision["derived_audio"] = None
        return source, decision

    # Validate the human-audited local model before looking at an old stem.
    # The model digest is also part of the cache key, so a replacement creates
    # a new derived-audio identity instead of reusing a stem from another model.
    audit = separator_model_audit(paths.root.parents[1])
    model_dir = paths.root.parents[1] / "models" / "separation"
    model_file = model_dir / SEPARATOR_MODEL_FILENAME
    if not audit["materialized"]:
        raise RuntimeError(
            "Separation was selected but the pinned UVR model failed the offline license/digest audit: "
            f"{model_file} ({audit.get('reason')}). Register it with "
            "`persona register-separator-model` before rerunning Prepare."
        )

    key = _digest(
        {
            "source_id": source_id,
            "source_sha256": decision["source_sha256"],
            "separator": decision["separator"],
            "separator_model_sha256": audit["sha256"],
            "policy": decision["policy"],
        }
    )[:32]
    stem_root = paths.cache / "analysis_stems" / key
    stem_root.mkdir(parents=True, exist_ok=True)
    manifest_path = stem_root / "manifest.json"
    existing = _existing_stem(
        manifest_path,
        source,
        separator_model_sha256=str(audit["sha256"]),
    )
    if existing is not None:
        decision.update(
            {
                "status": "reused",
                "derived_audio": str(existing.resolve()),
                "derived_sha256": sha256_file(existing),
                "manifest": str(manifest_path.resolve()),
                "separator_model": {
                    "sha256": audit["sha256"],
                    "size": audit["size"],
                },
            }
        )
        return existing, decision

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Separation was selected but the pinned audio-separator runtime is not installed. "
            "Install the optional separator environment or set prepare.separation_policy=off."
        )
    # Separation weights are shared runtime assets, not per-lineage model
    # artifacts. Derived stems stay inside the candidate lineage cache while
    # this pinned local model is never copied into or used as a publication
    # candidate.
    output_dir = stem_root / "tool-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        uv,
        "run",
        "--project",
        str(paths.root.parents[1] / "workers" / "asr"),
        "--no-sync",
        SEPARATOR_PACKAGE,
        str(source),
        "--model_filename",
        SEPARATOR_MODEL_FILENAME,
        "--model_file_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "FLAC",
        "--output_single_stem",
        "Vocals",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PERSONAVOICE_ROOT": str(paths.root.parents[1].resolve()),
        }
    )
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Pinned audio separation failed; canonical audio was preserved and no stem was published"
        ) from exc
    candidates = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".flac", ".wav"}
    )
    if not candidates:
        raise RuntimeError("Pinned audio separation produced no vocal stem")
    derived = stem_root / "vocals.flac"
    incoming = derived.with_name(f".{derived.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(candidates[0], incoming)
        os.replace(incoming, derived)
    finally:
        incoming.unlink(missing_ok=True)
    if derived.resolve() == source.resolve() or derived.stat().st_size <= 0:
        raise RuntimeError("Separator returned an invalid derived stem")
    decision.update(
        {
            "status": "created",
            "derived_audio": str(derived.resolve()),
            "derived_sha256": sha256_file(derived),
            "manifest": str(manifest_path.resolve()),
            "separator_model": {
                "sha256": audit["sha256"],
                "size": audit["size"],
            },
        }
    )
    atomic_write_json(
        manifest_path,
        {
            **decision,
            "canonical_audio": str(source),
            "canonical_preserved": True,
        },
    )
    return derived, decision
