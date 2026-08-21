from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from personavoice.atomic import atomic_write_text
from personavoice.media import sha256_file

REVISION_MARKER = ".personavoice-revision"
ASSET_CONTRACT_RELATIVE = Path("config/seed_vc_assets.json")
ASSET_ROOT_RELATIVE = Path("models/seed_vc/assets")
READY_MARKER_RELATIVE = Path(".runtime/seed-vc-models-ready")


def contract_path(repo_root: Path) -> Path:
    return repo_root / ASSET_CONTRACT_RELATIVE


def asset_root(repo_root: Path) -> Path:
    return repo_root / ASSET_ROOT_RELATIVE


def ready_marker(repo_root: Path) -> Path:
    return repo_root / READY_MARKER_RELATIVE


def contract_digest(repo_root: Path) -> str:
    path = contract_path(repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Seed-VC asset contract is missing: {path}")
    return sha256_file(path)


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.startswith(("/", "\\")):
        raise ValueError(f"{label} must stay inside the Seed-VC asset root: {value!r}")
    return path.as_posix()


def load_contract(repo_root: Path) -> dict[str, Any]:
    path = contract_path(repo_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Seed-VC asset contract is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Seed-VC asset contract schema_version must be 1")
    snapshots = value.get("snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("Seed-VC asset contract must contain snapshots")

    seen_dirs: set[str] = set()
    for name, raw in snapshots.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError("Seed-VC snapshot entries must be named objects")
        repo_id = raw.get("repo_id")
        revision = raw.get("revision")
        local_dir = _safe_relative(raw.get("local_dir"), label=f"{name}.local_dir")
        required = raw.get("required_files")
        hashes = raw.get("sha256", {})
        if not isinstance(repo_id, str) or "/" not in repo_id:
            raise ValueError(f"{name}.repo_id is invalid")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{name}.revision must be a full 40-character Git commit")
        if local_dir in seen_dirs:
            raise ValueError(f"duplicate Seed-VC local_dir: {local_dir}")
        seen_dirs.add(local_dir)
        if not isinstance(required, list) or not required:
            raise ValueError(f"{name}.required_files must be a non-empty list")
        normalized = [_safe_relative(item, label=f"{name}.required_files") for item in required]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{name}.required_files contains duplicates")
        if not isinstance(hashes, dict):
            raise ValueError(f"{name}.sha256 must be an object")
        for relative, digest in hashes.items():
            normalized_hash_path = _safe_relative(relative, label=f"{name}.sha256 key")
            if normalized_hash_path not in normalized:
                raise ValueError(f"{name}.sha256 references an undeclared file: {relative}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{name}.sha256[{relative!r}] must be a SHA256 hex digest")
    return value


def snapshot_directory(repo_root: Path, snapshot: dict[str, Any]) -> Path:
    return asset_root(repo_root) / str(snapshot["local_dir"])


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _snapshot_errors(
    repo_root: Path,
    name: str,
    snapshot: dict[str, Any],
    *,
    verify_hashes: bool,
) -> list[str]:
    directory = snapshot_directory(repo_root, snapshot)
    errors: list[str] = []
    revision = str(snapshot["revision"])
    marker = directory / REVISION_MARKER
    try:
        recorded = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    except OSError:
        recorded = None
    if recorded != revision:
        errors.append(f"{name}: revision marker does not match {revision}")

    hashes = snapshot.get("sha256") or {}
    for relative in snapshot["required_files"]:
        path = directory / str(relative)
        if not _nonempty_file(path):
            errors.append(f"{name}: required asset is missing or empty: {relative}")
            continue
        expected = hashes.get(relative)
        if verify_hashes and isinstance(expected, str):
            try:
                actual = sha256_file(path)
            except OSError as exc:
                errors.append(f"{name}: checksum read failed for {relative}: {exc}")
            else:
                if actual != expected:
                    errors.append(
                        f"{name}: checksum mismatch for {relative}; expected {expected}, got {actual}"
                    )
    return errors


def materialization_status(repo_root: Path, *, verify_hashes: bool = False) -> dict[str, Any]:
    try:
        contract = load_contract(repo_root)
        digest = contract_digest(repo_root)
    except Exception as exc:
        return {
            "ok": False,
            "contract_sha256": None,
            "snapshots": {},
            "errors": [f"asset contract error: {type(exc).__name__}: {exc}"],
        }

    snapshot_results: dict[str, dict[str, Any]] = {}
    all_errors: list[str] = []
    for name, snapshot in contract["snapshots"].items():
        errors = _snapshot_errors(
            repo_root,
            name,
            snapshot,
            verify_hashes=verify_hashes,
        )
        snapshot_results[name] = {
            "ok": not errors,
            "repo_id": snapshot["repo_id"],
            "revision": snapshot["revision"],
            "directory": str(snapshot_directory(repo_root, snapshot)),
            "errors": errors,
        }
        all_errors.extend(errors)

    marker = ready_marker(repo_root)
    try:
        ready_value = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    except OSError:
        ready_value = None
    ready_matches = ready_value == digest
    if not ready_matches:
        all_errors.append("Seed-VC ready marker does not match the current asset contract")
    return {
        "ok": not all_errors,
        "contract_sha256": digest,
        "ready_marker": ready_value,
        "ready_marker_matches": ready_matches,
        "snapshots": snapshot_results,
        "errors": all_errors,
    }


def materialize(
    repo_root: Path,
    *,
    cache_dir: Path,
    token: str | None = None,
) -> dict[str, list[str]]:
    contract = load_contract(repo_root)
    root = asset_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    reused: list[str] = []

    for name, snapshot in contract["snapshots"].items():
        directory = snapshot_directory(repo_root, snapshot)
        current_errors = _snapshot_errors(
            repo_root,
            name,
            snapshot,
            verify_hashes=True,
        )
        if not current_errors:
            reused.append(name)
            continue

        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(snapshot["repo_id"]),
            revision=str(snapshot["revision"]),
            local_dir=directory,
            cache_dir=cache_dir,
            allow_patterns=[str(value) for value in snapshot["required_files"]],
            token=token,
        )
        atomic_write_text(directory / REVISION_MARKER, str(snapshot["revision"]) + "\n")
        errors = _snapshot_errors(
            repo_root,
            name,
            snapshot,
            verify_hashes=True,
        )
        if errors:
            shutil.rmtree(directory, ignore_errors=True)
            raise RuntimeError(
                "Pinned Seed-VC asset materialization failed: " + "; ".join(errors)
            )
        downloaded.append(name)

    return {"downloaded": downloaded, "reused": reused}


def purge_materialization(repo_root: Path) -> None:
    shutil.rmtree(asset_root(repo_root), ignore_errors=True)
    ready_marker(repo_root).unlink(missing_ok=True)


def write_ready_marker(repo_root: Path) -> str:
    digest = contract_digest(repo_root)
    marker = ready_marker(repo_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(marker, digest + "\n")
    return digest
