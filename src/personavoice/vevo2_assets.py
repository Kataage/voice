from __future__ import annotations

import json
import re
import shutil
from pathlib import Path, PureWindowsPath
from urllib.request import Request, urlopen
from uuid import uuid4

from huggingface_hub import snapshot_download

from personavoice.atomic import atomic_write_text
from personavoice.media import sha256_file

REVISION_MARKER = ".personavoice-revision"
ASSET_CONTRACT_RELATIVE = Path("config/vevo2_assets.json")
ASSET_ROOT_RELATIVE = Path("models/vevo2/assets")
READY_MARKER_RELATIVE = Path(".runtime/vevo2-models-ready")


def contract_path(repo_root: Path) -> Path:
    return repo_root / ASSET_CONTRACT_RELATIVE


def asset_root(repo_root: Path) -> Path:
    return repo_root / ASSET_ROOT_RELATIVE


def ready_marker(repo_root: Path) -> Path:
    return repo_root / READY_MARKER_RELATIVE


def contract_digest(repo_root: Path) -> str:
    path = contract_path(repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Vevo2 asset contract is missing: {path}")
    return sha256_file(path)


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise ValueError(f"{label} must use canonical forward-slash separators: {value!r}")
    path = Path(value)
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in path.parts
        or value.startswith(("/", "\\"))
    ):
        raise ValueError(f"{label} must stay inside the Vevo2 asset root: {value!r}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{label} must name a file below the Vevo2 asset root")
    return normalized


def _validate_license(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must declare a non-empty license")
    return value.strip()


def load_contract(repo_root: Path) -> dict:
    path = contract_path(repo_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Vevo2 asset contract is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Vevo2 asset contract schema_version must be 1")

    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("Vevo2 asset contract must declare source provenance")
    if not isinstance(source.get("repository"), str) or not source["repository"]:
        raise ValueError("Vevo2 source.repository is invalid")
    if not isinstance(source.get("revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", source["revision"]
    ):
        raise ValueError("Vevo2 source.revision must be a full 40-character Git commit")
    _validate_license(source.get("license"), label="Vevo2 source.license")

    model = value.get("model")
    if not isinstance(model, dict):
        raise ValueError("Vevo2 asset contract must declare a model snapshot")
    repo_id = model.get("repo_id")
    revision = model.get("revision")
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise ValueError("Vevo2 model.repo_id is invalid")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Vevo2 model.revision must be a full 40-character Git commit")
    local_dir = _safe_relative(model.get("local_dir"), label="Vevo2 model.local_dir")
    _validate_license(model.get("license"), label="Vevo2 model.license")
    required = model.get("required_files")
    hashes = model.get("sha256")
    if not isinstance(required, list) or not required:
        raise ValueError("Vevo2 model.required_files must be a non-empty list")
    if not isinstance(hashes, dict):
        raise ValueError("Vevo2 model.sha256 must be an object")
    normalized_required = [
        _safe_relative(item, label="Vevo2 model.required_files") for item in required
    ]
    if len(normalized_required) != len(set(normalized_required)):
        raise ValueError("Vevo2 model.required_files contains duplicates")
    normalized_hashes: dict[str, str] = {}
    for raw_relative, digest in hashes.items():
        relative = _safe_relative(raw_relative, label="Vevo2 model.sha256 key")
        if relative not in normalized_required:
            raise ValueError(f"Vevo2 model.sha256 references undeclared file: {raw_relative}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Vevo2 model.sha256[{raw_relative!r}] must be a SHA256 digest")
        normalized_hashes[relative] = digest
    if set(normalized_hashes) != set(normalized_required):
        raise ValueError("Vevo2 model.sha256 must cover every required file")

    whisper = value.get("whisper")
    if not isinstance(whisper, dict):
        raise ValueError("Vevo2 asset contract must declare the Whisper dependency")
    if not isinstance(whisper.get("source_repository"), str) or not whisper[
        "source_repository"
    ]:
        raise ValueError("Vevo2 whisper.source_repository is invalid")
    if not isinstance(whisper.get("source_revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", whisper["source_revision"]
    ):
        raise ValueError("Vevo2 whisper.source_revision must be a full 40-character commit")
    if not isinstance(whisper.get("url"), str) or not whisper["url"].startswith("https://"):
        raise ValueError("Vevo2 whisper.url must be an HTTPS URL")
    whisper_local = _safe_relative(whisper.get("local_file"), label="Vevo2 whisper.local_file")
    whisper_hash = whisper.get("sha256")
    if not isinstance(whisper_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", whisper_hash):
        raise ValueError("Vevo2 whisper.sha256 must be a SHA256 digest")
    _validate_license(whisper.get("license"), label="Vevo2 whisper.license")

    model["local_dir"] = local_dir
    model["required_files"] = normalized_required
    model["sha256"] = normalized_hashes
    whisper["local_file"] = whisper_local
    return value


def model_directory(repo_root: Path, contract: dict | None = None) -> Path:
    value = contract if contract is not None else load_contract(repo_root)
    return asset_root(repo_root) / str(value["model"]["local_dir"])


def whisper_path(repo_root: Path, contract: dict | None = None) -> Path:
    value = contract if contract is not None else load_contract(repo_root)
    return asset_root(repo_root) / str(value["whisper"]["local_file"])


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _remove_directory(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _snapshot_errors(
    repo_root: Path,
    contract: dict,
    *,
    verify_hashes: bool,
) -> list[str]:
    model = contract["model"]
    directory = model_directory(repo_root, contract)
    marker = directory / REVISION_MARKER
    try:
        recorded = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    except OSError:
        recorded = None
    errors = []
    if recorded != model["revision"]:
        errors.append(f"Vevo2 model revision marker does not match {model['revision']}")
    for relative in model["required_files"]:
        path = directory / relative
        if not _nonempty_file(path):
            errors.append(f"Vevo2 model asset is missing or empty: {relative}")
            continue
        expected = model["sha256"].get(relative)
        if verify_hashes and isinstance(expected, str):
            try:
                actual = sha256_file(path)
            except OSError as exc:
                errors.append(f"Vevo2 model checksum read failed for {relative}: {exc}")
            else:
                if actual != expected:
                    errors.append(
                        f"Vevo2 model checksum mismatch for {relative}; "
                        f"expected {expected}, got {actual}"
                    )
    return errors


def _whisper_errors(repo_root: Path, contract: dict, *, verify_hashes: bool) -> list[str]:
    whisper = contract["whisper"]
    path = whisper_path(repo_root, contract)
    errors = []
    if not _nonempty_file(path):
        errors.append(f"Vevo2 Whisper medium asset is missing or empty: {path.name}")
    elif verify_hashes:
        try:
            actual = sha256_file(path)
        except OSError as exc:
            errors.append(f"Vevo2 Whisper checksum read failed: {exc}")
        else:
            if actual != whisper["sha256"]:
                errors.append(
                    "Vevo2 Whisper medium checksum mismatch; "
                    f"expected {whisper['sha256']}, got {actual}"
                )
    return errors


def materialization_status(repo_root: Path, *, verify_hashes: bool = False) -> dict:
    try:
        contract = load_contract(repo_root)
        digest = contract_digest(repo_root)
    except Exception as exc:
        return {
            "ok": False,
            "contract_sha256": None,
            "model": {},
            "whisper": {},
            "errors": [f"asset contract error: {type(exc).__name__}: {exc}"],
        }
    model_errors = _snapshot_errors(repo_root, contract, verify_hashes=verify_hashes)
    whisper_errors = _whisper_errors(repo_root, contract, verify_hashes=verify_hashes)
    marker = ready_marker(repo_root)
    try:
        ready_value = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    except OSError:
        ready_value = None
    ready_matches = ready_value == digest
    errors = model_errors + whisper_errors
    if not ready_matches:
        errors.append("Vevo2 ready marker does not match the current asset contract")
    return {
        "ok": not errors,
        "contract_sha256": digest,
        "ready_marker": ready_value,
        "ready_marker_matches": ready_matches,
        "source": contract["source"],
        "model": {
            "repo_id": contract["model"]["repo_id"],
            "revision": contract["model"]["revision"],
            "license": contract["model"]["license"],
            "directory": str(model_directory(repo_root, contract)),
            "errors": model_errors,
        },
        "whisper": {
            "source_revision": contract["whisper"]["source_revision"],
            "license": contract["whisper"]["license"],
            "path": str(whisper_path(repo_root, contract)),
            "errors": whisper_errors,
        },
        "errors": errors,
    }


def _download_whisper(repo_root: Path, contract: dict) -> None:
    whisper = contract["whisper"]
    target = whisper_path(repo_root, contract)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.download")
    try:
        request = Request(
            str(whisper["url"]),
            headers={"User-Agent": "PersonaVoice/0.4 Vevo2 materializer"},
        )
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(4 * 1024 * 1024):
                handle.write(chunk)
            handle.flush()
        if not _nonempty_file(temporary):
            raise RuntimeError("downloaded Whisper medium file is empty")
        actual = sha256_file(temporary)
        if actual != whisper["sha256"]:
            raise RuntimeError(
                "downloaded Whisper medium checksum mismatch: "
                f"expected {whisper['sha256']}, got {actual}"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(
    repo_root: Path,
    *,
    cache_dir: Path,
    token: str | None = None,
) -> dict[str, list[str]]:
    """Materialize Vevo2 FM assets through the explicit online setup path only."""

    contract = load_contract(repo_root)
    root = asset_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    reused: list[str] = []

    model_errors = _snapshot_errors(repo_root, contract, verify_hashes=True)
    model_dir = model_directory(repo_root, contract)
    if model_errors:
        ready_marker(repo_root).unlink(missing_ok=True)
        _remove_directory(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=str(contract["model"]["repo_id"]),
                revision=str(contract["model"]["revision"]),
                local_dir=model_dir,
                cache_dir=cache_dir,
                allow_patterns=[str(value) for value in contract["model"]["required_files"]],
                token=token,
            )
            atomic_write_text(model_dir / REVISION_MARKER, str(contract["model"]["revision"]) + "\n")
            errors = _snapshot_errors(repo_root, contract, verify_hashes=True)
            if errors:
                raise RuntimeError("Vevo2 model materialization failed: " + "; ".join(errors))
        except Exception:
            _remove_directory(model_dir)
            raise
        downloaded.append("model")
    else:
        reused.append("model")

    whisper_errors = _whisper_errors(repo_root, contract, verify_hashes=True)
    if whisper_errors:
        ready_marker(repo_root).unlink(missing_ok=True)
        _download_whisper(repo_root, contract)
        errors = _whisper_errors(repo_root, contract, verify_hashes=True)
        if errors:
            raise RuntimeError("Vevo2 Whisper materialization failed: " + "; ".join(errors))
        downloaded.append("whisper-medium")
    else:
        reused.append("whisper-medium")

    return {"downloaded": downloaded, "reused": reused}


def purge_materialization(repo_root: Path) -> None:
    _remove_directory(asset_root(repo_root))
    ready_marker(repo_root).unlink(missing_ok=True)


def write_ready_marker(repo_root: Path) -> str:
    digest = contract_digest(repo_root)
    marker = ready_marker(repo_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(marker, digest + "\n")
    return digest
