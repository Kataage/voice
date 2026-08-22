from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from personavoice.atomic import atomic_write_json, atomic_write_text
from personavoice.environment_contract import SETUP_TRANSACTION_MARKER, environment_contract
from personavoice.hardware import detect_irodori_backend
from personavoice.media import sha256_file
from personavoice.model_assets import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
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
    LFM_MODEL_ID,
    LFM_MODEL_REVISION,
    PYANNOTE_MODEL_ID,
    PYANNOTE_MODEL_REVISION,
    SEED_VC_SOURCE_REVISION,
    SENSE_MODEL_CMVN_SHA256,
    SENSE_MODEL_ID,
    SENSE_MODEL_TOKENIZER_SHA256,
    SENSE_MODEL_WEIGHT_SHA256,
)
from personavoice.process import CommandError, run
from personavoice.seed_vc_assets import (
    contract_digest as seed_vc_contract_digest,
)
from personavoice.seed_vc_assets import (
    materialize as materialize_seed_vc_assets,
)
from personavoice.seed_vc_assets import (
    ready_marker as seed_vc_ready_marker,
)
from personavoice.workers import local_model_env, worker

IRODORI_REPO = "https://github.com/Aratako/Irodori-TTS.git"
IRODORI_REVISION = IRODORI_SOURCE_REVISION
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEED_VC_REVISION = SEED_VC_SOURCE_REVISION
REVISION_MARKER = ".personavoice-revision"
IRODORI_LOCK_SWAP_MARKER = "irodori-lock-swap.json"
SUPPORTED_IRODORI_BACKENDS = {"cpu", "cu128", "rocm", "xpu"}
_LFM_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_ASR_REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)
_PYANNOTE_REQUIRED_FILES = (
    "config.yaml",
    "embedding/pytorch_model.bin",
    "segmentation/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)


def _irodori_swap_marker(repo_root: Path) -> Path:
    return repo_root / ".runtime" / IRODORI_LOCK_SWAP_MARKER


def _git_head(directory: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=directory, capture=True).stdout.strip()


def _restore_vendor_lock(irodori: Path) -> None:
    """Restore uv.lock to the pinned checkout's clean HEAD state."""

    vendor_lock = irodori / "uv.lock"
    tracked = run(
        ["git", "ls-files", "--error-unmatch", "--", "uv.lock"],
        cwd=irodori,
        capture=True,
        check=False,
    ).returncode == 0
    if tracked:
        run(
            ["git", "restore", "--source=HEAD", "--worktree", "--", "uv.lock"],
            cwd=irodori,
        )
    else:
        vendor_lock.unlink(missing_ok=True)


def _recover_irodori_lock_swap(repo_root: Path, irodori: Path) -> None:
    """Recover a managed uv.lock swap interrupted by process termination."""

    marker = _irodori_swap_marker(repo_root)
    if not marker.is_file():
        return
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Interrupted Irodori lock-swap marker is unreadable: {marker}. "
            "Inspect the vendor checkout before rerunning setup."
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise RuntimeError(
            f"Interrupted Irodori lock-swap marker has an unsupported format: {marker}"
        )

    expected_head = value.get("vendor_head")
    current_head = _git_head(irodori)
    if not isinstance(expected_head, str) or current_head != expected_head:
        raise RuntimeError(
            "An interrupted PersonaVoice Irodori lock swap belongs to a different vendor HEAD. "
            "Refusing automatic recovery; inspect vendor/Irodori-TTS before rerunning setup."
        )

    vendor_lock = irodori / "uv.lock"
    current_exists = vendor_lock.is_file()
    current_sha = sha256_file(vendor_lock) if current_exists else None
    original_exists = bool(value.get("original_exists"))
    original_sha = value.get("original_sha256")
    managed_sha = value.get("managed_sha256")
    original_state = current_exists == original_exists and (
        not current_exists or current_sha == original_sha
    )
    managed_state = current_exists and isinstance(managed_sha, str) and current_sha == managed_sha
    if managed_state:
        _restore_vendor_lock(irodori)
    elif not original_state:
        raise RuntimeError(
            "An interrupted PersonaVoice Irodori lock swap was found, but vendor/"
            "Irodori-TTS/uv.lock no longer matches either the original checkout or the "
            "audited temporary lock. Refusing to overwrite a possible local edit."
        )
    marker.unlink(missing_ok=True)


def _clone_pinned(repo_root: Path, name: str, url: str, revision: str) -> Path:
    destination = repo_root / "vendor" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        run(["git", "clone", "--filter=blob:none", url, destination], cwd=repo_root)
    git_dir = destination / ".git"
    if not git_dir.exists():
        raise RuntimeError(f"{destination} exists but is not a git checkout")
    if name == "Irodori-TTS":
        _recover_irodori_lock_swap(repo_root, destination)
    status = run(["git", "status", "--porcelain"], cwd=destination, capture=True).stdout.strip()
    if status:
        raise RuntimeError(
            f"Vendor checkout has local changes and will not be modified automatically: {destination}"
        )
    try:
        run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=destination,
            capture=True,
        )
    except CommandError:
        run(["git", "fetch", "--depth", "1", "origin", revision], cwd=destination)
    run(["git", "checkout", "--detach", revision], cwd=destination)
    return destination


def _worker_extras(selected_backend: str) -> dict[str, str | None]:
    """Map the Irodori backend to compatible isolated worker backends."""

    if selected_backend == "cu128":
        return {
            "asr": None,
            "diarization": "cu128",
            "sense": "cu128",
            "lfm": "cu128",
            "seed_vc": "cu124",
        }
    return {
        "asr": None,
        "diarization": "cpu",
        "sense": "cpu",
        "lfm": "cpu",
        "seed_vc": "cpu",
    }


def _install_irodori(repo_root: Path, irodori: Path, backend: str) -> None:
    """Sync Irodori from the audited lock with crash-safe checkout recovery."""

    managed_lock = repo_root / "locks" / "Irodori-TTS.uv.lock"
    if not managed_lock.is_file():
        raise FileNotFoundError(
            f"Audited Irodori lockfile is missing: {managed_lock}. "
            "Refusing an unlocked environment sync; restore the repository lockfile first."
        )

    marker = _irodori_swap_marker(repo_root)
    vendor_lock = irodori / "uv.lock"
    original_exists = vendor_lock.is_file()
    swap_state = {
        "schema_version": 2,
        "vendor": str(irodori.resolve()),
        "vendor_head": _git_head(irodori),
        "original_exists": original_exists,
        "original_sha256": sha256_file(vendor_lock) if original_exists else None,
        "managed_sha256": sha256_file(managed_lock),
    }
    atomic_write_json(marker, swap_state)
    args: list[str | Path] = [
        "uv",
        "sync",
        "--project",
        irodori,
        "--extra",
        backend,
        "--locked",
    ]
    try:
        atomic_write_text(vendor_lock, managed_lock.read_text(encoding="utf-8"))
        run(args, cwd=repo_root)
    finally:
        _restore_vendor_lock(irodori)
        marker.unlink(missing_ok=True)


def install_environments(repo_root: Path, *, backend: str | None = None) -> dict:
    if not shutil.which("uv"):
        raise RuntimeError("uv was not found in PATH")
    if not shutil.which("git"):
        raise RuntimeError("git was not found in PATH")
    selected_backend = backend or detect_irodori_backend()
    if selected_backend not in SUPPORTED_IRODORI_BACKENDS:
        expected = ", ".join(sorted(SUPPORTED_IRODORI_BACKENDS))
        raise ValueError(f"Unsupported Irodori backend {selected_backend!r}; choose one of: {expected}")

    worker_extras = _worker_extras(selected_backend)
    runtime = repo_root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    transaction_marker = runtime / SETUP_TRANSACTION_MARKER
    transaction_state = {
        "schema_version": 1,
        "irodori_backend": selected_backend,
        "worker_backends": worker_extras,
        "irodori_revision": IRODORI_REVISION,
        "seed_vc_revision": SEED_VC_REVISION,
        "environment_contract": environment_contract(repo_root),
    }
    # This marker is written before the first environment mutation. It is
    # deliberately kept on any failure so an old setup.json can never authorize
    # a partially replaced CPU/CUDA environment after an interrupted setup.
    atomic_write_json(transaction_marker, transaction_state)

    irodori = _clone_pinned(
        repo_root,
        "Irodori-TTS",
        IRODORI_REPO,
        IRODORI_REVISION,
    )
    seed = _clone_pinned(repo_root, "seed-vc", SEED_VC_REPO, SEED_VC_REVISION)
    _install_irodori(repo_root, irodori, selected_backend)

    synced = []
    for name in ("asr", "diarization", "sense", "lfm", "seed_vc"):
        worker(repo_root, name).sync(repo_root, extra=worker_extras[name])
        synced.append(name)

    setup_state = {
        "irodori_backend": selected_backend,
        "worker_backends": worker_extras,
        "irodori_revision": IRODORI_REVISION,
        "seed_vc_revision": SEED_VC_REVISION,
        "environment_contract": environment_contract(repo_root),
        "model_assets": {
            "irodori_model_sha256": IRODORI_MODEL_SHA256,
            "irodori_dacvae_sha256": IRODORI_DACVAE_SHA256,
            "irodori_text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
            "lfm_revision": LFM_MODEL_REVISION,
            "asr_revision": ASR_MODEL_REVISION,
            "seed_vc_asset_contract_sha256": seed_vc_contract_digest(repo_root),
        },
        "prepare_assets": {
            "pyannote_revision": PYANNOTE_MODEL_REVISION,
            "sense_weight_sha256": SENSE_MODEL_WEIGHT_SHA256,
            "sense_cmvn_sha256": SENSE_MODEL_CMVN_SHA256,
            "sense_tokenizer_sha256": SENSE_MODEL_TOKENIZER_SHA256,
        },
    }
    atomic_write_json(runtime / "setup.json", setup_state)
    transaction_marker.unlink(missing_ok=True)
    return {
        **setup_state,
        "workers": synced,
        "vendor": {"irodori": str(irodori), "seed_vc": str(seed)},
    }


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} checksum mismatch: expected {expected}, got {actual}. "
            "The local file may be corrupted or the upstream asset changed; "
            "remove the file and retry with an audited PersonaVoice revision."
        )


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _download_verified_file(
    *,
    model_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
    cache_dir: Path,
    sha256: str,
) -> tuple[Path, bool]:
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / filename
    existed = target.is_file()
    force_download = False
    if existed:
        try:
            _verify_sha256(target, sha256, label=f"{model_id}:{filename}")
            return target, True
        except RuntimeError:
            target.unlink(missing_ok=True)
            force_download = True

    hf_hub_download(
        repo_id=model_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    if not target.is_file():
        raise FileNotFoundError(f"Expected model file was not created: {target}")
    _verify_sha256(target, sha256, label=f"{model_id}:{filename}")
    return target, False


def _snapshot_pinned(
    *,
    model_id: str,
    revision: str,
    local_dir: Path,
    required_files: tuple[str, ...],
    cache_dir: Path,
    token: str | None = None,
) -> bool:
    revision_path = local_dir / REVISION_MARKER
    current_revision = (
        revision_path.read_text(encoding="utf-8").strip()
        if revision_path.is_file()
        else None
    )
    complete = all(_nonempty_file(local_dir / relative) for relative in required_files)
    if complete and current_revision == revision:
        return False

    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        cache_dir=cache_dir,
        token=token,
    )
    missing = [
        relative
        for relative in required_files
        if not _nonempty_file(local_dir / relative)
    ]
    if missing:
        raise FileNotFoundError(
            f"Pinned model download for {model_id}@{revision} completed but "
            f"required files are missing or empty: {', '.join(missing)}"
        )
    atomic_write_text(revision_path, revision + "\n")
    return True


def download_models(
    repo_root: Path,
    *,
    hf_token: str | None = None,
    include_seed_vc: bool = True,
) -> dict:
    env = local_model_env(repo_root, offline=False)
    hf_home = Path(env["HF_HOME"])
    hub_cache = Path(env["HUGGINGFACE_HUB_CACHE"])
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.getenv("HF_TOKEN")
    downloaded: list[str] = []
    reused: list[str] = []

    irodori_dir = repo_root / "models" / "irodori" / "v4.1-small"
    _, existed = _download_verified_file(
        model_id=IRODORI_MODEL_ID,
        filename=IRODORI_MODEL_FILENAME,
        revision=IRODORI_MODEL_REVISION,
        local_dir=irodori_dir,
        cache_dir=hub_cache,
        sha256=IRODORI_MODEL_SHA256,
    )
    (reused if existed else downloaded).append(
        f"{IRODORI_MODEL_ID}:{IRODORI_MODEL_FILENAME}"
    )

    dacvae_dir = repo_root / "models" / "irodori" / "dacvae"
    _, existed = _download_verified_file(
        model_id=IRODORI_DACVAE_ID,
        filename=IRODORI_DACVAE_FILENAME,
        revision=IRODORI_DACVAE_REVISION,
        local_dir=dacvae_dir,
        cache_dir=hub_cache,
        sha256=IRODORI_DACVAE_SHA256,
    )
    (reused if existed else downloaded).append(
        f"{IRODORI_DACVAE_ID}:{IRODORI_DACVAE_FILENAME}"
    )

    snapshot_download(
        repo_id=IRODORI_TEXT_ENCODER_ID,
        revision=IRODORI_TEXT_ENCODER_REVISION,
        cache_dir=hub_cache,
    )
    reused.append(f"{IRODORI_TEXT_ENCODER_ID}@{IRODORI_TEXT_ENCODER_REVISION}")

    lfm_dir = repo_root / "models" / "lfm" / "base"
    if _snapshot_pinned(
        model_id=LFM_MODEL_ID,
        revision=LFM_MODEL_REVISION,
        local_dir=lfm_dir,
        required_files=_LFM_REQUIRED_FILES,
        cache_dir=hub_cache,
    ):
        downloaded.append(f"{LFM_MODEL_ID}@{LFM_MODEL_REVISION}")
    else:
        reused.append(f"{LFM_MODEL_ID}@{LFM_MODEL_REVISION}")

    asr_dir = repo_root / "models" / "asr" / "large-v3"
    if _snapshot_pinned(
        model_id=ASR_MODEL_ID,
        revision=ASR_MODEL_REVISION,
        local_dir=asr_dir,
        required_files=_ASR_REQUIRED_FILES,
        cache_dir=hub_cache,
    ):
        downloaded.append(f"{ASR_MODEL_ID}@{ASR_MODEL_REVISION}")
    else:
        reused.append(f"{ASR_MODEL_ID}@{ASR_MODEL_REVISION}")

    pyannote_dir = repo_root / "models" / "pyannote" / "community-1"
    pyannote_revision_marker = pyannote_dir / REVISION_MARKER
    pyannote_is_pinned = (
        all(_nonempty_file(pyannote_dir / name) for name in _PYANNOTE_REQUIRED_FILES)
        and pyannote_revision_marker.is_file()
        and pyannote_revision_marker.read_text(encoding="utf-8").strip()
        == PYANNOTE_MODEL_REVISION
    )
    if not pyannote_is_pinned and not token:
        raise RuntimeError(
            f"HF_TOKEN is required to materialize the audited {PYANNOTE_MODEL_ID} snapshot. "
            "Accept its Hugging Face usage terms, then set HF_TOKEN in the environment. "
            "The token is never stored by PersonaVoice."
        )
    if _snapshot_pinned(
        model_id=PYANNOTE_MODEL_ID,
        revision=PYANNOTE_MODEL_REVISION,
        local_dir=pyannote_dir,
        required_files=_PYANNOTE_REQUIRED_FILES,
        cache_dir=hub_cache,
        token=token,
    ):
        downloaded.append(f"{PYANNOTE_MODEL_ID}@{PYANNOTE_MODEL_REVISION}")
    else:
        reused.append(f"{PYANNOTE_MODEL_ID}@{PYANNOTE_MODEL_REVISION}")

    runtime = repo_root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    sense_marker = runtime / "sense-model-ready"
    sense_local = repo_root / "models" / "sense" / "SenseVoiceSmall"
    sense_worker = worker(repo_root, "sense")
    sense_verified = False
    if sense_marker.exists():
        try:
            sense_worker.call(repo_root, "verify", {}, offline=True)
            sense_verified = True
        except Exception:
            sense_marker.unlink(missing_ok=True)
            shutil.rmtree(sense_local, ignore_errors=True)
    if sense_verified:
        reused.append(SENSE_MODEL_ID)
    else:
        sense_worker.call(
            repo_root,
            "download",
            {"online": True},
            offline=False,
        )
        atomic_write_text(sense_marker, "verified\n")
        downloaded.append(SENSE_MODEL_ID)

    if include_seed_vc:
        seed_assets = materialize_seed_vc_assets(
            repo_root,
            cache_dir=hub_cache,
            token=token,
        )
        downloaded.extend(f"Seed-VC:{name}" for name in seed_assets["downloaded"])
        reused.extend(f"Seed-VC:{name}" for name in seed_assets["reused"])
        seed_digest = seed_vc_contract_digest(repo_root)
        marker = seed_vc_ready_marker(repo_root)
        try:
            marker_matches = marker.is_file() and marker.read_text(encoding="utf-8").strip() == seed_digest
        except OSError:
            marker_matches = False
        if seed_assets["downloaded"] or not marker_matches:
            # Prove the pinned local views can instantiate with HF network access
            # disabled before publishing the ready marker. The worker itself
            # writes the contract digest only after a successful local-only load.
            worker(repo_root, "seed_vc").call(
                repo_root,
                "download",
                {"online": False},
                offline=True,
            )
            downloaded.append("Seed-VC-v2-local-runtime")
        else:
            reused.append("Seed-VC-v2-local-runtime")

    return {
        "downloaded": downloaded,
        "reused": reused,
        "hf_home": str(hf_home),
        "hub_cache": str(hub_cache),
    }
