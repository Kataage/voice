from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from personavoice.atomic import atomic_write_json, atomic_write_text
from personavoice.cuda_preflight import run_cuda_preflight
from personavoice.environment_contract import SETUP_TRANSACTION_MARKER, environment_contract
from personavoice.ffmpeg_materializer import ensure_ffmpeg_runtime
from personavoice.hardware import (
    backend_supports_gpu,
    cuda_backend_for_gpu,
    detect_irodori_backend,
    gpu_record,
    seed_vc_cuda_supported,
    selected_nvidia_gpu,
)
from personavoice.media import sha256_file
from personavoice.model_assets import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    ASR_MODEL_WEIGHT_SHA256,
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
    PYANNOTE_MODEL_ASSET_SHA256,
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
IRODORI_MANAGED_PROJECT = "Irodori-TTS.pyproject.toml"
SUPPORTED_IRODORI_BACKENDS = {"cpu", "cu126", "cu128", "rocm", "xpu"}
_LFM_REQUIRED_FILES = LFM_MODEL_REQUIRED_FILES
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


def _restore_vendor_file(irodori: Path, relative: str) -> None:
    path = irodori / relative
    tracked = run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=irodori,
        capture=True,
        check=False,
    ).returncode == 0
    if tracked:
        run(
            ["git", "restore", "--source=HEAD", "--worktree", "--", relative],
            cwd=irodori,
        )
    else:
        path.unlink(missing_ok=True)


def _restore_vendor_setup_files(irodori: Path) -> None:
    """Restore the pinned checkout after the managed project/lock overlay."""

    _restore_vendor_file(irodori, "pyproject.toml")
    _restore_vendor_file(irodori, "uv.lock")


def _file_swap_state(path: Path, managed: Path) -> dict:
    exists = path.is_file()
    return {
        "original_exists": exists,
        "original_sha256": sha256_file(path) if exists else None,
        "managed_sha256": sha256_file(managed),
    }


def _recover_irodori_lock_swap(repo_root: Path, irodori: Path) -> None:
    """Recover an interrupted managed Irodori project/lock overlay."""

    marker = _irodori_swap_marker(repo_root)
    if not marker.is_file():
        return
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Interrupted Irodori setup-overlay marker is unreadable: {marker}. "
            "Inspect the vendor checkout before rerunning setup."
        ) from exc
    schema = value.get("schema_version") if isinstance(value, dict) else None
    if schema not in {2, 3}:
        raise RuntimeError(
            f"Interrupted Irodori setup-overlay marker has an unsupported format: {marker}"
        )

    expected_head = value.get("vendor_head")
    current_head = _git_head(irodori)
    if not isinstance(expected_head, str) or current_head != expected_head:
        raise RuntimeError(
            "An interrupted PersonaVoice Irodori setup overlay belongs to a different vendor "
            "HEAD. Refusing automatic recovery; inspect vendor/Irodori-TTS before rerunning "
            "setup."
        )

    if schema == 2:
        # Backward-compatible recovery for markers written before the managed
        # pyproject overlay existed. Only uv.lock was swapped in that format.
        states = {
            "uv.lock": {
                "original_exists": bool(value.get("original_exists")),
                "original_sha256": value.get("original_sha256"),
                "managed_sha256": value.get("managed_sha256"),
            }
        }
    else:
        raw_states = value.get("files")
        if not isinstance(raw_states, dict):
            raise RuntimeError(f"Interrupted Irodori setup-overlay marker is invalid: {marker}")
        states = raw_states

    for relative, state in states.items():
        if relative not in {"pyproject.toml", "uv.lock"} or not isinstance(state, dict):
            raise RuntimeError(f"Interrupted Irodori setup-overlay marker is invalid: {marker}")
        path = irodori / relative
        current_exists = path.is_file()
        current_sha = sha256_file(path) if current_exists else None
        original_exists = bool(state.get("original_exists"))
        original_sha = state.get("original_sha256")
        managed_sha = state.get("managed_sha256")
        original_state = current_exists == original_exists and (
            not current_exists or current_sha == original_sha
        )
        managed_state = (
            current_exists
            and isinstance(managed_sha, str)
            and current_sha == managed_sha
        )
        if managed_state:
            _restore_vendor_file(irodori, relative)
        elif not original_state:
            raise RuntimeError(
                "An interrupted PersonaVoice Irodori setup overlay was found, but "
                f"vendor/Irodori-TTS/{relative} matches neither the pinned checkout nor the "
                "audited temporary overlay. Refusing to overwrite a possible local edit."
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


def _worker_extras(selected_backend: str, *, gpu=None) -> dict[str, str | None]:
    """Map the selected backend to safe isolated worker environments."""

    if selected_backend in {"cu126", "cu128"}:
        seed_backend = "cu124" if gpu is None or seed_vc_cuda_supported(gpu) else "cpu"
        return {
            "asr": None,
            "diarization": selected_backend,
            "sense": selected_backend,
            "lfm": selected_backend,
            # Seed-VC stays on its audited Torch 2.4 stack. Blackwell and newer
            # GPUs predate that wheel's cubins, so only this worker falls back.
            "seed_vc": seed_backend,
        }
    return {
        "asr": None,
        "diarization": "cpu",
        "sense": "cpu",
        "lfm": "cpu",
        "seed_vc": "cpu",
    }


def _install_irodori(repo_root: Path, irodori: Path, backend: str) -> None:
    """Sync Irodori from the audited project overlay and lock, then restore vendor files."""

    managed_project = repo_root / "locks" / IRODORI_MANAGED_PROJECT
    managed_lock = repo_root / "locks" / "Irodori-TTS.uv.lock"
    missing = [str(path) for path in (managed_project, managed_lock) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Audited Irodori dependency overlay is incomplete: "
            + ", ".join(missing)
            + ". Restore the repository lock files before setup."
        )

    marker = _irodori_swap_marker(repo_root)
    vendor_project = irodori / "pyproject.toml"
    vendor_lock = irodori / "uv.lock"
    swap_state = {
        "schema_version": 3,
        "vendor": str(irodori.resolve()),
        "vendor_head": _git_head(irodori),
        "files": {
            "pyproject.toml": _file_swap_state(vendor_project, managed_project),
            "uv.lock": _file_swap_state(vendor_lock, managed_lock),
        },
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
        atomic_write_text(vendor_project, managed_project.read_text(encoding="utf-8"))
        atomic_write_text(vendor_lock, managed_lock.read_text(encoding="utf-8"))
        run(args, cwd=repo_root)
    finally:
        _restore_vendor_setup_files(irodori)
        marker.unlink(missing_ok=True)


def _validate_cuda_backend(backend: str | None):
    """Return the selected GPU or reject an unsafe explicit/automatic CUDA stack."""

    if backend not in {"cu126", "cu128"}:
        return None
    gpu = selected_nvidia_gpu()
    if gpu is None:
        raise ValueError(
            f"The selected backend {backend} requires an NVIDIA GPU exposed as CUDA device 0, "
            "but no NVIDIA GPU could be selected. Check the driver and CUDA_VISIBLE_DEVICES, "
            "or use `--backend auto`."
        )
    if backend_supports_gpu(backend, gpu):
        return gpu

    preferred = cuda_backend_for_gpu(gpu)
    capability = gpu.compute_capability or "unknown"
    fallback = f"--backend {preferred}" if preferred in {"cu126", "cu128"} else "--backend cpu"
    raise ValueError(
        f"The selected NVIDIA GPU {gpu.name} has compute capability {capability}; "
        f"the audited {backend} PyTorch stack does not contain a compatible kernel image. "
        f"Use `--backend auto` (recommended) or `{fallback}`."
    )


def install_environments(repo_root: Path, *, backend: str | None = None) -> dict:
    if not shutil.which("uv"):
        raise RuntimeError("uv was not found in PATH")
    if not shutil.which("git"):
        raise RuntimeError("git was not found in PATH")
    selected_backend = backend or detect_irodori_backend()
    if selected_backend not in SUPPORTED_IRODORI_BACKENDS:
        expected = ", ".join(sorted(SUPPORTED_IRODORI_BACKENDS))
        raise ValueError(f"Unsupported Irodori backend {selected_backend!r}; choose one of: {expected}")

    selected_gpu = _validate_cuda_backend(selected_backend)
    ensure_ffmpeg_runtime(repo_root)
    selected_gpu_state = gpu_record(selected_gpu)
    worker_extras = _worker_extras(selected_backend, gpu=selected_gpu)
    runtime = repo_root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    transaction_marker = runtime / SETUP_TRANSACTION_MARKER
    transaction_state = {
        "schema_version": 1,
        "irodori_backend": selected_backend,
        "worker_backends": worker_extras,
        "selected_gpu": selected_gpu_state,
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

    if selected_gpu is not None:
        runtime_preflight = run_cuda_preflight(
            repo_root,
            irodori_project=irodori,
            gpu=selected_gpu,
            worker_backends=worker_extras,
        )
    else:
        runtime_preflight = {
            "ok": True,
            "skipped": True,
            "reason": f"backend {selected_backend} does not use the audited NVIDIA CUDA stack",
        }

    setup_state = {
        "irodori_backend": selected_backend,
        "worker_backends": worker_extras,
        "selected_gpu": selected_gpu_state,
        "irodori_revision": IRODORI_REVISION,
        "seed_vc_revision": SEED_VC_REVISION,
        "environment_contract": environment_contract(repo_root),
        "runtime_preflight": runtime_preflight,
        "model_assets": {
            "irodori_model_sha256": IRODORI_MODEL_SHA256,
            "irodori_dacvae_sha256": IRODORI_DACVAE_SHA256,
            "irodori_text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
            "lfm_revision": LFM_MODEL_REVISION,
            "lfm_model_sha256": LFM_MODEL_WEIGHT_SHA256,
            "lfm_asset_sha256": LFM_MODEL_ASSET_SHA256,
            "asr_revision": ASR_MODEL_REVISION,
            "asr_model_sha256": ASR_MODEL_WEIGHT_SHA256,
            "seed_vc_asset_contract_sha256": seed_vc_contract_digest(repo_root),
        },
        "prepare_assets": {
            "pyannote_revision": PYANNOTE_MODEL_REVISION,
            "pyannote_asset_sha256": PYANNOTE_MODEL_ASSET_SHA256,
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


def _snapshot_hashes_match(local_dir: Path, hashes: dict[str, str]) -> bool:
    try:
        return all(
            _nonempty_file(local_dir / relative)
            and sha256_file(local_dir / relative).lower() == expected.lower()
            for relative, expected in hashes.items()
        )
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
    sha256: dict[str, str] | None = None,
) -> bool:
    hashes = sha256 or {}
    unknown_hashes = sorted(set(hashes) - set(required_files))
    if unknown_hashes:
        raise ValueError(
            f"Pinned model hash contract for {model_id} contains undeclared files: "
            + ", ".join(unknown_hashes)
        )
    revision_path = local_dir / REVISION_MARKER
    current_revision = (
        revision_path.read_text(encoding="utf-8").strip()
        if revision_path.is_file()
        else None
    )
    complete = all(_nonempty_file(local_dir / relative) for relative in required_files)
    hashes_match = _snapshot_hashes_match(local_dir, hashes)
    if complete and hashes_match and current_revision == revision:
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
        shutil.rmtree(local_dir, ignore_errors=True)
        raise FileNotFoundError(
            f"Pinned model download for {model_id}@{revision} completed but "
            f"required files are missing or empty: {', '.join(missing)}"
        )
    if not _snapshot_hashes_match(local_dir, hashes):
        shutil.rmtree(local_dir, ignore_errors=True)
        raise RuntimeError(
            f"Pinned model download for {model_id}@{revision} failed the audited checksum contract"
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
        required_files=LFM_MODEL_REQUIRED_FILES,
        cache_dir=hub_cache,
        sha256=LFM_MODEL_ASSET_SHA256,
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
        sha256={"model.bin": ASR_MODEL_WEIGHT_SHA256},
    ):
        downloaded.append(f"{ASR_MODEL_ID}@{ASR_MODEL_REVISION}")
    else:
        reused.append(f"{ASR_MODEL_ID}@{ASR_MODEL_REVISION}")

    pyannote_dir = repo_root / "models" / "pyannote" / "community-1"
    pyannote_revision_marker = pyannote_dir / REVISION_MARKER
    pyannote_is_pinned = (
        all(_nonempty_file(pyannote_dir / name) for name in _PYANNOTE_REQUIRED_FILES)
        and _snapshot_hashes_match(pyannote_dir, PYANNOTE_MODEL_ASSET_SHA256)
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
        sha256=PYANNOTE_MODEL_ASSET_SHA256,
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
