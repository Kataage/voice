from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from personavoice.hardware import detect_irodori_backend
from personavoice.media import sha256_file
from personavoice.model_assets import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_ID,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_ID,
    IRODORI_MODEL_SHA256,
    IRODORI_TEXT_ENCODER_ID,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_ID,
    LFM_MODEL_REVISION,
)
from personavoice.process import CommandError, run
from personavoice.workers import local_model_env, worker

IRODORI_REPO = "https://github.com/Aratako/Irodori-TTS.git"
IRODORI_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEED_VC_REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"
REVISION_MARKER = ".personavoice-revision"


def _clone_pinned(repo_root: Path, name: str, url: str, revision: str) -> Path:
    destination = repo_root / "vendor" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        run(["git", "clone", "--filter=blob:none", url, destination], cwd=repo_root)
    git_dir = destination / ".git"
    if not git_dir.exists():
        raise RuntimeError(f"{destination} exists but is not a git checkout")
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
    """Map the Irodori backend to compatible isolated worker backends.

    The modern Torch workers use CUDA 12.8. Archived Seed-VC is pinned to
    Torch 2.4.0, whose newest official Windows/Linux CUDA wheel is CUDA 12.4.
    ROCm/XPU support is currently limited to Irodori; the other workers use CPU
    rather than silently resolving an arbitrary PyPI Torch build.
    """

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
    """Sync Irodori from our audited lock without dirtying the pinned checkout."""

    managed_lock = repo_root / "locks" / "Irodori-TTS.uv.lock"
    args: list[str | Path] = ["uv", "sync", "--project", irodori, "--extra", backend]
    if not managed_lock.exists():
        run(args, cwd=repo_root)
        return

    vendor_lock = irodori / "uv.lock"
    original = vendor_lock.read_bytes() if vendor_lock.exists() else None
    try:
        shutil.copy2(managed_lock, vendor_lock)
        run([*args, "--locked"], cwd=repo_root)
    finally:
        if original is None:
            vendor_lock.unlink(missing_ok=True)
        else:
            vendor_lock.write_bytes(original)


def install_environments(repo_root: Path, *, backend: str | None = None) -> dict:
    if not shutil.which("uv"):
        raise RuntimeError("uv was not found in PATH")
    if not shutil.which("git"):
        raise RuntimeError("git was not found in PATH")
    irodori = _clone_pinned(
        repo_root,
        "Irodori-TTS",
        IRODORI_REPO,
        IRODORI_REVISION,
    )
    seed = _clone_pinned(repo_root, "seed-vc", SEED_VC_REPO, SEED_VC_REVISION)
    selected_backend = backend or detect_irodori_backend()
    _install_irodori(repo_root, irodori, selected_backend)

    worker_extras = _worker_extras(selected_backend)
    synced = []
    for name in ("asr", "diarization", "sense", "lfm", "seed_vc"):
        worker(repo_root, name).sync(repo_root, extra=worker_extras[name])
        synced.append(name)

    runtime = repo_root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    setup_state = {
        "irodori_backend": selected_backend,
        "worker_backends": worker_extras,
        "irodori_revision": IRODORI_REVISION,
        "seed_vc_revision": SEED_VC_REVISION,
        "model_assets": {
            "irodori_model_sha256": IRODORI_MODEL_SHA256,
            "irodori_dacvae_sha256": IRODORI_DACVAE_SHA256,
            "irodori_text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
            "lfm_revision": LFM_MODEL_REVISION,
            "asr_revision": ASR_MODEL_REVISION,
        },
    }
    (runtime / "setup.json").write_text(
        json.dumps(setup_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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


def _download_verified_file(
    *,
    model_id: str,
    filename: str,
    local_dir: Path,
    cache_dir: Path,
    sha256: str,
) -> tuple[Path, bool]:
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / filename
    existed = target.exists()
    if not existed:
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=local_dir,
            cache_dir=cache_dir,
        )
    if not target.is_file():
        raise FileNotFoundError(f"Expected model file was not created: {target}")
    _verify_sha256(target, sha256, label=f"{model_id}:{filename}")
    return target, existed


def _snapshot_if_missing(
    *,
    model_id: str,
    local_dir: Path,
    marker: Path,
    cache_dir: Path,
    token: str | None = None,
    revision: str | None = None,
) -> bool:
    if marker.exists():
        return False
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        cache_dir=cache_dir,
        token=token,
    )
    if not marker.exists():
        raise FileNotFoundError(
            f"Model download for {model_id} completed but expected marker is missing: {marker}"
        )
    return True


def _snapshot_pinned(
    *,
    model_id: str,
    revision: str,
    local_dir: Path,
    required_file: str,
    cache_dir: Path,
) -> bool:
    revision_path = local_dir / REVISION_MARKER
    required_path = local_dir / required_file
    current_revision = (
        revision_path.read_text(encoding="utf-8").strip()
        if revision_path.is_file()
        else None
    )
    if required_path.is_file() and current_revision == revision:
        return False

    # An unmarked directory came from the older floating-revision setup. Remove
    # only PersonaVoice's materialized view; Hugging Face's shared cache is kept,
    # so unchanged blobs can still be reused without another network transfer.
    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Pinned model download for {model_id}@{revision} completed but "
            f"expected file is missing: {required_path}"
        )
    revision_path.write_text(revision + "\n", encoding="utf-8")
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
        local_dir=dacvae_dir,
        cache_dir=hub_cache,
        sha256=IRODORI_DACVAE_SHA256,
    )
    (reused if existed else downloaded).append(
        f"{IRODORI_DACVAE_ID}:{IRODORI_DACVAE_FILENAME}"
    )

    # The pinned Irodori v4 training configuration explicitly references this
    # ModernBERT commit. Ensuring it on every setup is cheap when cached and
    # guarantees that offline training can resolve the exact revision.
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
        required_file="config.json",
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
        required_file="model.bin",
        cache_dir=hub_cache,
    ):
        downloaded.append(f"{ASR_MODEL_ID}@{ASR_MODEL_REVISION}")
    else:
        reused.append(f"{ASR_MODEL_ID}@{ASR_MODEL_REVISION}")

    pyannote_dir = repo_root / "models" / "pyannote" / "community-1"
    pyannote_marker = pyannote_dir / "config.yaml"
    if not pyannote_marker.exists() and not token:
        raise RuntimeError(
            "HF_TOKEN is required for the first download of "
            "pyannote/speaker-diarization-community-1. Accept its Hugging Face usage terms, "
            "then set HF_TOKEN in the environment. The token is never stored by PersonaVoice."
        )
    if _snapshot_if_missing(
        model_id="pyannote/speaker-diarization-community-1",
        local_dir=pyannote_dir,
        marker=pyannote_marker,
        cache_dir=hub_cache,
        token=token,
    ):
        downloaded.append("pyannote/speaker-diarization-community-1")
    else:
        reused.append("pyannote/speaker-diarization-community-1")

    runtime = repo_root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    sense_marker = runtime / "sense-model-ready"
    if sense_marker.exists():
        reused.append("iic/SenseVoiceSmall")
    else:
        worker(repo_root, "sense").call(
            repo_root,
            "download",
            {"online": True},
            offline=False,
        )
        sense_marker.write_text("ready\n", encoding="utf-8")
        downloaded.append("iic/SenseVoiceSmall")

    if include_seed_vc:
        # Seed-VC has several transitive pretrained assets. Loading the wrapper online once
        # is the upstream-compatible way to materialize all of them; deep doctor verifies
        # the exact same load path in offline mode afterwards.
        seed_marker = runtime / "seed-vc-models-ready"
        if seed_marker.exists():
            reused.append("Seed-VC-v2-default-checkpoints")
        else:
            worker(repo_root, "seed_vc").call(
                repo_root,
                "download",
                {"online": True},
                offline=False,
            )
            seed_marker.write_text("ready\n", encoding="utf-8")
            downloaded.append("Seed-VC-v2-default-checkpoints")

    return {
        "downloaded": downloaded,
        "reused": reused,
        "hf_home": str(hf_home),
        "hub_cache": str(hub_cache),
    }
