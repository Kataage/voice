from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from personavoice.hardware import detect_irodori_backend
from personavoice.process import CommandError, run
from personavoice.workers import local_model_env, worker

IRODORI_REPO = "https://github.com/Aratako/Irodori-TTS.git"
IRODORI_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEED_VC_REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"

HF_MODELS = (
    "Aratako/Irodori-TTS-v4.1-Small",
    "Aratako/Semantic-DACVAE-Japanese-32dim",
    "sbintuitions/modernbert-ja-310m",
    "LiquidAI/LFM2.5-1.2B-JP-202606",
    "Systran/faster-whisper-large-v3",
)


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
        run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=destination, capture=True)
    except CommandError:
        run(["git", "fetch", "--depth", "1", "origin", revision], cwd=destination)
    run(["git", "checkout", "--detach", revision], cwd=destination)
    return destination


def install_environments(repo_root: Path, *, backend: str | None = None) -> dict:
    if not shutil.which("uv"):
        raise RuntimeError("uv was not found in PATH")
    if not shutil.which("git"):
        raise RuntimeError("git was not found in PATH")
    irodori = _clone_pinned(repo_root, "Irodori-TTS", IRODORI_REPO, IRODORI_REVISION)
    seed = _clone_pinned(repo_root, "seed-vc", SEED_VC_REPO, SEED_VC_REVISION)
    selected_backend = backend or detect_irodori_backend()
    run(["uv", "sync", "--project", irodori, "--extra", selected_backend], cwd=repo_root)
    synced = []
    for name in ("asr", "diarization", "sense", "lfm", "seed_vc"):
        worker(repo_root, name).sync(repo_root)
        synced.append(name)
    return {
        "irodori_revision": IRODORI_REVISION,
        "seed_vc_revision": SEED_VC_REVISION,
        "irodori_backend": selected_backend,
        "workers": synced,
        "vendor": {"irodori": str(irodori), "seed_vc": str(seed)},
    }


def download_models(repo_root: Path, *, hf_token: str | None = None, include_seed_vc: bool = True) -> dict:
    env = local_model_env(repo_root, offline=False)
    cache_dir = Path(env["HF_HOME"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.getenv("HF_TOKEN")
    local_models = {
        "Aratako/Irodori-TTS-v4.1-Small": repo_root / "models" / "irodori" / "v4.1-small",
        "LiquidAI/LFM2.5-1.2B-JP-202606": repo_root / "models" / "lfm" / "base",
        "Systran/faster-whisper-large-v3": repo_root / "models" / "asr" / "large-v3",
        "pyannote/speaker-diarization-community-1": repo_root / "models" / "pyannote" / "community-1",
    }
    downloaded = []
    for model_id, local_dir in local_models.items():
        snapshot_download(
            repo_id=model_id, local_dir=local_dir, cache_dir=cache_dir,
            token=token if model_id.startswith("pyannote/") else None,
        )
        downloaded.append(model_id)
    for model_id in ("Aratako/Semantic-DACVAE-Japanese-32dim", "sbintuitions/modernbert-ja-310m"):
        snapshot_download(repo_id=model_id, cache_dir=cache_dir)
        downloaded.append(model_id)
    worker(repo_root, "sense").call(repo_root, "download", {"online": True}, offline=False)
    downloaded.append("iic/SenseVoiceSmall")
    if include_seed_vc:
        worker(repo_root, "seed_vc").call(repo_root, "download", {"online": True}, offline=False)
        downloaded.append("Seed-VC-v2-default-checkpoints")
    return {"downloaded": downloaded, "cache": str(cache_dir)}
