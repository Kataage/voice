from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from personavoice.hardware import detect_irodori_backend
from personavoice.process import CommandError, run
from personavoice.workers import local_model_env, worker

IRODORI_REPO = "https://github.com/Aratako/Irodori-TTS.git"
IRODORI_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEED_VC_REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"


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
    run(
        ["uv", "sync", "--project", irodori, "--extra", selected_backend],
        cwd=repo_root,
    )
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


def _snapshot_if_missing(
    *,
    model_id: str,
    local_dir: Path,
    marker: Path,
    cache_dir: Path,
    token: str | None = None,
) -> bool:
    if marker.exists():
        return False
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        token=token,
    )
    if not marker.exists():
        raise FileNotFoundError(
            f"Model download for {model_id} completed but expected marker is missing: {marker}"
        )
    return True


def download_models(
    repo_root: Path,
    *,
    hf_token: str | None = None,
    include_seed_vc: bool = True,
) -> dict:
    env = local_model_env(repo_root, offline=False)
    cache_dir = Path(env["HF_HOME"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.getenv("HF_TOKEN")
    downloaded: list[str] = []
    reused: list[str] = []

    irodori_dir = repo_root / "models" / "irodori" / "v4.1-small"
    irodori_model = irodori_dir / "model.safetensors"
    if irodori_model.exists():
        reused.append("Aratako/Irodori-TTS-v4.1-Small:model.safetensors")
    else:
        irodori_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="Aratako/Irodori-TTS-v4.1-Small",
            filename="model.safetensors",
            local_dir=irodori_dir,
            cache_dir=cache_dir,
        )
        if not irodori_model.exists():
            raise FileNotFoundError(f"Expected Irodori checkpoint was not created: {irodori_model}")
        downloaded.append("Aratako/Irodori-TTS-v4.1-Small:model.safetensors")

    lfm_dir = repo_root / "models" / "lfm" / "base"
    if _snapshot_if_missing(
        model_id="LiquidAI/LFM2.5-1.2B-JP-202606",
        local_dir=lfm_dir,
        marker=lfm_dir / "config.json",
        cache_dir=cache_dir,
    ):
        downloaded.append("LiquidAI/LFM2.5-1.2B-JP-202606")
    else:
        reused.append("LiquidAI/LFM2.5-1.2B-JP-202606")

    asr_dir = repo_root / "models" / "asr" / "large-v3"
    if _snapshot_if_missing(
        model_id="Systran/faster-whisper-large-v3",
        local_dir=asr_dir,
        marker=asr_dir / "model.bin",
        cache_dir=cache_dir,
    ):
        downloaded.append("Systran/faster-whisper-large-v3")
    else:
        reused.append("Systran/faster-whisper-large-v3")

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
        cache_dir=cache_dir,
        token=token,
    ):
        downloaded.append("pyannote/speaker-diarization-community-1")
    else:
        reused.append("pyannote/speaker-diarization-community-1")

    # Irodori loads these by model id at runtime. Cache them once for offline use.
    for model_id in (
        "Aratako/Semantic-DACVAE-Japanese-32dim",
        "sbintuitions/modernbert-ja-310m",
    ):
        snapshot_download(repo_id=model_id, cache_dir=cache_dir)
        downloaded.append(model_id)

    sense_dir = repo_root / "models" / "sense" / "SenseVoiceSmall"
    if sense_dir.exists() and any(sense_dir.iterdir()):
        reused.append("iic/SenseVoiceSmall")
    else:
        worker(repo_root, "sense").call(
            repo_root,
            "download",
            {"online": True},
            offline=False,
        )
        downloaded.append("iic/SenseVoiceSmall")

    if include_seed_vc:
        # Seed-VC has several transitive pretrained assets. Loading the wrapper online once
        # is the upstream-compatible way to materialize all of them; deep doctor verifies
        # the exact same load path in offline mode afterwards.
        seed_marker = repo_root / ".runtime" / "seed-vc-models-ready"
        if seed_marker.exists():
            reused.append("Seed-VC-v2-default-checkpoints")
        else:
            worker(repo_root, "seed_vc").call(
                repo_root,
                "download",
                {"online": True},
                offline=False,
            )
            seed_marker.parent.mkdir(parents=True, exist_ok=True)
            seed_marker.write_text("ready\n", encoding="utf-8")
            downloaded.append("Seed-VC-v2-default-checkpoints")

    return {
        "downloaded": downloaded,
        "reused": reused,
        "cache": str(cache_dir),
    }
