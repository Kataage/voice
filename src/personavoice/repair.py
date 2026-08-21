from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def _is_materialization_failure(health: Any) -> bool:
    if not isinstance(health, dict) or health.get("ok") is not False:
        return False
    error = str(health.get("error") or "")
    # doctor adds this error after a worker health call succeeds. Re-downloading
    # model files cannot fix an unavailable CUDA runtime/driver.
    return not ("was installed for" in error and "runtime cannot see CUDA" in error)


def repair_failed_model_materializations(
    repo_root: Path,
    verification: dict[str, Any],
    *,
    include_seed_vc: bool,
) -> list[str]:
    """Discard only local model views that failed deep offline verification.

    The shared Hugging Face/ModelScope caches remain intact, so a subsequent
    explicit `download_models` call can normally relink/reuse healthy blobs.
    This function never performs network I/O itself and deliberately does not
    treat backend/CUDA visibility failures as model corruption.
    """

    repaired: list[str] = []
    worker_health = verification.get("worker_health")
    worker_health = worker_health if isinstance(worker_health, dict) else {}

    worker_materializations = {
        "asr": repo_root / "models" / "asr" / "large-v3",
        "lfm": repo_root / "models" / "lfm" / "base",
        "diarization": repo_root / "models" / "pyannote" / "community-1",
        "sense": repo_root / "models" / "sense" / "SenseVoiceSmall",
    }
    for name, path in worker_materializations.items():
        if _is_materialization_failure(worker_health.get(name)):
            shutil.rmtree(path, ignore_errors=True)
            if name == "sense":
                (repo_root / ".runtime" / "sense-model-ready").unlink(missing_ok=True)
            repaired.append(name)

    asset_integrity = verification.get("model_asset_integrity")
    asset_integrity = asset_integrity if isinstance(asset_integrity, dict) else {}
    asset_error = str(asset_integrity.get("error") or "")
    if "Irodori checkpoint checksum mismatch" in asset_error:
        shutil.rmtree(repo_root / "models" / "irodori" / "v4.1-small", ignore_errors=True)
        repaired.append("irodori")
    if "Irodori DACVAE checksum mismatch" in asset_error:
        shutil.rmtree(repo_root / "models" / "irodori" / "dacvae", ignore_errors=True)
        repaired.append("irodori_dacvae")
    if "Irodori base checkpoint or DACVAE checkpoint is missing" in asset_error:
        # Removing both materialized views is safe: download_models verifies the
        # two pinned hashes and reuses the shared cache where possible.
        shutil.rmtree(repo_root / "models" / "irodori" / "v4.1-small", ignore_errors=True)
        shutil.rmtree(repo_root / "models" / "irodori" / "dacvae", ignore_errors=True)
        repaired.extend(name for name in ("irodori", "irodori_dacvae") if name not in repaired)

    if include_seed_vc and _is_materialization_failure(worker_health.get("seed_vc")):
        seed_marker = repo_root / ".runtime" / "seed-vc-models-ready"
        # The Seed-VC worker removes this marker when its offline wrapper/model
        # load fails. Do not purge vendor/cache state here; the explicit download
        # path knows how to materialize all transitive upstream checkpoints.
        if not seed_marker.exists():
            repaired.append("seed_vc")

    return repaired
