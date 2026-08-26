from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from personavoice.config import TrainingConfig
from personavoice.environment import SECRET_ENV_KEYS
from personavoice.environment_contract import (
    environment_contract_status,
    require_current_environment,
    runtime_hardware_status,
)
from personavoice.executors import inspect_local_resources, preflight_local_full
from personavoice.hardware import hardware_report
from personavoice.lineage import backend_status, domain_backend_audit
from personavoice.media import sha256_file
from personavoice.modal_transport import detect_modal_auth
from personavoice.model_assets import (
    ASR_MODEL_REVISION,
    ASR_MODEL_WEIGHT_SHA256,
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_SHA256,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_ASSET_SHA256,
    LFM_MODEL_REQUIRED_FILES,
    LFM_MODEL_REVISION,
    LFM_MODEL_WEIGHT_SHA256,
    PYANNOTE_MODEL_ASSET_SHA256,
    PYANNOTE_MODEL_REVISION,
    QWEN_ASR_MODEL_LFS_OIDS,
    QWEN_ASR_MODEL_REQUIRED_FILES,
    QWEN_ASR_MODEL_REVISION,
    QWEN_FORCED_ALIGNER_MODEL_LFS_OIDS,
    QWEN_FORCED_ALIGNER_MODEL_REQUIRED_FILES,
    QWEN_FORCED_ALIGNER_MODEL_REVISION,
    SENSE_MODEL_CMVN_SHA256,
    SENSE_MODEL_TOKENIZER_SHA256,
    SENSE_MODEL_WEIGHT_SHA256,
    VEVO2_MODEL_ID,
    VEVO2_MODEL_LICENSE,
    VEVO2_MODEL_REVISION,
    VEVO2_SOURCE_LICENSE,
    VEVO2_SOURCE_REVISION,
)
from personavoice.process import run
from personavoice.runtime_dependencies import ffmpeg_runtime
from personavoice.seed_vc_assets import materialization_status as seed_vc_materialization_status
from personavoice.separation import separator_model_audit
from personavoice.setup_env import IRODORI_REVISION, REVISION_MARKER, SEED_VC_REVISION
from personavoice.training_plan import FamilyPlan, TrainingPlan
from personavoice.vevo2_assets import materialization_status as vevo2_materialization_status
from personavoice.workers import local_model_env, worker

WORKER_NAMES = ("asr", "diarization", "sense", "lfm", "seed_vc", "vevo2")
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
_SENSE_REQUIRED_FILES = (
    "model.pt",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
)

_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_token",
        "hf_token",
        "modal_token_id",
        "modal_token_secret",
        "password",
        "token",
        "token_id",
        "token_secret",
    }
)


def _secret_field(key: object) -> bool:
    folded = str(key).casefold().replace("-", "_")
    return (
        folded in _SECRET_FIELD_NAMES
        or folded == "authorization"
        or folded.startswith("authorization_")
        or folded.startswith("token_")
        or folded.endswith("_token")
        or "credential" in folded
        or "password" in folded
        or "secret" in folded
    )


def _without_secret_values(
    value: Any,
    *,
    secret_values: tuple[str, ...] | None = None,
) -> Any:
    """Omit secret fields and redact known process secrets wherever they occur."""

    if secret_values is None:
        secret_values = tuple(
            candidate for key in SECRET_ENV_KEYS if (candidate := os.environ.get(key, ""))
        )

    if isinstance(value, dict):
        return {
            str(key): _without_secret_values(child, secret_values=secret_values)
            for key, child in value.items()
            if not _secret_field(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_secret_values(child, secret_values=secret_values) for child in value]
    if isinstance(value, str) and any(secret in value for secret in secret_values):
        return "[redacted]"
    return value


def _preflight_plan(training: TrainingConfig) -> TrainingPlan:
    def family(name: str, *, enabled: bool, method: str, settings: dict[str, Any]) -> FamilyPlan:
        return FamilyPlan(
            family=name,
            enabled=enabled,
            method=method,
            dataset_fingerprint="preflight-only",
            training=settings,
            model_contract={},
            implementation_contract={},
            checkpoint_policy={},
            evaluation_policy={},
        )

    return TrainingPlan(
        persona="doctor-preflight",
        files=(),
        families=(
            family(
                "irodori",
                enabled=training.irodori.enabled,
                method=training.irodori.method,
                settings=training.irodori.model_dump(mode="json"),
            ),
            family(
                "lfm",
                enabled=training.lfm.enabled,
                method=training.lfm.method,
                settings=training.lfm.model_dump(mode="json"),
            ),
            family(
                "seed-vc",
                enabled=training.seed_vc.finetune,
                method="finetune",
                settings=training.seed_vc.model_dump(mode="json"),
            ),
        ),
    )


def training_preflight_status(
    repo_root: Path,
    training: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Report the same conservative local full-training admission check used by train."""

    setup = _setup_state(repo_root)
    setup_error: str | None = None
    try:
        current_setup = require_current_environment(repo_root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        setup_current = False
        setup_error = f"{type(exc).__name__}: {exc}"
    else:
        setup_current = True
        setup = current_setup
    backend = str(setup.get("irodori_backend") or "unknown")
    resources = inspect_local_resources(
        repo_root,
        backend=backend,
        setup_current=setup_current,
    )
    selected_training = TrainingConfig() if training is None else training
    preflight = preflight_local_full(_preflight_plan(selected_training), resources)
    result = {
        "ok": preflight.ok,
        "setup_current": setup_current,
        "setup_error": setup_error,
        "requested_full_families": list(preflight.full_families),
        "resources": asdict(resources),
        "requirements": {
            "gpu_total_mib": preflight.required_gpu_total_mib,
            "gpu_free_mib": preflight.required_gpu_free_mib,
            "ram_available_bytes": preflight.required_ram_available_bytes,
            "disk_free_bytes": preflight.required_disk_free_bytes,
        },
        "failures": [asdict(item) for item in preflight.failures],
    }
    return _without_secret_values(result)


def modal_readiness_status() -> dict[str, Any]:
    """Inspect only SDK presence and credential configuration; never contact Modal."""

    try:
        sdk_installed = find_spec("modal") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        sdk_installed = False
    auth = detect_modal_auth()
    return {
        "ready": sdk_installed and auth.configured,
        "sdk_installed": sdk_installed,
        "auth_configured": auth.configured,
        "auth_source": auth.source,
        "network_probe_performed": False,
    }


def _setup_state(repo_root: Path) -> dict:
    path = repo_root / ".runtime" / "setup.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _expected_worker_backend(name: str, setup: dict) -> str | None:
    if name == "asr":
        return "runtime-auto" if setup.get("irodori_backend") in {"cu126", "cu128"} else "cpu"
    backends = setup.get("worker_backends")
    value = backends.get(name) if isinstance(backends, dict) else None
    return None if value is None else str(value)


def _requires_cuda(value: str | None) -> bool:
    return bool(value and (value == "cuda" or value.startswith("cu")))


def _irodori_device(backend: str | None) -> str:
    if backend in {"cu126", "cu128", "rocm"}:
        return "cuda"
    if backend == "xpu":
        return "xpu"
    return "cpu"


def _vendor_integrity(repo_root: Path, relative: str, expected_revision: str) -> dict:
    directory = repo_root / "vendor" / relative
    if not (directory / ".git").exists():
        return {
            "ok": False,
            "expected_revision": expected_revision,
            "error": "vendor checkout is missing or is not a git repository",
        }
    try:
        head = run(["git", "rev-parse", "HEAD"], cwd=directory, capture=True).stdout.strip()
        status = run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture=True,
        ).stdout.strip()
    except Exception as exc:
        return {
            "ok": False,
            "expected_revision": expected_revision,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": head == expected_revision and not status,
        "head": head,
        "expected_revision": expected_revision,
        "clean": not status,
        "status": status,
    }


def _read_revision(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None
    except OSError:
        return None


def _read_integrity_ids(path: Path) -> dict[str, str] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        return None
    return value


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _model_asset_integrity(
    repo_root: Path,
    setup: dict,
    *,
    deep: bool,
    require_seed_vc: bool,
    seed_vc_status: dict,
    require_vevo2: bool,
    vevo2_status: dict,
) -> dict:
    expected_setup = {
        "irodori_model_sha256": IRODORI_MODEL_SHA256,
        "irodori_dacvae_sha256": IRODORI_DACVAE_SHA256,
        "irodori_text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
        "lfm_revision": LFM_MODEL_REVISION,
        "lfm_model_sha256": LFM_MODEL_WEIGHT_SHA256,
        "lfm_asset_sha256": LFM_MODEL_ASSET_SHA256,
        "asr_revision": ASR_MODEL_REVISION,
        "asr_model_sha256": ASR_MODEL_WEIGHT_SHA256,
        "qwen_asr_revision": QWEN_ASR_MODEL_REVISION,
        "qwen_forced_aligner_revision": QWEN_FORCED_ALIGNER_MODEL_REVISION,
        "seed_vc_asset_contract_sha256": seed_vc_status.get("contract_sha256"),
        "vevo2_source_revision": VEVO2_SOURCE_REVISION,
        "vevo2_model_id": VEVO2_MODEL_ID,
        "vevo2_model_revision": VEVO2_MODEL_REVISION,
        "vevo2_asset_contract_sha256": vevo2_status.get("contract_sha256"),
        "vevo2_source_license": VEVO2_SOURCE_LICENSE,
        "vevo2_model_license": VEVO2_MODEL_LICENSE,
    }
    expected_prepare = {
        "pyannote_revision": PYANNOTE_MODEL_REVISION,
        "pyannote_asset_sha256": PYANNOTE_MODEL_ASSET_SHA256,
        "sense_weight_sha256": SENSE_MODEL_WEIGHT_SHA256,
        "sense_cmvn_sha256": SENSE_MODEL_CMVN_SHA256,
        "sense_tokenizer_sha256": SENSE_MODEL_TOKENIZER_SHA256,
    }
    recorded = setup.get("model_assets") if isinstance(setup.get("model_assets"), dict) else {}
    recorded_prepare = (
        setup.get("prepare_assets") if isinstance(setup.get("prepare_assets"), dict) else {}
    )
    irodori = repo_root / "models" / "irodori" / "v4.1-small" / IRODORI_MODEL_FILENAME
    dacvae = repo_root / "models" / "irodori" / "dacvae" / IRODORI_DACVAE_FILENAME
    lfm_revision = _read_revision(repo_root / "models" / "lfm" / "base" / REVISION_MARKER)
    asr_revision = _read_revision(repo_root / "models" / "asr" / "large-v3" / REVISION_MARKER)
    qwen_dir = repo_root / "models" / "asr" / "qwen3-asr-1.7b"
    qwen_aligner_dir = repo_root / "models" / "asr" / "qwen3-forced-aligner-0.6b"
    qwen_revision = _read_revision(qwen_dir / REVISION_MARKER)
    qwen_aligner_revision = _read_revision(qwen_aligner_dir / REVISION_MARKER)
    qwen_integrity_ids = _read_integrity_ids(qwen_dir / "integrity_ids.json")
    qwen_aligner_integrity_ids = _read_integrity_ids(
        qwen_aligner_dir / "integrity_ids.json"
    )
    pyannote_revision = _read_revision(
        repo_root / "models" / "pyannote" / "community-1" / REVISION_MARKER
    )
    sense_marker = _read_revision(repo_root / ".runtime" / "sense-model-ready")

    result = {
        "ok": True,
        "setup_matches": recorded == expected_setup,
        "prepare_setup_matches": recorded_prepare == expected_prepare,
        "expected": expected_setup,
        "recorded": recorded,
        "expected_prepare": expected_prepare,
        "recorded_prepare": recorded_prepare,
        "lfm_revision": lfm_revision,
        "asr_revision": asr_revision,
        "qwen_asr_revision": qwen_revision,
        "qwen_forced_aligner_revision": qwen_aligner_revision,
        "qwen_asr_integrity_ids": qwen_integrity_ids,
        "qwen_forced_aligner_integrity_ids": qwen_aligner_integrity_ids,
        "pyannote_revision": pyannote_revision,
        "sense_verified_marker": sense_marker,
        "seed_vc": seed_vc_status,
        "vevo2": vevo2_status,
        "irodori_sha256": None,
        "dacvae_sha256": None,
        "lfm_asset_sha256": None,
        "asr_backend": str(setup.get("asr_backend") or "whisper-large-v3"),
        "asr_backend_status": backend_status(
            str(setup.get("asr_backend") or "whisper-large-v3")
        ),
        "domain_backend_audit": domain_backend_audit(),
    }
    errors = []
    if recorded != expected_setup:
        errors.append("setup model asset pins do not match this PersonaVoice revision")
    if recorded_prepare != expected_prepare:
        errors.append("setup preparation asset pins do not match this PersonaVoice revision")
    if not _nonempty_file(irodori) or not _nonempty_file(dacvae):
        errors.append("Irodori base checkpoint or DACVAE checkpoint is missing/empty")
    if lfm_revision != LFM_MODEL_REVISION:
        errors.append("LFM materialized revision does not match the audited revision")
    selected_asr = str(setup.get("asr_backend") or "whisper-large-v3")
    selected_asr_status = backend_status(selected_asr)
    if selected_asr_status.get("enabled") is not True:
        errors.append(
            "configured ASR backend is disabled: "
            + str(selected_asr_status.get("reason") or selected_asr)
        )
    elif selected_asr == "qwen3-asr-1.7b":
        if qwen_revision != QWEN_ASR_MODEL_REVISION:
            errors.append("Qwen ASR materialized revision does not match the audited revision")
        if qwen_aligner_revision != QWEN_FORCED_ALIGNER_MODEL_REVISION:
            errors.append(
                "Qwen forced aligner materialized revision does not match the audited revision"
            )
        if qwen_integrity_ids != QWEN_ASR_MODEL_LFS_OIDS:
            errors.append("Qwen ASR LFS integrity evidence is missing or stale")
        if qwen_aligner_integrity_ids != QWEN_FORCED_ALIGNER_MODEL_LFS_OIDS:
            errors.append("Qwen forced aligner LFS integrity evidence is missing or stale")
    elif asr_revision != ASR_MODEL_REVISION:
        errors.append("ASR materialized revision does not match the audited revision")
    if pyannote_revision != PYANNOTE_MODEL_REVISION:
        errors.append("pyannote materialized revision does not match the audited revision")
    if sense_marker != "verified":
        errors.append("SenseVoice assets have not been verified by the current setup")
    if require_seed_vc and not bool(seed_vc_status.get("ok")):
        seed_errors = seed_vc_status.get("errors")
        if isinstance(seed_errors, list) and seed_errors:
            errors.extend(f"Seed-VC: {value}" for value in seed_errors)
        else:
            errors.append("Seed-VC pinned assets are incomplete or stale")
    if require_vevo2 and not bool(vevo2_status.get("ok")):
        vevo_errors = vevo2_status.get("errors")
        if isinstance(vevo_errors, list) and vevo_errors:
            errors.extend(f"Vevo2: {value}" for value in vevo_errors)
        else:
            errors.append("Vevo2 pinned assets are incomplete or stale")

    if deep and _nonempty_file(irodori) and _nonempty_file(dacvae):
        try:
            irodori_sha = sha256_file(irodori)
            dacvae_sha = sha256_file(dacvae)
        except OSError as exc:
            errors.append(f"model checksum read failed: {exc}")
        else:
            result["irodori_sha256"] = irodori_sha
            result["dacvae_sha256"] = dacvae_sha
            if irodori_sha != IRODORI_MODEL_SHA256:
                errors.append("Irodori checkpoint checksum mismatch")
            if dacvae_sha != IRODORI_DACVAE_SHA256:
                errors.append("Irodori DACVAE checksum mismatch")
    lfm_dir = repo_root / "models" / "lfm" / "base"
    if deep and all(_nonempty_file(lfm_dir / name) for name in LFM_MODEL_REQUIRED_FILES):
        try:
            lfm_hashes = {
                name: sha256_file(lfm_dir / name) for name in LFM_MODEL_REQUIRED_FILES
            }
        except OSError as exc:
            errors.append(f"LFM asset checksum read failed: {exc}")
        else:
            result["lfm_asset_sha256"] = lfm_hashes
            if lfm_hashes != LFM_MODEL_ASSET_SHA256:
                errors.append("LFM base asset checksum contract mismatch")
    if errors:
        result["ok"] = False
        result["errors"] = errors
        result["error"] = errors[0]
    return result


def _irodori_health(repo_root: Path, setup: dict) -> dict:
    vendor = repo_root / "vendor" / "Irodori-TTS"
    checkpoint = repo_root / "models" / "irodori" / "v4.1-small" / IRODORI_MODEL_FILENAME
    codec = repo_root / "models" / "irodori" / "dacvae" / IRODORI_DACVAE_FILENAME
    backend = setup.get("irodori_backend")
    if backend is None:
        return {"ok": False, "error": "Irodori backend is not recorded; run `persona setup`"}
    if not (vendor / "infer.py").is_file():
        return {"ok": False, "error": "Irodori vendor checkout is missing"}
    if not _nonempty_file(checkpoint):
        return {"ok": False, "error": "Irodori base checkpoint is missing/empty"}
    if not _nonempty_file(codec):
        return {"ok": False, "error": "Irodori DACVAE checkpoint is missing/empty"}

    device = _irodori_device(str(backend))
    output_dir = repo_root / ".runtime" / "doctor"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "irodori-smoke.wav"
    output.unlink(missing_ok=True)
    try:
        run(
            [
                "uv",
                "run",
                "--project",
                vendor,
                "--no-sync",
                "python",
                vendor / "infer.py",
                "--checkpoint",
                checkpoint,
                "--codec-repo",
                codec,
                "--text",
                "動作確認です。",
                "--no-ref",
                "--output-wav",
                output,
                "--num-steps",
                "1",
                "--seconds",
                "0.5",
                "--num-candidates",
                "1",
                "--model-device",
                device,
                "--codec-device",
                device,
                "--model-precision",
                "fp32",
                "--codec-precision",
                "fp32",
                "--no-show-timings",
            ],
            cwd=vendor,
            env=local_model_env(repo_root, offline=True),
            capture=True,
        )
        if not _nonempty_file(output) or output.stat().st_size <= 44:
            return {"ok": False, "device": device, "error": "Irodori produced no valid WAV"}
        return {"ok": True, "device": device, "model_loaded": True, "smoke_inference": True}
    finally:
        output.unlink(missing_ok=True)


def report(
    repo_root: Path,
    *,
    deep: bool = False,
    require_seed_vc: bool = True,
    require_vevo2: bool = False,
) -> dict:
    ffmpeg_status = ffmpeg_runtime()
    required = {
        "uv": shutil.which("uv"),
        "git": shutil.which("git"),
        "ffmpeg": ffmpeg_status.ffmpeg,
        "ffprobe": ffmpeg_status.ffprobe,
    }
    commands_ok = bool(required["uv"] and required["git"] and ffmpeg_status.torchcodec_compatible)
    runtime = repo_root / ".runtime"
    setup = _setup_state(repo_root)
    selected_asr = str(setup.get("asr_backend") or "whisper-large-v3")
    lfm_dir = repo_root / "models" / "lfm" / "base"
    asr_dir = repo_root / "models" / "asr" / "large-v3"
    qwen_dir = repo_root / "models" / "asr" / "qwen3-asr-1.7b"
    qwen_aligner_dir = repo_root / "models" / "asr" / "qwen3-forced-aligner-0.6b"
    pyannote_dir = repo_root / "models" / "pyannote" / "community-1"
    sense_dir = repo_root / "models" / "sense" / "SenseVoiceSmall"
    seed_vc_status = seed_vc_materialization_status(repo_root, verify_hashes=deep)
    vevo2_status = vevo2_materialization_status(repo_root, verify_hashes=deep)
    separator_status = separator_model_audit(repo_root)
    models = {
        "irodori": _nonempty_file(repo_root / "models/irodori/v4.1-small/model.safetensors"),
        "irodori_dacvae": _nonempty_file(repo_root / "models/irodori/dacvae/weights.pth"),
        "lfm": all(_nonempty_file(lfm_dir / name) for name in LFM_MODEL_REQUIRED_FILES),
        "asr": all(_nonempty_file(asr_dir / name) for name in _ASR_REQUIRED_FILES),
        "asr_qwen": (
            all(_nonempty_file(qwen_dir / name) for name in QWEN_ASR_MODEL_REQUIRED_FILES)
            and _read_revision(qwen_dir / REVISION_MARKER) == QWEN_ASR_MODEL_REVISION
            and _read_integrity_ids(qwen_dir / "integrity_ids.json") == QWEN_ASR_MODEL_LFS_OIDS
        ),
        "asr_forced_aligner": (
            all(
                _nonempty_file(qwen_aligner_dir / name)
                for name in QWEN_FORCED_ALIGNER_MODEL_REQUIRED_FILES
            )
            and _read_revision(qwen_aligner_dir / REVISION_MARKER)
            == QWEN_FORCED_ALIGNER_MODEL_REVISION
            and _read_integrity_ids(qwen_aligner_dir / "integrity_ids.json")
            == QWEN_FORCED_ALIGNER_MODEL_LFS_OIDS
        ),
        "pyannote": all(_nonempty_file(pyannote_dir / name) for name in _PYANNOTE_REQUIRED_FILES),
        "sense": all(_nonempty_file(sense_dir / name) for name in _SENSE_REQUIRED_FILES)
        and _read_revision(runtime / "sense-model-ready") == "verified",
        "seed_vc_models": bool(seed_vc_status.get("ok")),
        "vevo2_models": bool(vevo2_status.get("ok")),
        # Separator weights are optional analysis assets.  Their absence does
        # not make a normal speech-only setup unusable, but the audit remains
        # visible so `auto`/`always` failures are never silent.
        "separator_model": bool(separator_status.get("materialized")),
        "seed_vc_vendor": _nonempty_file(repo_root / "vendor/seed-vc/inference_v2.py"),
        "vevo2_vendor": _nonempty_file(
            repo_root / "vendor/Amphion/models/svc/vevo2/vevo2_utils.py"
        ),
        "irodori_vendor": _nonempty_file(repo_root / "vendor/Irodori-TTS/infer.py"),
    }
    workers = {name: (repo_root / "workers" / name / ".venv").is_dir() for name in WORKER_NAMES}
    active_workers = tuple(
        name
        for name in WORKER_NAMES
        if (require_seed_vc or name != "seed_vc")
        and (require_vevo2 or name != "vevo2")
    )
    environment = environment_contract_status(repo_root, setup.get("environment_contract"))
    runtime_hardware = runtime_hardware_status(setup)
    seed_vc_runtime_hardware = (
        runtime_hardware_status(setup, worker_name="seed_vc")
        if require_seed_vc
        else {"ok": True, "skipped": True, "reason": "Seed-VC is not required"}
    )
    vevo2_runtime_hardware = (
        runtime_hardware_status(setup, worker_name="vevo2")
        if require_vevo2
        else {"ok": True, "skipped": True, "reason": "Vevo2 is not required"}
    )
    model_assets = _model_asset_integrity(
        repo_root,
        setup,
        deep=deep,
        require_seed_vc=require_seed_vc,
        seed_vc_status=seed_vc_status,
        require_vevo2=require_vevo2,
        vevo2_status=vevo2_status,
    )
    vendor_integrity = {
        "irodori": _vendor_integrity(repo_root, "Irodori-TTS", IRODORI_REVISION),
        "seed_vc": _vendor_integrity(repo_root, "seed-vc", SEED_VC_REVISION),
        "vevo2": _vendor_integrity(repo_root, "Amphion", VEVO2_SOURCE_REVISION),
    }
    worker_health: dict[str, dict] = {}
    if deep:
        for name in active_workers:
            if name == "asr" and backend_status(selected_asr).get("enabled") is not True:
                worker_health[name] = {
                    "ok": False,
                    "error": "configured ASR backend is disabled; no fallback is allowed",
                    "backend": selected_asr,
                }
                continue
            if not workers[name]:
                worker_health[name] = {"ok": False, "error": "worker .venv is missing"}
                continue
            try:
                health = worker(repo_root, name).call(
                    repo_root,
                    "health",
                    (
                        {
                            "deep": True,
                            "model": "qwen3-asr-1.7b",
                            "device": "auto",
                            "dtype": "auto",
                        }
                        if name == "asr" and selected_asr == "qwen3-asr-1.7b"
                        else {"deep": True, "model": "large-v3", "compute_type": "auto"}
                    ),
                )
                expected = _expected_worker_backend(name, setup)
                health["expected_backend"] = expected
                if _requires_cuda(expected) and not bool(health.get("cuda")):
                    health["ok"] = False
                    health["error"] = (
                        f"{name} was installed for {expected}, but its runtime cannot see CUDA. "
                        "Re-run `persona setup` and verify the NVIDIA driver."
                    )
                worker_health[name] = health
            except Exception as exc:
                worker_health[name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "expected_backend": _expected_worker_backend(name, setup),
                }
        if not runtime_hardware.get("ok"):
            worker_health["irodori"] = {
                "ok": False,
                "error": str(runtime_hardware.get("error")),
            }
        else:
            try:
                worker_health["irodori"] = _irodori_health(repo_root, setup)
            except Exception as exc:
                worker_health["irodori"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    lockfiles = {
        "root": (repo_root / "uv.lock").is_file(),
        **{name: (repo_root / "workers" / name / "uv.lock").is_file() for name in WORKER_NAMES},
        "irodori_managed": (repo_root / "locks" / "Irodori-TTS.uv.lock").is_file(),
    }
    required_model_keys = {
        "irodori",
        "irodori_dacvae",
        "lfm",
        "pyannote",
        "sense",
        "irodori_vendor",
    }
    if selected_asr == "qwen3-asr-1.7b":
        required_model_keys.update({"asr_qwen", "asr_forced_aligner"})
    else:
        required_model_keys.add("asr")
    required_vendor_keys = {"irodori"}
    required_worker_keys = set(active_workers)
    required_lock_keys = {"root", "asr", "diarization", "sense", "lfm", "irodori_managed"}
    if require_seed_vc:
        required_model_keys.update({"seed_vc_models", "seed_vc_vendor"})
        required_vendor_keys.add("seed_vc")
        required_lock_keys.add("seed_vc")
    if require_vevo2:
        required_model_keys.update({"vevo2_models", "vevo2_vendor"})
        required_vendor_keys.add("vevo2")
        required_lock_keys.add("vevo2")

    locks_ready = all(lockfiles[key] for key in required_lock_keys)
    vendors_ready = all(vendor_integrity[key].get("ok") for key in required_vendor_keys)
    reproducible = (
        locks_ready
        and bool(model_assets.get("ok"))
        and bool(environment.get("ok"))
        and bool(runtime_hardware.get("ok"))
        and bool(seed_vc_runtime_hardware.get("ok"))
        and bool(vevo2_runtime_hardware.get("ok"))
    )
    base_ready = (
        commands_ok
        and bool(setup)
        and reproducible
        and all(models[key] for key in required_model_keys)
        and all(workers[key] for key in required_worker_keys)
        and vendors_ready
    )
    deep_ready = all(bool(value.get("ok")) for value in worker_health.values()) if deep else True
    local_training_preflight = training_preflight_status(repo_root)
    modal = modal_readiness_status()
    result = {
        "python": sys.version.split()[0],
        "commands": required,
        "commands_ok": commands_ok,
        "ffmpeg_runtime": ffmpeg_status.as_dict(),
        "hardware": hardware_report(),
        "setup": _without_secret_values(setup),
        "environment_contract": environment,
        "runtime_hardware": runtime_hardware,
        "seed_vc_runtime_hardware": seed_vc_runtime_hardware,
        "vevo2_runtime_hardware": vevo2_runtime_hardware,
        "models": models,
        "asr_backend": selected_asr,
        "asr_backend_status": backend_status(selected_asr),
        "domain_backend_audit": domain_backend_audit(),
        "separator_model_audit": separator_status,
        "model_asset_integrity": model_assets,
        "workers": workers,
        "worker_health": worker_health if deep else None,
        "vendor_integrity": vendor_integrity,
        "lockfiles": lockfiles,
        "reproducible_environment": reproducible,
        "ready_offline": base_ready and deep_ready,
        "seed_vc_required": require_seed_vc,
        "vevo2_required": require_vevo2,
        "local_training_preflight": local_training_preflight,
        "modal": modal,
    }
    return _without_secret_values(result)
