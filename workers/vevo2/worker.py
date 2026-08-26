from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from uuid import uuid4

REVISION_MARKER = ".personavoice-revision"
EXPECTED_VENDOR_REVISION = "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a"
READY_MARKER = ".runtime/vevo2-models-ready"


def read_request(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Vevo2 worker request must be a JSON object")
    return value


def _root() -> Path:
    raw = os.environ.get("PERSONAVOICE_ROOT")
    if not raw:
        raise RuntimeError("PERSONAVOICE_ROOT is not set")
    return Path(raw).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise RuntimeError(f"{label} must use canonical forward-slash separators: {value!r}")
    relative = Path(value)
    windows = PureWindowsPath(value)
    if (
        relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in relative.parts
        or value.startswith(("/", "\\"))
    ):
        raise RuntimeError(f"{label} escapes the Vevo2 asset directory: {value!r}")
    normalized = relative.as_posix()
    if normalized in {"", "."}:
        raise RuntimeError(f"{label} must name an asset below the Vevo2 asset directory")
    return normalized


def _load_contract() -> dict:
    path = _root() / "config" / "vevo2_assets.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Vevo2 asset contract is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("Vevo2 asset contract schema_version must be 1")
    source = value.get("source")
    model = value.get("model")
    whisper = value.get("whisper")
    if not isinstance(source, dict) or not isinstance(model, dict) or not isinstance(whisper, dict):
        raise RuntimeError("Vevo2 asset contract is missing source/model/whisper provenance")
    if source.get("revision") != EXPECTED_VENDOR_REVISION:
        raise RuntimeError("Vevo2 source revision disagrees with the worker contract")
    revision = model.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Vevo2 model revision is not a full Git commit")
    local_dir = _canonical_relative(model.get("local_dir"), label="Vevo2 model.local_dir")
    required = model.get("required_files")
    hashes = model.get("sha256")
    if not isinstance(required, list) or not required or not isinstance(hashes, dict):
        raise RuntimeError("Vevo2 model required_files/sha256 contract is invalid")
    normalized = [_canonical_relative(item, label="Vevo2 model.required_files") for item in required]
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("Vevo2 model.required_files contains duplicates")
    normalized_hashes: dict[str, str] = {}
    for raw_relative, digest in hashes.items():
        relative = _canonical_relative(raw_relative, label="Vevo2 model.sha256 key")
        if relative not in normalized or not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise RuntimeError("Vevo2 model sha256 contract is invalid")
        normalized_hashes[relative] = digest
    if set(normalized_hashes) != set(normalized):
        raise RuntimeError("Vevo2 model sha256 contract does not cover every asset")
    whisper_relative = _canonical_relative(
        whisper.get("local_file"), label="Vevo2 whisper.local_file"
    )
    whisper_hash = whisper.get("sha256")
    if not isinstance(whisper_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", whisper_hash):
        raise RuntimeError("Vevo2 Whisper checksum contract is invalid")
    model["local_dir"] = local_dir
    model["required_files"] = normalized
    model["sha256"] = normalized_hashes
    whisper["local_file"] = whisper_relative
    return value


def _model_dir(contract: dict) -> Path:
    return _root() / "models" / "vevo2" / "assets" / str(contract["model"]["local_dir"])


def _whisper_path(contract: dict) -> Path:
    return _root() / "models" / "vevo2" / "assets" / str(contract["whisper"]["local_file"])


def _contract_digest() -> str:
    return _sha256(_root() / "config" / "vevo2_assets.json")


def _validate_assets(
    *,
    verify_hashes: bool,
    require_ready: bool = True,
) -> tuple[dict, Path, Path]:
    contract = _load_contract()
    model_dir = _model_dir(contract)
    errors: list[str] = []
    marker = model_dir / REVISION_MARKER
    recorded = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if recorded != contract["model"]["revision"]:
        errors.append("model revision marker mismatch")
    for relative in contract["model"]["required_files"]:
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing/empty model asset: {relative}")
            continue
        if verify_hashes:
            actual = _sha256(path)
            expected = contract["model"]["sha256"][relative]
            if actual != expected:
                errors.append(
                    f"model checksum mismatch for {relative}; expected {expected}, got {actual}"
                )
    whisper = _whisper_path(contract)
    if not whisper.is_file() or whisper.stat().st_size <= 0:
        errors.append("missing/empty Whisper medium asset")
    elif verify_hashes:
        actual = _sha256(whisper)
        expected = contract["whisper"]["sha256"]
        if actual != expected:
            errors.append(f"Whisper checksum mismatch; expected {expected}, got {actual}")
    if require_ready:
        ready = _root() / READY_MARKER
        ready_value = ready.read_text(encoding="utf-8").strip() if ready.is_file() else None
        if ready_value != _contract_digest():
            errors.append("Vevo2 ready marker does not match the asset contract")
    if errors:
        raise RuntimeError("Vevo2 pinned asset validation failed: " + "; ".join(errors))
    return contract, model_dir, whisper


def _setup_backend() -> str:
    path = _root() / ".runtime" / "setup.json"
    try:
        setup = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Vevo2 cannot read the audited setup state") from exc
    backends = setup.get("worker_backends") if isinstance(setup, dict) else None
    value = backends.get("vevo2") if isinstance(backends, dict) else None
    if value not in {"cpu", "cu124"}:
        raise RuntimeError(
            f"Vevo2 worker backend is not explicitly recorded as cpu/cu124: {value!r}. "
            "Run `persona setup --backend auto`."
        )
    return str(value)


def _vendor() -> Path:
    path = _root() / "vendor" / "Amphion"
    if not (path / ".git").exists() or not (path / "models" / "svc" / "vevo2" / "vevo2_utils.py").is_file():
        raise FileNotFoundError("Amphion/Vevo2 source is not installed. Run `persona setup` first.")
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Vevo2 source checkout integrity could not be verified") from exc
    if head != EXPECTED_VENDOR_REVISION:
        raise RuntimeError(
            f"Vevo2 source HEAD mismatch: expected {EXPECTED_VENDOR_REVISION}, got {head}"
        )
    if status:
        raise RuntimeError(
            "Vevo2 source checkout has local modifications or untracked files; "
            "restore the audited checkout before model work."
        )
    return path


def _device_and_dtype(requested_dtype: str) -> tuple[str, object, str]:
    import torch

    backend = _setup_backend()
    if requested_dtype not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported Vevo2 dtype {requested_dtype!r}; choose fp32 or fp16")
    if backend == "cu124":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Vevo2 is explicitly configured for cu124, but CUDA is unavailable. "
                "No CPU fallback is permitted; rerun setup after fixing the GPU runtime."
            )
        device = "cuda"
    else:
        device = "cpu"
    if requested_dtype == "fp16" and device != "cuda":
        raise RuntimeError("Vevo2 fp16 was requested with a CPU worker; use explicit fp32")
    torch_dtype = torch.float16 if requested_dtype == "fp16" else torch.float32
    return device, torch_dtype, backend


def _normalized_fmt_config(model_dir: Path) -> Path:
    import json5

    source = model_dir / "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json"
    config = json5.loads(source.read_text(encoding="utf-8"))
    try:
        config["model"]["coco"]["whisper_stats_path"] = str(
            (
                model_dir
                / "acoustic_modeling/fm_emilia101k_singnet7k_repa/whisper_stats.pt"
            ).resolve()
        )
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Vevo2 FM config has no model.coco.whisper_stats_path") from exc
    directory = _root() / ".runtime" / "vevo2"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "fm_emilia101k_singnet7k_repa.normalized.json"
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _load_pipeline(*, dtype_name: str = "fp32", require_ready: bool = True):
    contract, model_dir, whisper = _validate_assets(
        verify_hashes=True,
        require_ready=require_ready,
    )
    vendor = _vendor()
    device_name, _, backend = _device_and_dtype(dtype_name)

    fmt_config = _normalized_fmt_config(model_dir)
    old_cwd = Path.cwd()
    inserted = str(vendor)
    sys.path.insert(0, inserted)
    os.chdir(vendor)
    try:
        from models.svc.vevo2 import vevo2_utils

        original_load_model = vevo2_utils.whisper.load_model

        def load_local_model(_name, device=None, download_root=None, in_memory=None):
            del download_root, in_memory
            return original_load_model(str(whisper), device=device)

        vevo2_utils.whisper.load_model = load_local_model
        try:
            pipeline = vevo2_utils.Vevo2InferencePipeline(
                content_style_tokenizer_ckpt_path=str(
                    model_dir / "tokenizer/contentstyle_fvq16384_12.5hz"
                ),
                fmt_cfg_path=str(fmt_config),
                fmt_ckpt_path=str(
                    model_dir / "acoustic_modeling/fm_emilia101k_singnet7k_repa"
                ),
                vocoder_cfg_path=str(model_dir / "vocoder/config.json"),
                vocoder_ckpt_path=str(model_dir / "vocoder"),
                device=device_name,
            )
        finally:
            vevo2_utils.whisper.load_model = original_load_model
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
    if dtype_name == "fp16":
        # Keep this conversion explicit. Any unsupported layer/device error is
        # allowed to propagate; there is no automatic fp32 retry.
        pipeline._personavoice_dtype = "fp16"
    else:
        pipeline._personavoice_dtype = "fp32"
    pipeline._personavoice_device = device_name
    pipeline._personavoice_backend = backend
    pipeline._personavoice_contract = contract
    return pipeline


def _load_context(dtype_name: str):
    import torch

    pipeline = _load_pipeline(dtype_name=dtype_name)
    if dtype_name == "fp16":
        return pipeline, torch.autocast(device_type="cuda", dtype=torch.float16)
    return pipeline, nullcontext()


def _require_audio(payload: dict, key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Vevo2 {key} audio path is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Vevo2 {key} audio is missing or empty: {path}")
    return path


def _require_output_dir(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Vevo2 output_dir is required")
    directory = Path(value).expanduser().resolve()
    try:
        directory.relative_to(_root())
    except ValueError as exc:
        raise ValueError("Vevo2 output_dir must stay inside PERSONAVOICE_ROOT") from exc
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _runtime_result(*, pipeline, output: Path | None = None) -> dict:
    contract = pipeline._personavoice_contract
    result = {
        "ok": True,
        "backend": pipeline._personavoice_backend,
        "device": pipeline._personavoice_device,
        "cuda": pipeline._personavoice_device == "cuda",
        "dtype": pipeline._personavoice_dtype,
        "model_revision": contract["model"]["revision"],
        "asset_contract_sha256": _contract_digest(),
        "source_license": contract["source"]["license"],
        "model_license": contract["model"]["license"],
    }
    if output is not None:
        result["output"] = str(output)
    return result


def download(payload: dict) -> dict:
    del payload
    pipeline = _load_pipeline(dtype_name="fp32", require_ready=False)
    marker = _root() / READY_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_contract_digest() + "\n", encoding="utf-8")
    return _runtime_result(pipeline=pipeline)


def health(payload: dict) -> dict:
    deep = bool(payload.get("deep"))
    if deep:
        pipeline = _load_pipeline(dtype_name=str(payload.get("dtype") or "fp32"))
        result = _runtime_result(pipeline=pipeline)
        result["models_loaded"] = True
        return result
    contract, model_dir, whisper = _validate_assets(verify_hashes=False)
    requested_dtype = str(payload.get("dtype") or "fp32")
    device, _, backend = _device_and_dtype(requested_dtype)
    return {
        "ok": True,
        "backend": backend,
        "device": device,
        "cuda": device == "cuda",
        "dtype": requested_dtype,
        "model_revision": contract["model"]["revision"],
        "asset_contract_sha256": _contract_digest(),
        "model_dir": str(model_dir),
        "whisper": str(whisper),
        "models_loaded": False,
    }


def convert(payload: dict) -> dict:
    dtype_name = str(payload.get("dtype") or "fp32")
    source = _require_audio(payload, "source")
    target = _require_audio(payload, "target")
    output_dir = _require_output_dir(payload.get("output_dir"))
    steps = payload.get("flow_matching_steps", 32)
    if type(steps) is not int or not 1 <= steps <= 500:
        raise ValueError("Vevo2 flow_matching_steps must be an integer from 1 to 500")
    use_pitch_shift = payload.get("use_pitch_shift", False)
    if not isinstance(use_pitch_shift, bool):
        raise ValueError("Vevo2 use_pitch_shift must be boolean")

    pipeline, autocast = _load_context(dtype_name)
    output = output_dir / f"vevo2-fm-{uuid4().hex}.wav"
    old_cwd = Path.cwd()
    vendor = _vendor()
    sys.path.insert(0, str(vendor))
    os.chdir(vendor)
    try:
        import torch

        with torch.inference_mode(), autocast:
            generated = pipeline.inference_fm(
                src_wav_path=str(source),
                timbre_ref_wav_path=str(target),
                use_pitch_shift=use_pitch_shift,
                flow_matching_steps=steps,
                display_audio=False,
            )
        # The upstream FM pipeline returns a CPU waveform; save_audio performs
        # only the upstream 24 kHz output normalization.
        pipeline_module = sys.modules.get("models.svc.vevo2.vevo2_utils")
        if pipeline_module is None:
            raise RuntimeError("Vevo2 upstream utility module was not loaded")
        pipeline_module.save_audio(generated, output_path=str(output))
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == str(vendor):
            sys.path.pop(0)
    if not output.is_file() or output.stat().st_size <= 44:
        output.unlink(missing_ok=True)
        raise RuntimeError("Vevo2 completed without creating a valid WAV")
    result = _runtime_result(pipeline=pipeline, output=output)
    result.update(
        {
            "source_sha256": _sha256(source),
            "reference_sha256": _sha256(target),
            "flow_matching_steps": steps,
            "use_pitch_shift": use_pitch_shift,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["convert", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "convert":
        result = convert(payload)
    elif args.command == "download":
        result = download(payload)
    else:
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
