from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from uuid import uuid4

REVISION_MARKER = ".personavoice-revision"
EXPECTED_VENDOR_REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root() -> Path:
    return Path(os.environ["PERSONAVOICE_ROOT"]).resolve()


def _contract_path() -> Path:
    return _root() / "config" / "seed_vc_assets.json"


def _asset_root() -> Path:
    return _root() / "models" / "seed_vc" / "assets"


def _ready_marker() -> Path:
    return _root() / ".runtime" / "seed-vc-models-ready"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_digest() -> str:
    return _sha256(_contract_path())


def _canonical_relative(value, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise RuntimeError(f"{label} must use canonical forward-slash separators: {value!r}")
    relative = Path(value)
    windows = PureWindowsPath(value)
    if relative.is_absolute() or windows.is_absolute() or windows.drive or ".." in relative.parts:
        raise RuntimeError(f"{label} escapes the Seed-VC asset directory: {value!r}")
    normalized = relative.as_posix()
    if normalized in {"", "."}:
        raise RuntimeError(f"{label} must name a path below the asset directory")
    return normalized


def _load_contract() -> dict:
    path = _contract_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Seed-VC asset contract is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("Seed-VC asset contract schema_version must be 1")
    snapshots = value.get("snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        raise RuntimeError("Seed-VC asset contract contains no snapshots")

    seen_dirs: set[str] = set()
    for name, snapshot in snapshots.items():
        if not isinstance(name, str) or not name or not isinstance(snapshot, dict):
            raise RuntimeError("Seed-VC asset contract contains an invalid snapshot entry")
        repo_id = snapshot.get("repo_id")
        revision = snapshot.get("revision")
        local_dir = _canonical_relative(snapshot.get("local_dir"), label=f"{name}.local_dir")
        required = snapshot.get("required_files")
        hashes = snapshot.get("sha256") or {}
        if not isinstance(repo_id, str) or "/" not in repo_id:
            raise RuntimeError(f"{name}.repo_id is invalid")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"{name}.revision must be a full 40-character Git commit")
        if local_dir in seen_dirs:
            raise RuntimeError(f"duplicate Seed-VC local_dir: {local_dir}")
        seen_dirs.add(local_dir)
        if not isinstance(required, list) or not required or not isinstance(hashes, dict):
            raise RuntimeError(f"{name}.required_files/sha256 contract is invalid")
        normalized = [
            _canonical_relative(item, label=f"{name}.required_files") for item in required
        ]
        if len(normalized) != len(set(normalized)):
            raise RuntimeError(f"{name}.required_files contains duplicates")
        normalized_hashes: dict[str, str] = {}
        for raw_relative, digest in hashes.items():
            relative = _canonical_relative(raw_relative, label=f"{name}.sha256 key")
            if relative not in normalized:
                raise RuntimeError(f"{name}.sha256 references undeclared asset: {raw_relative}")
            if relative in normalized_hashes:
                raise RuntimeError(f"{name}.sha256 contains duplicate canonical paths")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError(f"{name}.sha256[{raw_relative!r}] is invalid")
            normalized_hashes[relative] = digest
        snapshot["local_dir"] = local_dir
        snapshot["required_files"] = normalized
        snapshot["sha256"] = normalized_hashes
    return value


def _snapshot_dir(snapshot: dict) -> Path:
    local_dir = _canonical_relative(snapshot.get("local_dir"), label="snapshot.local_dir")
    return (_asset_root() / local_dir).resolve()


def _validate_assets(*, verify_hashes: bool) -> dict[str, Path]:
    contract = _load_contract()
    directories: dict[str, Path] = {}
    errors: list[str] = []
    for name, snapshot in contract["snapshots"].items():
        directory = _snapshot_dir(snapshot)
        directories[name] = directory
        revision = snapshot["revision"]
        marker = directory / REVISION_MARKER
        try:
            recorded = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
        except OSError:
            recorded = None
        if recorded != revision:
            errors.append(f"{name}: revision marker mismatch")
        hashes = snapshot["sha256"]
        for raw_relative in snapshot["required_files"]:
            relative = _canonical_relative(raw_relative, label=f"{name}.required file")
            path = directory / relative
            try:
                valid = path.is_file() and path.stat().st_size > 0
            except OSError:
                valid = False
            if not valid:
                errors.append(f"{name}: missing/empty required asset: {raw_relative}")
                continue
            expected = hashes.get(relative)
            if verify_hashes and isinstance(expected, str):
                actual = _sha256(path)
                if actual != expected:
                    errors.append(
                        f"{name}: checksum mismatch for {raw_relative}; "
                        f"expected {expected}, got {actual}"
                    )
    if errors:
        raise RuntimeError("Seed-VC pinned asset validation failed: " + "; ".join(errors))
    return directories


def _atomic_ready_marker(value: str) -> None:
    marker = _ready_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    temp = marker.with_name(f".{marker.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(value + "\n", encoding="utf-8")
        temp.replace(marker)
    finally:
        temp.unlink(missing_ok=True)


def _ready_marker_matches_contract() -> bool:
    marker = _ready_marker()
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == _contract_digest()
    except OSError:
        return False


def vendor() -> Path:
    path = _root() / "vendor" / "seed-vc"
    if not (path / "inference_v2.py").is_file() or not (path / ".git").exists():
        raise FileNotFoundError("Seed-VC is not installed. Run `persona setup` first.")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Seed-VC vendor checkout integrity could not be verified") from exc
    if head != EXPECTED_VENDOR_REVISION:
        raise RuntimeError(
            f"Seed-VC vendor HEAD mismatch: expected {EXPECTED_VENDOR_REVISION}, got {head}. "
            "Run `persona setup` to restore the audited checkout."
        )
    if status:
        raise RuntimeError(
            "Seed-VC vendor checkout has tracked local modifications. "
            "Run `persona setup` after restoring the audited checkout."
        )
    return path


def _checkpoint(path: str | None, default: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve() if path else default.resolve()
    try:
        valid = candidate.is_file() and candidate.stat().st_size > 0
    except OSError:
        valid = False
    if not valid:
        raise FileNotFoundError(f"{label} checkpoint is missing or empty: {candidate}")
    return candidate


def _cpu_autocast_override(original_autocast):
    """Make Seed-VC's internal CPU autocast scopes explicit no-ops.

    The pinned Seed-VC wrapper opens ``torch.autocast`` internally. Torch 2.4
    disables CPU autocast when passed float32, but emits a warning and makes the
    behavior dependent on AMP policy. PersonaVoice deliberately executes the CPU
    fallback in plain fp32 instead. Non-CPU autocast calls are delegated unchanged.
    """

    def replacement(*args, **kwargs):
        device_type = kwargs.get("device_type")
        if device_type is None and args:
            device_type = args[0]
        if device_type == "cpu":
            return nullcontext()
        return original_autocast(*args, **kwargs)

    return replacement


def _declared_file(snapshot: dict, directory: Path, value, *, label: str) -> Path:
    relative = _canonical_relative(value, label=label)
    required = snapshot.get("required_files")
    if not isinstance(required, list) or relative not in required:
        raise RuntimeError(f"Seed-VC attempted undeclared local asset access: {label}={relative}")
    candidate = directory / relative
    try:
        valid = candidate.is_file() and candidate.stat().st_size > 0
    except OSError:
        valid = False
    if not valid:
        raise FileNotFoundError(f"Pinned Seed-VC asset is missing: {candidate}")
    return candidate


def _load_wrapper(
    *,
    ar_checkpoint: str | None = None,
    cfm_checkpoint: str | None = None,
    verify_hashes: bool,
):
    directories = _validate_assets(verify_hashes=verify_hashes)
    required_keys = {"seed_vc", "astral", "campplus", "whisper_small", "hubert", "bigvgan"}
    missing_keys = sorted(required_keys - set(directories))
    if missing_keys:
        raise RuntimeError(f"Seed-VC asset contract is missing snapshots: {', '.join(missing_keys)}")

    root = vendor()
    inserted = str(root)
    sys.path.insert(0, inserted)
    old = Path.cwd()
    os.chdir(root)
    try:
        import modules.v2.vc_wrapper as vc_module
        import torch
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        cfg_data = yaml.safe_load((root / "configs/v2/vc_wrapper.yaml").read_text(encoding="utf-8"))
        if not isinstance(cfg_data, dict):
            raise RuntimeError("Seed-VC v2 wrapper config is invalid")

        whisper = str(directories["whisper_small"])
        hubert = str(directories["hubert"])
        bigvgan = str(directories["bigvgan"])
        for key in ("content_extractor_narrow", "content_extractor_wide"):
            cfg_data[key]["tokenizer_name"] = whisper
            cfg_data[key]["ssl_model_name"] = hubert
        cfg_data["vocoder"]["pretrained_model_name_or_path"] = bigvgan

        contract = _load_contract()
        repo_to_snapshot = {
            str(snapshot["repo_id"]): (name, snapshot)
            for name, snapshot in contract["snapshots"].items()
            if isinstance(snapshot, dict) and isinstance(snapshot.get("repo_id"), str)
        }

        def local_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
            mapped = repo_to_snapshot.get(str(repo_id))
            if mapped is None:
                raise RuntimeError(
                    f"Seed-VC attempted undeclared Hugging Face access: {repo_id}:{model_filename}"
                )
            name, snapshot = mapped
            directory = directories[name]
            model_path = _declared_file(
                snapshot,
                directory,
                model_filename,
                label=f"{name}.model_filename",
            )
            if config_filename is None:
                return str(model_path)
            config_path = _declared_file(
                snapshot,
                directory,
                config_filename,
                label=f"{name}.config_filename",
            )
            return str(model_path), str(config_path)

        original_resolver = vc_module.load_custom_model_from_hf
        vc_module.load_custom_model_from_hf = local_model_from_hf
        try:
            wrapper = instantiate(DictConfig(cfg_data))
            seed_dir = directories["seed_vc"]
            wrapper.load_checkpoints(
                ar_checkpoint_path=str(
                    _checkpoint(
                        ar_checkpoint,
                        seed_dir / "v2" / "ar_base.pth",
                        label="Seed-VC AR",
                    )
                ),
                cfm_checkpoint_path=str(
                    _checkpoint(
                        cfm_checkpoint,
                        seed_dir / "v2" / "cfm_small.pth",
                        label="Seed-VC CFM",
                    )
                ),
            )
        finally:
            vc_module.load_custom_model_from_hf = original_resolver

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        wrapper.to(device)
        wrapper.eval()
        wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
        return wrapper, torch, device, dtype
    finally:
        os.chdir(old)
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)


def load_default_wrapper(*, verify_hashes: bool = False) -> dict:
    _wrapper, torch, device, dtype = _load_wrapper(verify_hashes=verify_hashes)
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "device": device.type,
        "dtype": str(dtype),
        "models_loaded": True,
        "asset_contract_sha256": _contract_digest(),
    }


def download(payload: dict) -> dict:
    # Root setup performs the only networked materialization. This command proves
    # that the resulting snapshots are complete and can load with no repo-ID fallback.
    result = load_default_wrapper(verify_hashes=True)
    _atomic_ready_marker(_contract_digest())
    return result


def convert(payload: dict) -> dict:
    import soundfile as sf

    if not _ready_marker_matches_contract():
        raise RuntimeError(
            "Seed-VC pinned assets are not finalized for this repository contract. "
            "Run `persona setup` first."
        )
    # Re-hash every inference-critical pinned weight before use. Seed-VC inference
    # is expensive enough that this small deterministic I/O cost is preferable to
    # silently accepting same-size corruption after setup.
    wrapper, torch, device, dtype = _load_wrapper(
        ar_checkpoint=payload.get("ar_checkpoint"),
        cfm_checkpoint=payload.get("cfm_checkpoint"),
        verify_hashes=True,
    )
    source = Path(payload["source"]).expanduser().resolve()
    target = Path(payload["target"]).expanduser().resolve()
    for label, path in (("source", source), ("target", target)):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Seed-VC {label} audio is missing or empty: {path}")

    output_dir = Path(payload["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"seed-vc-{uuid4().hex}.wav"
    full_audio = None

    original_autocast = torch.autocast
    if device.type == "cpu":
        torch.autocast = _cpu_autocast_override(original_autocast)
    try:
        generator = wrapper.convert_voice_with_streaming(
            source_audio_path=str(source),
            target_audio_path=str(target),
            diffusion_steps=int(payload.get("diffusion_steps", 30)),
            length_adjust=float(payload.get("length_adjust", 1.0)),
            intelligebility_cfg_rate=float(payload.get("intelligibility_cfg_rate", 0.7)),
            similarity_cfg_rate=float(payload.get("similarity_cfg_rate", 0.7)),
            top_p=float(payload.get("top_p", 0.9)),
            temperature=float(payload.get("temperature", 1.0)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            convert_style=bool(payload.get("convert_style", True)),
            anonymization_only=bool(payload.get("anonymization_only", False)),
            device=device,
            dtype=dtype,
            stream_output=True,
        )
        for _chunk, candidate in generator:
            if candidate is not None:
                full_audio = candidate
    finally:
        torch.autocast = original_autocast

    if not isinstance(full_audio, tuple) or len(full_audio) != 2:
        raise RuntimeError("Seed-VC completed without a final audio result")
    sample_rate, audio = full_audio
    sf.write(output, audio, int(sample_rate))
    if not output.is_file() or output.stat().st_size <= 44:
        output.unlink(missing_ok=True)
        raise RuntimeError("Seed-VC completed but no valid output WAV was created")
    return {"output": str(output)}


def health(payload: dict) -> dict:
    import torch

    try:
        _validate_assets(verify_hashes=bool(payload.get("deep")))
        if not _ready_marker_matches_contract():
            raise RuntimeError("Seed-VC ready marker does not match the pinned asset contract")
        result = {
            "ok": True,
            "vendor": str(vendor()),
            "cuda": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "asset_contract_sha256": _contract_digest(),
        }
        if payload.get("deep"):
            result.update(load_default_wrapper(verify_hashes=True))
        return result
    except Exception:
        _ready_marker().unlink(missing_ok=True)
        raise


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
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
