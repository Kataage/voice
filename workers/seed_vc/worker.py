from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root() -> Path:
    return Path(os.environ["PERSONAVOICE_ROOT"])


def _ready_marker() -> Path:
    return _root() / ".runtime" / "seed-vc-models-ready"


def vendor() -> Path:
    path = _root() / "vendor" / "seed-vc"
    if not (path / "inference_v2.py").exists():
        raise FileNotFoundError("Seed-VC is not installed. Run `persona setup` first.")
    return path


def load_default_wrapper() -> dict:
    root = vendor()
    sys.path.insert(0, str(root))
    old = Path.cwd()
    os.chdir(root)
    try:
        import torch
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        cfg = DictConfig(
            yaml.safe_load((root / "configs/v2/vc_wrapper.yaml").read_text(encoding="utf-8"))
        )
        wrapper = instantiate(cfg)
        wrapper.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
        return {
            "ok": True,
            "cuda": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "models_loaded": True,
        }
    finally:
        os.chdir(old)


def download(payload: dict) -> dict:
    result = load_default_wrapper()
    marker = _ready_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ready\n", encoding="utf-8")
    return result


def convert(payload: dict) -> dict:
    root = vendor()
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("*.wav"))
    args = [
        sys.executable,
        str(root / "inference_v2.py"),
        "--source",
        payload["source"],
        "--target",
        payload["target"],
        "--output",
        str(output_dir),
        "--diffusion-steps",
        str(payload.get("diffusion_steps", 30)),
        "--length-adjust",
        str(payload.get("length_adjust", 1.0)),
        "--intelligibility-cfg-rate",
        str(payload.get("intelligibility_cfg_rate", 0.7)),
        "--similarity-cfg-rate",
        str(payload.get("similarity_cfg_rate", 0.7)),
        "--top-p",
        str(payload.get("top_p", 0.9)),
        "--temperature",
        str(payload.get("temperature", 1.0)),
        "--repetition-penalty",
        str(payload.get("repetition_penalty", 1.0)),
        "--convert-style",
        "true" if payload.get("convert_style", True) else "false",
    ]
    if payload.get("ar_checkpoint"):
        args += ["--ar-checkpoint-path", payload["ar_checkpoint"]]
    if payload.get("cfm_checkpoint"):
        args += ["--cfm-checkpoint-path", payload["cfm_checkpoint"]]
    completed = subprocess.run(args, cwd=root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Seed-VC inference failed with exit code {completed.returncode}")
    created = [path for path in output_dir.glob("*.wav") if path not in before]
    if not created:
        raise RuntimeError("Seed-VC completed but no output wav was created")
    result = max(created, key=lambda path: path.stat().st_mtime_ns)
    return {"output": str(result.resolve())}


def health(payload: dict) -> dict:
    import torch

    result = {
        "ok": True,
        "vendor": str(vendor()),
        "cuda": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if payload.get("deep"):
        try:
            result.update(load_default_wrapper())
        except Exception:
            # A readiness marker is only a cache hint. If the transitive HF
            # assets were deleted/corrupted, invalidate it so the next
            # `persona setup` re-materializes them instead of looping forever.
            _ready_marker().unlink(missing_ok=True)
            raise
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
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
