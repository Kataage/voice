from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def vendor() -> Path:
    path = Path(os.environ["PERSONAVOICE_ROOT"]) / "vendor" / "seed-vc"
    if not (path / "inference_v2.py").exists():
        raise FileNotFoundError("Seed-VC is not installed. Run `persona setup` first.")
    return path


def download(payload: dict) -> dict:
    root = vendor()
    sys.path.insert(0, str(root))
    old = Path.cwd()
    os.chdir(root)
    try:
        import torch
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        cfg = DictConfig(yaml.safe_load((root / "configs/v2/vc_wrapper.yaml").read_text(encoding="utf-8")))
        wrapper = instantiate(cfg)
        wrapper.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
        return {"ok": True, "cuda": torch.cuda.is_available()}
    finally:
        os.chdir(old)


def convert(payload: dict) -> dict:
    root = vendor()
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("*.wav"))
    args = [
        sys.executable,
        str(root / "inference_v2.py"),
        "--source", payload["source"],
        "--target", payload["target"],
        "--output", str(output_dir),
        "--diffusion-steps", str(payload.get("diffusion_steps", 30)),
        "--length-adjust", str(payload.get("length_adjust", 1.0)),
        "--intelligibility-cfg-rate", str(payload.get("intelligibility_cfg_rate", 0.7)),
        "--similarity-cfg-rate", str(payload.get("similarity_cfg_rate", 0.7)),
        "--top-p", str(payload.get("top_p", 0.9)),
        "--temperature", str(payload.get("temperature", 1.0)),
        "--repetition-penalty", str(payload.get("repetition_penalty", 1.0)),
        "--convert-style", "true" if payload.get("convert_style", True) else "false",
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
        result = {"ok": True}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
