from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from personavoice.environment_contract import require_current_environment
from personavoice.model_assets import SEED_VC_SOURCE_REVISION
from personavoice.process import run, run_json
from personavoice.runtime_dependencies import ffmpeg_environment
from personavoice.worker_contracts import validate_worker_response


def _require_seed_vc_vendor_integrity(repo_root: Path) -> Path:
    """Fail closed before any Seed-VC worker process can import vendor code."""

    path = repo_root / "vendor" / "seed-vc"
    if not (path / "inference_v2.py").is_file() or not (path / ".git").exists():
        raise FileNotFoundError("Seed-VC is not installed. Run `persona setup` first.")
    try:
        head = run(["git", "rev-parse", "HEAD"], cwd=path, capture=True).stdout.strip()
        # Include non-ignored untracked files. A stray Python module inside the
        # vendor tree can affect import resolution just as a tracked edit can.
        status = run(["git", "status", "--porcelain"], cwd=path, capture=True).stdout.strip()
    except Exception as exc:
        raise RuntimeError("Seed-VC vendor checkout integrity could not be verified") from exc
    if head != SEED_VC_SOURCE_REVISION:
        raise RuntimeError(
            f"Seed-VC vendor HEAD mismatch: expected {SEED_VC_SOURCE_REVISION}, got {head}. "
            "Run `persona setup` to restore the audited checkout."
        )
    if status:
        raise RuntimeError(
            "Seed-VC vendor checkout has local modifications or non-ignored untracked files. "
            "Restore the audited checkout and run `persona setup` before model work."
        )
    return path


@dataclass(frozen=True)
class Worker:
    name: str
    project_dir: Path
    entrypoint: str = "worker.py"

    def call(
        self,
        repo_root: Path,
        command: str,
        payload: dict[str, Any],
        *,
        offline: bool = True,
    ) -> Any:
        # Every model worker executes from an isolated `.venv` with --no-sync.
        # Refuse to run it unless setup.json proves that environment was synced
        # from the exact dependency declarations and audited locks in this checkout.
        require_current_environment(repo_root)
        if self.name == "seed_vc":
            _require_seed_vc_vendor_integrity(repo_root)

        request_dir = repo_root / ".runtime" / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        request_path = request_dir / f"{self.name}-{uuid4().hex}.json"
        request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            env = local_model_env(repo_root, offline=offline)
            result = run_json(
                [
                    "uv",
                    "run",
                    "--project",
                    self.project_dir,
                    "--no-sync",
                    "python",
                    self.project_dir / self.entrypoint,
                    command,
                    "--request",
                    request_path,
                ],
                cwd=repo_root,
                env=env,
            )
            validate_worker_response(self.name, command, result)
            return result
        finally:
            request_path.unlink(missing_ok=True)

    def sync(self, repo_root: Path, *, extra: str | None = None) -> None:
        lockfile = self.project_dir / "uv.lock"
        if not lockfile.is_file():
            raise FileNotFoundError(
                f"Audited worker lockfile is missing for {self.name}: {lockfile}. "
                "Refusing an unlocked environment sync; restore the repository lockfile first."
            )
        args: list[str | Path] = [
            "uv",
            "sync",
            "--project",
            self.project_dir,
            "--locked",
        ]
        if extra:
            args.extend(["--extra", extra])
        run(args, cwd=repo_root)


def local_model_env(repo_root: Path, *, offline: bool = True) -> dict[str, str]:
    env = {
        "HF_HOME": str((repo_root / "models" / "hf-cache").resolve()),
        "HUGGINGFACE_HUB_CACHE": str(
            (repo_root / "models" / "hf-cache" / "hub").resolve()
        ),
        "MODELSCOPE_CACHE": str((repo_root / "models" / "modelscope-cache").resolve()),
        "PERSONAVOICE_ROOT": str(repo_root.resolve()),
        "TOKENIZERS_PARALLELISM": "false",
        # Make CUDA ordinal mapping deterministic across nvidia-smi/setup and all
        # model subprocesses. CUDA_VISIBLE_DEVICES is preserved by process.run.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        **ffmpeg_environment(),
    }
    if offline:
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    else:
        env.update({"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
    return env


def worker(repo_root: Path, name: str) -> Worker:
    project = repo_root / "workers" / name
    if not project.exists():
        raise FileNotFoundError(f"Worker project is missing: {project}")
    return Worker(name=name, project_dir=project)
