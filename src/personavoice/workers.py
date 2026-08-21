from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from personavoice.process import run, run_json


@dataclass(frozen=True)
class Worker:
    name: str
    project_dir: Path
    entrypoint: str = "worker.py"

    def call(self, repo_root: Path, command: str, payload: dict[str, Any], *, offline: bool = True) -> Any:
        request_dir = repo_root / ".runtime" / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        request_path = request_dir / f"{self.name}-{os.getpid()}.json"
        request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            env = local_model_env(repo_root, offline=offline)
            return run_json(
                [
                    "uv", "run", "--project", self.project_dir, "--no-sync", "python",
                    self.project_dir / self.entrypoint, command, "--request", request_path,
                ],
                cwd=repo_root,
                env=env,
            )
        finally:
            request_path.unlink(missing_ok=True)

    def sync(self, repo_root: Path) -> None:
        run(["uv", "sync", "--project", self.project_dir], cwd=repo_root)


def local_model_env(repo_root: Path, *, offline: bool = True) -> dict[str, str]:
    env = {
        "HF_HOME": str((repo_root / "models" / "hf-cache").resolve()),
        "MODELSCOPE_CACHE": str((repo_root / "models" / "modelscope-cache").resolve()),
        "PERSONAVOICE_ROOT": str(repo_root.resolve()),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if offline:
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return env


def worker(repo_root: Path, name: str) -> Worker:
    project = repo_root / "workers" / name
    if not project.exists():
        raise FileNotFoundError(f"Worker project is missing: {project}")
    return Worker(name=name, project_dir=project)
