from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


class CommandError(RuntimeError):
    pass


def run(
    args: Iterable[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    argv = [str(x) for x in args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=merged_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
        check=False,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise CommandError(f"Command failed: {' '.join(argv)}\n{detail}")
    return completed


def run_json(
    args: Iterable[str | Path], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> Any:
    completed = run(args, cwd=cwd, env=env, capture=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise CommandError(
        "Expected a JSON response from worker, but none was found.\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr or ''}"
    )
