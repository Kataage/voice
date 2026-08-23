from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class CommandError(RuntimeError):
    pass


# ASR normally emits a checkpoint/progress heartbeat at source start and while
# faster-whisper yields segments. Twenty minutes without any progress is long
# enough to cover model checksum/load and unusually slow decoding while still
# preventing a native CUDA/CTranslate2 stall from blocking a build forever.
ASR_STALL_TIMEOUT_SECONDS = 20 * 60
ASR_PROGRESS_POLL_SECONDS = 2.0


def _merged_environment(env: dict[str, str] | None) -> dict[str, str]:
    merged = os.environ.copy()
    if env:
        merged.update({key: str(value) for key, value in env.items()})
    return merged


def _command_error(argv: list[str], completed: subprocess.CompletedProcess[str]) -> CommandError:
    stderr = completed.stderr.strip() if completed.stderr else ""
    stdout = completed.stdout.strip() if completed.stdout else ""
    detail = stderr or stdout or f"exit code {completed.returncode}"
    return CommandError(f"Command failed: {' '.join(argv)}\n{detail}")


def run(
    args: Iterable[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    argv = [str(value) for value in args]
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=_merged_environment(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
        check=False,
    )
    if check and completed.returncode != 0:
        raise _command_error(argv, completed)
    return completed


def _request_argument(argv: list[str]) -> Path | None:
    try:
        index = argv.index("--request")
        raw = argv[index + 1]
    except (ValueError, IndexError):
        return None
    return Path(raw)


def _asr_progress_path(argv: list[str], cwd: Path | None) -> Path | None:
    """Return a trusted progress path only for PersonaVoice ASR batch requests."""

    if "batch_transcribe" not in argv:
        return None
    request_path = _request_argument(argv)
    if request_path is None or not request_path.name.startswith("asr-"):
        return None
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_checkpoint = payload.get("checkpoint_dir")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        return None
    checkpoint = Path(raw_checkpoint)
    if not checkpoint.is_absolute():
        return None

    root = (cwd or Path.cwd()).resolve()
    resolved = checkpoint.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None
    parts = relative.parts
    if (
        len(parts) != 5
        or parts[0] != "personas"
        or parts[2] != "cache"
        or parts[3] != "asr"
        or parts[4] != ".checkpoints"
    ):
        return None
    return resolved / "progress.json"


def _progress_signature(path: Path) -> bytes | None:
    """Use the tiny progress document itself as the atomic heartbeat signature."""

    try:
        return path.read_bytes()
    except OSError:
        return None


def _progress_summary(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "progress metadata unavailable"
    if not isinstance(value, dict):
        return "progress metadata unavailable"
    fields = []
    phase = value.get("phase")
    if isinstance(phase, str) and phase:
        fields.append(f"phase={phase}")
    current_id = value.get("current_id")
    if isinstance(current_id, str) and current_id:
        fields.append(f"current_id={current_id}")
    completed = value.get("completed")
    total = value.get("total")
    if isinstance(completed, int) and isinstance(total, int):
        fields.append(f"completed={completed}/{total}")
    device = value.get("device")
    compute_type = value.get("compute_type")
    if isinstance(device, str) and device:
        fields.append(f"device={device}")
    if isinstance(compute_type, str) and compute_type:
        fields.append(f"compute_type={compute_type}")
    processed = value.get("current_processed_seconds")
    if isinstance(processed, (int, float)) and not isinstance(processed, bool):
        fields.append(f"processed={float(processed):.1f}s")
    return ", ".join(fields) or "progress metadata unavailable"


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Best-effort crash-safe cleanup of uv plus all worker descendants."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_json_supervised(
    argv: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    progress_path: Path,
) -> subprocess.CompletedProcess[str]:
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=_merged_environment(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    last_signature = _progress_signature(progress_path)
    last_activity = time.monotonic()
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=ASR_PROGRESS_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                signature = _progress_signature(progress_path)
                now = time.monotonic()
                if signature != last_signature:
                    last_signature = signature
                    last_activity = now
                    continue
                if now - last_activity < ASR_STALL_TIMEOUT_SECONDS:
                    continue
                detail = _progress_summary(progress_path)
                _terminate_process_tree(process)
                # communicate() after timeout is safe to retry and preserves the
                # already captured output according to the subprocess contract.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.communicate(timeout=1)
                raise CommandError(
                    "ASR worker made no checkpoint/progress heartbeat for "
                    f"{ASR_STALL_TIMEOUT_SECONDS // 60} minutes ({detail}). "
                    "The worker process tree was terminated so the build cannot hang forever. "
                    "Successful item checkpoints remain reusable; rerun the same prepare/build "
                    "command without --force to resume."
                ) from None
    except BaseException:
        _terminate_process_tree(process)
        raise

    completed = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        raise _command_error(argv, completed)
    return completed


def run_json(
    args: Iterable[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> Any:
    argv = [str(value) for value in args]
    progress_path = _asr_progress_path(argv, cwd)
    if progress_path is None:
        completed = run(argv, cwd=cwd, env=env, capture=True)
    else:
        completed = _run_json_supervised(
            argv,
            cwd=cwd,
            env=env,
            progress_path=progress_path,
        )
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
