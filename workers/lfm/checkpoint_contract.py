from __future__ import annotations

import re
import shutil
from pathlib import Path

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_INCOMPLETE_SENTINEL = "checkpoint-is-incomplete.txt"
_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
_REQUIRED_TRAINER_FILES = (
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def checkpoint_complete(path: Path) -> bool:
    """Return whether a Trainer+PEFT checkpoint can resume exactly enough.

    Transformers restores model/adapter, optimizer, scheduler, trainer and RNG
    state when resuming. PersonaVoice intentionally requires all of those states
    and rejects JIT checkpoints whose incomplete sentinel survived a crash.
    """

    if not path.is_dir() or checkpoint_step(path) is None:
        return False
    if (path / _INCOMPLETE_SENTINEL).exists():
        return False
    if not all(_nonempty(path / name) for name in _REQUIRED_TRAINER_FILES):
        return False
    if not any(_nonempty(path / name) for name in _ADAPTER_WEIGHTS):
        return False
    return any(_nonempty(candidate) for candidate in path.glob("rng_state*.pth"))


def latest_complete_checkpoint(output: Path) -> Path | None:
    if not output.is_dir():
        return None
    candidates = [
        (step, path)
        for path in output.glob("checkpoint-*")
        if (step := checkpoint_step(path)) is not None and checkpoint_complete(path)
    ]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def prune_incomplete_checkpoints(output: Path) -> list[Path]:
    """Remove only numeric Trainer checkpoint directories that cannot be resumed.

    They are derived training artifacts, not user source data. Keeping them can
    cause a future generic Trainer 'last checkpoint' lookup to pick a corrupted
    newer directory over a valid older checkpoint.
    """

    removed: list[Path] = []
    if not output.is_dir():
        return removed
    for path in output.glob("checkpoint-*"):
        if path.is_dir() and checkpoint_step(path) is not None and not checkpoint_complete(path):
            shutil.rmtree(path)
            removed.append(path)
    return sorted(removed, key=lambda path: path.name)
