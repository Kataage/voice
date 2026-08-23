from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found < count:
        raise RuntimeError(
            f"Expected at least {count} patch anchor(s) in {path}, found {found}: {old[:100]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


patch(
    "src/personavoice/runtime_dependencies.py",
    "from personavoice.ffmpeg_contract import pinned_bin_dir, validate_pinned_runtime\n",
    "from personavoice.ffmpeg_contract import pinned_bin_dir, runtime_root, validate_pinned_runtime\n",
)
patch(
    "src/personavoice/runtime_dependencies.py",
    '''def _repo_root_if_available() -> Path | None:\n    explicit = os.getenv("PERSONAVOICE_ROOT")\n    if explicit:\n        candidate = Path(explicit).expanduser()\n        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "personavoice").is_dir():\n            return candidate\n    try:\n        return find_repo_root()\n    except RuntimeError:\n        return None\n''',
    '''def _repo_root_if_available() -> Path | None:\n    explicit = os.getenv("PERSONAVOICE_ROOT")\n    if explicit:\n        return Path(explicit).expanduser().resolve()\n    try:\n        return find_repo_root()\n    except RuntimeError:\n        return None\n''',
)
patch(
    "src/personavoice/runtime_dependencies.py",
    '''def _path_candidates() -> list[tuple[Path, str]]:\n    candidates: list[tuple[Path, str]] = []\n    repo_root = _repo_root_if_available()\n    if repo_root is not None and bool(validate_pinned_runtime(repo_root).get("ok")):\n        candidates.append((pinned_bin_dir(repo_root), "PersonaVoice:pinned"))\n\n    for name in ("ffmpeg", "ffprobe"):\n''',
    '''def _path_candidates() -> list[tuple[Path, str]]:\n    candidates: list[tuple[Path, str]] = []\n    for name in ("ffmpeg", "ffprobe"):\n''',
)
anchor = '''        return runtime if runtime.torchcodec_compatible else _incompatibility(runtime)\n\n    valid: list[FfmpegRuntime] = []\n'''
replacement = '''        return runtime if runtime.torchcodec_compatible else _incompatibility(runtime)\n\n    repo_root = _repo_root_if_available()\n    if repo_root is not None:\n        pinned_root = runtime_root(repo_root)\n        if pinned_root.exists():\n            status = validate_pinned_runtime(repo_root)\n            if not status["ok"]:\n                return FfmpegRuntime(\n                    ffmpeg=None,\n                    ffprobe=None,\n                    bin_dir=str(pinned_bin_dir(repo_root)),\n                    version_major=None,\n                    shared_libraries=False,\n                    torchcodec_compatible=False,\n                    source="PersonaVoice:pinned",\n                    error=(\n                        "PersonaVoice's pinned FFmpeg runtime is present but failed integrity "\n                        "validation. Rerun `persona setup --backend auto` to repair it; refusing "\n                        "to silently fall back to a different system FFmpeg runtime. "\n                        + "; ".join(str(value) for value in status["errors"])\n                    ),\n                )\n            pinned = _candidate_runtime(pinned_bin_dir(repo_root), "PersonaVoice:pinned")\n            if pinned is None:\n                return FfmpegRuntime(\n                    ffmpeg=None,\n                    ffprobe=None,\n                    bin_dir=str(pinned_bin_dir(repo_root)),\n                    version_major=None,\n                    shared_libraries=False,\n                    torchcodec_compatible=False,\n                    source="PersonaVoice:pinned",\n                    error="Pinned FFmpeg runtime is incomplete; rerun `persona setup --backend auto`.",\n                )\n            return pinned if pinned.torchcodec_compatible else _incompatibility(pinned)\n\n    valid: list[FfmpegRuntime] = []\n'''
patch("src/personavoice/runtime_dependencies.py", anchor, replacement)

print("post-PR14 final resolver fixes applied")
