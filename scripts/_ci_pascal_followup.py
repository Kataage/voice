from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found < count:
        raise RuntimeError(
            f"Expected at least {count} patch anchors in {path}, found {found}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


# The DLL directory must be registered before these imports. Mark the deliberate
# post-bootstrap imports so Ruff does not treat the ordering as accidental.
patch(
    "workers/diarization/worker.py",
    "import torch\nfrom huggingface_hub import snapshot_download\nfrom pyannote.audio import Pipeline\n",
    "import torch  # noqa: E402\n"
    "from huggingface_hub import snapshot_download  # noqa: E402\n"
    "from pyannote.audio import Pipeline  # noqa: E402\n",
)

# Environment fixtures now fingerprint the managed Irodori project overlay as
# well as the generated lock.
patch(
    "tests/test_environment_generation.py",
    '    _write(root / "locks" / "Irodori-TTS.uv.lock", b"irodori-lock")\n',
    '    _write(root / "locks" / "Irodori-TTS.pyproject.toml", b"irodori-project")\n'
    '    _write(root / "locks" / "Irodori-TTS.uv.lock", b"irodori-lock")\n',
)

# Unit tests isolate setup from the machine's real FFmpeg installation. Production
# setup still calls require_ffmpeg_runtime before mutating any environment.
patch(
    "tests/test_environment_generation.py",
    '    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: "/tool")\n',
    '    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: "/tool")\n'
    '    monkeypatch.setattr(setup_env, "require_ffmpeg_runtime", lambda: None)\n',
    count=2,
)

patch(
    "tests/test_environment_generation.py",
    '    with pytest.raises(FileNotFoundError, match="Audited Irodori lockfile is missing"):\n',
    '    with pytest.raises(FileNotFoundError, match="Audited Irodori dependency overlay is incomplete"):\n',
)

patch(
    "tests/test_environment_generation.py",
    '    restored: list[Path] = []\n'
    '    monkeypatch.setattr(setup_env, "_git_head", lambda _path: "head")\n'
    '    monkeypatch.setattr(setup_env, "_restore_vendor_lock", lambda path: restored.append(path))\n\n'
    '    setup_env._recover_irodori_lock_swap(tmp_path, vendor)\n'
    '    assert restored == [vendor]\n',
    '    restored: list[tuple[Path, str]] = []\n'
    '    monkeypatch.setattr(setup_env, "_git_head", lambda _path: "head")\n'
    '    monkeypatch.setattr(\n'
    '        setup_env,\n'
    '        "_restore_vendor_file",\n'
    '        lambda path, relative: restored.append((path, relative)),\n'
    '    )\n\n'
    '    setup_env._recover_irodori_lock_swap(tmp_path, vendor)\n'
    '    assert restored == [(vendor, "uv.lock")]\n',
)

# Prove the new environment contract detects changes to the Irodori dependency
# overlay, not only changes to worker locks.
append_path = ROOT / "tests/test_environment_generation.py"
text = append_path.read_text(encoding="utf-8")
anchor = "\ndef test_runtime_environment_contract_accepts_current_rejects_stale_and_recovers(\n"
insert = '''\ndef test_environment_contract_detects_irodori_project_overlay_change(tmp_path: Path):\n    _dependency_tree(tmp_path)\n    recorded = environment_contract(tmp_path)\n    assert environment_contract_status(tmp_path, recorded)["ok"]\n\n    (tmp_path / "locks" / "Irodori-TTS.pyproject.toml").write_bytes(b"new-overlay")\n    status = environment_contract_status(tmp_path, recorded)\n    assert not status["ok"]\n    assert "different dependency contract" in status["error"]\n\n\n'''
if anchor not in text:
    raise RuntimeError("Environment-contract insertion anchor not found")
append_path.write_text(text.replace(anchor, "\n" + insert + anchor.lstrip("\n"), 1), encoding="utf-8", newline="\n")

print("Pascal follow-up patches applied")
