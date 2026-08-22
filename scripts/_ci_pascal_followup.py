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

# Irodori's pinned v4 Small configs default to bf16 + TF32. Pascal (sm_6x)
# supports neither of those acceleration paths, so cu126 must force fp16 while
# keeping CUDA prefetch enabled.
patch(
    "src/personavoice/irodori.py",
    '    if backend == "cpu":\n'
    '        train_cfg["dataloader_cuda_prefetch"] = False\n'
    '        train_cfg["precision"] = "fp32"\n'
    '        train_cfg["allow_tf32"] = False\n',
    '    if backend == "cu126":\n'
    '        train_cfg["precision"] = "fp16"\n'
    '        train_cfg["allow_tf32"] = False\n'
    '    elif backend == "cpu":\n'
    '        train_cfg["dataloader_cuda_prefetch"] = False\n'
    '        train_cfg["precision"] = "fp32"\n'
    '        train_cfg["allow_tf32"] = False\n',
)

# If the user explicitly asks for cu128 on a known Pascal GPU, reject it before
# mutating environments instead of allowing a wheel that installs but has no
# executable kernel image for sm_61.
patch(
    "src/personavoice/setup_env.py",
    "from personavoice.hardware import detect_irodori_backend\n",
    "from personavoice.hardware import cuda_backend_for_gpu, detect_irodori_backend, nvidia_gpus\n",
)
patch(
    "src/personavoice/setup_env.py",
    "def install_environments(repo_root: Path, *, backend: str | None = None) -> dict:\n",
    '''def _validate_explicit_backend(backend: str | None) -> None:\n    if backend != "cu128":\n        return\n    gpus = nvidia_gpus()\n    if not gpus:\n        return\n    gpu0 = min(gpus, key=lambda gpu: gpu.index)\n    compatible = cuda_backend_for_gpu(gpu0)\n    if compatible == "cu126":\n        capability = gpu0.compute_capability or "unknown"\n        raise ValueError(\n            f"The selected NVIDIA GPU {gpu0.name} has compute capability {capability}; "\n            "the audited PyTorch CUDA 12.8 stack requires sm_70 or newer. "\n            "Use `--backend auto` or `--backend cu126` for this GPU."\n        )\n\n\ndef install_environments(repo_root: Path, *, backend: str | None = None) -> dict:\n''',
)
patch(
    "src/personavoice/setup_env.py",
    '    require_ffmpeg_runtime()\n    selected_backend = backend or detect_irodori_backend()\n',
    '    require_ffmpeg_runtime()\n'
    '    _validate_explicit_backend(backend)\n'
    '    selected_backend = backend or detect_irodori_backend()\n',
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
append_path.write_text(
    text.replace(anchor, "\n" + insert + anchor.lstrip("\n"), 1),
    encoding="utf-8",
    newline="\n",
)

print("Pascal follow-up patches applied")
