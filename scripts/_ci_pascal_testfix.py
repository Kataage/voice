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


# Irodori now overlays both pyproject.toml and uv.lock, and restores both in the
# finally path. Keep the existing atomic-restore test, but exercise the new pair.
patch(
    "tests/test_core.py",
    '''    managed = repo_root / "locks" / "Irodori-TTS.uv.lock"\n    managed.parent.mkdir(parents=True)\n    managed.write_bytes(b"audited-lock")\n    vendor_lock = vendor / "uv.lock"\n    if original is not None:\n        vendor_lock.write_bytes(original)\n    calls: list[list[str]] = []\n\n    def fake_run(args, **_kwargs):\n        calls.append([str(value) for value in args])\n\n    def fake_restore(path: Path):\n        if original is None:\n            (path / "uv.lock").unlink(missing_ok=True)\n        else:\n            (path / "uv.lock").write_bytes(original)\n\n    monkeypatch.setattr(setup_env, "_git_head", lambda _path: "audited-head")\n    monkeypatch.setattr(setup_env, "_restore_vendor_lock", fake_restore)\n    monkeypatch.setattr(setup_env, "run", fake_run)\n    setup_env._install_irodori(repo_root, vendor, "cpu")\n\n    assert any("--locked" in call for call in calls)\n    assert not (repo_root / ".runtime" / setup_env.IRODORI_LOCK_SWAP_MARKER).exists()\n    if original is None:\n        assert not vendor_lock.exists()\n    else:\n        assert vendor_lock.read_bytes() == original\n''',
    '''    managed_project = repo_root / "locks" / setup_env.IRODORI_MANAGED_PROJECT\n    managed_lock = repo_root / "locks" / "Irodori-TTS.uv.lock"\n    managed_project.parent.mkdir(parents=True)\n    managed_project.write_bytes(b"audited-project")\n    managed_lock.write_bytes(b"audited-lock")\n    vendor_project = vendor / "pyproject.toml"\n    vendor_project.write_bytes(b"upstream-project")\n    vendor_lock = vendor / "uv.lock"\n    if original is not None:\n        vendor_lock.write_bytes(original)\n    calls: list[list[str]] = []\n\n    def fake_run(args, **_kwargs):\n        calls.append([str(value) for value in args])\n\n    def fake_restore(path: Path):\n        (path / "pyproject.toml").write_bytes(b"upstream-project")\n        if original is None:\n            (path / "uv.lock").unlink(missing_ok=True)\n        else:\n            (path / "uv.lock").write_bytes(original)\n\n    monkeypatch.setattr(setup_env, "_git_head", lambda _path: "audited-head")\n    monkeypatch.setattr(setup_env, "_restore_vendor_setup_files", fake_restore)\n    monkeypatch.setattr(setup_env, "run", fake_run)\n    setup_env._install_irodori(repo_root, vendor, "cpu")\n\n    assert any("--locked" in call for call in calls)\n    assert not (repo_root / ".runtime" / setup_env.IRODORI_LOCK_SWAP_MARKER).exists()\n    assert vendor_project.read_bytes() == b"upstream-project"\n    if original is None:\n        assert not vendor_lock.exists()\n    else:\n        assert vendor_lock.read_bytes() == original\n''',
)

# Media atomicity tests mock the process itself; bypass the real host FFmpeg
# discovery so they continue testing publication semantics only.
patch(
    "tests/test_integrity.py",
    '    monkeypatch.setattr(media.subprocess, "run", fake_run)\n    with pytest.raises(subprocess.CalledProcessError):\n',
    '    monkeypatch.setattr(media.subprocess, "run", fake_run)\n'
    '    monkeypatch.setattr(media, "ffmpeg_command", lambda name: name)\n'
    '    with pytest.raises(subprocess.CalledProcessError):\n',
)
patch(
    "tests/test_integrity.py",
    '    monkeypatch.setattr(media.subprocess, "run", fake_run)\n    media.extract_lossless_audio(source, destination)\n',
    '    monkeypatch.setattr(media.subprocess, "run", fake_run)\n'
    '    monkeypatch.setattr(media, "ffmpeg_command", lambda name: name)\n'
    '    media.extract_lossless_audio(source, destination)\n',
)

# The managed Irodori project is now part of environment generation 3.
patch(
    "tests/test_seed_vc_pinned_assets.py",
    '    assert first["schema"] == 2\n',
    '    assert first["schema"] == 3\n',
)

# Materialization tests intentionally load the isolated ASR worker without its
# heavyweight environment. Stub the new runtime-only CTranslate2 policy imports.
patch(
    "tests/test_worker_materialization_hardening.py",
    '''def _load_asr_worker(monkeypatch):\n    faster_whisper = types.ModuleType("faster_whisper")\n''',
    '''def _load_asr_worker(monkeypatch):\n    ctranslate2 = types.ModuleType("ctranslate2")\n    ctranslate2.get_supported_compute_types = lambda _device: {"float32"}\n    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)\n\n    runtime_policy = types.ModuleType("runtime_policy")\n    runtime_policy.choose_compute_type = lambda _device: "float32"\n    monkeypatch.setitem(sys.modules, "runtime_policy", runtime_policy)\n\n    faster_whisper = types.ModuleType("faster_whisper")\n''',
)

# Lock/index tests cover both CUDA families so a future lock refresh cannot drop
# legacy-Pascal/Volta support while keeping modern Turing+ support green.
patch(
    "tests/test_core.py",
    '''def test_worker_backend_mapping_is_explicit():\n    cuda = _worker_extras("cu128")\n''',
    '''def test_worker_backend_mapping_is_explicit():\n    legacy_cuda = _worker_extras("cu126")\n    assert legacy_cuda["diarization"] == "cu126"\n    assert legacy_cuda["sense"] == "cu126"\n    assert legacy_cuda["lfm"] == "cu126"\n    assert legacy_cuda["seed_vc"] == "cu124"\n    cuda = _worker_extras("cu128")\n''',
)
patch(
    "tests/test_core.py",
    '    assert _requires_cuda("cu128") is True\n',
    '    assert _requires_cuda("cu126") is True\n    assert _requires_cuda("cu128") is True\n',
)
patch(
    "tests/test_core.py",
    '''    expectations = {\n        "diarization": "pytorch-cu128",\n        "sense": "pytorch-cu128",\n        "lfm": "pytorch-cu128",\n        "seed_vc": "pytorch-cu124",\n    }\n    for name, index in expectations.items():\n        text = (root / "workers" / name / "pyproject.toml").read_text(encoding="utf-8")\n        assert "explicit = true" in text\n        assert index in text\n        assert "pytorch-cpu" in text\n''',
    '''    expectations = {\n        "diarization": ("pytorch-cu126", "pytorch-cu128"),\n        "sense": ("pytorch-cu126", "pytorch-cu128"),\n        "lfm": ("pytorch-cu126", "pytorch-cu128"),\n        "seed_vc": ("pytorch-cu124",),\n    }\n    for name, indexes in expectations.items():\n        text = (root / "workers" / name / "pyproject.toml").read_text(encoding="utf-8")\n        assert "explicit = true" in text\n        for index in indexes:\n            assert index in text\n        assert "pytorch-cpu" in text\n''',
)

print("Pascal test-fixture patches applied")
