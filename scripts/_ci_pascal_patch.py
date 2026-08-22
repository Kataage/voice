from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8", newline="\n")


def replace(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise RuntimeError(f"Expected patch anchor not found in {path}: {old[:120]!r}")
    write(path, text.replace(old, new, 1))


def replace_block(path: str, start: str, end: str, new: str) -> None:
    text = read(path)
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeError(f"Start anchor not found in {path}: {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"End anchor not found in {path}: {end!r}")
    write(path, text[:start_index] + new + text[end_index:])


# setup_env: add a crash-safe managed pyproject+lock overlay so the pinned
# upstream checkout can gain the audited cu126 extra without staying modified.
replace(
    "src/personavoice/setup_env.py",
    "from personavoice.hardware import detect_irodori_backend\n",
    "from personavoice.hardware import detect_irodori_backend\n"
    "from personavoice.runtime_dependencies import require_ffmpeg_runtime\n",
)
replace(
    "src/personavoice/setup_env.py",
    'IRODORI_LOCK_SWAP_MARKER = "irodori-lock-swap.json"\n'
    'SUPPORTED_IRODORI_BACKENDS = {"cpu", "cu128", "rocm", "xpu"}\n',
    'IRODORI_LOCK_SWAP_MARKER = "irodori-lock-swap.json"\n'
    'IRODORI_MANAGED_PROJECT = "Irodori-TTS.pyproject.toml"\n'
    'SUPPORTED_IRODORI_BACKENDS = {"cpu", "cu126", "cu128", "rocm", "xpu"}\n',
)
replace_block(
    "src/personavoice/setup_env.py",
    "def _restore_vendor_lock(irodori: Path) -> None:\n",
    "def _clone_pinned(",
    '''def _restore_vendor_file(irodori: Path, relative: str) -> None:\n    path = irodori / relative\n    tracked = run(\n        ["git", "ls-files", "--error-unmatch", "--", relative],\n        cwd=irodori,\n        capture=True,\n        check=False,\n    ).returncode == 0\n    if tracked:\n        run(\n            ["git", "restore", "--source=HEAD", "--worktree", "--", relative],\n            cwd=irodori,\n        )\n    else:\n        path.unlink(missing_ok=True)\n\n\ndef _restore_vendor_setup_files(irodori: Path) -> None:\n    """Restore the pinned checkout after the managed project/lock overlay."""\n\n    _restore_vendor_file(irodori, "pyproject.toml")\n    _restore_vendor_file(irodori, "uv.lock")\n\n\ndef _file_swap_state(path: Path, managed: Path) -> dict:\n    exists = path.is_file()\n    return {\n        "original_exists": exists,\n        "original_sha256": sha256_file(path) if exists else None,\n        "managed_sha256": sha256_file(managed),\n    }\n\n\ndef _recover_irodori_lock_swap(repo_root: Path, irodori: Path) -> None:\n    """Recover an interrupted managed Irodori project/lock overlay."""\n\n    marker = _irodori_swap_marker(repo_root)\n    if not marker.is_file():\n        return\n    try:\n        value = json.loads(marker.read_text(encoding="utf-8"))\n    except (json.JSONDecodeError, OSError) as exc:\n        raise RuntimeError(\n            f"Interrupted Irodori setup-overlay marker is unreadable: {marker}. "\n            "Inspect the vendor checkout before rerunning setup."\n        ) from exc\n    schema = value.get("schema_version") if isinstance(value, dict) else None\n    if schema not in {2, 3}:\n        raise RuntimeError(\n            f"Interrupted Irodori setup-overlay marker has an unsupported format: {marker}"\n        )\n\n    expected_head = value.get("vendor_head")\n    current_head = _git_head(irodori)\n    if not isinstance(expected_head, str) or current_head != expected_head:\n        raise RuntimeError(\n            "An interrupted PersonaVoice Irodori setup overlay belongs to a different vendor "\n            "HEAD. Refusing automatic recovery; inspect vendor/Irodori-TTS before rerunning "\n            "setup."\n        )\n\n    if schema == 2:\n        # Backward-compatible recovery for markers written before the managed\n        # pyproject overlay existed. Only uv.lock was swapped in that format.\n        states = {\n            "uv.lock": {\n                "original_exists": bool(value.get("original_exists")),\n                "original_sha256": value.get("original_sha256"),\n                "managed_sha256": value.get("managed_sha256"),\n            }\n        }\n    else:\n        raw_states = value.get("files")\n        if not isinstance(raw_states, dict):\n            raise RuntimeError(f"Interrupted Irodori setup-overlay marker is invalid: {marker}")\n        states = raw_states\n\n    for relative, state in states.items():\n        if relative not in {"pyproject.toml", "uv.lock"} or not isinstance(state, dict):\n            raise RuntimeError(f"Interrupted Irodori setup-overlay marker is invalid: {marker}")\n        path = irodori / relative\n        current_exists = path.is_file()\n        current_sha = sha256_file(path) if current_exists else None\n        original_exists = bool(state.get("original_exists"))\n        original_sha = state.get("original_sha256")\n        managed_sha = state.get("managed_sha256")\n        original_state = current_exists == original_exists and (\n            not current_exists or current_sha == original_sha\n        )\n        managed_state = (\n            current_exists\n            and isinstance(managed_sha, str)\n            and current_sha == managed_sha\n        )\n        if managed_state:\n            _restore_vendor_file(irodori, relative)\n        elif not original_state:\n            raise RuntimeError(\n                "An interrupted PersonaVoice Irodori setup overlay was found, but "\n                f"vendor/Irodori-TTS/{relative} matches neither the pinned checkout nor the "\n                "audited temporary overlay. Refusing to overwrite a possible local edit."\n            )\n    marker.unlink(missing_ok=True)\n\n\n''',
)
replace_block(
    "src/personavoice/setup_env.py",
    "def _worker_extras(selected_backend: str) -> dict[str, str | None]:\n",
    "def _install_irodori(",
    '''def _worker_extras(selected_backend: str) -> dict[str, str | None]:\n    """Map the Irodori backend to compatible isolated worker backends."""\n\n    if selected_backend in {"cu126", "cu128"}:\n        return {\n            "asr": None,\n            "diarization": selected_backend,\n            "sense": selected_backend,\n            "lfm": selected_backend,\n            # Seed-VC intentionally remains on its audited Torch 2.4/CUDA 12.4 stack.\n            "seed_vc": "cu124",\n        }\n    return {\n        "asr": None,\n        "diarization": "cpu",\n        "sense": "cpu",\n        "lfm": "cpu",\n        "seed_vc": "cpu",\n    }\n\n\n''',
)
replace_block(
    "src/personavoice/setup_env.py",
    "def _install_irodori(repo_root: Path, irodori: Path, backend: str) -> None:\n",
    "def install_environments(",
    '''def _install_irodori(repo_root: Path, irodori: Path, backend: str) -> None:\n    """Sync Irodori from the audited project overlay and lock, then restore vendor files."""\n\n    managed_project = repo_root / "locks" / IRODORI_MANAGED_PROJECT\n    managed_lock = repo_root / "locks" / "Irodori-TTS.uv.lock"\n    missing = [str(path) for path in (managed_project, managed_lock) if not path.is_file()]\n    if missing:\n        raise FileNotFoundError(\n            "Audited Irodori dependency overlay is incomplete: "\n            + ", ".join(missing)\n            + ". Restore the repository lock files before setup."\n        )\n\n    marker = _irodori_swap_marker(repo_root)\n    vendor_project = irodori / "pyproject.toml"\n    vendor_lock = irodori / "uv.lock"\n    swap_state = {\n        "schema_version": 3,\n        "vendor": str(irodori.resolve()),\n        "vendor_head": _git_head(irodori),\n        "files": {\n            "pyproject.toml": _file_swap_state(vendor_project, managed_project),\n            "uv.lock": _file_swap_state(vendor_lock, managed_lock),\n        },\n    }\n    atomic_write_json(marker, swap_state)\n    args: list[str | Path] = [\n        "uv",\n        "sync",\n        "--project",\n        irodori,\n        "--extra",\n        backend,\n        "--locked",\n    ]\n    try:\n        atomic_write_text(vendor_project, managed_project.read_text(encoding="utf-8"))\n        atomic_write_text(vendor_lock, managed_lock.read_text(encoding="utf-8"))\n        run(args, cwd=repo_root)\n    finally:\n        _restore_vendor_setup_files(irodori)\n        marker.unlink(missing_ok=True)\n\n\n''',
)
replace(
    "src/personavoice/setup_env.py",
    '    if not shutil.which("git"):\n        raise RuntimeError("git was not found in PATH")\n'
    '    selected_backend = backend or detect_irodori_backend()\n',
    '    if not shutil.which("git"):\n        raise RuntimeError("git was not found in PATH")\n'
    '    require_ffmpeg_runtime()\n'
    '    selected_backend = backend or detect_irodori_backend()\n',
)

# Irodori runtime accepts both audited CUDA wheel families.
replace(
    "src/personavoice/irodori.py",
    'SUPPORTED_BACKENDS = {"cpu", "cu128", "rocm", "xpu"}',
    'SUPPORTED_BACKENDS = {"cpu", "cu126", "cu128", "rocm", "xpu"}',
)
replace(
    "src/personavoice/irodori.py",
    '    if backend in {"cu128", "rocm"}:\n',
    '    if backend in {"cu126", "cu128", "rocm"}:\n',
)

# CLI exposes the legacy-Pascal CUDA family explicitly while auto chooses it
# from nvidia-smi compute capability.
replace(
    "src/personavoice/cli.py",
    'SETUP_BACKENDS = {"auto", "cu128", "cpu", "rocm", "xpu"}',
    'SETUP_BACKENDS = {"auto", "cu126", "cu128", "cpu", "rocm", "xpu"}',
)
replace(
    "src/personavoice/cli.py",
    'help="Irodori backend: auto/cu128/cpu/rocm/xpu"',
    'help="Irodori backend: auto/cu126/cu128/cpu/rocm/xpu"',
)

# Doctor distinguishes command discovery from the shared FFmpeg runtime required
# by TorchCodec and understands cu126 as CUDA.
replace(
    "src/personavoice/doctor.py",
    "from personavoice.process import run\n",
    "from personavoice.process import run\n"
    "from personavoice.runtime_dependencies import ffmpeg_runtime\n",
)
replace(
    "src/personavoice/doctor.py",
    '        return "cuda" if setup.get("irodori_backend") == "cu128" else "cpu"\n',
    '        return "cuda" if setup.get("irodori_backend") in {"cu126", "cu128"} else "cpu"\n',
)
replace(
    "src/personavoice/doctor.py",
    '    if backend in {"cu128", "rocm"}:\n',
    '    if backend in {"cu126", "cu128", "rocm"}:\n',
)
replace(
    "src/personavoice/doctor.py",
    '    required = {name: shutil.which(name) for name in ("uv", "git", "ffmpeg", "ffprobe")}\n'
    '    runtime = repo_root / ".runtime"\n',
    '    ffmpeg_status = ffmpeg_runtime()\n'
    '    required = {\n'
    '        "uv": shutil.which("uv"),\n'
    '        "git": shutil.which("git"),\n'
    '        "ffmpeg": ffmpeg_status.ffmpeg,\n'
    '        "ffprobe": ffmpeg_status.ffprobe,\n'
    '    }\n'
    '    commands_ok = bool(required["uv"] and required["git"] and ffmpeg_status.torchcodec_compatible)\n'
    '    runtime = repo_root / ".runtime"\n',
)
replace(
    "src/personavoice/doctor.py",
    '        all(required.values())\n        and bool(setup)\n',
    '        commands_ok\n        and bool(setup)\n',
)
replace(
    "src/personavoice/doctor.py",
    '        "commands": required,\n        "commands_ok": all(required.values()),\n        "hardware": hardware_report(),\n',
    '        "commands": required,\n        "commands_ok": commands_ok,\n        "ffmpeg_runtime": ffmpeg_status.as_dict(),\n        "hardware": hardware_report(),\n',
)

# Environment-generation tests need the new managed overlay and cu126 mapping.
replace(
    "tests/test_environment_generation.py",
    '    _write(root / "locks" / "Irodori-TTS.uv.lock", b"irodori-lock")\n',
    '    _write(root / "locks" / "Irodori-TTS.pyproject.toml", b"irodori-project")\n'
    '    _write(root / "locks" / "Irodori-TTS.uv.lock", b"irodori-lock")\n',
)
replace(
    "tests/test_irodori_backend.py",
    '    assert irodori.backend_device("cu128") == "cuda"\n',
    '    assert irodori.backend_device("cu126") == "cuda"\n'
    '    assert irodori.backend_device("cu128") == "cuda"\n',
)

# Documentation reflects the audited CUDA split and Windows TorchCodec FFmpeg contract.
replace(
    "README.md",
    "必要: `uv`, Git, FFmpeg/ffprobe。NVIDIA GPU推奨。",
    "必要: `uv`, Git, FFmpeg/ffprobe。NVIDIA GPU推奨。WindowsではTorchCodec用にshared DLL付きFFmpeg 4〜8が必要で、bootstrapはWinGetのFFmpeg Shared 8.1.1を使用します。",
)
replace(
    "README.md",
    "Irodori backendは`persona setup --backend auto|cu128|cpu|rocm|xpu`で選択できます。NVIDIA時はmodern Torch workerをCUDA 12.8系、互換性のためTorch 2.4に固定しているSeed-VCをCUDA 12.4系へ明示的に解決します。",
    "Irodori backendは`persona setup --backend auto|cu126|cu128|cpu|rocm|xpu`で選択できます。`auto`はNVIDIA device 0のcompute capabilityを見てPascal 6.xをCUDA 12.6、7.0以上をCUDA 12.8へ分けます。互換性のためTorch 2.4に固定しているSeed-VCはCUDA 12.4系へ明示的に解決します。",
)

print("Pascal runtime integration patches applied")
