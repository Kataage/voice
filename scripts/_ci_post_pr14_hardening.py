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


# Setup is the only path allowed to obtain/materialize the pinned Windows FFmpeg archive.
patch(
    "src/personavoice/setup_env.py",
    "from personavoice.environment_contract import SETUP_TRANSACTION_MARKER, environment_contract\n",
    "from personavoice.environment_contract import SETUP_TRANSACTION_MARKER, environment_contract\n"
    "from personavoice.ffmpeg_materializer import ensure_ffmpeg_runtime\n",
)
patch(
    "src/personavoice/setup_env.py",
    "from personavoice.runtime_dependencies import require_ffmpeg_runtime\n",
    "",
)
patch(
    "src/personavoice/setup_env.py",
    "    require_ffmpeg_runtime()\n    selected_backend = backend or detect_irodori_backend()\n",
    "    ensure_ffmpeg_runtime(repo_root)\n    selected_backend = backend or detect_irodori_backend()\n",
)

# FFmpeg contract/materializer changes invalidate setup generations created before this contract.
patch(
    "src/personavoice/environment_contract.py",
    "ENVIRONMENT_CONTRACT_SCHEMA = 4\n",
    "ENVIRONMENT_CONTRACT_SCHEMA = 5\n",
)
patch(
    "src/personavoice/environment_contract.py",
    '            "runtime_dependencies_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "runtime_dependencies.py"\n'
    '            ),\n',
    '            "runtime_dependencies_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "runtime_dependencies.py"\n'
    '            ),\n'
    '            "ffmpeg_contract_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "ffmpeg_contract.py"\n'
    '            ),\n'
    '            "ffmpeg_materializer_sha256": _sha256(\n'
    '                repo_root / "src" / "personavoice" / "ffmpeg_materializer.py"\n'
    '            ),\n',
)

# Turing can use the cu128 wheel but does not have the BF16/TF32 path used by
# Irodori's upstream defaults. Ampere+ may use BF16/TF32; legacy audited CUDA is FP32.
anchor = '''def cuda_backend_for_gpu(gpu: GpuInfo) -> str:\n    """Return the safest audited PyTorch CUDA wheel family for one NVIDIA GPU."""\n\n    if backend_supports_gpu("cu128", gpu):\n        return "cu128"\n    if backend_supports_gpu("cu126", gpu):\n        return "cu126"\n    return "cpu"\n\n\n'''
precision = anchor + '''def irodori_training_precision(\n    backend: str,\n    *,\n    gpu: GpuInfo | None = None,\n) -> dict[str, str | bool]:\n    """Return audited Irodori precision flags for the actual CUDA-visible GPU."""\n\n    if backend not in {"cu126", "cu128"}:\n        return {"precision": "fp32", "allow_tf32": False}\n    selected = selected_nvidia_gpu() if gpu is None else gpu\n    capability = _parse_compute_capability(\n        selected.compute_capability if selected is not None else None\n    )\n    if (\n        backend == "cu128"\n        and selected is not None\n        and capability is not None\n        and capability >= (8, 0)\n        and backend_supports_gpu("cu128", selected)\n    ):\n        return {"precision": "bf16", "allow_tf32": True}\n    return {"precision": "fp32", "allow_tf32": False}\n\n\n'''
patch("src/personavoice/hardware.py", anchor, precision)
patch(
    "src/personavoice/irodori.py",
    "from personavoice.hardware import safe_batch_profile\n",
    "from personavoice.hardware import irodori_training_precision, safe_batch_profile\n",
)
patch(
    "src/personavoice/irodori.py",
    '    if backend == "cu126":\n'
    '        train_cfg["precision"] = "fp32"\n'
    '        train_cfg["allow_tf32"] = False\n'
    '    elif backend == "cpu":\n'
    '        train_cfg["dataloader_cuda_prefetch"] = False\n'
    '        train_cfg["precision"] = "fp32"\n'
    '        train_cfg["allow_tf32"] = False\n',
    '    if backend in {"cu126", "cu128"}:\n'
    '        precision = irodori_training_precision(backend)\n'
    '        train_cfg["precision"] = str(precision["precision"])\n'
    '        train_cfg["allow_tf32"] = bool(precision["allow_tf32"])\n'
    '    elif backend == "cpu":\n'
    '        train_cfg["dataloader_cuda_prefetch"] = False\n'
    '        train_cfg["precision"] = "fp32"\n'
    '        train_cfg["allow_tf32"] = False\n',
)

# Existing setup transaction tests isolate FFmpeg materialization from machine state.
patch(
    "tests/test_environment_generation.py",
    'monkeypatch.setattr(setup_env, "require_ffmpeg_runtime", lambda: None)',
    'monkeypatch.setattr(setup_env, "ensure_ffmpeg_runtime", lambda _root: None)',
    count=2,
)

# GPU-generation precision regression tests.
precision_test = ROOT / "tests" / "test_irodori_generation_precision.py"
precision_test.write_text(
    '''from __future__ import annotations\n\nfrom personavoice import hardware\n\n\ndef _gpu(capability: str) -> hardware.GpuInfo:\n    return hardware.GpuInfo(\n        index=0,\n        name="NVIDIA test GPU",\n        total_mib=16384,\n        free_mib=12000,\n        compute_capability=capability,\n        uuid="GPU-test",\n        pci_bus_id="00000000:01:00.0",\n        driver_version="600.00",\n    )\n\n\ndef test_irodori_precision_is_fp32_on_pascal_volta_and_turing(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    for capability, backend in (("6.1", "cu126"), ("7.0", "cu126"), ("7.5", "cu128")):\n        policy = hardware.irodori_training_precision(backend, gpu=_gpu(capability))\n        assert policy == {"precision": "fp32", "allow_tf32": False}\n\n\ndef test_irodori_precision_uses_bf16_tf32_only_on_audited_ampere_or_newer(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    for capability in ("8.0", "8.6", "8.9", "9.0", "10.0", "12.0"):\n        policy = hardware.irodori_training_precision("cu128", gpu=_gpu(capability))\n        assert policy == {"precision": "bf16", "allow_tf32": True}\n\n\ndef test_irodori_precision_fails_closed_for_unknown_or_mismatched_gpu(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    assert hardware.irodori_training_precision("cu128", gpu=_gpu("13.0")) == {\n        "precision": "fp32",\n        "allow_tf32": False,\n    }\n    assert hardware.irodori_training_precision("cpu", gpu=_gpu("8.6")) == {\n        "precision": "fp32",\n        "allow_tf32": False,\n    }\n''',
    encoding="utf-8",
    newline="\n",
)

# Windows setup becomes repository-local and reproducible; runtime remains local-only.
patch(
    "README.md",
    "必要: `uv`, Git, FFmpeg/ffprobe。NVIDIA GPU推奨。WindowsではTorchCodec用にshared DLL付きFFmpeg 4〜8が必要で、bootstrapはWinGetのFFmpeg Shared 8.1.1を使用します。",
    "必要: `uv`, Git。NVIDIA GPU推奨。WindowsではTorchCodec用の監査済みshared FFmpeg 8.1.1を`persona setup`が`.runtime/tools`へ自動materializeするため、システム全体へのFFmpeg/WinGetインストールは不要です。Linux/macOSではFFmpeg 4〜8の実行ファイルとshared librariesを用意してください。",
)
patch(
    "docs/TROUBLESHOOTING.md",
    "The recommended Windows bootstrap installs and verifies the audited shared build automatically:\n\n```powershell\n.\\scripts\\bootstrap.ps1\n```\n\nIf you need to install it manually:\n\n```powershell\nwinget install --id Gyan.FFmpeg.Shared --exact --version 8.1.1\n```\n\nThe WinGet shared package may not update the current PowerShell PATH immediately. PersonaVoice also scans the WinGet package directory directly, so rerunning bootstrap/setup in the same shell is supported. You can explicitly point to a compatible `bin` directory with `PERSONAVOICE_FFMPEG_BIN` if necessary.",
    "On Windows, `persona setup` obtains the exact Gyan shared FFmpeg 8.1.1 ZIP from its versioned official GitHub release, verifies the SHA256 independently published in Microsoft WinGet, extracts only its runtime `bin` tree into gitignored `.runtime/tools`, validates the required executable/DLL hashes, and atomically publishes the verified runtime. Normal build/inference/doctor paths never download FFmpeg.\n\n```powershell\nuv run --locked persona setup --backend auto\n```\n\nA compatible explicit override remains supported through `PERSONAVOICE_FFMPEG_BIN`; an invalid explicit override fails closed rather than being silently ignored. Existing compatible PATH/WinGet installations remain discoverable at runtime, but the default Windows setup is repository-local and reproducible.",
)

print("post-PR14 runtime hardening integration patches applied")
''