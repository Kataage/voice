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


# The caller owns the archive cache directory contract. Do not depend on a
# downloader implementation side effect to create the destination parent.
patch(
    "src/personavoice/ffmpeg_materializer.py",
    '    temp = archive.with_name(f".{archive.name}.{uuid4().hex}.tmp")\n    try:\n',
    '    archive.parent.mkdir(parents=True, exist_ok=True)\n'
    '    temp = archive.with_name(f".{archive.name}.{uuid4().hex}.tmp")\n'
    '    try:\n',
)

# External/shared FFmpeg compatibility is TorchCodec's core five-library
# requirement. The pinned Gyan archive contract separately verifies additional
# avdevice/avfilter DLL families for the exact local runtime we materialize.
patch(
    "src/personavoice/runtime_dependencies.py",
    '_WINDOWS_SHARED_DLL_PATTERNS = (\n'
    '    "avutil-*.dll",\n'
    '    "avcodec-*.dll",\n'
    '    "avformat-*.dll",\n'
    '    "avdevice-*.dll",\n'
    '    "avfilter-*.dll",\n'
    '    "swresample-*.dll",\n'
    '    "swscale-*.dll",\n'
    ')\n',
    '_WINDOWS_SHARED_DLL_PATTERNS = (\n'
    '    "avutil-*.dll",\n'
    '    "avcodec-*.dll",\n'
    '    "avformat-*.dll",\n'
    '    "swresample-*.dll",\n'
    '    "swscale-*.dll",\n'
    ')\n',
)
patch(
    "src/personavoice/runtime_dependencies.py",
    '            "FFmpeg executables were found, but the shared avutil/avcodec/avformat/"\n'
    '            "avdevice/avfilter/swresample/swscale DLLs required by TorchCodec were not "\n'
    '            "found beside them"\n',
    '            "FFmpeg executables were found, but the shared avutil/avcodec/avformat/"\n'
    '            "swresample/swscale DLLs required by TorchCodec were not found beside them"\n',
)

# Schema assertions should follow the exported contract constant rather than
# becoming stale every time the dependency/runtime contract intentionally bumps.
patch(
    "tests/test_gpu_runtime_contract.py",
    '    assert recorded["schema"] == 4\n',
    '    assert recorded["schema"] == env_contract.ENVIRONMENT_CONTRACT_SCHEMA\n',
)
patch(
    "tests/test_gpu_runtime_contract.py",
    '    monkeypatch.setattr(setup_env, "require_ffmpeg_runtime", lambda: None)\n',
    '    monkeypatch.setattr(setup_env, "ensure_ffmpeg_runtime", lambda _root: None)\n',
)
patch(
    "tests/test_seed_vc_pinned_assets.py",
    '    assert first["schema"] == 4\n',
    '    assert first["schema"] == ENVIRONMENT_CONTRACT_SCHEMA\n',
)
patch(
    "tests/test_seed_vc_pinned_assets.py",
    "from personavoice.environment_contract import environment_contract\n",
    "from personavoice.environment_contract import ENVIRONMENT_CONTRACT_SCHEMA, environment_contract\n",
)

print("post-PR14 follow-up fixes applied")
