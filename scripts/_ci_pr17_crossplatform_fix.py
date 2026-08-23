from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise RuntimeError(
            f"Expected exactly {count} patch anchors in {path}, found {found}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


patch(
    "src/personavoice/ffmpeg_materializer.py",
    "    FfmpegRuntime,\n    _candidate_runtime,\n",
    "    FfmpegRuntime,\n    _candidate_runtime,\n    _discover_ffmpeg_runtime,\n",
)
patch(
    "src/personavoice/ffmpeg_materializer.py",
    "    if platform.system() == \"Windows\":\n"
    "        runtime = materialize_windows_ffmpeg(repo_root)\n"
    "    else:\n"
    "        runtime = require_ffmpeg_runtime()\n"
    "    provenance = ffmpeg_provenance(runtime)\n",
    "    if platform.system() == \"Windows\":\n"
    "        runtime = materialize_windows_ffmpeg(repo_root)\n"
    "    else:\n"
    "        # Setup is the authorization boundary. Use raw discovery here so a\n"
    "        # deliberate system/override FFmpeg upgrade can be validated and\n"
    "        # recorded instead of being rejected against the previous setup's\n"
    "        # provenance. Normal runtime paths still use require_ffmpeg_runtime.\n"
    "        runtime = _discover_ffmpeg_runtime()\n"
    "        if (\n"
    "            runtime.ffmpeg is None\n"
    "            or runtime.ffprobe is None\n"
    "            or not runtime.torchcodec_compatible\n"
    "        ):\n"
    "            raise RuntimeError(runtime.error or \"A compatible FFmpeg runtime is required\")\n"
    "    provenance = ffmpeg_provenance(runtime)\n",
)
patch(
    "tests/test_ffmpeg_setup_provenance.py",
    'monkeypatch.setattr(ffmpeg_materializer, "require_ffmpeg_runtime", lambda: expected)',
    'monkeypatch.setattr(ffmpeg_materializer, "_discover_ffmpeg_runtime", lambda: expected)',
)

path = ROOT / "tests/test_ffmpeg_portability_followup.py"
text = path.read_text(encoding="utf-8")
anchor = "\ndef test_windows_bootstrap_never_installs_or_requires_ffmpeg_before_setup():\n"
insert = '''\ndef test_non_windows_setup_can_reauthorize_changed_ffmpeg(tmp_path, monkeypatch):\n    bin_dir = tmp_path / "ffmpeg-bin"\n    bin_dir.mkdir()\n    ffmpeg = bin_dir / "ffmpeg"\n    ffprobe = bin_dir / "ffprobe"\n    ffmpeg.write_bytes(b"new-ffmpeg-generation")\n    ffprobe.write_bytes(b"new-ffprobe-generation")\n    discovered = runtime_dependencies.FfmpegRuntime(\n        ffmpeg=str(ffmpeg),\n        ffprobe=str(ffprobe),\n        bin_dir=str(bin_dir),\n        version_major=8,\n        shared_libraries=True,\n        torchcodec_compatible=True,\n        source="PATH",\n        error=None,\n    )\n    monkeypatch.setattr(ffmpeg_materializer.platform, "system", lambda: "Linux")\n    monkeypatch.setattr(\n        ffmpeg_materializer,\n        "_discover_ffmpeg_runtime",\n        lambda: discovered,\n    )\n\n    result = ffmpeg_materializer.ensure_ffmpeg_runtime(tmp_path)\n    assert result is discovered\n    assert runtime_dependencies.recorded_ffmpeg_provenance(tmp_path) == (\n        runtime_dependencies.ffmpeg_provenance(discovered)\n    )\n\n\n'''
if text.count(anchor) != 1:
    raise RuntimeError("FFmpeg portability insertion anchor not found exactly once")
path.write_text(text.replace(anchor, "\n" + insert + anchor.lstrip("\n"), 1), encoding="utf-8", newline="\n")

print("PR17 cross-platform FFmpeg reauthorization fix applied")
