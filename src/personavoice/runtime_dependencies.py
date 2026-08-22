from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_TORCHCODEC_FFMPEG_MAJORS = frozenset(range(4, 9))
_WINDOWS_SHARED_DLL_PATTERNS = (
    "avutil-*.dll",
    "avcodec-*.dll",
    "avformat-*.dll",
    "swresample-*.dll",
    "swscale-*.dll",
)
_FFMPEG_VERSION_RE = re.compile(r"^ffmpeg version\s+(?:n)?(\d+)(?:\.|\b)", re.IGNORECASE)


@dataclass(frozen=True)
class FfmpegRuntime:
    ffmpeg: str | None
    ffprobe: str | None
    bin_dir: str | None
    version_major: int | None
    shared_libraries: bool
    torchcodec_compatible: bool
    source: str | None
    error: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def _version_major(executable: Path) -> int | None:
    try:
        completed = subprocess.run(
            [str(executable), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    match = _FFMPEG_VERSION_RE.match(first_line)
    return int(match.group(1)) if match else None


def _windows_shared_libraries(directory: Path) -> bool:
    return all(any(directory.glob(pattern)) for pattern in _WINDOWS_SHARED_DLL_PATTERNS)


def _path_candidates() -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    explicit = os.getenv("PERSONAVOICE_FFMPEG_BIN")
    if explicit:
        candidates.append((Path(explicit).expanduser(), "PERSONAVOICE_FFMPEG_BIN"))

    for name in ("ffmpeg", "ffprobe"):
        value = shutil.which(name)
        if value:
            path = Path(value)
            try:
                path = path.resolve()
            except OSError:
                pass
            candidates.append((path.parent, "PATH"))

    if platform.system() == "Windows":
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if package_root.is_dir():
                for package in package_root.glob("Gyan.FFmpeg.Shared_*"):
                    try:
                        for executable in package.rglob("ffmpeg.exe"):
                            candidates.append((executable.parent, "WinGet:Gyan.FFmpeg.Shared"))
                    except OSError:
                        continue

    deduped: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for directory, source in candidates:
        try:
            key = os.path.normcase(str(directory.resolve()))
        except OSError:
            key = os.path.normcase(str(directory))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((directory, source))
    return deduped


def _candidate_runtime(directory: Path, source: str) -> FfmpegRuntime | None:
    suffix = ".exe" if platform.system() == "Windows" else ""
    ffmpeg = directory / f"ffmpeg{suffix}"
    ffprobe = directory / f"ffprobe{suffix}"
    if not ffmpeg.is_file() or not ffprobe.is_file():
        return None

    major = _version_major(ffmpeg)
    if platform.system() == "Windows":
        shared = _windows_shared_libraries(directory)
        compatible = bool(shared and major in SUPPORTED_TORCHCODEC_FFMPEG_MAJORS)
    else:
        # On Unix the dynamic loader owns the final shared-library resolution.
        # The executable/version check is still useful, while deep doctor proves
        # TorchCodec can load the system libraries in the actual worker process.
        shared = True
        compatible = major in SUPPORTED_TORCHCODEC_FFMPEG_MAJORS

    return FfmpegRuntime(
        ffmpeg=str(ffmpeg),
        ffprobe=str(ffprobe),
        bin_dir=str(directory),
        version_major=major,
        shared_libraries=shared,
        torchcodec_compatible=compatible,
        source=source,
        error=None,
    )


def ffmpeg_runtime() -> FfmpegRuntime:
    valid: list[FfmpegRuntime] = []
    partial: list[FfmpegRuntime] = []
    for directory, source in _path_candidates():
        runtime = _candidate_runtime(directory, source)
        if runtime is None:
            continue
        partial.append(runtime)
        if runtime.torchcodec_compatible:
            valid.append(runtime)

    if valid:
        # Prefer the newest TorchCodec-supported major, then an explicitly
        # configured runtime over auto-discovered locations.
        valid.sort(
            key=lambda value: (
                value.version_major or -1,
                value.source == "PERSONAVOICE_FFMPEG_BIN",
            ),
            reverse=True,
        )
        return valid[0]

    if partial:
        best = partial[0]
        if best.version_major not in SUPPORTED_TORCHCODEC_FFMPEG_MAJORS:
            detail = (
                f"FFmpeg major {best.version_major!r} is not supported by the audited "
                "TorchCodec runtime; use FFmpeg 4 through 8"
            )
        elif platform.system() == "Windows" and not best.shared_libraries:
            detail = (
                "FFmpeg executables were found, but the shared avutil/avcodec/avformat/"
                "swresample/swscale DLLs required by TorchCodec were not found beside them"
            )
        else:
            detail = "FFmpeg was found but is not compatible with the audited TorchCodec runtime"
        return FfmpegRuntime(
            ffmpeg=best.ffmpeg,
            ffprobe=best.ffprobe,
            bin_dir=best.bin_dir,
            version_major=best.version_major,
            shared_libraries=best.shared_libraries,
            torchcodec_compatible=False,
            source=best.source,
            error=detail,
        )

    return FfmpegRuntime(
        ffmpeg=None,
        ffprobe=None,
        bin_dir=None,
        version_major=None,
        shared_libraries=False,
        torchcodec_compatible=False,
        source=None,
        error=(
            "FFmpeg/ffprobe were not found. On Windows install the audited shared FFmpeg 8 "
            "runtime with `winget install --id Gyan.FFmpeg.Shared --exact --version 8.1.1`."
        ),
    )


def require_ffmpeg_runtime() -> FfmpegRuntime:
    runtime = ffmpeg_runtime()
    if runtime.ffmpeg is None or runtime.ffprobe is None or not runtime.torchcodec_compatible:
        raise RuntimeError(runtime.error or "A compatible FFmpeg runtime is required")
    return runtime


def command(name: str) -> str:
    runtime = require_ffmpeg_runtime()
    if name == "ffmpeg" and runtime.ffmpeg:
        return runtime.ffmpeg
    if name == "ffprobe" and runtime.ffprobe:
        return runtime.ffprobe
    raise ValueError(f"Unsupported FFmpeg command: {name!r}")


def ffmpeg_environment() -> dict[str, str]:
    runtime = ffmpeg_runtime()
    if not runtime.bin_dir:
        return {}
    path = os.environ.get("PATH", "")
    entries = [runtime.bin_dir]
    if path:
        entries.append(path)
    return {
        "PERSONAVOICE_FFMPEG_BIN": runtime.bin_dir,
        "PATH": os.pathsep.join(entries),
    }
