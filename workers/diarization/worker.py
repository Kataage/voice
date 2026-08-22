from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

_DLL_DIRECTORY_HANDLES: list[object] = []
_AUDIO_SAMPLE_RATE = 16000


def _configure_windows_ffmpeg_dll_search() -> None:
    if os.name != "nt":
        return
    value = os.getenv("PERSONAVOICE_FFMPEG_BIN")
    if not value:
        return
    directory = Path(value)
    if not directory.is_dir():
        return
    # Python 3.8+ no longer searches arbitrary PATH directories for extension
    # dependencies as broadly as older Windows runtimes did. Keep the handle
    # alive for the entire worker process so TorchCodec can resolve FFmpeg's
    # avutil/avcodec/avformat/swresample DLLs while pyannote imports.
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(directory)))
    current_path = os.environ.get("PATH", "")
    if str(directory) not in current_path.split(os.pathsep):
        os.environ["PATH"] = str(directory) + (os.pathsep + current_path if current_path else "")


_configure_windows_ffmpeg_dll_search()

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from pyannote.audio import Pipeline  # noqa: E402

MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
MODEL_ASSET_SHA256 = {
    "config.yaml": "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
    "embedding/pytorch_model.bin": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
    "segmentation/pytorch_model.bin": "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
    "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
}
REVISION_MARKER = ".personavoice-revision"
REQUIRED_MODEL_FILES = (
    "config.yaml",
    "embedding/pytorch_model.bin",
    "segmentation/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_assets(local: Path) -> None:
    for relative, expected in MODEL_ASSET_SHA256.items():
        actual = _sha256(local / relative)
        if actual != expected:
            raise RuntimeError(
                f"pyannote checksum mismatch for {relative}: expected {expected}, got {actual}. "
                "Re-run `persona setup --download-models` with HF_TOKEN available."
            )


def _read_marker(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None
    except OSError:
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _materialization_complete(local: Path) -> bool:
    return all(_nonempty_file(local / relative) for relative in REQUIRED_MODEL_FILES)


def local_source() -> str:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "pyannote" / "community-1"
    marker = local / REVISION_MARKER
    if not _materialization_complete(local):
        missing = [
            relative
            for relative in REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned pyannote model is missing or incomplete: "
            f"{local} (invalid: {', '.join(missing)}). "
            "Run `persona setup --download-models`."
        )
    actual_revision = _read_marker(marker)
    if actual_revision != MODEL_REVISION:
        raise RuntimeError(
            "Local pyannote snapshot does not match the audited revision: "
            f"expected {MODEL_REVISION}, got {actual_revision!r}. "
            "Re-run `persona setup --download-models` with HF_TOKEN available."
        )
    _verify_assets(local)
    return str(local)


def _ffmpeg_executable() -> str:
    explicit = os.getenv("PERSONAVOICE_FFMPEG_BIN")
    if explicit:
        executable = Path(explicit) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if executable.is_file():
            return str(executable)
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    raise FileNotFoundError(
        "A verified FFmpeg executable is required to preload audio for pyannote. "
        "Run `persona setup` or `scripts/bootstrap.ps1` first."
    )


def _preload_audio(audio: str) -> dict:
    """Decode a local file to a pyannote waveform without TorchCodec file I/O."""

    source = Path(audio)
    if not source.is_file():
        raise FileNotFoundError(f"Diarization input is not a local file: {source}")
    completed = subprocess.run(
        [
            _ffmpeg_executable(),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_AUDIO_SAMPLE_RATE),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed to decode diarization audio: {stderr or 'unknown error'}")
    if not completed.stdout:
        raise RuntimeError("FFmpeg produced no PCM samples for diarization audio")
    if len(completed.stdout) % 4:
        raise RuntimeError("FFmpeg produced a malformed float32 PCM stream for diarization audio")

    # Convert the pipe-owned bytearray to an owned float32 tensor before the
    # local buffer goes out of scope. Shape is [channels, samples] as pyannote
    # expects for in-memory waveform input.
    waveform = torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32).clone().unsqueeze(0)
    return {"waveform": waveform, "sample_rate": _AUDIO_SAMPLE_RATE}


def load_pipeline() -> Pipeline:
    token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(local_source(), token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def annotation_rows(annotation) -> list[dict]:
    if annotation is None:
        return []
    return [
        {
            "start": round(float(segment.start), 4),
            "end": round(float(segment.end), 4),
            "speaker": str(speaker),
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]


def diarize_with_pipeline(pipeline: Pipeline, audio: str, *, force_one: bool = False) -> dict:
    output = pipeline(_preload_audio(audio), num_speakers=1 if force_one else None)
    diarization = output.speaker_diarization
    labels = [str(label) for label in diarization.labels()]
    embeddings = getattr(output, "speaker_embeddings", None)
    mapped = {}
    if embeddings is not None:
        for index, label in enumerate(labels):
            if index < len(embeddings):
                mapped[label] = [float(v) for v in embeddings[index].tolist()]
    exclusive = getattr(output, "exclusive_speaker_diarization", None) or diarization
    return {
        "turns": annotation_rows(diarization),
        "exclusive_turns": annotation_rows(exclusive),
        "speaker_embeddings": mapped,
    }


def diarize(payload: dict, *, force_one: bool = False) -> dict:
    pipeline = load_pipeline()
    return diarize_with_pipeline(pipeline, payload["audio"], force_one=force_one)


def embed_with_pipeline(pipeline: Pipeline, audio: str) -> dict:
    result = diarize_with_pipeline(pipeline, audio, force_one=True)
    embeddings = result["speaker_embeddings"]
    if not embeddings:
        raise RuntimeError("No speaker embedding could be extracted from identity audio")
    return {"embedding": next(iter(embeddings.values()))}


def embed(payload: dict) -> dict:
    pipeline = load_pipeline()
    return embed_with_pipeline(pipeline, payload["audio"])


def batch(payload: dict) -> dict:
    pipeline = load_pipeline()
    results: dict[str, list[dict]] = {"embeddings": [], "diarizations": []}
    for item in payload.get("embeddings") or []:
        item_id = str(item["id"])
        try:
            value = embed_with_pipeline(pipeline, str(item["audio"]))
            results["embeddings"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results["embeddings"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    for item in payload.get("diarizations") or []:
        item_id = str(item["id"])
        try:
            value = diarize_with_pipeline(pipeline, str(item["audio"]))
            results["diarizations"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results["diarizations"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    return results


def download(payload: dict) -> dict:
    root = Path(os.environ["PERSONAVOICE_ROOT"])
    local = root / "models" / "pyannote" / "community-1"
    token = os.getenv("HF_TOKEN")
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=local,
        cache_dir=Path(os.environ["HF_HOME"]),
        token=token,
    )
    if not _materialization_complete(local):
        missing = [
            relative
            for relative in REQUIRED_MODEL_FILES
            if not _nonempty_file(local / relative)
        ]
        raise FileNotFoundError(
            "Pinned pyannote download completed without required model files: "
            f"{', '.join(missing)}"
        )
    _verify_assets(local)
    _atomic_write_text(local / REVISION_MARKER, MODEL_REVISION + "\n")
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "path": str(local)}


def health(payload: dict) -> dict:
    result = {"ok": True, "cuda": torch.cuda.is_available()}
    if payload.get("deep"):
        pipeline = load_pipeline()
        result["model_loaded"] = pipeline is not None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["diarize", "embed", "batch", "download", "health"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "diarize":
        result = diarize(payload)
    elif args.command == "embed":
        result = embed(payload)
    elif args.command == "batch":
        result = batch(payload)
    elif args.command == "download":
        result = download(payload)
    else:
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
