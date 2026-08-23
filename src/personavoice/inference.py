from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from personavoice.captions import (
    annotate_text,
    build_caption,
    normalize_emotion,
    normalize_events,
)
from personavoice.config import PersonaConfig
from personavoice.hardware import selected_nvidia_gpu
from personavoice.irodori import (
    backend_device,
    base_checkpoint,
    codec_checkpoint,
    configured_backend,
    lora_adapter_complete,
    reference_files,
    speaker_embedding_complete,
    vendor_dir,
)
from personavoice.process import run
from personavoice.project import PersonaPaths
from personavoice.workers import local_model_env, worker

STYLE_PRESETS = {
    "whisper": "小声でささやくように、近い距離感で",
    "soft": "柔らかく優しく、落ち着いた声で",
    "excited": "テンション高めで、弾むように楽しそうに",
    "calm": "穏やかで落ち着いた声で",
    "happy": "明るく嬉しそうに",
    "sad": "悲しげで弱々しく",
    "angry": "苛立ちと怒りを込めて",
    "surprised": "驚いて一瞬息を呑むように",
}


def _ensure_authorized(cfg: PersonaConfig) -> None:
    if not cfg.consent.authorized:
        raise PermissionError("Voice generation is blocked because consent.authorized is not true.")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _safe_candidate_count(requested: int, *, backend: str) -> int:
    """Clamp batched Irodori candidates to the actual logical CUDA device 0."""

    gpu = selected_nvidia_gpu() if backend in {"cu126", "cu128"} else None
    if gpu is None or gpu.total_mib < 16000:
        return 1
    return min(requested, 4)


def _caption(style: str | None, emotion: str | None, events: list[str] | None) -> str:
    pieces = []
    if style:
        style_value = STYLE_PRESETS.get(style.strip().lower(), style.strip())
        pieces.append(style_value.rstrip("。"))
    if emotion or events:
        pieces.append(
            build_caption(
                emotion=emotion,
                events=events,
                chars_per_second=None,
            ).rstrip("。")
        )
    return "。".join(piece for piece in pieces if piece) + (
        "。" if pieces else "自然に話している。"
    )


def resolve_reference(paths: PersonaPaths, ref: str | Path | None) -> list[Path]:
    if ref is None:
        return [path for path in reference_files(paths.references) if _nonempty_file(path)]
    candidate = Path(ref).expanduser()
    if _nonempty_file(candidate):
        return [candidate.resolve()]
    named = paths.references / "by_emotion" / str(ref).lower()
    if named.is_dir():
        files = [path for path in sorted(named.glob("*.flac")) if _nonempty_file(path)]
        if files:
            return files
    raise FileNotFoundError(
        f"Reference {ref!r} is neither a non-empty audio file nor a known reference preset"
    )


def _best_lora_adapter(paths: PersonaPaths) -> Path | None:
    root = paths.models / "irodori" / "lora"
    candidates: list[tuple[float, Path]] = []
    for path in root.glob("checkpoint_best_val_loss_*"):
        if not path.is_dir() or not lora_adapter_complete(path):
            continue
        try:
            score = float(path.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((score, path))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    final = root / "checkpoint_final"
    return final if lora_adapter_complete(final) else None


def _verify_outputs(paths: list[Path]) -> list[Path]:
    missing = []
    for path in paths:
        try:
            valid = path.is_file() and path.stat().st_size > 44
        except OSError:
            valid = False
        if not valid:
            missing.append(path)
    if missing:
        raise RuntimeError(
            "Irodori finished without creating valid output WAV(s): "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _append_audio_reference_args(args: list[str | Path], refs: list[Path]) -> None:
    valid_refs = [path for path in refs if _nonempty_file(path)]
    if not valid_refs:
        raise FileNotFoundError("No non-empty Irodori audio reference is available")
    if len(valid_refs) == 1:
        args += ["--ref-wav", valid_refs[0]]
    else:
        args += ["--ref-wavs", *valid_refs]


def _append_reference_args(
    args: list[str | Path],
    paths: PersonaPaths,
    cfg: PersonaConfig,
    ref: str | Path | None,
) -> None:
    speaker = paths.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors"
    if ref is not None:
        _append_audio_reference_args(args, resolve_reference(paths, ref))
        return

    mode = cfg.inference.reference_mode
    if mode == "speaker-embed":
        if not speaker_embedding_complete(speaker):
            raise FileNotFoundError(
                "inference.reference_mode is 'speaker-embed', but no complete trained speaker "
                "embedding exists. Run `persona train` with Irodori Speaker Inversion enabled "
                "or choose reference_mode: auto/audio."
            )
        args += ["--ref-embed", speaker]
        return

    refs = [path for path in reference_files(paths.references) if _nonempty_file(path)]
    if mode == "audio":
        if not refs:
            raise FileNotFoundError(
                "inference.reference_mode is 'audio', but the reference bank has no valid audio. "
                "Run `persona prepare` or pass --ref explicitly."
            )
        _append_audio_reference_args(args, refs)
        return

    if speaker_embedding_complete(speaker):
        args += ["--ref-embed", speaker]
    elif refs:
        _append_audio_reference_args(args, refs)
    else:
        args += ["--no-ref"]


def synthesize(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    text: str,
    *,
    style: str | None = None,
    emotion: str | None = None,
    events: list[str] | None = None,
    ref: str | Path | None = None,
    candidates: int | None = None,
    seed: int | None = None,
    output: Path | None = None,
) -> list[Path]:
    _ensure_authorized(cfg)
    if not text.strip():
        text = annotate_text("", events)
    if not text.strip():
        raise ValueError("text is empty; provide text or a supported non-verbal --event")
    vendor = vendor_dir(repo_root)
    base = base_checkpoint(repo_root)
    codec = codec_checkpoint(repo_root)
    backend = configured_backend(repo_root)
    device = backend_device(backend)
    output = output or (paths.outputs / f"tts_{_stamp()}.wav")
    output.parent.mkdir(parents=True, exist_ok=True)

    requested = cfg.inference.default_candidates if candidates is None else candidates
    if requested < 1:
        raise ValueError("candidates must be at least 1")
    requested = _safe_candidate_count(requested, backend=backend)

    args: list[str | Path] = [
        "uv",
        "run",
        "--project",
        vendor,
        "--no-sync",
        "python",
        vendor / "infer.py",
        "--checkpoint",
        base,
        "--codec-repo",
        codec,
        "--model-device",
        device,
        "--codec-device",
        device,
        "--text",
        text,
        "--caption",
        _caption(style, emotion, events),
        "--num-steps",
        str(cfg.inference.default_num_steps),
        "--num-candidates",
        str(requested),
        "--decode-mode",
        "sequential",
        "--cfg-scale-text",
        str(cfg.inference.tts_cfg_scale),
        "--cfg-scale-caption",
        str(cfg.inference.tts_cfg_scale),
        "--output-wav",
        output,
    ]
    lora = _best_lora_adapter(paths)
    if lora is not None:
        args += ["--lora-adapter", lora]
    _append_reference_args(args, paths, cfg, ref)
    if seed is not None:
        args += ["--seed", str(seed)]
    run(args, cwd=vendor, env=local_model_env(repo_root))
    if requested == 1:
        return _verify_outputs([output])
    suffix = output.suffix or ".wav"
    generated = [
        output.with_name(f"{output.stem}_{index:03d}{suffix}")
        for index in range(1, requested + 1)
    ]
    return _verify_outputs(generated)


def _best_reference(paths: PersonaPaths) -> Path:
    refs = [path for path in reference_files(paths.references) if _nonempty_file(path)]
    if not refs:
        raise FileNotFoundError("No valid reference bank exists. Run `persona prepare` first.")
    return refs[0]


def reenact(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    source: Path,
    *,
    ref: str | Path | None = None,
    transfer_style: bool = True,
) -> Path:
    _ensure_authorized(cfg)
    if not _nonempty_file(source):
        raise FileNotFoundError(f"Source audio is missing or empty: {source}")
    output_dir = paths.outputs / "reenact" / _stamp()
    cfm = paths.models / "seed_vc" / "cfm.pth"
    target = resolve_reference(paths, ref)[0] if ref is not None else _best_reference(paths)
    result = worker(repo_root, "seed_vc").call(
        repo_root,
        "convert",
        {
            "source": str(source.resolve()),
            "target": str(target.resolve()),
            "output_dir": str(output_dir.resolve()),
            "diffusion_steps": cfg.inference.seed_vc_diffusion_steps,
            "similarity_cfg_rate": cfg.inference.seed_vc_similarity_cfg,
            "intelligibility_cfg_rate": cfg.inference.seed_vc_intelligibility_cfg,
            "convert_style": transfer_style,
            "cfm_checkpoint": str(cfm.resolve()) if _nonempty_file(cfm) else None,
        },
    )
    output = Path(result["output"])
    try:
        valid = output.is_file() and output.stat().st_size > 44
    except OSError:
        valid = False
    if not valid:
        raise RuntimeError(f"Seed-VC returned an invalid output: {output}")
    return output


def repeat(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig, source: Path) -> list[Path]:
    _ensure_authorized(cfg)
    if not _nonempty_file(source):
        raise FileNotFoundError(f"Source audio is missing or empty: {source}")
    asr = worker(repo_root, "asr").call(
        repo_root,
        "transcribe",
        {
            "audio": str(source.resolve()),
            "model": cfg.prepare.asr_model,
            "compute_type": cfg.prepare.asr_compute_type,
            "language": cfg.language,
        },
    )
    text = "".join(
        str(segment.get("text") or "").strip() for segment in asr.get("segments", [])
    ).strip()
    sense = worker(repo_root, "sense").call(
        repo_root,
        "analyze",
        {"audio": str(source.resolve()), "language": cfg.language},
    )
    if not text:
        if sense.get("events"):
            return [reenact(repo_root, paths, cfg, source, transfer_style=True)]
        raise RuntimeError("No speech or supported non-verbal event could be detected in the source audio")
    return synthesize(
        repo_root,
        paths,
        cfg,
        text,
        emotion=sense.get("emotion"),
        events=normalize_events(sense.get("events") or []),
        candidates=1,
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {
        "text": text,
        "voice": {"caption": "自然に話している。", "emotion": "NEUTRAL", "events": []},
    }


def _normalize_chat_plan(value: dict[str, Any]) -> dict[str, Any]:
    text_value = value.get("text")
    text = text_value.strip() if isinstance(text_value, str) else str(text_value or "").strip()
    raw_voice = value.get("voice")
    voice = raw_voice if isinstance(raw_voice, dict) else {}
    caption_value = voice.get("caption")
    caption = caption_value.strip() if isinstance(caption_value, str) else ""
    emotion_value = voice.get("emotion")
    emotion = normalize_emotion(emotion_value if isinstance(emotion_value, str) else None)
    events_value = voice.get("events")
    if isinstance(events_value, str):
        raw_events = [events_value]
    elif isinstance(events_value, (list, tuple)):
        raw_events = [str(item) for item in events_value if isinstance(item, str)]
    else:
        raw_events = []
    return {
        "text": text,
        "voice": {
            "caption": caption or "自然に話している。",
            "emotion": emotion,
            "events": normalize_events(raw_events),
        },
    }


def chat_turn(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    prompt: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    _ensure_authorized(cfg)
    system = (
        f"あなたは{cfg.name}として自然に会話します。"
        "返答はJSONのみ。形式は "
        "{\"text\":\"...\",\"voice\":{\"caption\":\"...\","
        "\"emotion\":\"NEUTRAL\",\"events\":[]}}。"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": prompt})
    adapter = paths.models / "lfm" / "adapter"
    result = worker(repo_root, "lfm").call(
        repo_root,
        "infer",
        {"messages": messages, "adapter": str(adapter) if adapter.is_dir() else None},
    )
    plan = _normalize_chat_plan(_extract_json(str(result.get("text") or "")))
    voice = plan["voice"]
    audio = synthesize(
        repo_root,
        paths,
        cfg,
        plan["text"],
        style=voice["caption"],
        emotion=voice["emotion"],
        events=voice["events"],
        candidates=1,
    )[0]
    plan["audio"] = str(audio)
    return plan
