from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from personavoice.artifacts import verify_publication
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
    irodori_full_artifact_complete,
    lora_adapter_complete,
    reference_files,
    speaker_embedding_complete,
    vendor_dir,
)
from personavoice.lfm_contract import (
    LFM_CONTRACT_FINGERPRINT,
    LFM_CONTRACT_SCHEMA_VERSION,
    LFM_MAX_NEW_TOKENS,
    LFM_RETRY_MAX_NEW_TOKENS,
    LFMContractError,
    build_lfm_messages,
    normalize_lfm_output,
)
from personavoice.process import run
from personavoice.profile import load_core_profile
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

VC_BACKENDS = ("seed-vc-v2", "vevo2-fm")


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


def _sha256_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


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
    # Irodori's pinned CLI accepts one caption string.  Keep the LFM caption as
    # the primary acting guidance, then append canonical emotion/event cues so
    # the normalized plan is acoustically representable without inventing any
    # event that was not present in ``voice.events``.
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


def _publication_contract_present(paths: PersonaPaths) -> bool:
    marker = paths.models / "publication.json"
    return marker.exists() or marker.is_symlink()


def _verified_publication(paths: PersonaPaths) -> dict[str, Any] | None:
    marker = paths.models / "publication.json"
    if marker.is_symlink():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    plan_fingerprint = value.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or re.fullmatch(
        r"[0-9a-f]{64}", plan_fingerprint
    ) is None:
        return None
    try:
        return verify_publication(
            paths.models,
            expected_plan_fingerprint=plan_fingerprint,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def _published_primary_artifact(
    paths: PersonaPaths,
    family: str,
) -> tuple[str, Path] | None:
    value = _verified_publication(paths)
    if value is None:
        return None
    artifacts = value["artifacts"]
    root = paths.models.resolve()
    for artifact in artifacts:
        if artifact.get("family") != family or artifact.get("component") != "primary":
            continue
        method = artifact["method"]
        relative = artifact.get("path")
        candidate = (root / relative).resolve()
        return method, candidate
    return None


def _published_artifact(paths: PersonaPaths, family: str, method: str) -> Path | None:
    selected = _published_primary_artifact(paths, family)
    if selected is None or selected[0] != method:
        return None
    return selected[1]


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
    *,
    mode_override: str | None = None,
    speaker_embedding_override: Path | None = None,
    allow_legacy_speaker_embedding: bool = True,
) -> None:
    speaker = speaker_embedding_override
    if speaker is None and allow_legacy_speaker_embedding:
        speaker = paths.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors"
    if ref is not None:
        _append_audio_reference_args(args, resolve_reference(paths, ref))
        return

    mode = mode_override or cfg.inference.reference_mode
    if mode == "none":
        args += ["--no-ref"]
        return
    if mode == "speaker-embed":
        if speaker is None or not speaker_embedding_complete(speaker):
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

    if speaker is not None and speaker_embedding_complete(speaker):
        args += ["--ref-embed", speaker]
    elif refs:
        _append_audio_reference_args(args, refs)
    else:
        args += ["--no-ref"]


def _effective_reference_mode(
    paths: PersonaPaths,
    cfg: PersonaConfig,
    ref: str | Path | None,
    *,
    mode_override: str | None,
    speaker_embedding_override: Path | None,
    allow_legacy_speaker_embedding: bool,
) -> str:
    """Describe the conditioning actually selected by ``_append_reference_args``."""

    if ref is not None:
        return "audio"
    mode = mode_override or cfg.inference.reference_mode
    if mode != "auto":
        return mode
    speaker = speaker_embedding_override
    if speaker is None and allow_legacy_speaker_embedding:
        speaker = paths.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors"
    if speaker is not None and speaker_embedding_complete(speaker):
        return "speaker-embed"
    if any(_nonempty_file(path) for path in reference_files(paths.references)):
        return "audio"
    return "none"


def _reference_fingerprint(
    paths: PersonaPaths,
    ref: str | Path | None,
    *,
    effective_mode: str,
    speaker_embedding: Path | None,
) -> str:
    """Return a content fingerprint without persisting local paths or audio."""

    files: list[dict[str, str]] = []
    try:
        if ref is not None:
            selected = resolve_reference(paths, ref)
        elif effective_mode == "audio":
            selected = [
                path for path in reference_files(paths.references) if _nonempty_file(path)
            ]
        elif effective_mode == "speaker-embed" and speaker_embedding is not None:
            selected = [speaker_embedding]
        else:
            selected = []
        for path in selected:
            digest = _sha256_file(path)
            if digest is not None:
                files.append({"sha256": digest, "kind": path.suffix.lower()})
    except (FileNotFoundError, OSError, RuntimeError):
        return "unavailable"
    payload = json.dumps(
        {"mode": effective_mode, "files": files},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    irodori_artifact: Path | None = None,
    irodori_method: str | None = None,
    base_only: bool = False,
    reference_mode: str | None = None,
    caption_conditioning: bool = True,
    duration_scale: float | None = None,
    trim_tail: bool | None = None,
    tail_window_size: int | None = None,
    tail_std_threshold: float | None = None,
    tail_mean_threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
    capture_logs: bool = False,
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

    effective_duration_scale = (
        cfg.inference.duration_scale if duration_scale is None else float(duration_scale)
    )
    effective_trim_tail = cfg.inference.trim_tail if trim_tail is None else bool(trim_tail)
    effective_tail_window_size = (
        cfg.inference.tail_window_size
        if tail_window_size is None
        else int(tail_window_size)
    )
    effective_tail_std_threshold = (
        cfg.inference.tail_std_threshold
        if tail_std_threshold is None
        else float(tail_std_threshold)
    )
    effective_tail_mean_threshold = (
        cfg.inference.tail_mean_threshold
        if tail_mean_threshold is None
        else float(tail_mean_threshold)
    )
    if not math.isfinite(effective_duration_scale) or effective_duration_scale <= 0:
        raise ValueError("duration_scale must be finite and greater than zero")
    if effective_tail_window_size <= 0:
        raise ValueError("tail_window_size must be greater than zero")
    if (
        not math.isfinite(effective_tail_std_threshold)
        or effective_tail_std_threshold < 0
        or not math.isfinite(effective_tail_mean_threshold)
        or effective_tail_mean_threshold < 0
    ):
        raise ValueError("tail thresholds must be finite and non-negative")

    checkpoint = base
    lora: Path | None = None
    speaker_override: Path | None = None
    full_persona = False
    method = "base-only" if base_only else "base"
    publication_present = _publication_contract_present(paths)
    if base_only and (irodori_artifact is not None or irodori_method is not None):
        raise ValueError("base_only cannot be combined with an Irodori candidate override")
    if irodori_artifact is not None:
        method = irodori_method
        if method == "full":
            if not irodori_full_artifact_complete(irodori_artifact):
                raise RuntimeError("Irodori full-model candidate failed portable verification")
            checkpoint = irodori_artifact / "model.safetensors"
            full_persona = True
            method = "full"
        elif method == "lora":
            if not lora_adapter_complete(irodori_artifact):
                raise RuntimeError("Irodori LoRA candidate is incomplete")
            lora = irodori_artifact
            method = "lora"
        elif method == "speaker-inversion":
            if not speaker_embedding_complete(irodori_artifact):
                raise RuntimeError("Irodori speaker-inversion candidate is incomplete")
            speaker_override = irodori_artifact
            method = "speaker-inversion"
        else:
            raise ValueError("irodori_method is required for a candidate override")
    elif not base_only:
        published = _published_primary_artifact(paths, "irodori")
        if published is not None:
            method, artifact = published
            if method == "full":
                checkpoint = artifact / "model.safetensors"
                full_persona = True
            elif method == "lora":
                lora = artifact
            elif method == "speaker-inversion":
                speaker_override = artifact
            # ``method`` is the value selected by the verified publication.
            # Keep it separate from the local branch variable used above.
            method = str(method)
        elif not publication_present:
            lora = _best_lora_adapter(paths)
            if lora is not None:
                method = "lora"

    caption = _caption(style, emotion, events) if caption_conditioning else None
    args: list[str | Path] = [
        "uv",
        "run",
        "--project",
        vendor,
        "--no-sync",
        "python",
        vendor / "infer.py",
        "--checkpoint",
        checkpoint,
        "--codec-repo",
        codec,
        "--model-device",
        device,
        "--codec-device",
        device,
        "--text",
        text,
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
        "--duration-scale",
        str(effective_duration_scale),
        "--trim-tail" if effective_trim_tail else "--no-trim-tail",
        "--tail-window-size",
        str(effective_tail_window_size),
        "--tail-std-threshold",
        str(effective_tail_std_threshold),
        "--tail-mean-threshold",
        str(effective_tail_mean_threshold),
        "--output-wav",
        output,
    ]
    if caption is not None:
        args += ["--caption", caption]
    if lora is not None:
        args += ["--lora-adapter", lora]
    effective_reference_mode = reference_mode
    if (
        effective_reference_mode is None
        and ref is None
        and full_persona
        and cfg.inference.reference_mode == "auto"
    ):
        # A full persona checkpoint owns its default identity in model weights.
        # Keep explicit audio/speaker modes available, but do not silently make
        # the standard full-model path depend on an external reference bank.
        effective_reference_mode = "none"
    _append_reference_args(
        args,
        paths,
        cfg,
        ref,
        mode_override=effective_reference_mode,
        speaker_embedding_override=speaker_override,
        allow_legacy_speaker_embedding=not publication_present,
    )
    selected_reference_mode = _effective_reference_mode(
        paths,
        cfg,
        ref,
        mode_override=effective_reference_mode,
        speaker_embedding_override=speaker_override,
        allow_legacy_speaker_embedding=not publication_present,
    )
    if seed is not None:
        args += ["--seed", str(seed)]
    completed = run(
        args,
        cwd=vendor,
        env=local_model_env(repo_root),
        capture=capture_logs,
    )
    stdout = getattr(completed, "stdout", "") or ""
    if requested == 1:
        generated = _verify_outputs([output])
    else:
        suffix = output.suffix or ".wav"
        generated = _verify_outputs(
            [
                output.with_name(f"{output.stem}_{index:03d}{suffix}")
                for index in range(1, requested + 1)
            ]
        )
    if metadata is not None:
        metadata.update(
            {
                "requested_text": text,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "method": method,
                "reference_mode": selected_reference_mode,
                "reference_fingerprint": _reference_fingerprint(
                    paths,
                    ref,
                    effective_mode=selected_reference_mode,
                    speaker_embedding=speaker_override,
                ),
                "duration_scale": effective_duration_scale,
                "trim_tail": effective_trim_tail,
                "tail_window_size": effective_tail_window_size,
                "tail_std_threshold": effective_tail_std_threshold,
                "tail_mean_threshold": effective_tail_mean_threshold,
                "command": [str(value) for value in args],
                "stdout": stdout,
            }
        )
    return generated


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
    backend: str | None = None,
    output_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    _ensure_authorized(cfg)
    if not _nonempty_file(source):
        raise FileNotFoundError(f"Source audio is missing or empty: {source}")
    selected_backend = cfg.vc_backend if backend is None else backend
    if selected_backend not in VC_BACKENDS:
        raise ValueError(
            f"Unsupported VC backend {selected_backend!r}; choose one of {', '.join(VC_BACKENDS)}"
        )
    output_dir = output_dir or (paths.outputs / "reenact" / _stamp())
    target = resolve_reference(paths, ref)[0] if ref is not None else _best_reference(paths)
    if selected_backend == "seed-vc-v2":
        cfm = paths.models / "seed_vc" / "cfm.pth"
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
    else:
        result = worker(repo_root, "vevo2").call(
            repo_root,
            "convert",
            {
                "source": str(source.resolve()),
                "target": str(target.resolve()),
                "output_dir": str(output_dir.resolve()),
                "flow_matching_steps": cfg.inference.vevo2_flow_matching_steps,
                "use_pitch_shift": cfg.inference.vevo2_use_pitch_shift,
                "dtype": cfg.inference.vevo2_dtype,
            },
        )
    output = Path(result["output"])
    try:
        valid = output.is_file() and output.stat().st_size > 44
    except OSError:
        valid = False
    if not valid:
        raise RuntimeError(f"{selected_backend} returned an invalid output: {output}")
    if metadata is not None:
        metadata.update(
            {
                "backend": selected_backend,
                "source": str(source.resolve()),
                "reference": str(target.resolve()),
                "worker_result": result,
            }
        )
    return output


def repeat(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    source: Path,
    *,
    backend: str | None = None,
) -> list[Path]:
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
            return [
                reenact(
                    repo_root,
                    paths,
                    cfg,
                    source,
                    transfer_style=True,
                    backend=backend,
                )
            ]
        raise RuntimeError(
            "No speech or supported non-verbal event could be detected in the source audio"
        )
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
    profile = load_core_profile(paths.core_profile, persona_name=cfg.name)
    messages, message_diagnostics = build_lfm_messages(profile, history, prompt)
    publication_present = _publication_contract_present(paths)
    published = _published_primary_artifact(paths, "lfm")
    full_model = published[1] if published is not None and published[0] == "full" else None
    published_adapter = (
        published[1] if published is not None and published[0] == "lora" else None
    )
    legacy_adapter = paths.models / "lfm" / "adapter"
    worker_client = worker(repo_root, "lfm")
    model_payload = {
        "full_model": (
            str(full_model) if full_model is not None and full_model.is_dir() else None
        ),
        "adapter": (
            str(published_adapter)
            if published_adapter is not None
            else (
                str(legacy_adapter)
                if full_model is None and not publication_present and legacy_adapter.is_dir()
                else None
            )
        ),
    }
    attempts = 0
    failures: list[dict[str, str]] = []
    plan = None
    active_messages = messages
    for attempt in range(2):
        attempts += 1
        result = worker_client.call(
            repo_root,
            "infer",
            {
                "messages": active_messages,
                **model_payload,
                "temperature": 0.1,
                "max_new_tokens": (
                    LFM_MAX_NEW_TOKENS if attempt == 0 else LFM_RETRY_MAX_NEW_TOKENS
                ),
            },
        )
        raw = result.get("text") if isinstance(result, dict) else None
        try:
            plan = normalize_lfm_output(raw)
            break
        except LFMContractError as exc:
            failures.append({"code": exc.code, "message": str(exc)})
            if attempt == 1:
                raise RuntimeError(
                    "LFM output contract failed after two bounded attempts: "
                    + ", ".join(item["code"] for item in failures)
                ) from exc
            active_messages, retry_diagnostics = build_lfm_messages(
                profile,
                history,
                prompt,
                repair_notice=(
                    f"前回の違反コードは{exc.code}。空出力、繰り返し文字、malformed JSONを返さず、"
                    "spoken textまたはsupported non-verbal-only planを、指定schemaで返す。"
                ),
            )
            message_diagnostics = tuple(
                list(message_diagnostics) + list(retry_diagnostics) + ["bounded_retry"]
            )
    if plan is None:  # pragma: no cover - the loop either returns or raises
        raise RuntimeError("LFM did not produce a normalized plan")
    voice = plan.as_voice()
    audio = synthesize(
        repo_root,
        paths,
        cfg,
        plan.text,
        style=plan.caption,
        emotion=plan.emotion,
        events=list(plan.events),
        candidates=1,
    )[0]
    normalized = plan.as_dict()
    normalized["audio"] = str(audio)
    normalized["provenance"] = {
        "profile": {
            "schema_version": profile.schema_version,
            "fingerprint": profile.fingerprint,
        },
        "lfm": {
            "contract_schema_version": LFM_CONTRACT_SCHEMA_VERSION,
            "contract_fingerprint": LFM_CONTRACT_FINGERPRINT,
            "attempts": attempts,
            "recovered": plan.recovered,
            "source": plan.source,
            "diagnostics": list(plan.diagnostics),
            "bounded_failures": failures,
        },
        "message_diagnostics": list(message_diagnostics),
        "irodori_handoff": {
            "text": plan.text,
            "voice": voice,
            "non_verbal_only": plan.non_verbal_only,
        },
    }
    return normalized
