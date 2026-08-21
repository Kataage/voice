from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from personavoice.captions import annotate_text, build_caption
from personavoice.config import PersonaConfig
from personavoice.hardware import nvidia_gpus
from personavoice.irodori import base_checkpoint, reference_files, vendor_dir
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
        return reference_files(paths.references)
    candidate = Path(ref)
    if candidate.exists():
        return [candidate.resolve()]
    named = paths.references / "by_emotion" / str(ref).lower()
    if named.exists():
        files = sorted(named.glob("*.flac"))
        if files:
            return files
    raise FileNotFoundError(f"Reference {ref!r} is neither a file nor a known reference preset")


def _irodori_refs(paths: PersonaPaths, ref: str | Path | None) -> list[Path]:
    return resolve_reference(paths, ref)


def _best_lora_adapter(paths: PersonaPaths) -> Path | None:
    root = paths.models / "irodori" / "lora"
    candidates: list[tuple[float, Path]] = []
    for path in root.glob("checkpoint_best_val_loss_*"):
        if not path.is_dir():
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
    return final if final.exists() else None


def _verify_outputs(paths: list[Path]) -> list[Path]:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size <= 44]
    if missing:
        raise RuntimeError(
            "Irodori finished without creating valid output WAV(s): "
            + ", ".join(str(path) for path in missing)
        )
    return paths


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
    output = output or (paths.outputs / f"tts_{_stamp()}.wav")
    output.parent.mkdir(parents=True, exist_ok=True)

    requested = cfg.inference.default_candidates if candidates is None else candidates
    if requested < 1:
        raise ValueError("candidates must be at least 1")
    gpus = nvidia_gpus()
    if not gpus or max(gpu.total_mib for gpu in gpus) < 16000:
        requested = 1
    else:
        requested = min(requested, 4)

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
    speaker = paths.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors"
    refs = _irodori_refs(paths, ref)
    if cfg.inference.reference_mode != "audio" and speaker.exists() and ref is None:
        args += ["--ref-embed", speaker]
    elif refs:
        if len(refs) == 1:
            args += ["--ref-wav", refs[0]]
        else:
            args += ["--ref-wavs", *refs]
    else:
        args += ["--no-ref"]
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
    refs = reference_files(paths.references)
    if not refs:
        raise FileNotFoundError("No reference bank exists. Run `persona prepare` first.")
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
    output_dir = paths.outputs / "reenact"
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
            "cfm_checkpoint": str(cfm.resolve()) if cfm.exists() else None,
        },
    )
    output = Path(result["output"])
    if not output.is_file() or output.stat().st_size <= 44:
        raise RuntimeError(f"Seed-VC returned an invalid output: {output}")
    return output


def repeat(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig, source: Path) -> list[Path]:
    _ensure_authorized(cfg)
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
        str(segment.get("text") or "").strip()
        for segment in asr.get("segments", [])
    ).strip()
    sense = worker(repo_root, "sense").call(
        repo_root,
        "analyze",
        {"audio": str(source.resolve()), "language": cfg.language},
    )
    if not text:
        if sense.get("events"):
            return [reenact(repo_root, paths, cfg, source, transfer_style=True)]
        raise RuntimeError(
            "No speech or supported non-verbal event could be detected in the source audio"
        )
    return synthesize(
        repo_root,
        paths,
        cfg,
        text,
        emotion=sense.get("emotion"),
        events=sense.get("events") or [],
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
        "voice": {
            "caption": "自然に話している。",
            "emotion": "NEUTRAL",
            "events": [],
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
        {"messages": messages, "adapter": str(adapter) if adapter.exists() else None},
    )
    plan = _extract_json(result["text"])
    voice = plan.get("voice") if isinstance(plan.get("voice"), dict) else {}
    audio = synthesize(
        repo_root,
        paths,
        cfg,
        str(plan.get("text") or ""),
        style=voice.get("caption"),
        emotion=voice.get("emotion"),
        events=voice.get("events") or [],
        candidates=1,
    )[0]
    plan["audio"] = str(audio)
    return plan
