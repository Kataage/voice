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
        pieces.append(build_caption(emotion=emotion, events=events, chars_per_second=None).rstrip("。"))
    return "。".join(piece for piece in pieces if piece) + ("。" if pieces else "自然に話している。")


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
    requested = candidates or cfg.inference.default_candidates
    gpus = nvidia_gpus()
    if not gpus or max(g.total_mib for g in gpus) < 16000:
        requested = min(requested, 1)
    else:
        requested = min(max(1, requested), 4)
    args: list[str | Path] = [
        "uv", "run", "--project", vendor, "--no-sync", "python", vendor / "infer.py",
        "--checkpoint", base,
        "--text", text,
        "--caption", _caption(style, emotion, events),
        "--num-steps", str(cfg.inference.default_num_steps),
        "--num-candidates", str(requested),
        "--decode-mode", "sequential",
        "--output-wav", output,
    ]
    lora = paths.models / "irodori" / "lora" / "checkpoint_final"
    if lora.exists():
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
        return [output]
    suffix = output.suffix or ".wav"
    return [output.with_name(f"{output.stem}_{i:03d}{suffix}") for i in range(1, requested + 1)]


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
    result = worker(repo_root, "seed_vc").call(
        repo_root,
        "convert",
        {
            "source": str(source.resolve()),
            "target": str((resolve_reference(paths, ref)[0] if ref is not None else _best_reference(paths)).resolve()),
            "output_dir": str(output_dir.resolve()),
            "diffusion_steps": cfg.inference.seed_vc_diffusion_steps,
            "similarity_cfg_rate": cfg.inference.seed_vc_similarity_cfg,
            "intelligibility_cfg_rate": cfg.inference.seed_vc_intelligibility_cfg,
            "convert_style": transfer_style,
            "cfm_checkpoint": str(cfm.resolve()) if cfm.exists() else None,
        },
    )
    return Path(result["output"])


def repeat(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig, source: Path) -> list[Path]:
    asr = worker(repo_root, "asr").call(
        repo_root, "transcribe", {"audio": str(source.resolve()), "model": cfg.prepare.asr_model, "language": cfg.language}
    )
    text = "".join(str(seg.get("text") or "").strip() for seg in asr.get("segments", [])).strip()
    sense = worker(repo_root, "sense").call(
        repo_root, "analyze", {"audio": str(source.resolve()), "language": cfg.language}
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
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    return {"text": text, "voice": {"caption": "自然に話している。", "emotion": "NEUTRAL", "events": []}}


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
        "返答はJSONのみ。形式は {\"text\":\"...\",\"voice\":{\"caption\":\"...\",\"emotion\":\"NEUTRAL\",\"events\":[]}}。"
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
