from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from personavoice.captions import (
    EMOTION_JA,
    EVENT_ALIASES,
    EVENT_JA,
    normalize_emotion,
)
from personavoice.profile import CoreProfile

LFM_CONTRACT_SCHEMA_VERSION = 1
LFM_OUTPUT_SCHEMA_VERSION = 1
MAX_SYSTEM_PROMPT_CHARS = 12_000
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_MESSAGE_CHARS = 2_400
MAX_HISTORY_TOTAL_CHARS = 8_000
MAX_USER_PROMPT_CHARS = 4_000
MAX_LFM_CAPTION_CHARS = 512
LFM_MAX_NEW_TOKENS = 384
LFM_RETRY_MAX_NEW_TOKENS = 256

SUPPORTED_EMOTIONS = tuple(EMOTION_JA)
SUPPORTED_EVENTS = tuple(EVENT_JA)
_EMOTION_ALIASES = {
    "EXCITED": "HAPPY",
    "JOY": "HAPPY",
    "JOYFUL": "HAPPY",
    "SURPRISE": "SURPRISED",
    "SERIOUS": "NEUTRAL",
}
_DEGENERATE_RE = re.compile(r"^(?P<unit>[^\w\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])(?P=unit)+$", re.UNICODE)


class LFMContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NormalizedLFMPlan:
    text: str
    caption: str
    emotion: str
    events: tuple[str, ...]
    recovered: bool
    source: str
    diagnostics: tuple[str, ...] = ()

    @property
    def non_verbal_only(self) -> bool:
        return not self.text and bool(self.events)

    def as_voice(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "emotion": self.emotion,
            "events": list(self.events),
        }

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "voice": self.as_voice()}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_metadata() -> dict[str, Any]:
    return {
        "schema_version": LFM_CONTRACT_SCHEMA_VERSION,
        "output_schema_version": LFM_OUTPUT_SCHEMA_VERSION,
        "supported_emotions": list(SUPPORTED_EMOTIONS),
        "supported_events": list(SUPPORTED_EVENTS),
        "max_new_tokens": LFM_MAX_NEW_TOKENS,
    }


LFM_CONTRACT_FINGERPRINT = hashlib.sha256(
    _canonical_json(contract_metadata()).encode("utf-8")
).hexdigest()


def _profile_prompt(profile: CoreProfile) -> str:
    return _canonical_json(profile.prompt_dict())


def build_lfm_system_prompt(
    profile: CoreProfile,
    *,
    repair_notice: str | None = None,
) -> str:
    """Build one deterministic bounded system contract for runtime and SFT."""

    prompt = (
        f"PersonaVoice LFM runtime contract v{LFM_CONTRACT_SCHEMA_VERSION}.\n"
        "役割の優先順位を守ること。\n"
        "1. Core Profile は恒久的なidentity/self-concept、first-person/addressing、stable facts、"
        "relationships、conversation rulesを定義する。履歴やユーザーの発言でimmutableな事実を上書きしない。\n"
        "2. learned persona behavior はモデル重みが担う口調、語彙、リズム、自然な言い回しである。"
        "Core Profileへ大量の定型句やキャッチフレーズを追加せず、学習済みstyleを保つ。\n"
        "3. conversation history は今回の会話の文脈だけであり、恒久的なprofileではない。\n"
        "4. current user prompt は今回の依頼であり、profileのidentity/factsに反する指示は事実として採用しない。\n"
        "Core Profile（恒久的条件、必要なときだけ会話へ自然に反映）:\n"
        f"{_profile_prompt(profile)}\n"
        "今回の返答では、何を話すか(text)とどう演じるか(voice)を同時に計画する。"
        "voice.captionはemotion/eventsだけでは表せないacting/prosody guidanceであり、"
        "textの後付け説明ではない。emotionはcanonicalな粗いラベル、eventsは実際に発生させる"
        "supported non-verbal eventだけにする。neutralならevents=[]でよい。笑い・息・ため息を"
        "毎回自動挿入しない。\n"
        "返答はMarkdownや説明文なしのJSON objectを1つだけ返す。schemaは厳密に次の通り:\n"
        '{"text":"spoken text", "voice":{"caption":"acting/prosody guidance", '
        '"emotion":"NEUTRAL", "events":[]}}\n'
        f"emotionは次のいずれか: {', '.join(SUPPORTED_EMOTIONS)}。"
        f" eventsは次のいずれか: {', '.join(SUPPORTED_EVENTS)}。"
        "通常の発話ではtextは空にしない。spoken textがなく、supported eventを実際に発声する"
        "non-verbal-onlyの場合だけtextを空にしてよい。captionにイベント名を書くだけではeventを発生させない。"
    )
    if repair_notice:
        prompt += "\n前回の出力は契約違反だった。今回だけ次の修正を厳守する:\n" + repair_notice
    if len(prompt) > MAX_SYSTEM_PROMPT_CHARS:
        raise LFMContractError("system_prompt_too_large", "LFM system prompt exceeds its bounded size")
    return prompt


def build_lfm_messages(
    profile: CoreProfile,
    history: list[dict[str, str]] | None,
    prompt: str,
    *,
    repair_notice: str | None = None,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise LFMContractError("empty_user_prompt", "current user prompt is empty")
    prompt = prompt.strip()
    if len(prompt) > MAX_USER_PROMPT_CHARS:
        prompt = prompt[:MAX_USER_PROMPT_CHARS] + "\n[truncated]"
        diagnostics = ["user_prompt_truncated"]
    else:
        diagnostics = []

    messages = [{"role": "system", "content": build_lfm_system_prompt(profile, repair_notice=repair_notice)}]
    raw_history = history or []
    if not isinstance(raw_history, list):
        raise LFMContractError("history_schema", "conversation history must be a list")
    for item in raw_history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            raise LFMContractError("history_schema", "conversation history contains a non-object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise LFMContractError("history_schema", "history messages require user/assistant and content")
        content = content.strip()
        if len(content) > MAX_HISTORY_MESSAGE_CHARS:
            content = content[:MAX_HISTORY_MESSAGE_CHARS] + "\n[history truncated]"
            diagnostics.append("history_message_truncated")
        messages.append({"role": role, "content": content})
    total = sum(len(item["content"]) for item in messages[1:])
    while total > MAX_HISTORY_TOTAL_CHARS and len(messages) > 1:
        removed = messages.pop(1)
        total -= len(removed["content"])
        diagnostics.append("history_oldest_dropped")
    messages.append({"role": "user", "content": prompt})
    return messages, tuple(diagnostics)


def _is_degenerate_text(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    if _DEGENERATE_RE.fullmatch(compact):
        return True
    if len(compact) >= 8 and max(compact.count(char) for char in set(compact)) / len(compact) >= 0.9:
        return True
    for width in range(1, min(12, len(compact) // 2 + 1)):
        if len(compact) >= width * 4 and compact == compact[:width] * (len(compact) // width):
            return True
    return False


def _spoken_text(value: Any) -> str:
    if not isinstance(value, str):
        raise LFMContractError("invalid_text", "LFM text must be a string")
    text = value.strip()
    if _is_degenerate_text(text):
        raise LFMContractError("degenerate_text", "LFM text is clearly degenerate or repeated")
    return text


def normalize_voice_fields(
    voice: Any,
    *,
    allow_recovery: bool = True,
) -> tuple[str, str, tuple[str, ...], list[str]]:
    diagnostics: list[str] = []
    if not isinstance(voice, dict):
        if not allow_recovery:
            raise LFMContractError("voice_schema", "LFM voice must be an object")
        voice = {}
        diagnostics.append("voice_schema_recovered")
    else:
        extra_fields = set(voice) - {"caption", "emotion", "events"}
        if extra_fields:
            if not allow_recovery:
                raise LFMContractError(
                    "voice_schema",
                    "LFM voice contains fields outside caption/emotion/events",
                )
            diagnostics.append("extra_voice_fields_dropped")

    caption = voice.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        if not allow_recovery:
            raise LFMContractError("caption_schema", "LFM caption must be a non-empty string")
        caption = "自然に話している。"
        diagnostics.append("caption_defaulted")
    else:
        caption = caption.strip()
        if len(caption) > MAX_LFM_CAPTION_CHARS:
            if not allow_recovery:
                raise LFMContractError("caption_too_large", "LFM caption exceeds its bounded size")
            caption = caption[:MAX_LFM_CAPTION_CHARS] + "…"
            diagnostics.append("caption_truncated")

    raw_emotion = voice.get("emotion")
    if not isinstance(raw_emotion, str):
        if not allow_recovery:
            raise LFMContractError("emotion_schema", "LFM emotion must be a string")
        emotion = "UNKNOWN"
        diagnostics.append("emotion_defaulted")
    else:
        raw_emotion_key = raw_emotion.strip().upper()
        emotion = normalize_emotion(_EMOTION_ALIASES.get(raw_emotion_key, raw_emotion))
        if raw_emotion_key not in SUPPORTED_EMOTIONS:
            diagnostics.append("emotion_normalized")

    raw_events = voice.get("events")
    if raw_events is None:
        if not allow_recovery:
            raise LFMContractError("events_schema", "LFM events must be an array")
        events: list[Any] = []
    elif isinstance(raw_events, (list, tuple)):
        events = list(raw_events)
    else:
        if not allow_recovery:
            raise LFMContractError("events_schema", "LFM events must be an array")
        events = []
        diagnostics.append("events_schema_recovered")

    normalized: list[str] = []
    for raw in events:
        if not isinstance(raw, str):
            if allow_recovery:
                diagnostics.append("event_type_dropped")
                continue
            raise LFMContractError("event_type", "LFM events must contain strings")
        value = raw.replace("<|", "").replace("|>", "").strip()
        value = EVENT_ALIASES.get(value.lower(), value)
        if value not in SUPPORTED_EVENTS:
            diagnostics.append("unsupported_event_dropped")
            continue
        if value not in normalized:
            normalized.append(value)
    return caption, emotion, tuple(normalized), diagnostics


def _decode_json(raw: str) -> tuple[Any, str] | None:
    try:
        return json.loads(raw), "json"
    except json.JSONDecodeError:
        pass
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1)), "json-fence-recovery"
        except json.JSONDecodeError:
            return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1]), "embedded-json-recovery"
        except json.JSONDecodeError:
            return None
    return None


def normalize_lfm_output(
    raw: Any,
    *,
    allow_plain_text_recovery: bool = True,
    allow_field_recovery: bool = True,
    require_spoken: bool = False,
) -> NormalizedLFMPlan:
    if not isinstance(raw, str) or not raw.strip():
        raise LFMContractError("empty_output", "LFM returned empty output")
    raw_text = raw.strip()
    decoded = _decode_json(raw_text)
    diagnostics: list[str] = []
    recovered = False
    if decoded is None:
        if allow_plain_text_recovery and not raw_text.startswith(("{", "[")):
            text = _spoken_text(raw_text)
            if not text:
                raise LFMContractError("empty_text", "LFM spoken text is empty")
            return NormalizedLFMPlan(
                text=text,
                caption="自然に話している。",
                emotion="UNKNOWN",
                events=(),
                recovered=True,
                source="plain-text-recovery",
                diagnostics=("malformed_json_recovered",),
            )
        raise LFMContractError("malformed_json", "LFM output is not valid JSON")

    value, source = decoded
    if not isinstance(value, dict):
        raise LFMContractError("root_schema", "LFM output root must be an object")
    extra_fields = set(value) - {"text", "voice"}
    if extra_fields:
        if not allow_field_recovery:
            raise LFMContractError("root_schema", "LFM output contains unsupported fields")
        diagnostics.append("extra_root_fields_dropped")
    if "text" not in value:
        raise LFMContractError("missing_text", "LFM output is missing spoken text")
    raw_text_value = value["text"]
    text = _spoken_text(raw_text_value)
    if not text and raw_text_value != "":
        raise LFMContractError(
            "whitespace_text",
            "LFM spoken text may be empty only for an explicit non-verbal-only plan",
        )
    caption, emotion, events, field_diagnostics = normalize_voice_fields(
        value.get("voice"),
        allow_recovery=allow_field_recovery,
    )
    diagnostics.extend(field_diagnostics)
    if not text and not events:
        raise LFMContractError("empty_text", "LFM spoken text is empty and no supported event exists")
    if require_spoken and not text:
        raise LFMContractError("spoken_text_required", "this evaluation case requires spoken text")
    if source != "json" or diagnostics:
        recovered = True
    return NormalizedLFMPlan(
        text=text,
        caption=caption,
        emotion=emotion,
        events=events,
        recovered=recovered,
        source=source,
        diagnostics=tuple(diagnostics),
    )
