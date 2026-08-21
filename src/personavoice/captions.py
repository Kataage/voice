from __future__ import annotations

EMOTION_JA = {
    "HAPPY": "嬉しそうに",
    "SAD": "悲しそうに",
    "ANGRY": "怒りを含んで",
    "NEUTRAL": "自然で落ち着いて",
    "FEARFUL": "不安や怖さを含んで",
    "DISGUSTED": "嫌悪感を含んで",
    "SURPRISED": "驚きを含んで",
    "UNKNOWN": "自然に",
}

EVENT_JA = {
    "Laughter": "笑い声を交えながら",
    "Cry": "泣き声を含みながら",
    "Breath": "自然な息遣いやため息を含めて",
    "Cough": "咳を含みながら",
    "Sneeze": "くしゃみを含みながら",
    "Applause": "拍手が聞こえる状況で",
    "BGM": "背景音楽がある状況で",
}

EVENT_EMOJI = {
    "Laughter": "🤭",
    "Cry": "😭",
    "Breath": "😮‍💨",
    "Cough": "😷",
    "Sneeze": "🤧",
    "Applause": "👏",
}

EVENT_ALIASES = {
    "laugh": "Laughter",
    "laughter": "Laughter",
    "chuckle": "Laughter",
    "giggle": "Laughter",
    "cry": "Cry",
    "sob": "Cry",
    "breath": "Breath",
    "sigh": "Breath",
    "gasp": "Breath",
    "cough": "Cough",
    "sneeze": "Sneeze",
    "applause": "Applause",
    "bgm": "BGM",
}


def normalize_emotion(value: str | None) -> str:
    if not value:
        return "UNKNOWN"
    value = value.replace("<|", "").replace("|>", "").strip().upper()
    return value if value in EMOTION_JA else "UNKNOWN"


def normalize_events(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        value = raw.replace("<|", "").replace("|>", "").strip()
        value = EVENT_ALIASES.get(value.lower(), value)
        if value in {"Speech", "Event_UNK", "nospeech", "Sing", "Speech_Noise"}:
            continue
        if value and value not in out:
            out.append(value)
    return out


def build_caption(
    *,
    emotion: str | None,
    events: list[str] | None,
    chars_per_second: float | None,
) -> str:
    emo = normalize_emotion(emotion)
    event_list = normalize_events(events)
    parts = [EMOTION_JA[emo]]
    for event in event_list:
        if event in EVENT_JA:
            parts.append(EVENT_JA[event])
    if chars_per_second is not None:
        if chars_per_second >= 7.0:
            parts.append("やや早口で")
        elif chars_per_second <= 3.0:
            parts.append("ゆっくりめに")
    return "、".join(parts) + "話している。"


def annotate_text(text: str, events: list[str] | None) -> str:
    result = text.strip()
    event_list = normalize_events(events)
    emojis = "".join(EVENT_EMOJI[e] for e in event_list if e in EVENT_EMOJI)
    if not result:
        return emojis
    if emojis and emojis not in result:
        result = f"{result}{emojis}"
    return result
