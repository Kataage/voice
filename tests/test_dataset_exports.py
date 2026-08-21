from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.dataset import export_lfm, replace_utterances, write_jsonl


def _row(
    row_id: str,
    *,
    start: float,
    text: str,
    target: bool,
    speaker: str,
    quality: float = 0.9,
    emotion: str = "NEUTRAL",
) -> dict:
    return {
        "id": row_id,
        "source_id": "source",
        "start": start,
        "end": start + 1.0,
        "speaker": speaker,
        "target": target,
        "speaker_similarity": 0.95 if target else None,
        "speaker_coverage": 1.0,
        "overlap_ratio": 0.0,
        "text": text,
        "text_annotated": text,
        "emotion": emotion,
        "events": [],
        "caption": f"{emotion}で自然に話している。",
        "audio_path": None,
        "quality": quality,
    }


def test_write_jsonl_keeps_previous_export_when_generation_fails(tmp_path: Path):
    output = tmp_path / "train.jsonl"
    output.write_text('{"old":true}\n', encoding="utf-8")

    def rows():
        yield {"new": 1}
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        write_jsonl(output, rows())

    assert output.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(tmp_path.glob(".train.jsonl.*.tmp"))


def test_lfm_export_collapses_consecutive_target_segments_into_one_reply(tmp_path: Path):
    db = tmp_path / "master.sqlite3"
    output = tmp_path / "lfm.jsonl"
    replace_utterances(
        db,
        [
            _row("u1", start=0, text="今日はどうだった？", target=False, speaker="SPEAKER_01"),
            _row("u2", start=1, text="めっちゃ楽しかった。", target=True, speaker="SPEAKER_00"),
            _row(
                "u3",
                start=2,
                text="また行きたいな。",
                target=True,
                speaker="SPEAKER_00",
                quality=0.95,
                emotion="HAPPY",
            ),
            _row("u4", start=3, text="何が一番よかった？", target=False, speaker="SPEAKER_01"),
            _row("u5", start=4, text="景色かな。", target=True, speaker="SPEAKER_00"),
        ],
    )

    assert export_lfm(db, output, "alice") == 2
    examples = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    first_user = examples[0]["prompt"][1]["content"]
    first_answer = json.loads(examples[0]["completion"][0]["content"])
    assert [message["role"] for message in examples[0]["prompt"]] == ["system", "user"]
    assert [message["role"] for message in examples[0]["completion"]] == ["assistant"]
    assert "messages" not in examples[0]
    assert "相手: 今日はどうだった？" in first_user
    assert "SPEAKER_01" not in first_user
    assert first_answer["text"] == "めっちゃ楽しかった。また行きたいな。"
    assert first_answer["voice"]["emotion"] == "HAPPY"

    second_answer = json.loads(examples[1]["completion"][0]["content"])
    assert second_answer["text"] == "景色かな。"


def test_lfm_export_does_not_create_self_response_from_target_monologue(tmp_path: Path):
    db = tmp_path / "master.sqlite3"
    output = tmp_path / "lfm.jsonl"
    replace_utterances(
        db,
        [
            _row("u1", start=0, text="独り言その1。", target=True, speaker="SPEAKER_00"),
            _row("u2", start=1, text="独り言その2。", target=True, speaker="SPEAKER_00"),
        ],
    )

    assert export_lfm(db, output, "alice") == 0
    assert output.read_text(encoding="utf-8") == ""
