from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.captions import annotate_text, build_caption, normalize_events
from personavoice.config import PersonaConfig
from personavoice.dataset import load_utterances, replace_utterances
from personavoice.pipeline import _batch_results, _prepare_fingerprint, _turn_rows
from personavoice.project import init_persona, safe_name
from personavoice.speaker import (
    cosine_similarity,
    dominant_speaker,
    overlap_ratio,
    select_target_speaker,
)
from personavoice.state import StateStore


def test_safe_name():
    assert safe_name("alice-01") == "alice-01"
    with pytest.raises(ValueError):
        safe_name("../alice")


def test_persona_init(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    assert cfg.consent.authorized is True
    assert paths.raw.exists() and paths.identity.exists()
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    assert state["schema_version"] == 2


def test_state_resume(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)
    with store.running("x", "abc"):
        store.set_result("x", {"value": 1})
    assert store.is_complete("x", "abc")
    assert store.stage("x")["result"]["value"] == 1


def test_speaker_math():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    label, score = select_target_speaker(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        [[0.95, 0.05]],
        threshold=0.5,
    )
    assert label == "A" and score > 0.9
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "A"},
        {"start": 1.0, "end": 3.0, "speaker": "B"},
    ]
    speaker, coverage = dominant_speaker(0.0, 1.0, turns)
    assert speaker == "A" and coverage == pytest.approx(1.0)
    assert overlap_ratio(0.0, 3.0, turns) == pytest.approx(1.0 / 3.0)


def test_caption_aliases():
    assert normalize_events(["laugh", "sigh"]) == ["Laughter", "Breath"]
    assert "🤭" in annotate_text("あはは", ["laugh"])
    assert annotate_text("", ["sigh"]) == "😮‍💨"
    caption = build_caption(emotion="happy", events=["laugh"], chars_per_second=8)
    assert "嬉し" in caption and "笑い" in caption and "早口" in caption


def test_dataset_roundtrip(tmp_path: Path):
    db = tmp_path / "master.sqlite3"
    row = {
        "id": "x",
        "source_id": "s",
        "start": 0.0,
        "end": 2.0,
        "speaker": "A",
        "target": True,
        "speaker_similarity": 0.9,
        "speaker_coverage": 1.0,
        "overlap_ratio": 0.0,
        "text": "こんにちは",
        "text_annotated": "こんにちは",
        "emotion": "NEUTRAL",
        "events": [],
        "caption": "自然に話している。",
        "audio_path": None,
        "quality": 0.9,
    }
    replace_utterances(db, [row])
    loaded = load_utterances(db)
    assert loaded[0]["text"] == "こんにちは"


def test_word_aligned_long_turn_split_preserves_all_text():
    words = []
    for index, token in enumerate("ABCDEFGHIJ"):
        words.append(
            {
                "start": index * 1.0,
                "end": index * 1.0 + 0.5,
                "word": token,
                "probability": 0.95,
            }
        )
    asr = {"segments": [{"start": 0.0, "end": 10.0, "text": "ABCDEFGHIJ", "words": words}]}
    turns = [{"start": 0.0, "end": 10.0, "speaker": "A"}]
    rows = _turn_rows(asr, turns, max_seconds=3.0)
    assert len(rows) > 1
    assert "".join(row["text"] for row in rows) == "ABCDEFGHIJ"
    assert all(row["end"] - row["start"] <= 3.2 for row in rows)


def test_prepare_fingerprint_changes_when_identity_changes(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    (paths.raw / "source.wav").write_bytes(b"raw")
    identity = paths.identity / "id.wav"
    identity.write_bytes(b"one")
    first = _prepare_fingerprint(paths, cfg)
    identity.write_bytes(b"two-two")
    second = _prepare_fingerprint(paths, cfg)
    assert first != second


def test_batch_results_fails_loudly():
    with pytest.raises(RuntimeError, match="ASR failed"):
        _batch_results(
            [{"id": "x", "ok": False, "error": "boom"}],
            operation="ASR",
        )
