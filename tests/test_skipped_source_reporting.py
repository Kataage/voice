from __future__ import annotations

import json
from pathlib import Path

from personavoice.dataset import replace_utterances
from personavoice.pipeline import _skipped_source_report


def _inventory_row(source_id: str, path: str) -> dict:
    return {
        "path": path,
        "duplicate_paths": [],
        "absolute_path": f"/tmp/{path}",
        "size_bytes": 1,
        "sha256": source_id + "0" * (64 - len(source_id)),
        "probe": {},
    }


def test_zero_utterance_source_is_published_as_skipped(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    silent_id = "a" * 16
    target_id = "b" * 16
    (dataset / "source_inventory.json").write_text(
        json.dumps(
            [
                _inventory_row(silent_id, "silent.wav"),
                _inventory_row(target_id, "target.wav"),
            ]
        ),
        encoding="utf-8",
    )
    replace_utterances(
        dataset / "master.sqlite3",
        [
            {
                "id": f"{target_id}_000001",
                "source_id": target_id,
                "source_path": "target.wav",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "target": True,
                "speaker_similarity": 0.9,
                "speaker_coverage": 1.0,
                "overlap_ratio": 0.0,
                "text": "a",
                "text_annotated": "a",
                "emotion": "NEUTRAL",
                "events": [],
                "caption": "",
                "audio_path": None,
                "quality": 1.0,
            }
        ],
    )

    skipped = json.loads((dataset / "skipped_sources.json").read_text(encoding="utf-8"))
    assert skipped == [
        {
            "source_id": silent_id,
            "source_path": "silent.wav",
            "reason": "no_utterances",
            "detected_speakers": [],
            "utterances": 0,
        }
    ]


def test_identity_rejection_enriches_zero_utterance_source(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source_id = "c" * 16
    (dataset / "source_inventory.json").write_text(
        json.dumps([_inventory_row(source_id, "silent.wav")]),
        encoding="utf-8",
    )
    replace_utterances(dataset / "master.sqlite3", [])

    report = _skipped_source_report(
        dataset / "skipped_sources.json",
        {
            source_id: {
                "reason": "authorized_speaker_below_identity_threshold",
                "best_similarity": 0.1,
                "threshold": 0.5,
            }
        },
    )

    assert len(report) == 1
    assert report[0]["source_id"] == source_id
    assert report[0]["reason"] == "authorized_speaker_below_identity_threshold"
    assert report[0]["best_similarity"] == 0.1
    assert report[0]["threshold"] == 0.5
