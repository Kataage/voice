from __future__ import annotations

import hashlib
import json
import math
import shutil
import wave
from pathlib import Path

import pytest

from personavoice import vc_evaluation
from personavoice.config import PersonaConfig
from personavoice.project import PersonaPaths


def _wav(path: Path, *, seconds: float = 0.6, frequency: float = 220.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 16000
    samples = [
        int(12000 * math.sin(2 * math.pi * frequency * index / rate))
        for index in range(round(rate * seconds))
    ]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
    return path


def _persona(tmp_path: Path) -> tuple[PersonaPaths, PersonaConfig]:
    root = tmp_path / "alice"
    paths = PersonaPaths(root)
    paths.dataset.mkdir(parents=True)
    paths.references.mkdir(parents=True)
    return paths, PersonaConfig(name="alice", consent={"authorized": True})


def test_manifest_is_deterministic_and_prioritizes_nonverbal_buckets(tmp_path: Path) -> None:
    paths, cfg = _persona(tmp_path)
    _wav(paths.references / "target.wav", frequency=180.0)
    rows = []
    for index, (text, events) in enumerate(
        (("普通の発話", []), ("笑って話す", ["laughter"]), ("", ["breath"]))
    ):
        source = _wav(paths.dataset / "clips" / f"clip-{index}.wav", frequency=220 + index * 20)
        rows.append(
            {
                "id": f"clip-{index}",
                "source_id": "recording-a",
                "target": True,
                "quality": 0.9,
                "audio_path": str(source),
                "text": text,
                "events": events,
                "start": index,
                "end": index + 0.6,
            }
        )
    (paths.dataset / "master.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    result = vc_evaluation.build_vc_evaluation_manifest(paths, cfg, limit=3, seed=11)
    manifest = paths.dataset / vc_evaluation.VC_MANIFEST_FILENAME
    loaded = vc_evaluation.load_vc_manifest(paths, manifest)

    assert result["sample_count"] == 3
    assert result["bucket_counts"] == {
        "mixed_speech_event": 1,
        "nonverbal_only": 1,
        "normal_speech": 1,
    }
    assert all(row["reference_audio"] == "references/target.wav" for row in loaded)
    assert len({row["reference_sha256"] for row in loaded}) == 1
    assert result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()

    source = paths.dataset / "clips" / "clip-0.wav"
    source.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="source checksum mismatch"):
        vc_evaluation.load_vc_manifest(paths, manifest)


def test_manifest_requires_prepared_data_without_fabricating_samples(tmp_path: Path) -> None:
    paths, cfg = _persona(tmp_path)
    _wav(paths.references / "target.wav")
    with pytest.raises(RuntimeError, match="master.json"):
        vc_evaluation.build_vc_evaluation_manifest(paths, cfg)


def test_canonical_manifest_rejects_non_japanese_personas(tmp_path: Path) -> None:
    paths, _ = _persona(tmp_path)
    cfg = PersonaConfig(name="alice", language="en", prepare={"language": "en"})

    with pytest.raises(RuntimeError, match="Japanese-only"):
        vc_evaluation.build_vc_evaluation_manifest(paths, cfg)


def test_evaluation_writes_machine_and_human_reports_and_keeps_default_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, cfg = _persona(tmp_path)
    reference = _wav(paths.references / "target.wav", frequency=180.0)
    rows = []
    for index, (text, events) in enumerate(
        (("普通の発話", []), ("笑って話す", ["laughter"]), ("", ["breath"]))
    ):
        source = _wav(paths.dataset / "clips" / f"clip-{index}.wav", frequency=220 + index * 20)
        rows.append(
            {
                "schema_version": 1,
                "id": f"clip-{index}",
                "source_id": "recording-a",
                "source_audio": f"dataset/clips/clip-{index}.wav",
                "reference_audio": "references/target.wav",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "text": text,
                "language": "ja",
                "bucket": vc_evaluation._bucket(text, events),
                "events": events,
                "selection_seed": 1,
            }
        )
    manifest = paths.dataset / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    def fake_reenact(_root, _paths, _cfg, source, *, output_dir, **_kwargs):
        output = output_dir / f"{source.stem}.wav"
        shutil.copy2(source, output)
        return output

    class FakeWorker:
        def __init__(self, name: str):
            self.name = name

        def call(self, _root, command: str, payload: dict):
            if command == "embed":
                return {"embedding": [1.0, 0.0, 0.0]}
            if command == "transcribe":
                return {
                    "language": "ja",
                    "duration": 0.6,
                    "language_probability": 1.0,
                    "segments": [{"start": 0.0, "end": 0.6, "text": "普通の発話", "words": []}],
                }
            if command == "analyze":
                return {"raw": "", "emotion": "NEUTRAL", "events": ["laughter", "breath"], "tags": []}
            raise AssertionError((self.name, command, payload))

    monkeypatch.setattr(vc_evaluation, "reenact", fake_reenact)
    monkeypatch.setattr(vc_evaluation, "worker", lambda _root, name: FakeWorker(name))

    report = vc_evaluation.evaluate_vc(tmp_path, paths, cfg, manifest)
    report_path = paths.root / report["report"]
    markdown_path = paths.root / report["report_markdown"]

    assert report["backends"] == ["seed-vc-v2", "vevo2-fm"]
    assert report["same_input_contract"]["same_manifest"] is True
    assert report["decision_gate"]["default_changed"] is False
    assert report["decision_gate"]["recommended_default"] == "seed-vc-v2"
    assert report["status"] == "pending target-machine validation"
    assert report_path.is_file()
    assert markdown_path.is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["results"]["vevo2-fm"]["samples"]
    assert "pending target-machine validation" in markdown_path.read_text(encoding="utf-8")
