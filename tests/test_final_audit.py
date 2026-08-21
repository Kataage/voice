from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personavoice import inference, media, state, training
from personavoice.config import PersonaConfig
from personavoice.project import init_persona


def test_config_pins_audited_asr_and_irodori_reference_limit():
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "alice", "prepare": {"asr_model": "small"}})
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "alice", "prepare": {"reference_seconds": 121}})
    assert PersonaConfig.model_validate(
        {"name": "alice", "prepare": {"reference_seconds": 120}}
    ).prepare.reference_seconds == 120


def test_persona_yaml_name_must_match_directory(tmp_path: Path):
    directory = tmp_path / "alice"
    directory.mkdir()
    path = directory / "persona.yaml"
    path.write_text(yaml.safe_dump({"name": "bob"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match directory"):
        PersonaConfig.load(path)


def test_inventory_deduplicates_identical_media_but_records_provenance(
    tmp_path: Path,
    monkeypatch,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.wav").write_bytes(b"same recording")
    (raw / "copy.wav").write_bytes(b"same recording")
    monkeypatch.setattr(media, "ffprobe", lambda _path: {"ok": True})

    rows = media.inventory(raw)
    assert len(rows) == 1
    assert rows[0]["path"] == "a.wav"
    assert rows[0]["duplicate_paths"] == ["copy.wav"]


def test_inventory_fingerprint_changes_when_materialization_root_changes(tmp_path: Path):
    first = tmp_path / "one" / "raw"
    second = tmp_path / "two" / "raw"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "a.wav").write_bytes(b"same")
    (second / "a.wav").write_bytes(b"same")
    assert media.inventory_fingerprint(first) != media.inventory_fingerprint(second)


def test_prepare_cache_policy_changes_when_worker_lock_changes(tmp_path: Path, monkeypatch):
    for name in ("asr", "diarization", "sense"):
        lock = tmp_path / "workers" / name / "uv.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"{name}-one", encoding="utf-8")
    monkeypatch.setattr(state, "_repo_root", lambda: tmp_path)
    before = state._prepare_cache_policy()
    (tmp_path / "workers" / "asr" / "uv.lock").write_text("asr-two", encoding="utf-8")
    assert state._prepare_cache_policy() != before


def test_training_fingerprint_changes_when_training_lock_changes(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    (paths.dataset / "irodori_source.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    for relative in (
        Path("locks/Irodori-TTS.uv.lock"),
        Path("workers/lfm/uv.lock"),
        Path("workers/seed_vc/uv.lock"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("one", encoding="utf-8")
    before = training._fingerprint(paths, cfg)
    (tmp_path / "workers" / "lfm" / "uv.lock").write_text("two", encoding="utf-8")
    assert training._fingerprint(paths, cfg) != before


def test_worker_sources_fail_closed_on_audited_local_models():
    root = Path(__file__).resolve().parents[1]
    asr = (root / "workers" / "asr" / "worker.py").read_text(encoding="utf-8")
    sense = (root / "workers" / "sense" / "worker.py").read_text(encoding="utf-8")
    lfm = (root / "workers" / "lfm" / "worker.py").read_text(encoding="utf-8")
    lfm_train = (root / "workers" / "lfm" / "train.py").read_text(encoding="utf-8")

    assert "supports only" in asr
    assert "return MODEL_ID" not in sense
    assert "local_files_only=True" in lfm
    assert "b31023f2d69b95fbd7876898f8de9fae90e8afbd" in lfm
    assert ".personavoice-revision" in lfm
    assert "_verify_base(base)" in lfm_train


def test_chat_plan_normalizes_malformed_voice_types():
    plan = inference._normalize_chat_plan(
        {
            "text": 123,
            "voice": {
                "caption": ["bad"],
                "emotion": 999,
                "events": "laugh",
            },
        }
    )
    assert plan["text"] == "123"
    assert plan["voice"]["caption"] == "自然に話している。"
    assert plan["voice"]["emotion"] == "UNKNOWN"
    assert plan["voice"]["events"] == ["Laughter"]


def test_reenact_uses_unique_output_directory(tmp_path: Path, monkeypatch):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    reference = paths.references / "ref.flac"
    reference.write_bytes(b"reference")

    stamps = iter(["one", "two"])
    monkeypatch.setattr(inference, "_stamp", lambda: next(stamps))

    output_dirs: list[Path] = []

    class FakeWorker:
        def call(self, _repo_root, command, payload):
            assert command == "convert"
            output_dir = Path(payload["output_dir"])
            output_dirs.append(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / "converted.wav"
            output.write_bytes(b"0" * 64)
            return {"output": str(output)}

    monkeypatch.setattr(inference, "worker", lambda _root, _name: FakeWorker())
    first = inference.reenact(tmp_path, paths, cfg, source)
    second = inference.reenact(tmp_path, paths, cfg, source)

    assert first.parent.name == "one"
    assert second.parent.name == "two"
    assert output_dirs[0] != output_dirs[1]


def test_seed_vc_deep_health_invalidates_stale_readiness_marker():
    root = Path(__file__).resolve().parents[1]
    seed = (root / "workers" / "seed_vc" / "worker.py").read_text(encoding="utf-8")
    assert "_ready_marker().unlink(missing_ok=True)" in seed
