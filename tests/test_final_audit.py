from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import pytest
import yaml

from personavoice import dataset, inference, irodori, media, speaker, state, training
from personavoice.config import PersonaConfig
from personavoice.project import init_persona


def _write_torch_step(path: Path, step: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trainer/data.pkl", pickle.dumps({"step": step}, protocol=4))
        archive.writestr("trainer/version", "3\n")


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


def test_inventory_rejects_truncated_source_id_collision(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    first = raw / "a.wav"
    second = raw / "b.wav"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    monkeypatch.setattr(media, "ffprobe", lambda _path: {"ok": True})
    digests = {
        first: "0123456789abcdef" + "0" * 48,
        second: "0123456789abcdef" + "1" * 48,
    }
    monkeypatch.setattr(media, "sha256_file", lambda path: digests[path])
    with pytest.raises(RuntimeError, match="truncated source ID"):
        media.inventory(raw)


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


def test_target_speaker_absence_returns_non_matching_sentinel():
    label, score = speaker.select_target_speaker(
        {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.8, 0.2]},
        [[0.0, 1.0]],
        threshold=0.5,
    )
    assert label == speaker.TARGET_NOT_FOUND
    assert score < 0.5


def test_target_speaker_structural_errors_still_fail_loud():
    with pytest.raises(ValueError, match="no speaker embeddings"):
        speaker.select_target_speaker({}, [[1.0]], threshold=0.5)
    with pytest.raises(ValueError, match="Multiple speakers"):
        speaker.select_target_speaker(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            [],
            threshold=0.5,
        )


def test_dataset_records_sources_without_selected_target(tmp_path: Path):
    master = tmp_path / "dataset" / "master.sqlite3"
    dataset.replace_utterances(
        master,
        [
            {
                "id": "source_000001",
                "source_id": "source",
                "source_path": "other-speakers.wav",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "target": False,
                "speaker_similarity": None,
                "speaker_coverage": 1.0,
                "overlap_ratio": 0.0,
                "text": "hello",
                "text_annotated": "hello",
                "emotion": "NEUTRAL",
                "events": [],
                "caption": "自然に話している。",
                "audio_path": None,
                "quality": 1.0,
            }
        ],
    )
    skipped = json.loads((master.parent / "skipped_sources.json").read_text(encoding="utf-8"))
    assert skipped == [
        {
            "source_id": "source",
            "source_path": "other-speakers.wav",
            "reason": "authorized_speaker_not_selected",
            "detected_speakers": ["SPEAKER_00"],
            "utterances": 1,
        }
    ]


def test_prepare_result_with_no_usable_target_audio_fails(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = state.StateStore(paths.state)
    with pytest.raises(RuntimeError, match="no usable authorized-speaker"):
        store.set_result("prepare", {"usable_tts_utterances": 0})


def test_enabled_lfm_training_rejects_silent_data_shortage(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    (paths.dataset / "lfm_train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fewer than two valid conversational"):
        training.train_lfm(tmp_path, paths, cfg)


def test_enabled_seed_vc_training_rejects_silent_data_shortage(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    cfg.training.seed_vc_finetune = True
    with pytest.raises(RuntimeError, match="fewer than two valid target-speaker"):
        training.train_seed_vc(tmp_path, paths, cfg)


def test_irodori_resume_uses_highest_numeric_step(tmp_path: Path):
    for name in ("checkpoint_900", "checkpoint_1000"):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        _write_torch_step(
            checkpoint / "trainer_state.pt",
            int(checkpoint.name.removeprefix("checkpoint_")),
        )
    (tmp_path / "checkpoint_best_val_loss_1200_0.2").mkdir()
    assert irodori._latest_resume(tmp_path) == tmp_path / "checkpoint_1000"

    speaker_900 = tmp_path / "checkpoint_900.speaker.safetensors"
    speaker_1000 = tmp_path / "checkpoint_1000.speaker.safetensors"
    speaker_900.write_bytes(b"x")
    speaker_1000.write_bytes(b"x")
    assert irodori._latest_numeric_checkpoint([speaker_900, speaker_1000]) == speaker_1000


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
