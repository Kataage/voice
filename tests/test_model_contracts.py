from __future__ import annotations

from pathlib import Path

from personavoice import state, training
from personavoice.config import PersonaConfig
from personavoice.project import init_persona


def test_prepare_cache_policy_changes_with_preprocessing_model_contract(monkeypatch):
    original = state._prepare_cache_policy()
    monkeypatch.setattr(state, "ASR_MODEL_REVISION", "different-asr-revision")
    assert state._prepare_cache_policy() != original


def test_training_fingerprint_changes_with_base_model_contract(tmp_path: Path, monkeypatch):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    paths.dataset.mkdir(parents=True, exist_ok=True)
    (paths.dataset / "irodori_source.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (paths.dataset / "lfm_train.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    original = training._fingerprint(paths, cfg)
    monkeypatch.setattr(training, "IRODORI_REVISION", "different-irodori-revision")
    assert training._fingerprint(paths, cfg) != original


def test_untracked_training_artifacts_are_detected_and_invalidated(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    marker = paths.models / "lfm" / "adapter" / "adapter_config.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")

    assert training._has_training_artifacts(paths) is True
    training._invalidate_training_artifacts(paths)
    assert training._has_training_artifacts(paths) is False


def test_workers_enforce_audited_local_model_contracts():
    root = Path(__file__).resolve().parents[1]
    asr = (root / "workers" / "asr" / "worker.py").read_text(encoding="utf-8")
    diarization = (root / "workers" / "diarization" / "worker.py").read_text(encoding="utf-8")
    sense = (root / "workers" / "sense" / "worker.py").read_text(encoding="utf-8")

    assert "edaa852ec7e145841d8ffdb056a99866b5f0a478" in asr
    assert ".personavoice-revision" in asr
    assert "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee" in diarization
    assert ".personavoice-revision" in diarization
    assert "trust_remote_code=False" in sense
    assert "disable_update=True" in sense
    assert "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea" in sense
    assert "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5" in sense
    assert "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8" in sense
    assert "sense-model-ready" in sense
    assert "_atomic_write_text" in sense
    assert '"verified\\n"' in sense
