from pathlib import Path

from personavoice import state, training
from personavoice.config import PersonaConfig
from personavoice.project import init_persona


def test_prepare_policy_changes_when_pipeline_code_changes(tmp_path: Path, monkeypatch):
    pipeline = tmp_path / "src" / "personavoice" / "pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("one", encoding="utf-8")
    monkeypatch.setattr(state, "_repo_root", lambda: tmp_path)

    before = state._prepare_cache_policy()
    pipeline.write_text("two", encoding="utf-8")

    assert state._prepare_cache_policy() != before


def test_training_fingerprint_changes_when_training_implementation_changes(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    implementation = tmp_path / "workers" / "lfm" / "train.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("one", encoding="utf-8")

    before = training._fingerprint(paths, cfg)
    implementation.write_text("two", encoding="utf-8")

    assert training._fingerprint(paths, cfg) != before
