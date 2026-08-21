from __future__ import annotations

from pathlib import Path

from personavoice import training
from personavoice.config import PersonaConfig
from personavoice.project import PersonaPaths


def test_seed_vc_finetune_auto_continues_partial_stages(tmp_path: Path, monkeypatch):
    paths = PersonaPaths(tmp_path / "personas" / "alice")
    audio = paths.dataset / "seed_vc" / "audio"
    audio.mkdir(parents=True)
    (audio / "a.flac").write_bytes(b"a")
    (audio / "b.flac").write_bytes(b"b")

    cfg = PersonaConfig(name="alice")
    cfg.training.seed_vc_finetune = True
    cfg.training.seed_vc_max_steps = 100

    class HealthyWorker:
        def call(self, *_args, **_kwargs):
            return {"ok": True, "cuda": True}

    monkeypatch.setattr(training, "worker", lambda *_args, **_kwargs: HealthyWorker())
    monkeypatch.setattr(training, "_seed_vc_training_progress", lambda *_args: (0, None))

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    first = checkpoints / "CFM_epoch_1_step_40.pth"
    second = checkpoints / "CFM_epoch_2_step_60.pth"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls = []

    def fake_stage(_repo_root, **kwargs):
        calls.append((kwargs["completed_steps"], kwargs["initial_checkpoint"]))
        if kwargs["completed_steps"] == 0:
            return 40, first
        assert kwargs["completed_steps"] == 40
        assert kwargs["initial_checkpoint"] == first
        return 100, second

    monkeypatch.setattr(training, "_run_seed_vc_stage", fake_stage)

    result = training.train_seed_vc(tmp_path, paths, cfg)
    assert calls == [(0, None), (40, first)]
    assert Path(result).read_bytes() == b"second"


def test_seed_vc_checkpoint_selection_ignores_empty_checkpoint(tmp_path: Path):
    empty = tmp_path / "CFM_epoch_9_step_999.pth"
    valid = tmp_path / "CFM_epoch_8_step_900.pth"
    empty.write_bytes(b"")
    valid.write_bytes(b"valid")

    assert training._latest_seed_vc_checkpoint(tmp_path) == valid
