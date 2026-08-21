from __future__ import annotations

import importlib.util
from pathlib import Path

from personavoice.config import PersonaConfig
from personavoice.project import init_persona
from personavoice.training import _fingerprint


def _contract_module():
    path = Path(__file__).resolve().parents[1] / "workers" / "lfm" / "checkpoint_contract.py"
    spec = importlib.util.spec_from_file_location("lfm_checkpoint_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_complete_checkpoint(path: Path, *, binary_weight: bool = False) -> None:
    path.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "rng_state.pth",
    ):
        (path / name).write_bytes(b"state")
    weight = "adapter_model.bin" if binary_weight else "adapter_model.safetensors"
    (path / weight).write_bytes(b"adapter")


def test_lfm_resume_uses_highest_complete_checkpoint_and_prunes_partial(tmp_path: Path):
    contract = _contract_module()
    older = tmp_path / "checkpoint-100"
    newer_partial = tmp_path / "checkpoint-200"
    newest_complete = tmp_path / "checkpoint-300"
    _write_complete_checkpoint(older)
    _write_complete_checkpoint(newest_complete, binary_weight=True)
    _write_complete_checkpoint(newer_partial)
    (newer_partial / "checkpoint-is-incomplete.txt").write_text("saving", encoding="utf-8")

    assert contract.latest_complete_checkpoint(tmp_path) == newest_complete
    removed = contract.prune_incomplete_checkpoints(tmp_path)
    assert removed == [newer_partial]
    assert not newer_partial.exists()
    assert older.exists() and newest_complete.exists()


def test_lfm_resume_rejects_checkpoint_without_optimizer_scheduler_or_rng(tmp_path: Path):
    contract = _contract_module()
    missing_optimizer = tmp_path / "checkpoint-10"
    missing_scheduler = tmp_path / "checkpoint-20"
    missing_rng = tmp_path / "checkpoint-30"
    for path in (missing_optimizer, missing_scheduler, missing_rng):
        _write_complete_checkpoint(path)
    (missing_optimizer / "optimizer.pt").unlink()
    (missing_scheduler / "scheduler.pt").unlink()
    (missing_rng / "rng_state.pth").unlink()

    assert contract.latest_complete_checkpoint(tmp_path) is None
    assert set(contract.prune_incomplete_checkpoints(tmp_path)) == {
        missing_optimizer,
        missing_scheduler,
        missing_rng,
    }


def test_lfm_resume_ignores_non_numeric_checkpoint_directories(tmp_path: Path):
    contract = _contract_module()
    unrelated = tmp_path / "checkpoint-final"
    unrelated.mkdir()
    assert contract.checkpoint_step(unrelated) is None
    assert contract.prune_incomplete_checkpoints(tmp_path) == []
    assert unrelated.exists()


def test_training_fingerprint_includes_lfm_checkpoint_contract(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    (paths.dataset / "irodori_source.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    helper = tmp_path / "workers" / "lfm" / "checkpoint_contract.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("revision = 1\n", encoding="utf-8")

    first = _fingerprint(paths, cfg)
    helper.write_text("revision = 2\n", encoding="utf-8")
    second = _fingerprint(paths, cfg)
    assert first != second
