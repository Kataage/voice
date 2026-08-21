from pathlib import Path

from personavoice import training


def test_seed_vc_checkpoint_selection_uses_highest_numeric_step(tmp_path: Path):
    older = tmp_path / "CFM_epoch_10_step_900.pth"
    newer = tmp_path / "CFM_epoch_2_step_1000.pth"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")

    assert training._latest_seed_vc_checkpoint(tmp_path) == newer


def test_seed_vc_progress_accumulates_stage_offset_and_local_step(tmp_path: Path):
    runs = tmp_path / "runs"
    first = runs / "personavoice_alice_stage_0000000000"
    second = runs / "personavoice_alice_stage_0000000500"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "CFM_epoch_10_step_500.pth").write_bytes(b"first")
    latest = second / "CFM_epoch_3_step_250.pth"
    latest.write_bytes(b"second")

    completed, checkpoint = training._seed_vc_training_progress(tmp_path, "alice")
    assert completed == 750
    assert checkpoint == latest


def test_seed_vc_progress_ignores_partial_stage_without_checkpoint(tmp_path: Path):
    runs = tmp_path / "runs"
    complete = runs / "personavoice_alice_stage_0000000000"
    partial = runs / "personavoice_alice_stage_0000000500"
    complete.mkdir(parents=True)
    partial.mkdir(parents=True)
    checkpoint = complete / "CFM_epoch_10_step_500.pth"
    checkpoint.write_bytes(b"complete")
    (partial / "config.yml").write_text("partial: true\n", encoding="utf-8")

    completed, selected = training._seed_vc_training_progress(tmp_path, "alice")
    assert completed == 500
    assert selected == checkpoint
