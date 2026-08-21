from pathlib import Path

from personavoice import training


def test_seed_vc_checkpoint_selection_uses_highest_numeric_step(tmp_path: Path):
    older = tmp_path / "CFM_epoch_10_step_900.pth"
    newer = tmp_path / "CFM_epoch_2_step_1000.pth"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")

    assert training._latest_seed_vc_checkpoint(tmp_path) == newer
