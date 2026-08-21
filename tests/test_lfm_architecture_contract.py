from pathlib import Path


def test_lfm_lora_targets_match_dense_lfm2_module_names():
    root = Path(__file__).resolve().parents[1]
    source = (root / "workers" / "lfm" / "train.py").read_text(encoding="utf-8")

    for target in ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"):
        assert f'"{target}"' in source
    assert '"o_proj"' not in source
    assert "_validate_lora_targets(model)" in source
