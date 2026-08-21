from pathlib import Path


def test_lfm_lora_targets_use_exact_attention_contract():
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "workers" / "lfm" / "train.py").read_text(encoding="utf-8")
    contract = (root / "workers" / "lfm" / "model_contract.py").read_text(encoding="utf-8")

    for target in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert f'"{target}"' in contract
    assert '"o_proj"' not in contract
    assert '"in_proj"' not in contract
    assert '"w1"' not in contract
    assert '"w2"' not in contract
    assert '"w3"' not in contract
    assert '".self_attn."' in contract
    assert "audited_attention_lora_targets(model)" in trainer
