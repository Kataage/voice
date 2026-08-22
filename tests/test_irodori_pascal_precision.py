from __future__ import annotations

from pathlib import Path

import yaml

from personavoice import irodori
from personavoice.model_assets import IRODORI_TEXT_ENCODER_ID, IRODORI_TEXT_ENCODER_REVISION


def test_cu126_training_config_uses_fp16_without_tf32(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.yaml"
    output = tmp_path / "patched.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "text_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                    "text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
                    "caption_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                },
                "train": {
                    "precision": "bf16",
                    "allow_tf32": True,
                    "dataloader_cuda_prefetch": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        irodori,
        "safe_batch_profile",
        lambda *, backend: {
            "batch_size": 1,
            "gradient_accumulation_steps": 12,
            "num_workers": 2,
            "gradient_checkpointing": True,
        },
    )

    irodori._patched_config(source, output, max_steps=100, backend="cu126")
    patched = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert patched["train"]["precision"] == "fp16"
    assert patched["train"]["allow_tf32"] is False
    assert patched["train"]["dataloader_cuda_prefetch"] is True
