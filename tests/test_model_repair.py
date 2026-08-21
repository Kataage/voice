from pathlib import Path

from personavoice.repair import repair_failed_model_materializations


def test_model_repair_discards_failed_materialization_but_not_cuda_mismatch(tmp_path: Path):
    asr = tmp_path / "models" / "asr" / "large-v3"
    lfm = tmp_path / "models" / "lfm" / "base"
    asr.mkdir(parents=True)
    lfm.mkdir(parents=True)
    (asr / "model.bin").write_bytes(b"broken")
    (lfm / "config.json").write_text("{}", encoding="utf-8")

    repaired = repair_failed_model_materializations(
        tmp_path,
        {
            "worker_health": {
                "asr": {"ok": False, "error": "FileNotFoundError: missing tokenizer"},
                "lfm": {
                    "ok": False,
                    "error": "lfm was installed for cu128, but its runtime cannot see CUDA.",
                },
            },
            "model_asset_integrity": {"ok": True},
        },
        include_seed_vc=False,
    )

    assert repaired == ["asr"]
    assert not asr.exists()
    assert lfm.exists()


def test_model_repair_handles_checksum_and_invalidated_seed_marker(tmp_path: Path):
    irodori = tmp_path / "models" / "irodori" / "v4.1-small"
    irodori.mkdir(parents=True)
    (irodori / "model.safetensors").write_bytes(b"broken")

    repaired = repair_failed_model_materializations(
        tmp_path,
        {
            "worker_health": {
                "seed_vc": {"ok": False, "error": "FileNotFoundError: missing checkpoint"},
            },
            "model_asset_integrity": {
                "ok": False,
                "error": "Irodori checkpoint checksum mismatch",
            },
        },
        include_seed_vc=True,
    )

    assert repaired == ["irodori", "seed_vc"]
    assert not irodori.exists()
