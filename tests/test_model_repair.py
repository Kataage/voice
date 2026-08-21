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


def test_model_repair_never_deletes_for_stale_or_missing_environment(tmp_path: Path):
    asr = tmp_path / "models" / "asr" / "large-v3"
    lfm = tmp_path / "models" / "lfm" / "base"
    asr.mkdir(parents=True)
    lfm.mkdir(parents=True)
    (asr / "model.bin").write_bytes(b"healthy")
    (lfm / "config.json").write_text("{}", encoding="utf-8")

    repaired = repair_failed_model_materializations(
        tmp_path,
        {
            "worker_health": {
                "asr": {
                    "ok": False,
                    "error": "RuntimeError: PersonaVoice local environments are stale for the current repository dependency contract.",
                },
                "lfm": {"ok": False, "error": "worker .venv is missing"},
            },
            "model_asset_integrity": {"ok": True},
        },
        include_seed_vc=False,
    )

    assert repaired == []
    assert asr.exists()
    assert lfm.exists()


def test_model_repair_never_deletes_for_gpu_runtime_failures(tmp_path: Path):
    diarization = tmp_path / "models" / "pyannote" / "community-1"
    sense = tmp_path / "models" / "sense" / "SenseVoiceSmall"
    diarization.mkdir(parents=True)
    sense.mkdir(parents=True)

    repaired = repair_failed_model_materializations(
        tmp_path,
        {
            "worker_health": {
                "diarization": {"ok": False, "error": "CUDA driver initialization failed"},
                "sense": {"ok": False, "error": "RuntimeError: CUDA out of memory"},
            },
            "model_asset_integrity": {"ok": True},
        },
        include_seed_vc=False,
    )

    assert repaired == []
    assert diarization.exists()
    assert sense.exists()


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
