from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import pytest
import yaml

from personavoice import training as training_module
from personavoice.config import PersonaConfig, TrainingConfig
from personavoice.project import PersonaPaths


def _write_torch_step(path: Path, step: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trainer/data.pkl", pickle.dumps({"step": step}, protocol=4))
        archive.writestr("trainer/version", "3\n")


def test_new_training_defaults_use_full_model_methods_and_local_private_gate():
    training = TrainingConfig()

    assert training.schema_version == 2
    assert training.executor == "auto"
    assert training.remote_data_authorized is False
    assert training.irodori.enabled is True
    assert training.irodori.method == "full"
    assert training.irodori.auxiliary_speaker_inversion is False
    assert training.irodori.max_steps == 4000
    assert training.lfm.enabled is True
    assert training.lfm.method == "full"
    assert training.lfm.epochs == 3.0
    assert training.lfm.learning_rate == pytest.approx(2e-5)
    assert training.seed_vc.finetune is False
    assert training.quality_gate.enabled is True
    assert training.quality_gate.require_validation is True
    assert training.quality_gate.min_lfm_expected_similarity == pytest.approx(0.35)
    assert training.quality_gate.max_lfm_expected_cer == pytest.approx(0.85)
    assert training.quality_gate.max_lfm_expected_wer == pytest.approx(1.0)
    assert training.quality_gate.min_lfm_required_phrase_coverage == pytest.approx(1.0)
    assert training.quality_gate.max_lfm_base_similarity_regression == pytest.approx(0.1)
    assert training.was_migrated is False
    assert training.migration_notes == ()


def test_legacy_defaults_preserve_lora_plus_auxiliary_speaker_inversion():
    training = TrainingConfig.model_validate({"irodori_max_steps": 4000})

    assert training.was_migrated is True
    assert training.migrated_from_schema_version == 1
    assert training.irodori.enabled is True
    assert training.irodori.method == "lora"
    assert training.irodori.auxiliary_speaker_inversion is True
    assert training.lfm.enabled is True
    assert training.lfm.method == "lora"
    assert training.lfm.learning_rate == pytest.approx(2e-4)
    assert training.seed_vc.finetune is False
    assert training.irodori_lora is True
    assert training.irodori_speaker_inversion is True
    assert training.lfm_lora is True
    assert any("Preserved Irodori behavior" in note for note in training.migration_notes)


@pytest.mark.parametrize(
    (
        "speaker_inversion",
        "lora",
        "enabled",
        "method",
        "auxiliary_speaker_inversion",
    ),
    [
        (True, True, True, "lora", True),
        (False, True, True, "lora", False),
        (True, False, True, "speaker-inversion", False),
        (False, False, False, "lora", False),
    ],
)
def test_every_legacy_irodori_flag_combination_is_losslessly_mapped(
    speaker_inversion: bool,
    lora: bool,
    enabled: bool,
    method: str,
    auxiliary_speaker_inversion: bool,
):
    training = TrainingConfig.model_validate(
        {
            "irodori_speaker_inversion": speaker_inversion,
            "irodori_lora": lora,
        }
    )

    assert training.irodori.enabled is enabled
    assert training.irodori.method == method
    assert training.irodori.auxiliary_speaker_inversion is auxiliary_speaker_inversion
    assert training.irodori_speaker_inversion is speaker_inversion
    assert training.irodori_lora is lora


def test_legacy_numeric_and_backend_values_are_preserved():
    training = TrainingConfig.model_validate(
        {
            "schema_version": 1,
            "irodori_speaker_inversion": False,
            "irodori_lora": True,
            "lfm_lora": False,
            "seed_vc_finetune": True,
            "irodori_max_steps": 123,
            "speaker_inversion_max_steps": 45,
            "lfm_epochs": 4.5,
            "lfm_learning_rate": 0.0003,
            "lfm_lora_r": 8,
            "lfm_lora_alpha": 24,
            "seed_vc_max_steps": 678,
        }
    )

    assert training.irodori.max_steps == 123
    assert training.irodori.speaker_inversion_max_steps == 45
    assert training.lfm.enabled is False
    assert training.lfm.method == "lora"
    assert training.lfm.epochs == 4.5
    assert training.lfm.learning_rate == pytest.approx(0.0003)
    assert training.lfm.lora_r == 8
    assert training.lfm.lora_alpha == 24
    assert training.seed_vc.finetune is True
    assert training.seed_vc.max_steps == 678


@pytest.mark.parametrize(
    "training",
    [
        {"irodori_lora": True, "irodori": {"method": "full"}},
        {"schema_version": 2, "lfm_lora": True},
        {"schema_version": 1, "executor": "local"},
    ],
)
def test_mixed_legacy_and_nested_training_fields_are_rejected(training: dict):
    with pytest.raises(ValueError, match="legacy|schema_version 1"):
        TrainingConfig.model_validate(training)


@pytest.mark.parametrize(
    "training",
    [
        {"schema_version": 3},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"executor": "ssh"},
        {"irodori": {"method": "adapter"}},
        {"irodori": {"unknown": True}},
        {"lfm": {"learning_rate": 0}},
        {"quality_gate": {"max_cer": 1.01}},
        {"quality_gate": {"min_speaker_similarity": -0.01}},
        {"quality_gate": {"min_lfm_expected_similarity": 1.01}},
        {"quality_gate": {"max_lfm_expected_cer": 2.01}},
        {"quality_gate": {"max_lfm_expected_wer": -0.01}},
        {"quality_gate": {"min_lfm_required_phrase_coverage": 1.01}},
        {"quality_gate": {"max_lfm_base_similarity_regression": -0.01}},
        {"irodori_max_steps": 0},
        {"irodori_lora": True, "unknown_legacy_field": 1},
    ],
)
def test_training_schema_remains_strict(training: dict):
    with pytest.raises(ValueError):
        TrainingConfig.model_validate(training)


def test_canonical_dump_contains_only_schema_v2_fields():
    training = TrainingConfig.model_validate({"lfm_lora": True})
    dumped = training.canonical_dict()

    assert set(dumped) == {
        "schema_version",
        "executor",
        "remote_data_authorized",
        "irodori",
        "lfm",
        "seed_vc",
        "quality_gate",
    }
    assert not TrainingConfig.LEGACY_KEYS.intersection(dumped)
    assert "migration_notes" not in dumped
    assert "migrated_from_schema_version" not in dumped


def test_loaded_legacy_config_is_not_rewritten_until_explicitly_saved(tmp_path: Path):
    persona = tmp_path / "alice"
    persona.mkdir()
    source = persona / "persona.yaml"
    original = yaml.safe_dump(
        {
            "name": "alice",
            "consent": {"authorized": True},
            "training": {
                "irodori_speaker_inversion": True,
                "irodori_lora": True,
                "lfm_lora": True,
                "seed_vc_finetune": False,
            },
        },
        sort_keys=False,
    )
    source.write_text(original, encoding="utf-8")

    config = PersonaConfig.load(source)

    assert config.was_migrated is True
    assert source.read_text(encoding="utf-8") == original
    serialized = config.migrated_yaml()
    parsed = yaml.safe_load(serialized)
    assert parsed["training"]["schema_version"] == 2
    assert parsed["training"]["irodori"]["method"] == "lora"
    assert parsed["training"]["irodori"]["auxiliary_speaker_inversion"] is True
    assert not TrainingConfig.LEGACY_KEYS.intersection(parsed["training"])

    notes = config.save_migrated(source)
    reloaded = PersonaConfig.load(source)
    assert notes == config.migration_notes
    assert notes
    assert reloaded.was_migrated is False
    assert reloaded.training.canonical_dict() == config.training.canonical_dict()


def test_legacy_python_seed_vc_aliases_remain_assignable():
    training = TrainingConfig()

    training.seed_vc_finetune = True
    training.seed_vc_max_steps = 100

    assert training.seed_vc.finetune is True
    assert training.seed_vc.max_steps == 100


def test_schema_v2_fields_remain_assignable_after_legacy_migration():
    training = TrainingConfig.model_validate({"irodori_lora": True})

    training.executor = "local"
    training.remote_data_authorized = True

    assert training.executor == "local"
    assert training.remote_data_authorized is True
    assert training.was_migrated is True
    assert training.migration_notes

    with pytest.raises(ValueError):
        training.executor = "ssh"  # type: ignore[assignment]


def test_migration_metadata_cannot_be_supplied_as_external_input():
    with pytest.raises(ValueError, match="metadata is read-only"):
        TrainingConfig.model_validate({"migration_notes": ["forged"]})

    with pytest.raises(ValueError, match="metadata is read-only"):
        PersonaConfig.model_validate(
            {
                "name": "alice",
                "training": {"migration_notes": ["forged"]},
            }
        )


def test_v03_artifact_adoption_requires_exact_completed_input_and_config_lineage(
    tmp_path: Path,
) -> None:
    paths = PersonaPaths(tmp_path / "personas" / "alice")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"audio":"a.flac","text":"hello"}\n', encoding="utf-8", newline="\n"
    )
    (paths.dataset / "lfm_train.jsonl").write_text(
        '{"messages":[{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
        newline="\n",
    )
    cfg = PersonaConfig.model_validate(
        {
            "name": "alice",
            "training": {
                "schema_version": 1,
                "irodori_speaker_inversion": True,
                "irodori_lora": True,
                "lfm_lora": True,
            },
        }
    )
    fingerprints = training_module._legacy_v03_fingerprints(paths, cfg)
    assert fingerprints == frozenset(
        {
            "2c06e9a51edbf6bed95492a94cc115c011fd7204a87ddbee3ee1c916f87f8e44",
            "5f251f84d110e0d33083e0dd21d1756b09b60bb72c307f4f795f84fbdcf12877",
        }
    )
    recorded = sorted(fingerprints)[0]
    previous = {
        "status": "complete",
        "fingerprint": recorded,
        "result": {
            "train_schema": 8,
            "fingerprint": recorded,
            "irodori": {},
            "lfm_adapter": None,
            "seed_vc_cfm": None,
        },
    }

    assert training_module._legacy_v03_lineage_verified(previous, paths, cfg)

    (paths.dataset / "lfm_train.jsonl").write_text(
        '{"messages":[{"role":"assistant","content":"changed"}]}\n',
        encoding="utf-8",
        newline="\n",
    )
    assert not training_module._legacy_v03_lineage_verified(previous, paths, cfg)

    (paths.dataset / "lfm_train.jsonl").write_text(
        '{"messages":[{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
        newline="\n",
    )
    cfg.training.lfm.epochs = 4.0
    assert not training_module._legacy_v03_lineage_verified(previous, paths, cfg)
    previous["status"] = "error"
    assert not training_module._legacy_v03_lineage_verified(previous, paths, cfg)


def test_v03_checkpoint_is_losslessly_seeded_but_final_adapter_is_not_rebound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v03" / "checkpoint_0000010"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (source / "adapter_model.safetensors").write_bytes(b"legacy-adapter")
    _write_torch_step(source / "trainer_state.pt", 10)
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    destination = tmp_path / "schema9" / source.name

    migrated = training_module._migrate_legacy_checkpoint_directory(
        source,
        destination,
        destination_complete=training_module.lora_resume_checkpoint_complete,
    )

    assert migrated is True
    assert training_module.lora_resume_checkpoint_complete(destination)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
    assert not (destination / "provenance.json").exists()
