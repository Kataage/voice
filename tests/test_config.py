from __future__ import annotations

import pytest

from personavoice.config import PersonaConfig


def test_config_rejects_unsafe_persona_name():
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "../escape"})


def test_config_rejects_unknown_keys_instead_of_silently_ignoring_them():
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "alice", "trianing": {}})


def test_config_requires_one_consistent_language():
    with pytest.raises(ValueError, match="must match"):
        PersonaConfig.model_validate(
            {
                "name": "alice",
                "language": "ja",
                "prepare": {"language": "en"},
            }
        )


def test_config_rejects_invalid_prepare_ranges():
    with pytest.raises(ValueError, match="max_clip_seconds"):
        PersonaConfig.model_validate(
            {
                "name": "alice",
                "prepare": {
                    "min_clip_seconds": 10.0,
                    "max_clip_seconds": 5.0,
                    "reference_clip_max_seconds": 12.0,
                },
            }
        )


def test_config_rejects_non_positive_training_and_inference_values():
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "alice", "training": {"irodori_max_steps": 0}})
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "alice", "inference": {"default_candidates": 0}})


def test_config_rejects_unsupported_backend_names():
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "alice", "tts_backend": "unknown-tts"})


def test_legacy_v03_config_loads_with_inference_boundary_defaults(tmp_path):
    persona_dir = tmp_path / "alice"
    persona_dir.mkdir()
    config_path = persona_dir / "persona.yaml"
    config_path.write_text(
        "name: alice\nlanguage: ja\ninference:\n  default_candidates: 1\n",
        encoding="utf-8",
    )
    loaded = PersonaConfig.load(config_path)
    assert loaded.name == "alice"

    config = PersonaConfig.model_validate(
        {
            "name": "alice",
            "language": "ja",
            "prepare": {"language": "ja"},
            "inference": {"default_candidates": 1},
        }
    )
    assert config.inference.duration_scale == 1.0
    assert config.inference.trim_tail is True
    assert config.inference.tail_window_size == 20
    assert config.inference.reference_mode == "auto"


def test_inference_boundary_settings_are_validated_independently():
    with pytest.raises(ValueError):
        PersonaConfig.model_validate({"name": "alice", "inference": {"duration_scale": 0}})
    with pytest.raises(ValueError):
        PersonaConfig.model_validate(
            {"name": "alice", "inference": {"tail_window_size": 0}}
        )
    assert (
        PersonaConfig.model_validate(
            {"name": "alice", "inference": {"reference_mode": "none"}}
        ).inference.reference_mode
        == "none"
    )
