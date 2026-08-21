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
