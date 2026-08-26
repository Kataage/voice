from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from personavoice.atomic import atomic_write_text


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConsentConfig(StrictConfigModel):
    authorized: bool = False
    scope: str = "local-private"
    notes: str = ""


class PrepareConfig(StrictConfigModel):
    language: str = Field(default="ja", min_length=1)
    # PersonaVoice setup/cache contracts audit and materialize exactly this ASR model.
    # Allowing an arbitrary model name here could silently bypass those guarantees.
    asr_model: Literal["large-v3"] = "large-v3"
    asr_compute_type: str = Field(default="auto", min_length=1)
    min_clip_seconds: float = Field(default=1.0, gt=0)
    max_clip_seconds: float = Field(default=18.0, gt=0)
    merge_gap_seconds: float = Field(default=0.45, ge=0)
    max_overlap_ratio: float = Field(default=0.08, ge=0, le=1)
    min_identity_similarity: float = Field(default=0.45, ge=-1, le=1)
    # Pinned Irodori v4.1 Small supports a combined reference window up to 120 s.
    reference_seconds: float = Field(default=40.0, gt=0, le=120.0)
    reference_clip_max_seconds: float = Field(default=12.0, gt=0)
    keep_nonverbal_only: bool = True
    use_sensevoice: bool = True

    @model_validator(mode="after")
    def validate_clip_ranges(self) -> Self:
        if self.max_clip_seconds < self.min_clip_seconds:
            raise ValueError("prepare.max_clip_seconds must be >= prepare.min_clip_seconds")
        if self.reference_clip_max_seconds < self.min_clip_seconds:
            raise ValueError(
                "prepare.reference_clip_max_seconds must be >= prepare.min_clip_seconds"
            )
        return self


class TrainingConfig(StrictConfigModel):
    irodori_speaker_inversion: bool = True
    irodori_lora: bool = True
    lfm_lora: bool = True
    seed_vc_finetune: bool = False
    irodori_max_steps: int = Field(default=4000, ge=1)
    speaker_inversion_max_steps: int = Field(default=2000, ge=1)
    lfm_epochs: float = Field(default=3.0, gt=0)
    lfm_learning_rate: float = Field(default=2e-4, gt=0)
    lfm_lora_r: int = Field(default=16, ge=1)
    lfm_lora_alpha: int = Field(default=32, ge=1)
    seed_vc_max_steps: int = Field(default=1000, ge=1)


class InferenceConfig(StrictConfigModel):
    default_candidates: int = Field(default=3, ge=1, le=16)
    default_num_steps: int = Field(default=24, ge=1, le=500)
    tts_cfg_scale: float = Field(default=3.0, ge=0, le=100)
    reference_mode: Literal["auto", "none", "speaker-embed", "audio"] = "auto"
    # Explicit inference-only boundary controls.  They are intentionally kept
    # out of Prepare/training fingerprints so existing personas can use them
    # without regenerating data or retraining.
    duration_scale: float = Field(default=1.0, gt=0, le=4)
    trim_tail: bool = True
    tail_window_size: int = Field(default=20, ge=1, le=4096)
    tail_std_threshold: float = Field(default=0.05, ge=0, le=10)
    tail_mean_threshold: float = Field(default=0.1, ge=0, le=10)
    seed_vc_diffusion_steps: int = Field(default=30, ge=1, le=500)
    seed_vc_similarity_cfg: float = Field(default=0.7, ge=0)
    seed_vc_intelligibility_cfg: float = Field(default=0.7, ge=0)


class PersonaConfig(StrictConfigModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    language: str = Field(default="ja", min_length=1)
    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    tts_backend: Literal["irodori-v4.1-small"] = "irodori-v4.1-small"
    vc_backend: Literal["seed-vc-v2"] = "seed-vc-v2"
    brain_backend: Literal["lfm2.5-1.2b-jp-202606"] = "lfm2.5-1.2b-jp-202606"

    @model_validator(mode="after")
    def validate_language_consistency(self) -> Self:
        if self.prepare.language != self.language:
            raise ValueError(
                "persona language and prepare.language must match; use one consistent language"
            )
        return self

    @classmethod
    def load(cls, path: Path) -> PersonaConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Persona config must contain a YAML mapping: {path}")
        config = cls.model_validate(value)
        # A persona.yaml is namespaced by its directory. Allowing the embedded
        # name to drift would change speaker labels/model run names while the
        # command still targets a different persona directory.
        if path.name == "persona.yaml" and path.parent.name != config.name:
            raise ValueError(
                f"Persona config name {config.name!r} does not match directory "
                f"{path.parent.name!r}: {path}"
            )
        return config

    def save(self, path: Path) -> None:
        atomic_write_text(
            path,
            yaml.safe_dump(
                self.model_dump(mode="json"),
                allow_unicode=True,
                sort_keys=False,
            ),
        )
