from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ConsentConfig(BaseModel):
    authorized: bool = False
    scope: str = "local-private"
    notes: str = ""


class PrepareConfig(BaseModel):
    language: str = "ja"
    asr_model: str = "large-v3"
    asr_compute_type: str = "auto"
    min_clip_seconds: float = 1.0
    max_clip_seconds: float = 18.0
    merge_gap_seconds: float = 0.45
    max_overlap_ratio: float = 0.08
    min_identity_similarity: float = 0.45
    reference_seconds: float = 40.0
    reference_clip_max_seconds: float = 12.0
    keep_nonverbal_only: bool = True
    use_sensevoice: bool = True


class TrainingConfig(BaseModel):
    irodori_speaker_inversion: bool = True
    irodori_lora: bool = True
    lfm_lora: bool = True
    seed_vc_finetune: bool = False
    irodori_max_steps: int = 4000
    speaker_inversion_max_steps: int = 2000
    lfm_epochs: float = 3.0
    lfm_learning_rate: float = 2e-4
    lfm_lora_r: int = 16
    lfm_lora_alpha: int = 32
    seed_vc_max_steps: int = 1000


class InferenceConfig(BaseModel):
    default_candidates: int = 3
    default_num_steps: int = 24
    tts_cfg_scale: float = 3.0
    reference_mode: Literal["auto", "speaker-embed", "audio"] = "auto"
    seed_vc_diffusion_steps: int = 30
    seed_vc_similarity_cfg: float = 0.7
    seed_vc_intelligibility_cfg: float = 0.7


class PersonaConfig(BaseModel):
    name: str
    language: str = "ja"
    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    tts_backend: str = "irodori-v4.1-small"
    vc_backend: str = "seed-vc-v2"
    brain_backend: str = "lfm2.5-1.2b-jp-202606"

    @classmethod
    def load(cls, path: Path) -> PersonaConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.write_text(
            yaml.safe_dump(
                self.model_dump(mode="json"),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
