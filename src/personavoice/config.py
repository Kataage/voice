from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ConsentConfig(BaseModel):
    authorized: bool = False
    scope: str = "local-private"
    notes: str = ""


class PersonaConfig(BaseModel):
    name: str
    language: str = "ja"
    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    tts_backend: str = "irodori"
    vc_backend: str = "seed-vc"
    brain_backend: str = "lfm2.5-jp"

    @classmethod
    def load(cls, path: Path) -> "PersonaConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
