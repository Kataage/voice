from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from personavoice.atomic import atomic_write_text
from personavoice.captions import normalize_emotion

PROFILE_SCHEMA_VERSION = 1
PROFILE_FILENAME = "core_profile.yaml"
MAX_PROFILE_PROMPT_CHARS = 8_000
_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"


class _ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProfileIdentity(_ProfileModel):
    display_name: str = Field(min_length=1, max_length=64)
    self_concept: str = Field(default="", max_length=800)
    role: str = Field(default="", max_length=400)
    background: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    first_person: str = Field(default="私", min_length=1, max_length=32)
    preferred_address: str = Field(default="", max_length=64)


class StableFact(_ProfileModel):
    key: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=500)
    immutable: bool = True


class ProfileRelationship(_ProfileModel):
    name: str = Field(min_length=1, max_length=80)
    relation: str = Field(min_length=1, max_length=160)
    address: str = Field(default="", max_length=64)
    notes: str = Field(default="", max_length=400)


class ProfileExpression(_ProfileModel):
    baseline_emotion: str = Field(default="UNKNOWN", min_length=1, max_length=32)
    tendencies: tuple[str, ...] = Field(default_factory=tuple, max_length=16)

    @field_validator("baseline_emotion")
    @classmethod
    def canonicalize_emotion(cls, value: str) -> str:
        return normalize_emotion(value)


class CoreProfile(_ProfileModel):
    """Persistent identity constraints used for runtime conditioning only.

    Learned diction, catchphrases and rhythm intentionally do not belong here.
    The profile is loaded on every chat turn and is not part of any Prepare,
    Irodori, VC, or optimizer-checkpoint fingerprint.
    """

    SCHEMA_VERSION: ClassVar[int] = PROFILE_SCHEMA_VERSION
    schema_version: Literal[1] = PROFILE_SCHEMA_VERSION
    identity: ProfileIdentity
    stable_facts: tuple[StableFact, ...] = Field(default_factory=tuple, max_length=32)
    relationships: tuple[ProfileRelationship, ...] = Field(default_factory=tuple, max_length=24)
    conversation_rules: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    expression: ProfileExpression = Field(default_factory=ProfileExpression)

    @model_validator(mode="after")
    def validate_profile_budget(self) -> CoreProfile:
        serialized = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(serialized) > MAX_PROFILE_PROMPT_CHARS:
            raise ValueError(
                f"core profile exceeds the bounded prompt budget of {MAX_PROFILE_PROMPT_CHARS} characters"
            )
        return self

    @classmethod
    def default(cls, persona_name: str) -> CoreProfile:
        if not re.fullmatch(_NAME_RE, persona_name):
            raise ValueError(f"invalid persona name for default core profile: {persona_name!r}")
        return cls(identity=ProfileIdentity(display_name=persona_name))

    @classmethod
    def load(cls, path: Path, *, persona_name: str) -> CoreProfile:
        if not path.exists():
            return cls.default(persona_name)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValueError(f"Core Profile is unreadable: {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Core Profile must contain a YAML mapping: {path}")
        profile = cls.model_validate(raw)
        if profile.identity.display_name != persona_name:
            raise ValueError(
                f"Core Profile display_name {profile.identity.display_name!r} does not match "
                f"persona {persona_name!r}: {path}"
            )
        return profile

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def yaml_text(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        )

    def save(self, path: Path) -> None:
        atomic_write_text(path, self.yaml_text())

    def prompt_dict(self) -> dict[str, Any]:
        """Return only persistent conditioning data in stable JSON-compatible form."""

        return self.model_dump(mode="json")


def load_core_profile(path: Path, *, persona_name: str) -> CoreProfile:
    """Load a profile, using a safe deterministic default for old personas."""

    return CoreProfile.load(path, persona_name=persona_name)
