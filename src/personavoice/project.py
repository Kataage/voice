from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from personavoice.atomic import atomic_write_json
from personavoice.config import ConsentConfig, PersonaConfig
from personavoice.profile import PROFILE_FILENAME, CoreProfile

PERSONA_DIRS = (
    "raw",
    "identity",
    "dataset",
    "references",
    "models",
    "outputs",
    "cache",
    "logs",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def safe_name(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("persona name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    return value


@dataclass(frozen=True)
class PersonaPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "persona.yaml"

    @property
    def core_profile(self) -> Path:
        return self.root / PROFILE_FILENAME

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def identity(self) -> Path:
        return self.root / "identity"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def references(self) -> Path:
        return self.root / "references"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


def init_persona(repo_root: Path, name: str, *, authorized: bool = False) -> PersonaPaths:
    name = safe_name(name)
    root = repo_root / "personas" / name
    root.mkdir(parents=True, exist_ok=True)
    for dirname in PERSONA_DIRS:
        directory = root / dirname
        directory.mkdir(exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)

    config = PersonaConfig(name=name, consent=ConsentConfig(authorized=authorized))
    if not (root / "persona.yaml").exists():
        config.save(root / "persona.yaml")
    profile_path = root / PROFILE_FILENAME
    if not profile_path.exists():
        CoreProfile.default(name).save(profile_path)

    if not (root / "state.json").exists():
        state = {
            "schema_version": 2,
            "persona": name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "stages": {},
        }
        atomic_write_json(root / "state.json", state)
    return PersonaPaths(root=root)


def get_persona(repo_root: Path, name: str) -> PersonaPaths:
    name = safe_name(name)
    root = repo_root / "personas" / name
    if not root.exists():
        raise FileNotFoundError(f"Unknown persona {name!r}. Run `persona init {name}` first.")
    return PersonaPaths(root=root)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "personavoice").exists():
            return candidate
    raise RuntimeError("PersonaVoice repository root was not found")
