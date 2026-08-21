from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from personavoice.config import ConsentConfig, PersonaConfig

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


@dataclass(frozen=True)
class PersonaPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "persona.yaml"

    @property
    def state(self) -> Path:
        return self.root / "state.json"


def init_persona(repo_root: Path, name: str, *, authorized: bool = False) -> PersonaPaths:
    root = repo_root / "personas" / name
    root.mkdir(parents=True, exist_ok=True)
    for dirname in PERSONA_DIRS:
        directory = root / dirname
        directory.mkdir(exist_ok=True)
        marker = directory / ".gitkeep"
        marker.touch(exist_ok=True)

    config = PersonaConfig(name=name, consent=ConsentConfig(authorized=authorized))
    if not (root / "persona.yaml").exists():
        config.save(root / "persona.yaml")

    if not (root / "state.json").exists():
        state = {
            "schema_version": 1,
            "persona": name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "stages": {},
        }
        (root / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return PersonaPaths(root=root)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "personavoice").exists():
            return candidate
    raise RuntimeError("PersonaVoice repository root was not found")
