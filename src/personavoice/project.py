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
_LINEAGE_ID_RE = re.compile(r"^pl-[0-9a-f]{32}$")


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
    # ``None`` preserves the v0.3/v0.4 legacy layout for old personas and
    # narrow LFM-only exports.  New upstream Prepare work uses an immutable
    # generation root selected by this field.
    lineage_id: str | None = None

    @property
    def lineage_root(self) -> Path:
        if self.lineage_id is None:
            return self.root
        return self.root / "generations" / "prepare" / self.lineage_id

    @property
    def generation_root(self) -> Path:
        return self.lineage_root

    @property
    def generations(self) -> Path:
        return self.root / "generations"

    @property
    def lineage_record(self) -> Path:
        return self.lineage_root / "lineage.json"

    def for_lineage(self, lineage_id: str) -> PersonaPaths:
        value = str(lineage_id)
        if not _LINEAGE_ID_RE.fullmatch(value):
            raise ValueError(f"Invalid Prepare lineage id: {lineage_id!r}")
        return PersonaPaths(root=self.root, lineage_id=value)

    def ensure_lineage(self) -> None:
        if self.lineage_id is None:
            return
        for dirname in ("dataset", "references", "models", "outputs", "cache"):
            directory = self.lineage_root / dirname
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".gitkeep").touch(exist_ok=True)

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
        return self.lineage_root / "dataset"

    @property
    def references(self) -> Path:
        return self.lineage_root / "references"

    @property
    def models(self) -> Path:
        return self.lineage_root / "models"

    @property
    def outputs(self) -> Path:
        return self.lineage_root / "outputs"

    @property
    def cache(self) -> Path:
        return self.lineage_root / "cache"

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
    (root / "generations" / "prepare").mkdir(parents=True, exist_ok=True)
    (root / "generations" / "activation-history").mkdir(parents=True, exist_ok=True)

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
