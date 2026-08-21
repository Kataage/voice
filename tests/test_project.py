from pathlib import Path

from personavoice.config import PersonaConfig
from personavoice.project import PERSONA_DIRS, init_persona


def test_init_persona_creates_contract(tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)

    assert paths.root == tmp_path / "personas" / "alice"
    assert (paths.root / "persona.yaml").is_file()
    assert (paths.root / "state.json").is_file()
    for dirname in PERSONA_DIRS:
        assert (paths.root / dirname).is_dir()

    config = PersonaConfig.load(paths.root / "persona.yaml")
    assert config.name == "alice"
    assert config.consent.authorized is True


def test_init_persona_is_idempotent(tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice")
    original = (paths.root / "persona.yaml").read_text(encoding="utf-8")

    init_persona(tmp_path, "alice", authorized=True)

    assert (paths.root / "persona.yaml").read_text(encoding="utf-8") == original
