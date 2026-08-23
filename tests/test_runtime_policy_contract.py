from __future__ import annotations

from pathlib import Path

from personavoice.environment_contract import ENVIRONMENT_CONTRACT_SCHEMA, environment_contract


def test_subprocess_runtime_policy_is_part_of_environment_generation(tmp_path: Path):
    process = tmp_path / "src" / "personavoice" / "process.py"
    process.parent.mkdir(parents=True)
    process.write_text("generation-one\n", encoding="utf-8")

    first = environment_contract(tmp_path)
    process.write_text("generation-two\n", encoding="utf-8")
    second = environment_contract(tmp_path)

    assert ENVIRONMENT_CONTRACT_SCHEMA >= 6
    assert first["runtime_policy"]["process_sha256"] != second["runtime_policy"]["process_sha256"]
    assert first != second
