from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from personavoice import cli
from personavoice.config import PersonaConfig
from personavoice.project import init_persona


def test_train_and_build_forward_one_run_executor_override(monkeypatch, tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    calls: list[tuple[str | None, bool]] = []

    monkeypatch.setattr(cli, "_load", lambda _name: (tmp_path, paths, cfg))
    monkeypatch.setattr(cli.console, "status", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(cli, "_print", lambda _value: None)
    monkeypatch.setattr(cli, "prepare_persona", lambda *_args, **_kwargs: {"ok": True})

    def fake_train(_root, _paths, _cfg, *, force: bool, executor: str | None):
        calls.append((executor, force))
        return {"ok": True}

    monkeypatch.setattr(cli, "train_persona", fake_train)

    cli.train("alice", force=True, executor="MODAL")
    cli.build("alice", force=False, evaluate_after=False, executor="local")

    assert calls == [("modal", True), ("local", False)]
    assert cfg.training.executor == "auto"
    assert PersonaConfig.load(paths.config).training.executor == "auto"


def test_executor_override_rejects_unknown_value() -> None:
    with pytest.raises(typer.BadParameter, match="Unsupported training executor"):
        cli._executor_override("ssh")


def test_cli_rejects_unknown_executor_before_loading_persona(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_load",
        lambda _name: pytest.fail("invalid executor must be rejected before persona loading"),
    )

    result = CliRunner().invoke(cli.app, ["train", "alice", "--executor", "ssh"])

    assert result.exit_code == 2
    assert "Unsupported training executor" in result.stderr


def test_cli_help_exposes_executor_and_explicit_migration_commands() -> None:
    runner = CliRunner()

    root_help = runner.invoke(cli.app, ["--help"])
    train_help = runner.invoke(cli.app, ["train", "--help"])
    build_help = runner.invoke(cli.app, ["build", "--help"])

    assert root_help.exit_code == 0
    assert "migrate-config" in root_help.stdout
    assert train_help.exit_code == 0 and "--executor" in train_help.stdout
    assert build_help.exit_code == 0 and "--executor" in build_help.stdout


def test_migrate_config_never_writes_until_the_explicit_non_dry_run(
    monkeypatch, tmp_path: Path
) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    legacy = {
        "name": "alice",
        "consent": {"authorized": True},
        "training": {
            "irodori_speaker_inversion": True,
            "irodori_lora": True,
            "lfm_lora": True,
            "seed_vc_finetune": False,
        },
    }
    original = yaml.safe_dump(legacy, sort_keys=False)
    paths.config.write_text(original, encoding="utf-8")
    reports: list[dict] = []

    def load(_name: str):
        return tmp_path, paths, PersonaConfig.load(paths.config)

    monkeypatch.setattr(cli, "_load", load)
    monkeypatch.setattr(cli, "_print", reports.append)

    cli.migrate_config("alice", dry_run=True)

    assert paths.config.read_text(encoding="utf-8") == original
    assert reports[-1]["migration_required"] is True
    assert reports[-1]["written"] is False
    assert reports[-1]["notes"]

    cli.migrate_config("alice", dry_run=False)

    migrated = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    assert migrated["training"]["schema_version"] == 2
    assert migrated["training"]["irodori"]["method"] == "lora"
    assert migrated["training"]["irodori"]["auxiliary_speaker_inversion"] is True
    assert reports[-1]["written"] is True


def test_consent_refuses_to_piggyback_an_implicit_legacy_migration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = init_persona(tmp_path, "alice", authorized=False)
    original = yaml.safe_dump(
        {
            "name": "alice",
            "training": {"irodori_lora": True},
        },
        sort_keys=False,
    )
    paths.config.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_load",
        lambda _name: (tmp_path, paths, PersonaConfig.load(paths.config)),
    )

    with pytest.raises(typer.BadParameter, match="will not save.*implicitly"):
        cli.consent("alice", authorized=True)

    assert paths.config.read_text(encoding="utf-8") == original


def test_cli_root_loader_honors_allowlist_without_overriding_process_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    secret = "root-env-secret-never-reported"
    (tmp_path / ".env").write_text(
        f"HF_TOKEN={secret}\nPYTHONPATH=must-not-load\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "find_repo_root", lambda: tmp_path)
    monkeypatch.setenv("HF_TOKEN", "process-wins")
    original_pythonpath = os.environ.get("PYTHONPATH")

    assert cli._repo_root() == tmp_path

    assert os.environ["HF_TOKEN"] == "process-wins"
    assert os.environ.get("PYTHONPATH") == original_pythonpath
