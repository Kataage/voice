from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.environment import (
    ENV_ALLOWLIST,
    NONSECRET_ENV_DEFAULTS,
    SECRET_ENV_KEYS,
    EnvironmentFileError,
    load_root_environment,
)
from personavoice.process import _merged_environment


def test_root_env_loads_only_allowlisted_keys_and_never_reports_values(
    tmp_path: Path,
    caplog,
    capsys,
):
    secret = "hf_private_value_that_must_not_be_reported"
    modal_secret = "modal_private_value_that_must_not_be_reported"
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"HF_TOKEN={secret}",
                "MODAL_TOKEN_ID=modal-id",
                f'MODAL_TOKEN_SECRET="{modal_secret}"',
                "PERSONAVOICE_MODAL_GPU=A100 # local override",
                "PERSONAVOICE_MODAL_FUNCTION=train-blue",
                "PYTHONPATH=should-not-load",
                "UNRELATED_SECRET=also-ignored",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}

    report = load_root_environment(tmp_path, environ=environ)

    assert environ["HF_TOKEN"] == secret
    assert environ["MODAL_TOKEN_SECRET"] == modal_secret
    assert environ["PERSONAVOICE_MODAL_GPU"] == "A100"
    assert environ["PERSONAVOICE_MODAL_FUNCTION"] == "train-blue"
    assert "PYTHONPATH" not in environ
    assert "UNRELATED_SECRET" not in environ
    assert report.loaded_from_file == (
        "HF_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "PERSONAVOICE_MODAL_FUNCTION",
        "PERSONAVOICE_MODAL_GPU",
    )
    assert report.ignored_keys == ("PYTHONPATH", "UNRELATED_SECRET")
    public_report = repr(report)
    assert secret not in public_report
    assert modal_secret not in public_report
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert modal_secret not in captured.out + captured.err
    assert not caplog.records


def test_parent_process_environment_always_wins(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "HF_TOKEN=file-token\nPERSONAVOICE_MODAL_GPU=H100\n",
        encoding="utf-8",
    )
    environ = {"HF_TOKEN": "process-token", "PERSONAVOICE_MODAL_GPU": "L40S"}

    report = load_root_environment(tmp_path, environ=environ)

    assert environ["HF_TOKEN"] == "process-token"
    assert environ["PERSONAVOICE_MODAL_GPU"] == "L40S"
    assert set(report.existing_preserved) >= {"HF_TOKEN", "PERSONAVOICE_MODAL_GPU"}
    assert "HF_TOKEN" not in report.loaded_from_file


def test_missing_env_applies_only_nonsecret_defaults(tmp_path: Path):
    environ: dict[str, str] = {}

    report = load_root_environment(tmp_path, environ=environ)

    assert report.file_found is False
    assert environ == dict(NONSECRET_ENV_DEFAULTS)
    assert set(report.defaults_applied) == set(NONSECRET_ENV_DEFAULTS)
    assert not SECRET_ENV_KEYS.intersection(environ)


def test_defaults_can_be_disabled_for_a_pure_env_file_load(tmp_path: Path):
    (tmp_path / ".env").write_text("MODAL_ENVIRONMENT=staging\n", encoding="utf-8")
    environ: dict[str, str] = {}

    report = load_root_environment(tmp_path, environ=environ, apply_defaults=False)

    assert environ == {"MODAL_ENVIRONMENT": "staging"}
    assert report.defaults_applied == ()


def test_env_parser_supports_export_quotes_comments_and_no_interpolation(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "export MODAL_ENVIRONMENT='team environment' # comment",
                'PERSONAVOICE_MODAL_APP="voice\\ntrainer"',
                "HF_TOKEN=${MODAL_TOKEN_SECRET}",
                "MODAL_TOKEN_ID=first",
                "MODAL_TOKEN_ID=last",
                "MODAL_TOKEN_SECRET=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}

    report = load_root_environment(tmp_path, environ=environ, apply_defaults=False)

    assert environ["MODAL_ENVIRONMENT"] == "team environment"
    assert environ["PERSONAVOICE_MODAL_APP"] == "voice\ntrainer"
    assert environ["HF_TOKEN"] == "${MODAL_TOKEN_SECRET}"
    assert environ["MODAL_TOKEN_ID"] == "last"
    assert "MODAL_TOKEN_SECRET" not in environ
    assert report.empty_keys == ("MODAL_TOKEN_SECRET",)


def test_malformed_secret_entry_never_echoes_its_value(tmp_path: Path):
    secret = "do-not-echo-this-secret"
    (tmp_path / ".env").write_text(f'HF_TOKEN="{secret}\n', encoding="utf-8")

    with pytest.raises(EnvironmentFileError) as captured:
        load_root_environment(tmp_path, environ={})

    assert secret not in str(captured.value)


def test_environment_contract_and_example_cover_the_same_allowlist():
    repository = Path(__file__).resolve().parents[1]
    assignments = {}
    for raw_line in (repository / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        assignments[key] = value

    assert set(assignments) == set(ENV_ALLOWLIST)
    assert all(assignments[key] == "" for key in SECRET_ENV_KEYS)
    for key, value in NONSECRET_ENV_DEFAULTS.items():
        assert assignments[key] == value
    assert assignments["MODAL_ENVIRONMENT"] == ""


def test_nonsecret_defaults_match_modal_transport_and_deployment_contracts():
    from personavoice.modal_app import ModalAppContract
    from personavoice.modal_transport import ModalSettings

    environ = dict(NONSECRET_ENV_DEFAULTS)
    transport = ModalSettings.from_env(environ)
    deployment = ModalAppContract.from_env(environ)

    assert transport.app_name == deployment.app_name == "personavoice-training"
    assert transport.function_name == "train"
    assert transport.volume_name == deployment.volume_name == "personavoice-training"
    assert deployment.gpu == "A100-40GB"
    assert deployment.hf_secret_name == "personavoice-huggingface"
    assert deployment.timeout_seconds == 86_400
    assert deployment.max_retries == 2


def test_root_credentials_are_never_inherited_by_child_processes(monkeypatch) -> None:
    sentinels = {
        "HF_TOKEN": "hf-child-secret",
        "MODAL_TOKEN_ID": "modal-child-id",
        "MODAL_TOKEN_SECRET": "modal-child-secret",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)

    inherited = _merged_environment(
        {
            **sentinels,
            "PERSONAVOICE_ROOT": "portable-runtime-root",
        }
    )

    assert not SECRET_ENV_KEYS.intersection(inherited)
    assert inherited["PERSONAVOICE_ROOT"] == "portable-runtime-root"
