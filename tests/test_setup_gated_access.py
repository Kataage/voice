from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import typer
from huggingface_hub.errors import GatedRepoError
from requests import Response
from rich.console import Console

from personavoice import cli


def test_download_models_gated_access_is_actionable_and_redacts_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Response()
    response.status_code = 401
    response.url = (
        "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/revision/config.yaml"
    )
    denied = GatedRepoError("access denied", response=response)

    def blocked_download(*_args, **_kwargs):
        raise denied

    output = StringIO()
    monkeypatch.setattr(cli, "download_models", blocked_download)
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False))
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_test_token")

    with pytest.raises(typer.Exit) as raised:
        cli._download_models_or_explain(tmp_path, include_seed_vc=True)

    assert raised.value.exit_code == 2
    rendered = output.getvalue()
    assert "Hugging Face access was denied" in rendered
    assert "pyannote/speaker-diarization-community-1" in rendered
    assert "https://huggingface.co/settings/tokens" in rendered
    assert "HF_TOKEN" in rendered
    assert "hf_super_secret_test_token" not in rendered
