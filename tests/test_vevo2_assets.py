from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from personavoice import vevo2_assets
from personavoice.model_assets import (
    VEVO2_MODEL_ASSET_SHA256,
    VEVO2_MODEL_LICENSE,
    VEVO2_MODEL_REVISION,
    VEVO2_SOURCE_LICENSE,
    VEVO2_SOURCE_REVISION,
    VEVO2_WHISPER_MODEL_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]


def test_vevo2_contract_matches_audited_constants_and_separates_licenses():
    contract = vevo2_assets.load_contract(ROOT)

    assert contract["source"]["revision"] == VEVO2_SOURCE_REVISION
    assert contract["model"]["revision"] == VEVO2_MODEL_REVISION
    assert contract["model"]["sha256"] == VEVO2_MODEL_ASSET_SHA256
    assert contract["model"]["license"] == VEVO2_MODEL_LICENSE
    assert contract["source"]["license"] == VEVO2_SOURCE_LICENSE
    assert contract["whisper"]["sha256"] == VEVO2_WHISPER_MODEL_SHA256
    assert contract["source"]["license"] != contract["model"]["license"]


def test_vevo2_materialization_status_is_pending_without_local_assets(tmp_path: Path):
    (tmp_path / "config").mkdir()
    shutil.copy2(ROOT / "config" / "vevo2_assets.json", tmp_path / "config/vevo2_assets.json")

    result = vevo2_assets.materialization_status(tmp_path)

    assert result["ok"] is False
    assert result["contract_sha256"] == vevo2_assets.contract_digest(tmp_path)
    assert any("missing or empty" in error for error in result["errors"])
    assert "ready marker" in " ".join(result["errors"])


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "C:\\\\escape", ""])
def test_vevo2_asset_contract_rejects_nonportable_paths(tmp_path: Path, bad: str):
    contract = json.loads((ROOT / "config" / "vevo2_assets.json").read_text(encoding="utf-8"))
    contract["model"]["local_dir"] = bad
    path = tmp_path / "config" / "vevo2_assets.json"
    path.parent.mkdir()
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match="local_dir"):
        vevo2_assets.load_contract(tmp_path)
