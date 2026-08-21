from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from personavoice import irodori, setup_env
from personavoice.model_assets import (
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_REVISION,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_REVISION,
    IRODORI_MODEL_SHA256,
    IRODORI_SOURCE_REVISION,
    SEED_VC_SOURCE_REVISION,
)

ROOT = Path(__file__).resolve().parents[1]


def _vendor(root: Path) -> Path:
    vendor = root / "vendor" / "Irodori-TTS"
    (vendor / ".git").mkdir(parents=True)
    (vendor / "infer.py").write_text("# pinned\n", encoding="utf-8")
    return vendor


def _git_result(stdout: str):
    return SimpleNamespace(stdout=stdout)


def test_vendor_runtime_requires_pinned_clean_checkout(tmp_path: Path, monkeypatch):
    vendor = _vendor(tmp_path)

    def clean_run(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return _git_result(IRODORI_SOURCE_REVISION + "\n")
        if args[:3] == ["git", "status", "--porcelain"]:
            return _git_result("")
        raise AssertionError(args)

    monkeypatch.setattr(irodori, "run", clean_run)
    assert irodori.vendor_dir(tmp_path) == vendor

    def wrong_head(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return _git_result("0" * 40 + "\n")
        return _git_result("")

    monkeypatch.setattr(irodori, "run", wrong_head)
    with pytest.raises(RuntimeError, match="vendor HEAD mismatch"):
        irodori.vendor_dir(tmp_path)

    def dirty_run(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return _git_result(IRODORI_SOURCE_REVISION + "\n")
        return _git_result(" M infer.py\n")

    monkeypatch.setattr(irodori, "run", dirty_run)
    with pytest.raises(RuntimeError, match="tracked local modifications"):
        irodori.vendor_dir(tmp_path)


def test_irodori_runtime_rejects_corrupt_pinned_assets(tmp_path: Path, monkeypatch):
    base = tmp_path / "models" / "irodori" / "v4.1-small" / IRODORI_MODEL_FILENAME
    codec = tmp_path / "models" / "irodori" / "dacvae" / IRODORI_DACVAE_FILENAME
    base.parent.mkdir(parents=True)
    codec.parent.mkdir(parents=True)
    base.write_bytes(b"corrupt-base")
    codec.write_bytes(b"corrupt-codec")
    monkeypatch.setattr(irodori, "sha256_file", lambda _path: "0" * 64)

    with pytest.raises(RuntimeError, match="base checkpoint checksum mismatch"):
        irodori.base_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="DACVAE checksum mismatch"):
        irodori.codec_checkpoint(tmp_path)


def test_online_base_materialization_replaces_corruption_and_rehashes(
    tmp_path: Path,
    monkeypatch,
):
    base = tmp_path / "models" / "irodori" / "v4.1-small" / IRODORI_MODEL_FILENAME
    base.parent.mkdir(parents=True)
    base.write_bytes(b"corrupt")

    def fake_sha(path: Path) -> str:
        if path == base and path.read_bytes() == b"audited":
            return IRODORI_MODEL_SHA256
        return "0" * 64

    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / IRODORI_MODEL_FILENAME
        destination.write_bytes(b"audited")
        return str(destination)

    monkeypatch.setattr(irodori, "sha256_file", fake_sha)
    monkeypatch.setattr(irodori, "hf_hub_download", fake_download)

    assert irodori.base_checkpoint(tmp_path, online=True) == base
    assert base.read_bytes() == b"audited"
    assert len(calls) == 1
    assert calls[0]["revision"] == IRODORI_MODEL_REVISION


def test_runtime_hash_constants_and_source_pins_stay_aligned():
    for value in (
        IRODORI_SOURCE_REVISION,
        SEED_VC_SOURCE_REVISION,
        IRODORI_MODEL_REVISION,
        IRODORI_DACVAE_REVISION,
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", value)
    assert len(IRODORI_MODEL_SHA256) == 64
    assert len(IRODORI_DACVAE_SHA256) == 64
    assert setup_env.IRODORI_REVISION == IRODORI_SOURCE_REVISION
    assert setup_env.SEED_VC_REVISION == SEED_VC_SOURCE_REVISION


def test_lock_scripts_cannot_drift_from_audited_irodori_source_pin():
    shell = (ROOT / "scripts" / "lock_all.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "scripts" / "lock_all.ps1").read_text(encoding="utf-8")
    assert f'IRODORI_REVISION="{IRODORI_SOURCE_REVISION}"' in shell
    assert f'$IrodoriRevision = "{IRODORI_SOURCE_REVISION}"' in powershell
