from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from personavoice import setup_env
from personavoice.model_assets import (
    ASR_MODEL_WEIGHT_SHA256,
    LFM_MODEL_WEIGHT_SHA256,
    PYANNOTE_MODEL_ASSET_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_snapshot_pinned_repairs_corruption_even_when_revision_marker_matches(
    tmp_path: Path,
    monkeypatch,
):
    local = tmp_path / "model"
    local.mkdir()
    (local / "config.json").write_bytes(b"config")
    (local / "weight.bin").write_bytes(b"corrupt")
    (local / setup_env.REVISION_MARKER).write_text("abc123\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_bytes(b"config")
        (target / "weight.bin").write_bytes(b"audited")

    monkeypatch.setattr(setup_env, "snapshot_download", fake_snapshot_download)
    changed = setup_env._snapshot_pinned(
        model_id="org/model",
        revision="abc123",
        local_dir=local,
        required_files=("config.json", "weight.bin"),
        cache_dir=tmp_path / "cache",
        sha256={"weight.bin": _digest(b"audited")},
    )

    assert changed is True
    assert len(calls) == 1
    assert calls[0]["revision"] == "abc123"
    assert (local / "weight.bin").read_bytes() == b"audited"
    assert (local / setup_env.REVISION_MARKER).read_text(encoding="utf-8").strip() == "abc123"


def test_snapshot_pinned_never_finalizes_download_with_wrong_checksum(
    tmp_path: Path,
    monkeypatch,
):
    local = tmp_path / "model"

    def fake_snapshot_download(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_bytes(b"config")
        (target / "weight.bin").write_bytes(b"wrong")

    monkeypatch.setattr(setup_env, "snapshot_download", fake_snapshot_download)
    with pytest.raises(RuntimeError, match="failed the audited checksum contract"):
        setup_env._snapshot_pinned(
            model_id="org/model",
            revision="abc123",
            local_dir=local,
            required_files=("config.json", "weight.bin"),
            cache_dir=tmp_path / "cache",
            sha256={"weight.bin": _digest(b"audited")},
        )

    assert not local.exists()


def test_snapshot_pinned_rejects_hashes_for_undeclared_files(tmp_path: Path):
    with pytest.raises(ValueError, match="undeclared files"):
        setup_env._snapshot_pinned(
            model_id="org/model",
            revision="abc123",
            local_dir=tmp_path / "model",
            required_files=("config.json",),
            cache_dir=tmp_path / "cache",
            sha256={"weight.bin": "0" * 64},
        )


def test_root_and_isolated_worker_checksum_constants_stay_aligned():
    asr = (ROOT / "workers" / "asr" / "worker.py").read_text(encoding="utf-8")
    lfm = (ROOT / "workers" / "lfm" / "worker.py").read_text(encoding="utf-8")
    diarization = (ROOT / "workers" / "diarization" / "worker.py").read_text(
        encoding="utf-8"
    )

    assert f'PINNED_MODEL_WEIGHT_SHA256 = "{ASR_MODEL_WEIGHT_SHA256}"' in asr
    assert f'MODEL_WEIGHT_SHA256 = "{LFM_MODEL_WEIGHT_SHA256}"' in lfm
    for relative, digest in PYANNOTE_MODEL_ASSET_SHA256.items():
        assert f'"{relative}": "{digest}"' in diarization


def test_worker_runtime_paths_verify_hashes_before_model_load():
    asr = (ROOT / "workers" / "asr" / "worker.py").read_text(encoding="utf-8")
    lfm = (ROOT / "workers" / "lfm" / "worker.py").read_text(encoding="utf-8")
    diarization = (ROOT / "workers" / "diarization" / "worker.py").read_text(
        encoding="utf-8"
    )

    assert asr.index("_verify_weight(local)\n    return str(local)") < asr.index("def make_model")
    assert lfm.index("_verify_weight(local)\n    return str(local)") < lfm.index("def load_base")
    assert diarization.index("_verify_assets(local)\n    return str(local)") < diarization.index(
        "def load_pipeline"
    )
