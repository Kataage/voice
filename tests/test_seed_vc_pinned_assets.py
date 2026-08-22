from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from personavoice.environment_contract import environment_contract
from personavoice.seed_vc_assets import (
    contract_digest,
    load_contract,
    materialization_status,
    materialize,
    write_ready_marker,
)
from personavoice.setup_env import SEED_VC_REVISION

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_REVISIONS = {
    "seed_vc": "4105b66f617cedef76c18288a479036773562e36",
    "astral": "4a2e9679f76eb03753adc8c503e3c23bb9c22f26",
    "campplus": "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
    "whisper_small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "hubert": "ff022d095678a2995f3c49bab18a96a9e553f782",
    "bigvgan": "633ff708ed5b74903e86ff1298cf4a98e921c513",
}


def _write_contract(repo_root: Path, *, payload: bytes = b"asset") -> str:
    digest = hashlib.sha256(payload).hexdigest()
    contract = {
        "schema_version": 1,
        "snapshots": {
            "fixture": {
                "repo_id": "example/fixture",
                "revision": "a" * 40,
                "local_dir": "fixture",
                "required_files": ["weights.bin"],
                "sha256": {"weights.bin": digest},
            }
        },
    }
    path = repo_root / "config" / "seed_vc_assets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract), encoding="utf-8")
    return digest


def test_repository_seed_vc_contract_pins_every_transitive_snapshot():
    contract = load_contract(ROOT)
    assert set(contract["snapshots"]) == set(EXPECTED_REVISIONS)
    for name, revision in EXPECTED_REVISIONS.items():
        snapshot = contract["snapshots"][name]
        assert snapshot["revision"] == revision
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
        assert snapshot["required_files"]
        for relative, digest in snapshot["sha256"].items():
            assert relative in snapshot["required_files"]
            assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_seed_vc_contract_rejects_cross_platform_escape_spellings(tmp_path: Path):
    _write_contract(tmp_path)
    path = tmp_path / "config" / "seed_vc_assets.json"
    value = json.loads(path.read_text(encoding="utf-8"))

    value["snapshots"]["fixture"]["local_dir"] = "C:escape"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="asset root"):
        load_contract(tmp_path)

    value["snapshots"]["fixture"]["local_dir"] = "fixture"
    value["snapshots"]["fixture"]["required_files"] = ["dir\\weights.bin"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="forward-slash"):
        load_contract(tmp_path)


def test_seed_vc_contract_canonicalizes_checksum_paths(tmp_path: Path):
    payload = b"asset"
    _write_contract(tmp_path, payload=payload)
    path = tmp_path / "config" / "seed_vc_assets.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value["snapshots"]["fixture"]["sha256"].pop("weights.bin")
    value["snapshots"]["fixture"]["required_files"] = ["./weights.bin"]
    value["snapshots"]["fixture"]["sha256"] = {"weights.bin": digest}
    path.write_text(json.dumps(value), encoding="utf-8")

    contract = load_contract(tmp_path)
    snapshot = contract["snapshots"]["fixture"]
    assert snapshot["required_files"] == ["weights.bin"]
    assert snapshot["sha256"] == {"weights.bin": digest}


def test_seed_vc_materializer_uses_exact_revision_and_repairs_only_bad_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    payload = b"asset"
    _write_contract(tmp_path, payload=payload)
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / "weights.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(kwargs["local_dir"])

    from personavoice import seed_vc_assets

    monkeypatch.setattr(seed_vc_assets, "snapshot_download", fake_snapshot_download)
    cache = tmp_path / "cache"

    first = materialize(tmp_path, cache_dir=cache)
    assert first == {"downloaded": ["fixture"], "reused": []}
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "example/fixture"
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["allow_patterns"] == ["weights.bin"]
    assert (tmp_path / "models/seed_vc/assets/fixture/.personavoice-revision").read_text(
        encoding="utf-8"
    ).strip() == "a" * 40

    second = materialize(tmp_path, cache_dir=cache)
    assert second == {"downloaded": [], "reused": ["fixture"]}
    assert len(calls) == 1

    (tmp_path / "models/seed_vc/assets/fixture/weights.bin").write_bytes(b"corrupt")
    third = materialize(tmp_path, cache_dir=cache)
    assert third == {"downloaded": ["fixture"], "reused": []}
    assert len(calls) == 2


def test_seed_vc_repair_invalidates_ready_marker_before_mutation(
    tmp_path: Path,
    monkeypatch,
):
    payload = b"asset"
    _write_contract(tmp_path, payload=payload)
    directory = tmp_path / "models/seed_vc/assets/fixture"
    directory.mkdir(parents=True)
    (directory / "weights.bin").write_bytes(payload)
    (directory / ".personavoice-revision").write_text("a" * 40 + "\n", encoding="utf-8")
    write_ready_marker(tmp_path)
    (directory / "weights.bin").write_bytes(b"corrupt")

    from personavoice import seed_vc_assets

    def fail_download(**_kwargs):
        assert not (tmp_path / ".runtime/seed-vc-models-ready").exists()
        raise RuntimeError("simulated interrupted repair")

    monkeypatch.setattr(seed_vc_assets, "snapshot_download", fail_download)
    with pytest.raises(RuntimeError, match="simulated interrupted repair"):
        materialize(tmp_path, cache_dir=tmp_path / "cache")
    assert not (tmp_path / ".runtime/seed-vc-models-ready").exists()
    assert not directory.exists()


def test_seed_vc_ready_marker_is_bound_to_contract_digest(tmp_path: Path):
    payload = b"asset"
    _write_contract(tmp_path, payload=payload)
    directory = tmp_path / "models/seed_vc/assets/fixture"
    directory.mkdir(parents=True)
    (directory / "weights.bin").write_bytes(payload)
    (directory / ".personavoice-revision").write_text("a" * 40 + "\n", encoding="utf-8")

    before = materialization_status(tmp_path, verify_hashes=True)
    assert before["ok"] is False
    assert before["ready_marker_matches"] is False

    written = write_ready_marker(tmp_path)
    assert written == contract_digest(tmp_path)
    after = materialization_status(tmp_path, verify_hashes=True)
    assert after["ok"] is True
    assert after["ready_marker"] == written


def test_environment_contract_changes_with_seed_vc_asset_contract(tmp_path: Path):
    _write_contract(tmp_path, payload=b"one")
    first = environment_contract(tmp_path)
    path = tmp_path / "config" / "seed_vc_assets.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["snapshots"]["fixture"]["revision"] = "b" * 40
    path.write_text(json.dumps(value), encoding="utf-8")
    second = environment_contract(tmp_path)

    assert first["schema"] == 3
    assert first["seed_vc"]["asset_contract_sha256"] != second["seed_vc"][
        "asset_contract_sha256"
    ]


def test_seed_vc_worker_uses_direct_local_only_runtime():
    source = (ROOT / "workers" / "seed_vc" / "worker.py").read_text(encoding="utf-8")
    assert 'str(root / "inference_v2.py")' not in source
    assert "Seed-VC inference failed with exit code" not in source
    assert "wrapper.convert_voice_with_streaming" in source
    assert 'cfg_data[key]["tokenizer_name"] = whisper' in source
    assert 'cfg_data[key]["ssl_model_name"] = hubert' in source
    assert 'cfg_data["vocoder"]["pretrained_model_name_or_path"] = bigvgan' in source
    assert "attempted undeclared Hugging Face access" in source
    assert "attempted undeclared local asset access" in source
    assert "verify_hashes=True" in source


def test_seed_vc_worker_vendor_revision_matches_setup_pin():
    source = (ROOT / "workers" / "seed_vc" / "worker.py").read_text(encoding="utf-8")
    match = re.search(r'^EXPECTED_VENDOR_REVISION = "([0-9a-f]{40})"$', source, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == SEED_VC_REVISION
