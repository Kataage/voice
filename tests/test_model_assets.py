from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from personavoice import irodori, setup_env
from personavoice.environment_contract import environment_contract
from personavoice.model_assets import (
    IRODORI_TEXT_ENCODER_ID,
    IRODORI_TEXT_ENCODER_REVISION,
)


def test_snapshot_pinned_replaces_unmarked_materialization(tmp_path: Path, monkeypatch):
    local = tmp_path / "model"
    local.mkdir()
    (local / "config.json").write_text("stale", encoding="utf-8")
    (local / "stale.bin").write_bytes(b"stale")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("pinned", encoding="utf-8")
        (target / "weights.bin").write_bytes(b"weights")

    monkeypatch.setattr(setup_env, "snapshot_download", fake_snapshot_download)
    changed = setup_env._snapshot_pinned(
        model_id="org/model",
        revision="abc123",
        local_dir=local,
        required_files=("config.json", "weights.bin"),
        cache_dir=tmp_path / "cache",
    )

    assert changed is True
    assert calls[0]["revision"] == "abc123"
    assert not (local / "stale.bin").exists()
    assert (local / setup_env.REVISION_MARKER).read_text(encoding="utf-8").strip() == "abc123"


def test_snapshot_pinned_reuses_only_complete_exact_materialization(tmp_path: Path, monkeypatch):
    local = tmp_path / "model"
    local.mkdir()
    (local / "config.json").write_text("pinned", encoding="utf-8")
    (local / "weights.bin").write_bytes(b"weights")
    (local / setup_env.REVISION_MARKER).write_text("abc123\n", encoding="utf-8")

    def unexpected_download(**_kwargs):
        raise AssertionError("exact pinned materialization must be reused")

    monkeypatch.setattr(setup_env, "snapshot_download", unexpected_download)
    changed = setup_env._snapshot_pinned(
        model_id="org/model",
        revision="abc123",
        local_dir=local,
        required_files=("config.json", "weights.bin"),
        cache_dir=tmp_path / "cache",
    )
    assert changed is False


def test_snapshot_pinned_rematerializes_when_secondary_asset_is_missing(
    tmp_path: Path,
    monkeypatch,
):
    local = tmp_path / "model"
    local.mkdir()
    (local / "config.json").write_text("pinned", encoding="utf-8")
    (local / setup_env.REVISION_MARKER).write_text("abc123\n", encoding="utf-8")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        (target / "config.json").write_text("pinned", encoding="utf-8")
        (target / "weights.bin").write_bytes(b"weights")

    monkeypatch.setattr(setup_env, "snapshot_download", fake_snapshot_download)
    changed = setup_env._snapshot_pinned(
        model_id="org/model",
        revision="abc123",
        local_dir=local,
        required_files=("config.json", "weights.bin"),
        cache_dir=tmp_path / "cache",
    )

    assert changed is True
    assert len(calls) == 1
    assert (local / "weights.bin").read_bytes() == b"weights"


def test_snapshot_pinned_never_finalizes_incomplete_download(tmp_path: Path, monkeypatch):
    local = tmp_path / "model"

    def incomplete_snapshot_download(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("pinned", encoding="utf-8")
        (target / "weights.bin").write_bytes(b"")

    monkeypatch.setattr(setup_env, "snapshot_download", incomplete_snapshot_download)
    with pytest.raises(FileNotFoundError, match="required files are missing or empty"):
        setup_env._snapshot_pinned(
            model_id="org/model",
            revision="abc123",
            local_dir=local,
            required_files=("config.json", "weights.bin"),
            cache_dir=tmp_path / "cache",
        )

    assert not (local / setup_env.REVISION_MARKER).exists()


def test_verified_file_rejects_checksum_mismatch(tmp_path: Path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        setup_env._verify_sha256(path, "0" * 64, label="asset")


def test_prepare_manifest_passes_local_codec_and_recorded_cpu_backend(
    tmp_path: Path,
    monkeypatch,
):
    vendor = tmp_path / "vendor" / "Irodori-TTS"
    vendor.mkdir(parents=True)
    (vendor / "prepare_manifest.py").write_text("", encoding="utf-8")
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "setup.json").write_text(
        json.dumps(
            {
                "irodori_backend": "cpu",
                "environment_contract": environment_contract(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    codec = tmp_path / "models" / "irodori" / "dacvae" / "weights.pth"
    codec.parent.mkdir(parents=True)
    codec.write_bytes(b"codec")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    latents = tmp_path / "latents"
    captured: list[str] = []

    def fake_run(args, **_kwargs):
        captured.extend(str(value) for value in args)
        manifest.write_text('{"text":"x","latent_path":"x.pt","num_frames":1}\n', encoding="utf-8")

    monkeypatch.setattr(irodori, "vendor_dir", lambda _root: vendor)
    monkeypatch.setattr(irodori, "run", fake_run)
    irodori.prepare_manifest(tmp_path, source, manifest, latents)

    assert captured[captured.index("--codec-repo") + 1] == str(codec)
    assert captured[captured.index("--device") + 1] == "cpu"


def test_patched_irodori_config_requires_audited_text_encoder(tmp_path: Path):
    source = tmp_path / "config.yaml"
    destination = tmp_path / "patched.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "text_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                    "text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
                    "caption_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                },
                "train": {
                    "batch_size": 16,
                    "gradient_accumulation_steps": 1,
                    "num_workers": 8,
                    "max_steps": 3000,
                    "precision": "bf16",
                    "allow_tf32": True,
                    "dataloader_cuda_prefetch": True,
                },
            }
        ),
        encoding="utf-8",
    )

    irodori._patched_config(source, destination, max_steps=100, backend="cpu")
    patched = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert patched["train"]["precision"] == "fp32"
    assert patched["train"]["allow_tf32"] is False
    assert patched["train"]["dataloader_cuda_prefetch"] is False

    bad = yaml.safe_load(source.read_text(encoding="utf-8"))
    bad["model"]["text_encoder_revision"] = "floating-main"
    source.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited text encoder"):
        irodori._patched_config(source, destination, max_steps=100, backend="cpu")
