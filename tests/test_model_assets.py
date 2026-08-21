from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from personavoice import irodori, setup_env
from personavoice.environment_contract import environment_contract
from personavoice.model_assets import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    IRODORI_DACVAE_FILENAME,
    IRODORI_DACVAE_ID,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_FILENAME,
    IRODORI_MODEL_ID,
    IRODORI_MODEL_SHA256,
    IRODORI_TEXT_ENCODER_ID,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_ID,
    LFM_MODEL_REVISION,
    PYANNOTE_MODEL_ID,
    PYANNOTE_MODEL_REVISION,
)


def _write_required_snapshot_files(local_dir: Path, required: tuple[str, ...]) -> None:
    for relative in required:
        path = local_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_download_models_uses_local_irodori_cache(tmp_path: Path, monkeypatch):
    calls: list[dict] = []
    snapshot_calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        filename = kwargs["filename"]
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    def fake_snapshot(**kwargs):
        snapshot_calls.append(kwargs)
        local_dir = kwargs.get("local_dir")
        if local_dir is not None:
            local = Path(local_dir)
            if kwargs["repo_id"] == LFM_MODEL_ID:
                _write_required_snapshot_files(local, setup_env._LFM_REQUIRED_FILES)
            elif kwargs["repo_id"] == ASR_MODEL_ID:
                _write_required_snapshot_files(local, setup_env._ASR_REQUIRED_FILES)
            elif kwargs["repo_id"] == PYANNOTE_MODEL_ID:
                _write_required_snapshot_files(local, setup_env._PYANNOTE_REQUIRED_FILES)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(setup_env, "hf_hub_download", fake_download)
    monkeypatch.setattr(setup_env, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(setup_env, "_verify_sha256", lambda *_args, **_kwargs: None)

    class FakeSense:
        def call(self, *_args, **_kwargs):
            return {"ok": True}

    class FakeSeed:
        def call(self, *_args, **_kwargs):
            return {"ok": True}

    def fake_worker(_root, name):
        return FakeSense() if name == "sense" else FakeSeed()

    monkeypatch.setattr(setup_env, "worker", fake_worker)
    monkeypatch.setattr(
        setup_env,
        "materialize_seed_vc_assets",
        lambda *_args, **_kwargs: {"downloaded": [], "reused": []},
    )
    monkeypatch.setattr(setup_env, "seed_vc_contract_digest", lambda _root: "seed-digest")
    marker = tmp_path / ".runtime" / "seed-vc-models-ready"
    marker.parent.mkdir(parents=True)
    marker.write_text("seed-digest\n", encoding="utf-8")
    monkeypatch.setattr(setup_env, "seed_vc_ready_marker", lambda _root: marker)

    setup_env.download_models(tmp_path)

    assert any(
        call["repo_id"] == IRODORI_MODEL_ID
        and call["filename"] == IRODORI_MODEL_FILENAME
        and Path(call["local_dir"]) == tmp_path / "models" / "irodori" / "v4.1-small"
        for call in calls
    )
    assert any(
        call["repo_id"] == IRODORI_DACVAE_ID
        and call["filename"] == IRODORI_DACVAE_FILENAME
        and Path(call["local_dir"]) == tmp_path / "models" / "irodori" / "dacvae"
        for call in calls
    )
    assert any(
        call["repo_id"] == LFM_MODEL_ID and call["revision"] == LFM_MODEL_REVISION
        for call in snapshot_calls
    )
    assert any(
        call["repo_id"] == ASR_MODEL_ID and call["revision"] == ASR_MODEL_REVISION
        for call in snapshot_calls
    )
    assert any(
        call["repo_id"] == PYANNOTE_MODEL_ID and call["revision"] == PYANNOTE_MODEL_REVISION
        for call in snapshot_calls
    )


def test_download_models_rejects_unverified_irodori_bytes(tmp_path: Path, monkeypatch):
    def fake_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        path = local_dir / kwargs["filename"]
        path.write_bytes(b"tampered")
        return str(path)

    monkeypatch.setattr(setup_env, "hf_hub_download", fake_download)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        setup_env._download_verified_file(
            model_id=IRODORI_MODEL_ID,
            filename=IRODORI_MODEL_FILENAME,
            local_dir=tmp_path / "model",
            cache_dir=tmp_path / "cache",
            sha256=IRODORI_MODEL_SHA256,
        )


def test_verify_sha256_rejects_mismatch(tmp_path: Path):
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
    monkeypatch.setattr(irodori, "sha256_file", lambda path: IRODORI_DACVAE_SHA256 if path == codec else "0" * 64)
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
    data = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert data["train"]["batch_size"] == 1
    assert data["train"]["precision"] == "fp32"
    assert data["train"]["dataloader_cuda_prefetch"] is False


def test_patched_irodori_config_rejects_text_encoder_drift(tmp_path: Path):
    source = tmp_path / "config.yaml"
    destination = tmp_path / "patched.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "text_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                    "text_encoder_revision": "main",
                },
                "train": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no longer matches the audited text encoder"):
        irodori._patched_config(source, destination, max_steps=100, backend="cpu")


def test_pinned_revisions_are_full_commits():
    for revision in (LFM_MODEL_REVISION, ASR_MODEL_REVISION, PYANNOTE_MODEL_REVISION):
        assert len(revision) == 40
        int(revision, 16)


def test_pinned_irodori_hashes_are_full_sha256():
    for digest in (IRODORI_MODEL_SHA256, IRODORI_DACVAE_SHA256):
        assert len(digest) == 64
        int(digest, 16)
