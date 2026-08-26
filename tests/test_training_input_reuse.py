from __future__ import annotations

import json
import os
from pathlib import Path

from personavoice.project import PersonaPaths
from personavoice.training_inputs import ensure_irodori_manifest, read_valid_manifest


def _legacy(tmp_path: Path) -> tuple[PersonaPaths, Path]:
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    source = paths.dataset / "irodori_source.jsonl"
    source.write_text(
        json.dumps(
            {
                "audio": str((paths.dataset / "clips" / "a.flac").resolve()),
                "text": "テストです。",
                "caption": "自然に話している。",
                "speaker": "alice",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    latent = paths.cache / "irodori_latents" / "000.pt"
    latent.parent.mkdir(parents=True)
    latent.write_bytes(b"latent")
    manifest = paths.dataset / "irodori_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "text": "テストです。",
                "caption": "自然に話している。",
                "speaker_id": "json:alice",
                "latent_path": os.path.relpath(latent, start=manifest.parent),
                "num_frames": 10,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    stat = manifest.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns - 1))
    return paths, manifest


def test_adopts_complete_v030_manifest_without_reencoding(tmp_path: Path, monkeypatch) -> None:
    paths, legacy = _legacy(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(
        "personavoice.training_inputs.prepare_manifest",
        lambda *_args, **_kwargs: calls.append(object()),
    )

    selected = ensure_irodori_manifest(tmp_path, paths, conditioning="speaker")

    assert selected == legacy
    assert calls == []
    assert (paths.dataset / "irodori_manifest.contract.json").is_file()


def test_no_speaker_view_reuses_same_latent_bytes(tmp_path: Path, monkeypatch) -> None:
    paths, legacy = _legacy(tmp_path)
    monkeypatch.setattr(
        "personavoice.training_inputs.prepare_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not encode")),
    )

    selected = ensure_irodori_manifest(tmp_path, paths, conditioning="none")
    rows = read_valid_manifest(selected)

    assert selected != legacy
    assert rows is not None and "speaker_id" not in rows[0]
    assert (paths.cache / "irodori_latents" / "000.pt").read_bytes() == b"latent"


def test_training_method_does_not_participate_in_latent_contract(tmp_path: Path) -> None:
    paths, _ = _legacy(tmp_path)
    first = ensure_irodori_manifest(tmp_path, paths, conditioning="speaker")
    second = ensure_irodori_manifest(tmp_path, paths, conditioning="speaker")
    assert first == second
