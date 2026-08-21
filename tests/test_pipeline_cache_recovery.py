from __future__ import annotations

import json
from pathlib import Path

from personavoice.pipeline import _dump, _read_cache_json


def test_corrupt_prepare_json_cache_is_removed_and_recomputed(tmp_path: Path):
    cache = tmp_path / "cache" / "asr" / "source.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{truncated", encoding="utf-8")

    assert _read_cache_json(cache) is None
    assert not cache.exists()


def test_non_object_prepare_json_cache_is_not_reused(tmp_path: Path):
    cache = tmp_path / "cache" / "sense" / "clip.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("[]\n", encoding="utf-8")

    assert _read_cache_json(cache) is None
    assert not cache.exists()


def test_prepare_json_dump_is_complete_json(tmp_path: Path):
    path = tmp_path / "dataset" / "skipped_sources.json"
    value = [{"source_id": "abc", "reason": "authorized_speaker_below_identity_threshold"}]

    _dump(path, value)

    assert json.loads(path.read_text(encoding="utf-8")) == value
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
