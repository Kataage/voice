from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import inference, setup_env
from personavoice.api import ui
from personavoice.captions import annotate_text, build_caption, normalize_events
from personavoice.config import PersonaConfig
from personavoice.dataset import load_utterances, replace_utterances
from personavoice.doctor import _requires_cuda
from personavoice.pipeline import _batch_results, _prepare_fingerprint, _turn_rows
from personavoice.project import init_persona, safe_name
from personavoice.setup_env import _worker_extras
from personavoice.speaker import (
    cosine_similarity,
    dominant_speaker,
    overlap_ratio,
    select_target_speaker,
)
from personavoice.state import StateStore
from personavoice.training import _invalidate_training_artifacts
from personavoice.workers import local_model_env


def test_safe_name():
    assert safe_name("alice-01") == "alice-01"
    with pytest.raises(ValueError):
        safe_name("../alice")


def test_persona_init(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    assert cfg.consent.authorized is True
    assert paths.raw.exists() and paths.identity.exists()
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    assert state["schema_version"] == 2


def test_state_resume(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)
    with store.running("x", "abc"):
        store.set_result("x", {"value": 1})
    assert store.is_complete("x", "abc")
    assert store.stage("x")["result"]["value"] == 1


def _stale_prepare_paths(paths) -> list[Path]:
    return [
        paths.cache / "asr" / "old.json",
        paths.cache / "diarization" / "old.json",
        paths.cache / "identity" / "old.json",
        paths.cache / "sense" / "old.json",
        paths.dataset / "clips" / "old.flac",
    ]


def _write_stale(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")


def test_prepare_changed_fingerprint_invalidates_unsafe_subcaches(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)
    with store.running("prepare", "first"):
        pass
    stale = _stale_prepare_paths(paths)
    _write_stale(stale)
    with store.running("prepare", "second"):
        assert all(not path.exists() for path in stale)


def test_prepare_force_after_complete_invalidates_unsafe_subcaches(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)
    with store.running("prepare", "same"):
        pass
    stale = _stale_prepare_paths(paths)
    _write_stale(stale)
    # Reaching running again with the same completed fingerprint represents --force.
    with store.running("prepare", "same"):
        assert all(not path.exists() for path in stale)


def test_prepare_failed_resume_keeps_expensive_subcaches(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    store = StateStore(paths.state)
    with pytest.raises(RuntimeError), store.running("prepare", "same"):
        raise RuntimeError("interrupted")
    stale = _stale_prepare_paths(paths)
    _write_stale(stale)
    with store.running("prepare", "same"):
        assert all(path.exists() for path in stale)


def test_speaker_math():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    label, score = select_target_speaker(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        [[0.95, 0.05]],
        threshold=0.5,
    )
    assert label == "A" and score > 0.9
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "A"},
        {"start": 1.0, "end": 3.0, "speaker": "B"},
    ]
    speaker, coverage = dominant_speaker(0.0, 1.0, turns)
    assert speaker == "A" and coverage == pytest.approx(1.0)
    assert overlap_ratio(0.0, 3.0, turns) == pytest.approx(1.0 / 3.0)


def test_caption_aliases():
    assert normalize_events(["laugh", "sigh"]) == ["Laughter", "Breath"]
    assert "🤭" in annotate_text("あはは", ["laugh"])
    assert annotate_text("", ["sigh"]) == "😮‍💨"
    caption = build_caption(emotion="happy", events=["laugh"], chars_per_second=8)
    assert "嬉し" in caption and "笑い" in caption and "早口" in caption


def test_dataset_roundtrip(tmp_path: Path):
    db = tmp_path / "master.sqlite3"
    row = {
        "id": "x",
        "source_id": "s",
        "start": 0.0,
        "end": 2.0,
        "speaker": "A",
        "target": True,
        "speaker_similarity": 0.9,
        "speaker_coverage": 1.0,
        "overlap_ratio": 0.0,
        "text": "こんにちは",
        "text_annotated": "こんにちは",
        "emotion": "NEUTRAL",
        "events": [],
        "caption": "自然に話している。",
        "audio_path": None,
        "quality": 0.9,
    }
    replace_utterances(db, [row])
    loaded = load_utterances(db)
    assert loaded[0]["text"] == "こんにちは"


def test_word_aligned_long_turn_split_preserves_all_text():
    words = []
    for index, token in enumerate("ABCDEFGHIJ"):
        words.append(
            {
                "start": index * 1.0,
                "end": index * 1.0 + 0.5,
                "word": token,
                "probability": 0.95,
            }
        )
    asr = {"segments": [{"start": 0.0, "end": 10.0, "text": "ABCDEFGHIJ", "words": words}]}
    turns = [{"start": 0.0, "end": 10.0, "speaker": "A"}]
    rows = _turn_rows(asr, turns, max_seconds=3.0)
    assert len(rows) > 1
    assert "".join(row["text"] for row in rows) == "ABCDEFGHIJ"
    assert all(row["end"] - row["start"] <= 3.2 for row in rows)


def test_prepare_fingerprint_changes_when_identity_changes(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    (paths.raw / "source.wav").write_bytes(b"raw")
    identity = paths.identity / "id.wav"
    identity.write_bytes(b"one")
    first = _prepare_fingerprint(paths, cfg)
    identity.write_bytes(b"two-two")
    second = _prepare_fingerprint(paths, cfg)
    assert first != second


def test_batch_results_fails_loudly():
    with pytest.raises(RuntimeError, match="ASR failed"):
        _batch_results(
            [{"id": "x", "ok": False, "error": "boom"}],
            operation="ASR",
        )


def test_training_invalidation_removes_stale_outputs(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    stale = [
        paths.models / "irodori" / "lora" / "adapter.bin",
        paths.models / "lfm" / "adapter" / "adapter_config.json",
        paths.models / "seed_vc" / "cfm.pth",
        paths.cache / "irodori_latents" / "000.pt",
        paths.dataset / "irodori_manifest.jsonl",
        paths.cache / "irodori_lora.yaml",
    ]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")
    _invalidate_training_artifacts(paths)
    assert all(not path.exists() for path in stale)


def test_huggingface_cache_layout_is_consistent(tmp_path: Path):
    env = local_model_env(tmp_path, offline=True)
    hf_home = Path(env["HF_HOME"])
    hub_cache = Path(env["HUGGINGFACE_HUB_CACHE"])
    assert hub_cache.parent == hf_home
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_worker_backend_mapping_is_explicit():
    cuda = _worker_extras("cu128")
    assert cuda["diarization"] == "cu128"
    assert cuda["sense"] == "cu128"
    assert cuda["lfm"] == "cu128"
    assert cuda["seed_vc"] == "cu124"
    cpu = _worker_extras("cpu")
    assert all(value in {None, "cpu"} for value in cpu.values())
    assert _requires_cuda("cu128") is True
    assert _requires_cuda("cu124") is True
    assert _requires_cuda("cpu") is False


def test_worker_pyprojects_pin_pytorch_indexes():
    root = Path(__file__).resolve().parents[1]
    expectations = {
        "diarization": "pytorch-cu128",
        "sense": "pytorch-cu128",
        "lfm": "pytorch-cu128",
        "seed_vc": "pytorch-cu124",
    }
    for name, index in expectations.items():
        text = (root / "workers" / name / "pyproject.toml").read_text(encoding="utf-8")
        assert "explicit = true" in text
        assert index in text
        assert "pytorch-cpu" in text


@pytest.mark.parametrize("original", [None, b"upstream-lock"])
def test_irodori_locked_sync_restores_vendor_checkout(
    tmp_path: Path,
    monkeypatch,
    original: bytes | None,
):
    repo_root = tmp_path
    vendor = repo_root / "vendor" / "Irodori-TTS"
    vendor.mkdir(parents=True)
    managed = repo_root / "locks" / "Irodori-TTS.uv.lock"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"audited-lock")
    vendor_lock = vendor / "uv.lock"
    if original is not None:
        vendor_lock.write_bytes(original)
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append([str(value) for value in args])

    monkeypatch.setattr(setup_env, "run", fake_run)
    setup_env._install_irodori(repo_root, vendor, "cpu")

    assert calls and "--locked" in calls[0]
    if original is None:
        assert not vendor_lock.exists()
    else:
        assert vendor_lock.read_bytes() == original


def test_best_irodori_adapter_prefers_lowest_validation_loss(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    root = paths.models / "irodori" / "lora"
    (root / "checkpoint_final").mkdir(parents=True)
    worse = root / "checkpoint_best_val_loss_0001000_0.900000"
    better = root / "checkpoint_best_val_loss_0002000_0.400000"
    worse.mkdir()
    better.mkdir()
    assert inference._best_lora_adapter(paths) == better


def test_extract_json_falls_back_on_malformed_braces():
    raw = "普通の返答 {not valid json}"
    result = inference._extract_json(raw)
    assert result["text"] == raw
    assert result["voice"]["emotion"] == "NEUTRAL"


def test_synthesize_forwards_cfg_and_checks_output(tmp_path: Path, monkeypatch):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    cfg.inference.default_candidates = 1
    cfg.inference.tts_cfg_scale = 4.25
    vendor = tmp_path / "vendor" / "Irodori-TTS"
    vendor.mkdir(parents=True)
    base = tmp_path / "model.safetensors"
    base.write_bytes(b"model")
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(inference, "vendor_dir", lambda _root: vendor)
    monkeypatch.setattr(inference, "base_checkpoint", lambda _root: base)
    monkeypatch.setattr(inference, "nvidia_gpus", lambda: [])

    def fake_run(args, **_kwargs):
        argv = [str(value) for value in args]
        captured["argv"] = argv
        output = Path(argv[argv.index("--output-wav") + 1])
        output.write_bytes(b"RIFF" + b"0" * 64)

    monkeypatch.setattr(inference, "run", fake_run)
    outputs = inference.synthesize(tmp_path, paths, cfg, "テスト", candidates=1)
    assert len(outputs) == 1 and outputs[0].exists()
    argv = captured["argv"]
    assert argv[argv.index("--cfg-scale-text") + 1] == "4.25"
    assert argv[argv.index("--cfg-scale-caption") + 1] == "4.25"


def test_ui_does_not_insert_llm_text_as_html():
    html = ui()
    assert "textContent=text||''" in html
    assert "${x.text" not in html
