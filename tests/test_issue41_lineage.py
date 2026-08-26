from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from personavoice.artifacts import PublicationItem, publish_training_candidates
from personavoice.asr_contract import normalize_asr_result
from personavoice.config import PersonaConfig
from personavoice.dataset import export_lfm, replace_utterances
from personavoice.lineage import (
    DomainBackendDisabledError,
    activate_generation,
    build_lineage_record,
    lineage_identity,
    prepare_lineage_seed,
    resolve_alignment,
    resolve_backend,
)
from personavoice.model_assets import SEPARATOR_MODEL_FILENAME
from personavoice.project import PersonaPaths, init_persona
from personavoice.separation import (
    materialize_analysis_audio,
    register_separator_model,
    separation_decision,
)
from personavoice.training_plan import FamilyPlan, TrainingPlan


def test_new_default_and_fail_closed_domain_contract() -> None:
    cfg = PersonaConfig(name="alice")
    assert cfg.prepare.asr_model == "qwen3-asr-1.7b"
    assert resolve_backend("qwen").model_id == "Qwen/Qwen3-ASR-1.7B"
    assert resolve_backend("whisper-large-v3").kind == "legacy-reference"
    assert resolve_alignment("qwen3-asr-1.7b").key == "qwen3-forced-aligner-0.6b"
    with pytest.raises(DomainBackendDisabledError, match="GPL-3.0|commercial-use"):
        resolve_backend("qwen3-asr-1.7b-ja-anime-galgame-hf")
    with pytest.raises(DomainBackendDisabledError, match="disabled"):
        resolve_alignment("qwen3-asr-1.7b", "domain-ctc")
    with pytest.raises(ValueError, match="cannot be attached"):
        resolve_alignment("qwen3-asr-1.7b", "whisper-native")


def test_qwen_normalization_does_not_fabricate_whisper_confidence(tmp_path: Path) -> None:
    result = normalize_asr_result(
        {
            "language": "ja",
            "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "はい", "words": []}],
        },
        backend="qwen3-asr-1.7b",
        source_audio=tmp_path / "source.wav",
    )
    segment = result["segments"][0]
    assert segment["avg_logprob"] is None
    assert segment["no_speech_prob"] is None
    assert segment["confidence"] is None
    assert result["provenance"]["contract_version"] == "asr-normalized-v1"
    assert "alignment" not in result


def test_prepare_lineage_changes_for_asr_alignment_and_separator_policy(tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig(name="alice")
    first = prepare_lineage_seed(
        paths,
        cfg,
        prepare_schema=4,
        cache_policy_version="policy-a",
    )
    first_id, first_fingerprint = lineage_identity(first)
    changed = dict(first)
    changed["asr"] = {**first["asr"], "revision": "different"}
    changed["alignment"] = {**first["alignment"], "revision": "different"}
    changed["separation"] = {**first["separation"], "policy": "always"}
    second_id, second_fingerprint = lineage_identity(changed)
    assert first_id != second_id
    assert first_fingerprint != second_fingerprint
    assert paths.for_lineage(first_id).root == paths.root
    assert paths.for_lineage(first_id).dataset != paths.for_lineage(second_id).dataset


def test_separator_is_analysis_only_and_requires_audited_local_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    model_source = tmp_path / "download" / SEPARATOR_MODEL_FILENAME
    model_source.parent.mkdir(parents=True)
    model_source.write_bytes(b"audited separator fixture")
    audit = register_separator_model(
        repo_root,
        model_source,
        source_url="https://example.invalid/audited-model",
        model_terms="local test terms; no redistribution",
    )
    assert audit["materialized"] is True
    source = tmp_path / "personas" / "alice" / "raw" / "game_music.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"canonical source")
    paths = PersonaPaths(tmp_path / "personas" / "alice").for_lineage("pl-" + "a" * 32)
    paths.ensure_lineage()

    assert separation_decision(source, policy="off")["selected"] is False
    assert separation_decision(source, policy="auto")["selected"] is True

    monkeypatch.setattr("personavoice.separation.shutil.which", lambda _name: "audio-separator")

    def fake_separator(command, **_kwargs):
        output_dir = Path(command[command.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "Vocals.wav").write_bytes(b"derived vocal")

    monkeypatch.setattr("personavoice.separation.subprocess.run", fake_separator)
    derived, decision = materialize_analysis_audio(
        source,
        paths,
        "source-id",
        policy="auto",
        metadata={"music_heavy": True},
    )
    assert derived != source
    assert derived.is_file()
    assert derived.is_relative_to(paths.cache)
    assert source.read_bytes() == b"canonical source"
    assert decision["analysis_only"] is True
    assert decision["separator_model"]["sha256"] == hashlib.sha256(model_source.read_bytes()).hexdigest()


class _TemplateTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        # The fake still exercises the exact chat-template code path; every
        # accepted example has a deterministic three-token budget.
        return {"input_ids": [[1, 2, 3]]}


def _quality_row(
    row_id: str,
    *,
    start: float,
    target: bool,
    text: str,
    events: list[str] | None = None,
    audio: Path,
) -> dict:
    return {
        "id": row_id,
        "source_id": "recording",
        "start": start,
        "end": start + 1.0,
        "speaker": "TARGET" if target else "OTHER",
        "target": target,
        "speaker_similarity": 0.95 if target else None,
        "speaker_coverage": 1.0,
        "overlap_ratio": 0.0,
        "text": text,
        "text_annotated": text,
        "emotion": "NEUTRAL",
        "events": events or [],
        "caption": "自然に話している。",
        "audio_path": str(audio),
        "quality": 0.9,
        "asr_backend": "qwen3-asr-1.7b",
        "asr_model_id": "Qwen/Qwen3-ASR-1.7B",
        "asr_model_revision": "a" * 40,
        "asr_language_probability": None,
        "asr_segment_confidence": [],
        "transcript_hash": "b" * 64,
        "alignment_backend": "qwen3-forced-aligner-0.6b",
        "alignment_model_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "alignment_model_revision": "c" * 40,
        "alignment_hash": "d" * 64,
        "boundary_evidence": {"source": "fixture", "clip_start": start, "clip_end": start + 1.0},
    }


def test_lfm_quality_report_uses_chat_template_and_keeps_short_nonverbal_rows(tmp_path: Path) -> None:
    db = tmp_path / "master.sqlite3"
    output = tmp_path / "lfm.jsonl"
    report = tmp_path / "lfm_quality_report.json"
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    replace_utterances(
        db,
        [
            _quality_row("u1", start=0.0, target=False, text="元気？", audio=audio),
            _quality_row("u2", start=1.0, target=True, text="はい", audio=audio),
            _quality_row("u3", start=2.0, target=False, text="うん", audio=audio),
            _quality_row("u4", start=3.0, target=True, text="", events=["breath"], audio=audio),
        ],
    )
    count = export_lfm(
        db,
        output,
        "alice",
        report_path=report,
        lineage_metadata={
            "lineage_id": "pl-" + "a" * 32,
            "lineage_fingerprint": "b" * 64,
            "master_fingerprint": "c" * 64,
        },
        tokenizer=_TemplateTokenizer(),
        max_tokens=3,
    )
    assert count == 2
    values = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert json.loads(values[0]["completion"][0]["content"])["text"] == "はい"
    assert json.loads(values[1]["completion"][0]["content"])["voice"]["events"] == ["Breath"]
    saved_report = json.loads(report.read_text(encoding="utf-8"))
    assert saved_report["accepted_count"] == 2
    assert saved_report["token_count_source"] == "pinned_lfm_chat_template"
    assert saved_report["lineage"]["lineage_id"] == "pl-" + "a" * 32


def _activation_plan(lineage_id: str, lineage_fingerprint: str, master_fingerprint: str) -> TrainingPlan:
    families = tuple(
        FamilyPlan(
            family=name,
            enabled=name == "irodori",
            method=method,
            dataset_fingerprint="d" * 64,
            training={},
            model_contract={},
            implementation_contract={},
            checkpoint_policy={},
            evaluation_policy={},
        )
        for name, method in (("irodori", "full"), ("lfm", "full"), ("seed-vc", "finetune"))
    )
    return TrainingPlan(
        persona="alice",
        files=(),
        families=families,
        prepare_lineage={
            "lineage_id": lineage_id,
            "lineage_fingerprint": lineage_fingerprint,
            "master_fingerprint": master_fingerprint,
        },
    )


def test_activation_is_explicit_atomic_and_keeps_previous_generation(tmp_path: Path) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    lineage_id = "pl-" + "a" * 32
    lineage_fingerprint = "b" * 64
    master_fingerprint_value = "c" * 64
    candidate_paths = paths.for_lineage(lineage_id)
    candidate_paths.ensure_lineage()
    record = build_lineage_record(
        {
            "contract_version": "prepare-lineage-v1",
            "schema_version": 1,
            "raw_inventory_fingerprint": "d" * 64,
            "identity_inventory_fingerprint": "e" * 64,
            "asr": {},
            "alignment": {},
            "separation": {},
            "prepare_schema": 4,
            "prepare_cache_policy": "policy",
        },
        lineage_id=lineage_id,
        lineage_fingerprint=lineage_fingerprint,
        master_fingerprint=master_fingerprint_value,
        source_count=1,
        utterance_count=1,
    )
    (candidate_paths.lineage_record).write_text(json.dumps(record), encoding="utf-8")
    plan = _activation_plan(lineage_id, lineage_fingerprint, master_fingerprint_value)
    candidate = tmp_path / "candidate-seed-vc.pth"
    candidate.write_bytes(b"model")
    family_fingerprint = plan.family("seed-vc").fingerprint
    publish_training_candidates(
        candidate_paths.models,
        plan=plan,
        items=[
            PublicationItem(
                family="seed-vc",
                method="finetune",
                family_fingerprint=family_fingerprint,
                candidate=candidate,
                destination=candidate_paths.models / "seed_vc" / "cfm.pth",
            )
        ],
        quality={"passed": True},
    )
    activated = activate_generation(paths, lineage_id, plan_fingerprint=plan.fingerprint)
    assert activated["active_lineage_id"] == lineage_id
    assert json.loads((paths.generations / "active.json").read_text(encoding="utf-8"))["active_lineage_id"] == lineage_id
