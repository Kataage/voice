from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from personavoice.atomic import atomic_write_json
from personavoice.dataset import export_irodori, export_lfm, replace_utterances
from personavoice.lineage import (
    DomainBackendDisabledError,
    activate_generation,
    effective_paths,
    resolve_alignment,
    resolve_backend,
    separator_contract,
)
from personavoice.project import init_persona


def _write_generation(
    paths,
    lineage_id: str,
    generation_id: str,
    generation_fingerprint: str,
) -> None:
    lineage = paths.for_lineage(lineage_id)
    lineage.ensure_lineage()
    atomic_write_json(
        lineage.lineage_record,
        {
            "schema_version": 1,
            "contract_version": "prepare-lineage-v1",
            "lineage_id": lineage_id,
            "lineage_fingerprint": "a" * 64,
            "master_fingerprint": "b" * 64,
            "sources": 2,
            "utterances": 2,
            "active": False,
        },
    )
    candidate = paths.for_generation(lineage_id, generation_id)
    candidate.ensure_lineage()
    families = {}
    for family in ("irodori", "lfm", "seed_vc"):
        artifact = candidate.models / family / "lineage.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"{family}:{generation_id}\n", encoding="utf-8")
        families[family] = {
            "status": "complete",
            "artifacts": [
                {
                    "path": str(artifact.relative_to(candidate.generation_root)),
                    "size": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        }
    atomic_write_json(
        candidate.generation_manifest,
        {
            "schema_version": 1,
            "kind": "personavoice-v03-generation",
            "architecture": "v0.3-pre-full-fine-tuning",
            "lineage_id": lineage_id,
            "lineage_fingerprint": "a" * 64,
            "master_fingerprint": "b" * 64,
            "generation_id": generation_id,
            "generation_fingerprint": generation_fingerprint,
            "validation": {"passed": True},
            "families": families,
        },
    )


def test_asr_alignment_and_separator_contracts_are_fail_closed():
    assert resolve_backend("openai/whisper-large-v3").key == "whisper-large-v3"
    assert resolve_backend("Qwen/Qwen3-ASR-1.7B").key == "qwen3-asr-1.7b"
    with pytest.raises(DomainBackendDisabledError):
        resolve_backend("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")

    alignment = resolve_alignment("qwen3-asr-1.7b")
    assert alignment.key == "qwen3-forced-aligner-0.6b"
    assert alignment.model_id == "Qwen/Qwen3-ForcedAligner-0.6B"
    with pytest.raises(ValueError):
        resolve_alignment("qwen3-asr-1.7b", "whisper-native")
    with pytest.raises(DomainBackendDisabledError):
        resolve_alignment("qwen3-asr-1.7b", "domain-ctc")

    separator = separator_contract()
    assert separator["version"] == "0.44.2"
    assert separator["analysis_only"] is True
    assert separator["offline_requires_local_manifest"] is True


def test_generation_activation_preserves_candidates_and_supports_verified_rollback(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    lineage_id = "pl-" + "1" * 32
    first_id = "gen-" + "2" * 32
    second_id = "gen-" + "3" * 32
    _write_generation(paths, lineage_id, first_id, "c" * 64)
    _write_generation(paths, lineage_id, second_id, "d" * 64)

    first_pointer = activate_generation(
        paths,
        lineage_id,
        generation_id=first_id,
        generation_fingerprint="c" * 64,
    )
    assert first_pointer["active_generation_id"] == first_id
    assert effective_paths(paths).generation_id == first_id

    second_pointer = activate_generation(
        paths,
        lineage_id,
        generation_id=second_id,
        generation_fingerprint="d" * 64,
    )
    assert second_pointer["active_generation_id"] == second_id
    assert effective_paths(paths).generation_id == second_id
    assert list((paths.generations / "activation-history").glob("*.json"))

    first_artifact = (
        paths.for_generation(lineage_id, first_id).models / "lfm" / "lineage.json"
    )
    first_artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        activate_generation(
            paths,
            lineage_id,
            generation_id=first_id,
            generation_fingerprint="c" * 64,
        )
    assert effective_paths(paths).generation_id == second_id


def _quality_row(
    row_id: str,
    *,
    start: float,
    target: bool,
    text: str,
    audio: Path,
    complete_provenance: bool = True,
) -> dict:
    row = {
        "id": row_id,
        "source_id": "source",
        "start": start,
        "end": start + 1.0,
        "speaker": "SPEAKER_00" if target else "SPEAKER_01",
        "target": target,
        "speaker_similarity": 0.95 if target and complete_provenance else None,
        "speaker_coverage": 0.9,
        "overlap_ratio": 0.0,
        "text": text,
        "text_annotated": text,
        "emotion": "NEUTRAL",
        "events": [],
        "caption": "自然に話している。",
        "audio_path": str(audio) if target else None,
        "quality": 0.9,
    }
    if target and complete_provenance:
        row.update(
            {
                "asr_backend": "qwen3-asr-1.7b",
                "asr_model_revision": "qwen-revision",
                "asr_segment_confidence": 0.9,
                "transcript_hash": "t" * 64,
                "alignment_backend": "qwen3-forced-aligner-0.6b",
                "alignment_model_revision": "align-revision",
                "alignment_hash": "a" * 64,
                "boundary_evidence": {"start": start, "end": start + 1.0},
            }
        )
    return row


def test_quality_reports_record_provenance_rejections_and_short_response_retention(
    tmp_path: Path,
):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture-audio")
    database = tmp_path / "master.sqlite3"
    lfm_output = tmp_path / "lfm.jsonl"
    lfm_report = tmp_path / "lfm-quality.json"
    irodori_output = tmp_path / "irodori.jsonl"
    irodori_report = tmp_path / "irodori-quality.json"
    rows = [
        _quality_row("context-1", start=0.0, target=False, text="元気？", audio=audio),
        _quality_row("target-1", start=1.0, target=True, text="はい。", audio=audio),
        _quality_row("context-2", start=2.0, target=False, text="本当？", audio=audio),
        _quality_row(
            "target-2",
            start=3.0,
            target=True,
            text="ええ。",
            audio=audio,
            complete_provenance=False,
        ),
    ]
    replace_utterances(database, rows)
    metadata = {
        "lineage_id": "pl-" + "4" * 32,
        "lineage_fingerprint": "e" * 64,
        "master_fingerprint": "f" * 64,
    }

    assert (
        export_lfm(
            database,
            lfm_output,
            "alice",
            report_path=lfm_report,
            lineage_metadata=metadata,
            max_tokens=128,
        )
        == 1
    )
    assert (
        export_irodori(
            database,
            irodori_output,
            "alice",
            report_path=irodori_report,
            lineage_metadata=metadata,
            require_provenance=True,
        )
        == 1
    )

    lfm = json.loads(lfm_report.read_text(encoding="utf-8"))
    irodori = json.loads(irodori_report.read_text(encoding="utf-8"))
    assert lfm["accepted_count"] == 1
    assert lfm["rejected_count"] == 1
    assert lfm["rejection_reasons"]["insufficient_target_speaker_evidence"] == 1
    assert lfm["token_count_source"] in {"heuristic", "mixed_token_count_sources"}
    assert lfm["valid_short_or_nonverbal_retention"] is True
    assert irodori["accepted_count"] == 1
    assert irodori["rejection_reasons"]["missing_target_speaker_evidence"] == 1
    assert irodori["lineage"] == metadata
