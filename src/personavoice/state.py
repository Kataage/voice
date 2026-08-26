from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from personavoice.atomic import atomic_write_json
from personavoice.dataset import SCHEMA_VERSION as DATASET_SCHEMA_VERSION
from personavoice.lfm_contract import (
    LFM_CONTRACT_FINGERPRINT,
    LFM_CONTRACT_SCHEMA_VERSION,
)
from personavoice.model_assets import (
    ASR_MODEL_REVISION,
    LFM_MODEL_REVISION,
    PYANNOTE_MODEL_REVISION,
    QWEN_ASR_MODEL_REVISION,
    QWEN_FORCED_ALIGNER_MODEL_REVISION,
    SENSE_MODEL_CMVN_SHA256,
    SENSE_MODEL_TOKENIZER_SHA256,
    SENSE_MODEL_WEIGHT_SHA256,
    SEPARATOR_SOURCE_REVISION,
    SEPARATOR_VERSION,
)
from personavoice.stage_lock import stage_lock
from personavoice.worker_contracts import purge_invalid_prepare_caches

SECRET_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "OPENAI_API_KEY",
    "PERSONAVOICE_SECRET",
)

PREPARE_RESULT_SCHEMA = 4
TRAIN_RESULT_SCHEMA = 9
LEGACY_TRAIN_RESULT_SCHEMA = 8


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_contract(path: Path) -> str:
    """Hash audited text contracts independently of checkout line endings."""

    try:
        if not path.is_file():
            return "missing"
        raw = path.read_bytes()
    except OSError:
        return "unreadable"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        normalized = raw
    else:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _prepare_cache_policy() -> str:
    """Return the exact preprocessing/model/environment implementation contract."""

    repo = _repo_root()
    contract = {
        "schema": 14,
        "prepare_result_schema": PREPARE_RESULT_SCHEMA,
        "dataset_schema": DATASET_SCHEMA_VERSION,
        "asr_revision": ASR_MODEL_REVISION,
        "qwen_asr_revision": QWEN_ASR_MODEL_REVISION,
        "qwen_forced_aligner_revision": QWEN_FORCED_ALIGNER_MODEL_REVISION,
        "separator_version": SEPARATOR_VERSION,
        "separator_source_revision": SEPARATOR_SOURCE_REVISION,
        "pyannote_revision": PYANNOTE_MODEL_REVISION,
        "sense_weight_sha256": SENSE_MODEL_WEIGHT_SHA256,
        "sense_cmvn_sha256": SENSE_MODEL_CMVN_SHA256,
        "sense_tokenizer_sha256": SENSE_MODEL_TOKENIZER_SHA256,
        "asr_lock_sha256": _file_contract(repo / "workers" / "asr" / "uv.lock"),
        "diarization_lock_sha256": _file_contract(repo / "workers" / "diarization" / "uv.lock"),
        "sense_lock_sha256": _file_contract(repo / "workers" / "sense" / "uv.lock"),
        "pipeline_code_sha256": _file_contract(repo / "src" / "personavoice" / "pipeline.py"),
        "prepare_checkpoints_code_sha256": _file_contract(
            repo / "src" / "personavoice" / "prepare_checkpoints.py"
        ),
        "media_code_sha256": _file_contract(repo / "src" / "personavoice" / "media.py"),
        "speaker_code_sha256": _file_contract(repo / "src" / "personavoice" / "speaker.py"),
        "captions_code_sha256": _file_contract(repo / "src" / "personavoice" / "captions.py"),
        "dataset_code_sha256": _file_contract(repo / "src" / "personavoice" / "dataset.py"),
        "lineage_code_sha256": _file_contract(repo / "src" / "personavoice" / "lineage.py"),
        "asr_contract_code_sha256": _file_contract(
            repo / "src" / "personavoice" / "asr_contract.py"
        ),
        "separation_code_sha256": _file_contract(
            repo / "src" / "personavoice" / "separation.py"
        ),
        # VC backend routing is intentionally excluded: adding or changing a
        # zero-shot VC runtime must not invalidate expensive Prepare/ASR/
        # diarization/SenseVoice caches. Prepare-specific worker code and locks
        # remain bound above and below; the complete launcher/response contract
        # is audited by environment_contract instead.
        "asr_worker_code_sha256": _file_contract(repo / "workers" / "asr" / "worker.py"),
        "diarization_worker_code_sha256": _file_contract(
            repo / "workers" / "diarization" / "worker.py"
        ),
        "sense_worker_code_sha256": _file_contract(repo / "workers" / "sense" / "worker.py"),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"14-{hashlib.sha256(encoded).hexdigest()[:20]}"


# Keep the historical root-cache migration marker stable for already materialized
# Whisper personas.  New upstream semantics are scoped by the independent
# Prepare-lineage fingerprint (which includes the dynamic implementation
# contract) and therefore never reuse this legacy root cache accidentally.
PREPARE_CACHE_POLICY_VERSION = "14-8fa7f248e19dab94265c"
PREPARE_CACHE_POLICY_COMPATIBILITY = {
    # The LFM export contract is a training-only change.  Keep the
    # previous Prepare generation reusable; profile/runtime and LFM export
    # updates must not force ASR, diarization, or SenseVoice work.
    "14-8fa7f248e19dab94265c": frozenset(
        {
            "14-5632a9a5a5b360e5430a",
            "14-89127e1d568497e01210",
            "14-b19d85f2c6e8eac470cf",
            "14-9b93893d6b990319b60e",
            "12-6ef53c9f266fd6794c3e",
            "12-1d31ef1abd217bcf5c4f",
        }
    ),
}


def _prepare_policy_compatible(recorded: Any) -> bool:
    if recorded == PREPARE_CACHE_POLICY_VERSION:
        return True
    compatible = PREPARE_CACHE_POLICY_COMPATIBILITY.get(PREPARE_CACHE_POLICY_VERSION)
    return bool(compatible and recorded in compatible)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _adapter_weight(path: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if _nonempty_file(candidate):
            return candidate
    return None


def _irodori_lora_complete(path: Path) -> bool:
    return _nonempty_file(path / "adapter_config.json") and _adapter_weight(path) is not None


def _lfm_adapter_complete(path: Path) -> bool:
    if not _irodori_lora_complete(path):
        return False
    marker = path / ".personavoice-base-revision"
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == LFM_MODEL_REVISION
    except OSError:
        return False


def _jsonl_contract(path: Path, *, path_key: str | None = None) -> int | None:
    if not path.is_file():
        return None
    count = 0
    lfm_marker_mode: bool | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None
                if path.name == "lfm_train.jsonl":
                    marker = value.get("lfm_contract")
                    has_marker = marker is not None
                    if lfm_marker_mode is not None and has_marker != lfm_marker_mode:
                        return None
                    lfm_marker_mode = has_marker
                    if has_marker and (
                        not isinstance(marker, dict)
                        or type(marker.get("schema_version")) is not int
                        or marker.get("schema_version") != LFM_CONTRACT_SCHEMA_VERSION
                        or marker.get("fingerprint") != LFM_CONTRACT_FINGERPRINT
                    ):
                        return None
                count += 1
                if path_key is not None:
                    raw_path = value.get(path_key)
                    if not isinstance(raw_path, str) or not raw_path:
                        return None
                    artifact = Path(raw_path)
                    if not _nonempty_file(artifact):
                        try:
                            if artifact.is_file() and artifact.stat().st_size == 0:
                                artifact.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return None
    except (OSError, json.JSONDecodeError):
        return None
    return count


def _json_object_list(path: Path) -> list[dict[str, Any]] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return None
    return value


def _sqlite_contract(path: Path, *, expected_utterances: int) -> bool:
    """Validate the lightweight canonical dataset contract without a full DB scan."""

    if not _nonempty_file(path) or expected_utterances < 0:
        return False
    try:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA query_only = ON")
            schema_row = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            count_row = connection.execute("SELECT COUNT(*) FROM utterances").fetchone()
        finally:
            connection.close()
    except (sqlite3.DatabaseError, OSError):
        return False
    if schema_row is None or count_row is None:
        return False
    return (
        str(schema_row[0]) == str(DATASET_SCHEMA_VERSION)
        and int(count_row[0]) == expected_utterances
    )


def _canonical_prepare_json_complete(
    dataset: Path,
    numeric_values: dict[str, int],
    *,
    expected_usable_seconds: float,
) -> bool:
    inventory = _json_object_list(dataset / "source_inventory.json")
    master = _json_object_list(dataset / "master.json")
    skipped = _json_object_list(dataset / "skipped_sources.json")
    if inventory is None or master is None or skipped is None:
        return False
    if len(inventory) != numeric_values["sources"]:
        return False
    if len(master) != numeric_values["utterances"]:
        return False
    if len(skipped) != numeric_values["skipped_sources"]:
        return False

    source_ids: set[str] = set()
    for source in inventory:
        digest = source.get("sha256")
        if not isinstance(digest, str) or len(digest) < 16:
            return False
        source_id = digest[:16]
        if source_id in source_ids:
            return False
        source_ids.add(source_id)
    utterance_ids: set[str] = set()
    target_count = 0
    usable_count = 0
    usable_seconds = 0.0
    target_sources: set[str] = set()
    for row in master:
        item_id = row.get("id")
        source_id = row.get("source_id")
        if not isinstance(item_id, str) or not item_id or item_id in utterance_ids:
            return False
        if not isinstance(source_id, str) or source_id not in source_ids:
            return False
        utterance_ids.add(item_id)
        if bool(row.get("target")):
            target_count += 1
            target_sources.add(source_id)
            if row.get("audio_path") and row.get("text_annotated"):
                start = _safe_float(row.get("start"))
                end = _safe_float(row.get("end"))
                if start is None or end is None or end < start:
                    return False
                usable_count += 1
                usable_seconds += end - start

    if target_count != numeric_values["target_utterances"]:
        return False
    if usable_count != numeric_values["usable_tts_utterances"]:
        return False
    if round(usable_seconds, 2) != round(expected_usable_seconds, 2):
        return False

    skipped_ids: set[str] = set()
    for item in skipped:
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or source_id not in source_ids:
            return False
        if source_id in skipped_ids or source_id in target_sources:
            return False
        skipped_ids.add(source_id)
    return True


def _prepare_artifacts_complete(persona_root: Path, result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    required_keys = {
        "prepare_schema",
        "sources",
        "skipped_sources",
        "utterances",
        "target_utterances",
        "usable_tts_utterances",
        "usable_seconds",
        "references",
        "irodori_examples",
        "lfm_examples",
        "seed_vc_examples",
        "master_db",
    }
    if not required_keys.issubset(result):
        return False
    if _safe_int(result.get("prepare_schema")) != PREPARE_RESULT_SCHEMA:
        return False
    numeric_values: dict[str, int] = {}
    for key in (
        "sources",
        "skipped_sources",
        "utterances",
        "target_utterances",
        "usable_tts_utterances",
        "references",
        "irodori_examples",
        "lfm_examples",
        "seed_vc_examples",
    ):
        value = _safe_int(result.get(key))
        if value is None or value < 0:
            return False
        numeric_values[key] = value
    usable_seconds = _safe_float(result.get("usable_seconds"))
    if usable_seconds is None or usable_seconds < 0.0:
        return False
    if numeric_values["usable_tts_utterances"] == 0:
        return False
    if numeric_values["skipped_sources"] > numeric_values["sources"]:
        return False
    if numeric_values["target_utterances"] > numeric_values["utterances"]:
        return False
    if numeric_values["usable_tts_utterances"] > numeric_values["target_utterances"]:
        return False

    lineage_id = result.get("lineage_id")
    if isinstance(lineage_id, str) and lineage_id:
        lineage_fingerprint = result.get("lineage_fingerprint")
        master_fingerprint = result.get("master_fingerprint")
        if (
            re.fullmatch(r"pl-[0-9a-f]{32}", lineage_id) is None
            or not isinstance(lineage_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", lineage_fingerprint) is None
            or not isinstance(master_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", master_fingerprint) is None
        ):
            return False
        generation = persona_root / "generations" / "prepare" / lineage_id
        dataset = generation / "dataset"
        references = generation / "references"
        lineage_record = generation / "lineage.json"
        try:
            record = json.loads(lineage_record.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if (
            not isinstance(record, dict)
            or record.get("lineage_id") != lineage_id
            or record.get("lineage_fingerprint") != lineage_fingerprint
            or record.get("master_fingerprint") != master_fingerprint
            or _safe_int(record.get("sources")) != numeric_values["sources"]
            or _safe_int(record.get("utterances")) != numeric_values["utterances"]
        ):
            return False
    else:
        # Backward-compatible historical root layout.
        dataset = persona_root / "dataset"
        references = persona_root / "references"
        if result.get("lineage_fingerprint") is not None or result.get("master_fingerprint") is not None:
            return False
    if not all(
        _nonempty_file(path)
        for path in (
            dataset / "source_inventory.json",
            dataset / "skipped_sources.json",
            dataset / "master.json",
            dataset / "master.sqlite3",
        )
    ):
        return False
    if not _canonical_prepare_json_complete(
        dataset,
        numeric_values,
        expected_usable_seconds=usable_seconds,
    ):
        return False
    if isinstance(lineage_id, str) and lineage_id:
        try:
            master_rows = _json_object_list(dataset / "master.json")
            from personavoice.lineage import master_fingerprint

            if master_rows is None or master_fingerprint(master_rows) != result.get(
                "master_fingerprint"
            ):
                return False
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    recorded_master = result.get("master_db")
    if not isinstance(recorded_master, str) or not recorded_master:
        return False
    master_db = dataset / "master.sqlite3"
    try:
        recorded_path = Path(recorded_master)
        if not recorded_path.is_absolute():
            recorded_path = (persona_root / recorded_path).resolve()
        if recorded_path != master_db.resolve():
            return False
    except OSError:
        return False
    if not _sqlite_contract(master_db, expected_utterances=numeric_values["utterances"]):
        return False

    raw_contracts = (
        (dataset / "irodori_source.jsonl", "audio", result.get("irodori_examples")),
        (dataset / "lfm_train.jsonl", None, result.get("lfm_examples")),
        (dataset / "seed_vc" / "manifest.jsonl", "audio", result.get("seed_vc_examples")),
    )
    for path, path_key, raw_expected in raw_contracts:
        expected = _safe_int(raw_expected)
        actual = _jsonl_contract(path, path_key=path_key)
        if expected is None or expected < 0 or actual is None or actual != expected:
            return False

    bank = references / "bank.json"
    try:
        bank_value = json.loads(bank.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(bank_value, dict) or not isinstance(bank_value.get("files"), list):
        return False
    reference_files = bank_value["files"]
    if len(reference_files) != numeric_values["references"]:
        return False
    for raw_path in reference_files:
        if not isinstance(raw_path, str) or not raw_path:
            return False
        reference_path = Path(raw_path)
        if not reference_path.is_absolute():
            reference_path = persona_root / reference_path
        if not _nonempty_file(reference_path):
            return False
    for report_key in ("lfm_quality_report", "irodori_quality_report"):
        report_value = result.get(report_key)
        if report_value is not None:
            report_path = Path(report_value)
            if not report_path.is_absolute():
                report_path = persona_root / report_path
            if not _nonempty_file(report_path):
                return False
    return True


def _legacy_train_artifacts_complete(result: Any, *, expected_fingerprint: str) -> bool:
    if not isinstance(result, dict):
        return False
    if not {"train_schema", "fingerprint", "irodori", "lfm_adapter", "seed_vc_cfm"}.issubset(
        result
    ):
        return False
    if _safe_int(result.get("train_schema")) != LEGACY_TRAIN_RESULT_SCHEMA:
        return False
    if result.get("fingerprint") != expected_fingerprint:
        return False
    irodori = result.get("irodori")
    if not isinstance(irodori, dict) or "base" not in irodori:
        return False
    base = irodori.get("base")
    if not isinstance(base, str) or not base or not _nonempty_file(Path(base)):
        return False
    speaker = irodori.get("speaker_embedding")
    if speaker is not None and (
        not isinstance(speaker, str) or not speaker or not _nonempty_file(Path(speaker))
    ):
        return False
    lora = irodori.get("lora_adapter")
    if lora is not None and (
        not isinstance(lora, str) or not lora or not _irodori_lora_complete(Path(lora))
    ):
        return False
    lfm = result["lfm_adapter"]
    if lfm is not None and (
        not isinstance(lfm, str) or not lfm or not _lfm_adapter_complete(Path(lfm))
    ):
        return False
    seed = result["seed_vc_cfm"]
    return seed is None or (isinstance(seed, str) and bool(seed) and _nonempty_file(Path(seed)))


_SECRET_REDACTION = "[redacted]"


def _process_secret_values() -> tuple[str, ...]:
    """Return configured credential values without exposing their names or values."""

    return tuple(
        sorted(
            {
                value
                for key in SECRET_ENV_KEYS
                if isinstance((value := os.environ.get(key)), str) and value
            },
            key=len,
            reverse=True,
        )
    )


def _redact_process_secrets(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = value
    for secret in _process_secret_values():
        redacted = redacted.replace(secret, _SECRET_REDACTION)
    return redacted


def _secret_free(value: Any, *, secret_values: tuple[str, ...] | None = None) -> bool:
    if secret_values is None:
        secret_values = _process_secret_values()
    secret_fragments = ("token", "secret", "password", "credential", "authorization")
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if any(fragment in key_text.casefold() for fragment in secret_fragments):
                return False
            if any(secret in key_text for secret in secret_values):
                return False
            if not _secret_free(child, secret_values=secret_values):
                return False
    elif isinstance(value, (list, tuple)):
        return all(_secret_free(child, secret_values=secret_values) for child in value)
    elif isinstance(value, str):
        return not any(secret in value for secret in secret_values)
    return True


def _secure_state_before_save(state: dict[str, Any]) -> None:
    """Reject credential-bearing payloads and redact diagnostic strings in-place."""

    stages = state.get("stages")
    if not isinstance(stages, dict):
        return
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        for field in ("result", "progress"):
            if field in stage and not _secret_free(stage[field]):
                raise ValueError("State result/progress may not contain credentials or secret values")
        error = stage.get("error")
        if isinstance(error, str):
            stage["error"] = _redact_process_secrets(error)



def _portable_persona_path(persona_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    root = persona_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _v03_generation_valid(
    result: Any,
    *,
    expected_fingerprint: str,
    persona_root: Path,
    require_validated: bool,
) -> bool:
    """Validate v0.3 candidate-family metadata without a v0.4 publication layer."""

    if not isinstance(result, dict):
        return False
    if _safe_int(result.get("train_schema")) != TRAIN_RESULT_SCHEMA:
        return False
    if result.get("fingerprint") != expected_fingerprint:
        return False
    lineage_id = result.get("lineage_id")
    generation_id = result.get("generation_id")
    generation_fingerprint = result.get("generation_fingerprint")
    lineage_fingerprint = result.get("prepare_lineage_fingerprint")
    master = result.get("master_fingerprint")
    if (
        not isinstance(lineage_id, str)
        or re.fullmatch(r"pl-[0-9a-f]{32}", lineage_id) is None
        or not isinstance(generation_id, str)
        or re.fullmatch(r"gen-[0-9a-f]{32}", generation_id) is None
        or not isinstance(generation_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", generation_fingerprint) is None
        or not isinstance(lineage_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", lineage_fingerprint) is None
        or not isinstance(master, str)
        or re.fullmatch(r"[0-9a-f]{64}", master) is None
    ):
        return False
    generation_value = _portable_persona_path(
        persona_root,
        result.get("generation_manifest"),
    )
    if generation_value is None or not generation_value.is_file():
        return False
    try:
        manifest = json.loads(generation_value.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    if (
        manifest.get("kind") != "personavoice-v03-generation"
        or manifest.get("architecture") != "v0.3-pre-full-fine-tuning"
        or manifest.get("lineage_id") != lineage_id
        or manifest.get("lineage_fingerprint") != lineage_fingerprint
        or manifest.get("master_fingerprint") != master
        or manifest.get("generation_id") != generation_id
        or manifest.get("generation_fingerprint") != generation_fingerprint
    ):
        return False
    if require_validated:
        validation = manifest.get("validation")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            return False
    families = manifest.get("families")
    if not isinstance(families, dict) or set(families) != {"irodori", "lfm", "seed_vc"}:
        return False
    candidate_root = generation_value.parent
    for family in families.values():
        if not isinstance(family, dict):
            return False
        status = family.get("status")
        if status not in {"complete", "not_requested"}:
            return False
        if status == "not_requested":
            if family.get("artifacts") not in (None, []):
                return False
            continue
        artifacts = family.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return False
        for item in artifacts:
            if not isinstance(item, dict):
                return False
            relative = item.get("path")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                return False
            path = (candidate_root / relative).resolve()
            try:
                path.relative_to(candidate_root.resolve())
            except ValueError:
                return False
            if not _nonempty_file(path):
                return False
            if type(item.get("size")) is not int or item["size"] != path.stat().st_size:
                return False
            if item.get("sha256") != _file_contract(path):
                return False
    try:
        from personavoice.lineage import load_lineage
        from personavoice.project import PersonaPaths

        record = load_lineage(PersonaPaths(persona_root), lineage_id)
    except (OSError, TypeError, ValueError):
        return False
    return (
        isinstance(record, dict)
        and record.get("lineage_fingerprint") == lineage_fingerprint
        and record.get("master_fingerprint") == master
    )


def _train_artifacts_complete(
    result: Any,
    *,
    expected_fingerprint: str,
    persona_root: Path | None = None,
) -> bool:
    schema = _safe_int(result.get("train_schema")) if isinstance(result, dict) else None
    if schema == LEGACY_TRAIN_RESULT_SCHEMA:
        return _legacy_train_artifacts_complete(
            result,
            expected_fingerprint=expected_fingerprint,
        )
    if schema != TRAIN_RESULT_SCHEMA or persona_root is None:
        return False
    return _v03_generation_valid(
        result,
        expected_fingerprint=expected_fingerprint,
        persona_root=persona_root,
        require_validated=True,
    )


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: dict[str, Any]) -> None:
        _secure_state_before_save(state)
        state["updated_at"] = _now()
        atomic_write_json(self.path, state)

    def stage(self, name: str) -> dict[str, Any]:
        return self.load().setdefault("stages", {}).get(name, {})

    def is_complete(self, name: str, fingerprint: str) -> bool:
        stage = self.stage(name)
        if name == "prepare" and not _prepare_policy_compatible(stage.get("cache_policy_version")):
            return False
        if stage.get("status") != "complete" or stage.get("fingerprint") != fingerprint:
            return False
        if name == "prepare":
            return _prepare_artifacts_complete(self.path.parent, stage.get("result"))
        if name == "train":
            return _train_artifacts_complete(
                stage.get("result"),
                expected_fingerprint=fingerprint,
                persona_root=self.path.parent,
            )
        return True

    def is_trained(self, fingerprint: str) -> bool:
        """Return true for a verified schema-v9 candidate, before publication."""

        stage = self.stage("train")
        if stage.get("status") not in {"trained", "complete"}:
            return False
        return _v03_generation_valid(
            stage.get("result"),
            expected_fingerprint=fingerprint,
            persona_root=self.path.parent,
            require_validated=stage.get("status") == "complete",
        )

    def set_result(self, name: str, result: dict[str, Any]) -> None:
        if name == "prepare" and "usable_tts_utterances" in result:
            usable = _safe_int(result.get("usable_tts_utterances"))
            if usable is None or usable <= 0:
                raise RuntimeError(
                    "Preparation found no usable authorized-speaker utterances. Sources where the "
                    "authorized speaker was not selected are listed in dataset/skipped_sources.json. "
                    "Add/clean identity reference audio, add source recordings containing the target "
                    "speaker, or deliberately review prepare.min_identity_similarity."
                )
        if not _secret_free(result):
            raise ValueError("State result may not contain credentials or secret values")
        state = self.load()
        state.setdefault("stages", {}).setdefault(name, {})["result"] = result
        self.save(state)

    def set_progress(self, name: str, progress: dict[str, Any]) -> None:
        if not _secret_free(progress):
            raise ValueError("Training progress may not contain credentials or secret values")
        state = self.load()
        stage = state.setdefault("stages", {}).setdefault(name, {})
        stage["progress"] = progress
        self.save(state)

    def set_status(self, name: str, status: str, *, error: str | None = None) -> None:
        if status not in {"running", "trained", "complete", "failed"}:
            raise ValueError(f"Unsupported stage status: {status!r}")
        state = self.load()
        stage = state.setdefault("stages", {}).setdefault(name, {})
        stage["status"] = status
        stage["error"] = _redact_process_secrets(error)
        if status in {"trained", "complete", "failed"}:
            stage["finished_at"] = _now()
        self.save(state)

    def _invalidate_prepare_derived(self, root: Path | None = None) -> None:
        persona_root = root or self.path.parent
        for relative in (
            Path("cache/audio"),
            Path("cache/asr"),
            Path("cache/diarization"),
            Path("cache/identity"),
            Path("cache/sense"),
            Path("dataset/clips"),
        ):
            shutil.rmtree(persona_root / relative, ignore_errors=True)

    @contextmanager
    def running(
        self,
        name: str,
        fingerprint: str,
        *,
        force: bool = False,
        success_status: str = "complete",
        lineage: bool = False,
        lineage_cache_root: Path | None = None,
    ) -> Iterator[dict[str, Any]]:
        # The OS lock is the source of truth for liveness. It is released by the
        # kernel on normal exit, exceptions, crashes, and forced process death,
        # so long-running jobs never depend on arbitrary stale timeouts or PID
        # reuse heuristics. Acquire before mutating state/cache so a second
        # process cannot invalidate artifacts owned by the active stage.
        if success_status not in {"trained", "complete"}:
            raise ValueError(f"Unsupported success status: {success_status!r}")
        with stage_lock(self.path.parent, name):
            state = self.load()
            stage = state.setdefault("stages", {}).setdefault(name, {})
            # Preserve the last durable stage record while the OS lock is held.
            # Migration callers must not authorize reuse from a snapshot read
            # before acquiring the lock because another process may finish or
            # replace that record in between.
            previous_stage = deepcopy(stage)
            if name == "prepare":
                old_fingerprint = stage.get("fingerprint")
                old_policy = stage.get("cache_policy_version")
                must_invalidate = (
                    force
                    or not _prepare_policy_compatible(old_policy)
                    or (old_fingerprint is not None and old_fingerprint != fingerprint)
                )
                if must_invalidate:
                    if lineage and lineage_cache_root is not None:
                        self._invalidate_prepare_derived(lineage_cache_root.parent)
                    elif not lineage:
                        self._invalidate_prepare_derived()
                purge_invalid_prepare_caches(
                    self.path.parent,
                    cache_root=lineage_cache_root if lineage else None,
                )

            try:
                hostname = socket.gethostname()
            except OSError:
                hostname = None
            stage.update(
                {
                    "status": "running",
                    "fingerprint": fingerprint,
                    "started_at": _now(),
                    "finished_at": None,
                    "error": None,
                    "runner": {
                        "lock_protocol": 1,
                        "run_id": uuid4().hex,
                        "pid": os.getpid(),
                        "hostname": hostname,
                    },
                }
            )
            if name == "prepare":
                stage["cache_policy_version"] = PREPARE_CACHE_POLICY_VERSION
            self.save(state)
            try:
                yield previous_stage
            except Exception as exc:
                state = self.load()
                stage = state.setdefault("stages", {}).setdefault(name, {})
                stage.update(
                    {
                        "status": "failed",
                        "finished_at": _now(),
                        "error": _redact_process_secrets(str(exc)),
                    }
                )
                self.save(state)
                raise
            else:
                state = self.load()
                stage = state.setdefault("stages", {}).setdefault(name, {})
                stage.update(
                    {
                        "status": success_status,
                        "finished_at": _now(),
                        "error": None,
                    }
                )
                self.save(state)
