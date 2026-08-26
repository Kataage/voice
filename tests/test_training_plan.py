from __future__ import annotations

import json
from pathlib import Path

import pytest

import personavoice.training as training
import personavoice.training_plan as training_plan
from personavoice.config import PersonaConfig
from personavoice.model_assets import LFM_MODEL_ASSET_SHA256
from personavoice.pipeline import _prepare_fingerprint
from personavoice.project import PersonaPaths
from personavoice.training_bundle import canonical_plan_bytes
from personavoice.training_plan import TrainingPlan, build_training_plan, verify_plan_files


def _prepared(tmp_path: Path) -> tuple[Path, PersonaPaths, PersonaConfig, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"audio":"local-only.flac","text":"test"}\n', encoding="utf-8"
    )
    (paths.dataset / "lfm_train.jsonl").write_text(
        '{"messages":[{"role":"assistant","content":"test"}]}\n',
        encoding="utf-8",
    )
    manifest = paths.dataset / "irodori_manifest.jsonl"
    manifest.write_text('{"latent_path":"cache/latent.pt","text":"test"}\n', encoding="utf-8")
    return repo_root, paths, PersonaConfig(name="alice"), manifest


def test_plan_is_executor_and_remote_authorization_independent(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    local = cfg.model_copy(deep=True)
    local.training.executor = "local"
    remote = cfg.model_copy(deep=True)
    remote.training.executor = "modal"
    remote.training.remote_data_authorized = True

    local_plan = build_training_plan(repo_root, paths, local, irodori_manifest=manifest)
    remote_plan = build_training_plan(repo_root, paths, remote, irodori_manifest=manifest)

    assert local_plan.as_dict() == remote_plan.as_dict()
    assert local_plan.fingerprint == remote_plan.fingerprint
    assert local_plan.family("lfm").model_contract["base_assets_sha256"] == (LFM_MODEL_ASSET_SHA256)


def test_plan_binds_shared_executor_implementation_without_invalidating_family(
    tmp_path: Path,
) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    assert set(plan.executor_contract) == {
        "src/personavoice/artifacts.py",
        "src/personavoice/config.py",
        "src/personavoice/environment.py",
        "src/personavoice/executors.py",
        "src/personavoice/modal_app.py",
        "src/personavoice/modal_transport.py",
        "src/personavoice/process.py",
        "src/personavoice/training.py",
        "src/personavoice/training_bundle.py",
        "src/personavoice/training_plan.py",
        "src/personavoice/workers.py",
    }
    training._verify_plan_implementation(plan, repo_root)

    changed_contract = dict(plan.executor_contract)
    changed_contract["src/personavoice/training.py"] = "0" * 64
    changed = TrainingPlan(
        persona=plan.persona,
        files=plan.files,
        families=plan.families,
        executor_contract=changed_contract,
    )
    assert changed.fingerprint != plan.fingerprint
    assert changed.family("irodori").fingerprint == plan.family("irodori").fingerprint
    with pytest.raises(RuntimeError, match="executor implementation"):
        training._verify_plan_implementation(changed, repo_root)


def test_train_stage_fingerprint_reenters_on_executor_change_without_losing_family_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    plan_before = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    stage_before = training._fingerprint(paths, cfg)
    original = training._file_contract

    def changed_contract(path: Path) -> str:
        if path.name == "modal_transport.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(training, "_file_contract", changed_contract)
    stage_after = training._fingerprint(paths, cfg)
    plan_after = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    assert stage_after != stage_before
    assert plan_after.family("irodori").fingerprint == plan_before.family("irodori").fingerprint
    assert plan_after.family("lfm").fingerprint == plan_before.family("lfm").fingerprint


def test_family_checkpoint_fingerprint_ignores_publication_thresholds(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    first = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    stricter = cfg.model_copy(deep=True)
    stricter.training.quality_gate.max_cer = 0.1
    second = build_training_plan(repo_root, paths, stricter, irodori_manifest=manifest)

    assert first.family("irodori").fingerprint == second.family("irodori").fingerprint
    assert first.fingerprint != second.fingerprint


def test_full_family_fingerprints_ignore_only_method_inapplicable_knobs(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    first = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    changed = cfg.model_copy(deep=True)
    changed.training.irodori.auxiliary_speaker_inversion = True
    changed.training.irodori.speaker_inversion_max_steps += 111
    changed.training.lfm.lora_r += 8
    changed.training.lfm.lora_alpha += 16
    second = build_training_plan(repo_root, paths, changed, irodori_manifest=manifest)

    assert first.fingerprint != second.fingerprint
    assert first.family("irodori").fingerprint == second.family("irodori").fingerprint
    assert first.family("lfm").fingerprint == second.family("lfm").fingerprint
    assert first.family("irodori").auxiliary_fingerprint is None
    assert second.family("irodori").auxiliary_fingerprint is not None

    primary_changed = changed.model_copy(deep=True)
    primary_changed.training.irodori.max_steps += 1
    primary_changed.training.lfm.learning_rate *= 0.5
    third = build_training_plan(repo_root, paths, primary_changed, irodori_manifest=manifest)
    assert third.family("irodori").fingerprint != second.family("irodori").fingerprint
    assert third.family("lfm").fingerprint != second.family("lfm").fingerprint


def test_lora_and_speaker_inversion_fingerprints_use_exact_method_knobs(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    cfg.training.irodori.method = "lora"
    cfg.training.irodori.auxiliary_speaker_inversion = True
    cfg.training.lfm.method = "lora"
    first = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    changed_aux = cfg.model_copy(deep=True)
    changed_aux.training.irodori.speaker_inversion_max_steps += 1
    changed_aux.training.lfm.lora_r += 1
    second = build_training_plan(repo_root, paths, changed_aux, irodori_manifest=manifest)
    assert first.family("irodori").fingerprint == second.family("irodori").fingerprint
    assert first.family("irodori").auxiliary_fingerprint != (
        second.family("irodori").auxiliary_fingerprint
    )
    assert first.family("lfm").fingerprint != second.family("lfm").fingerprint

    speaker = cfg.model_copy(deep=True)
    speaker.training.irodori.auxiliary_speaker_inversion = False
    speaker.training.irodori.method = "speaker-inversion"
    speaker_first = build_training_plan(repo_root, paths, speaker, irodori_manifest=manifest)
    changed_unused_primary_budget = speaker.model_copy(deep=True)
    changed_unused_primary_budget.training.irodori.max_steps += 999
    speaker_second = build_training_plan(
        repo_root,
        paths,
        changed_unused_primary_budget,
        irodori_manifest=manifest,
    )
    assert speaker_first.family("irodori").fingerprint == (
        speaker_second.family("irodori").fingerprint
    )
    changed_speaker_budget = speaker.model_copy(deep=True)
    changed_speaker_budget.training.irodori.speaker_inversion_max_steps += 1
    speaker_third = build_training_plan(
        repo_root,
        paths,
        changed_speaker_budget,
        irodori_manifest=manifest,
    )
    assert speaker_first.family("irodori").fingerprint != (
        speaker_third.family("irodori").fingerprint
    )


def test_evaluation_contract_change_replans_without_invalidating_family_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    first = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    stage_before = training._fingerprint(paths, cfg)
    prepare_before = _prepare_fingerprint(paths, cfg)
    original_plan_contract = training_plan.normalized_source_sha256
    original_stage_contract = training._source_contract

    def changed_plan_contract(path: Path) -> str:
        if path.name == "evaluation.py":
            return "0" * 64
        return original_plan_contract(path)

    def changed_stage_contract(path: Path) -> str:
        if path.name == "evaluation.py":
            return "0" * 64
        return original_stage_contract(path)

    monkeypatch.setattr(training_plan, "normalized_source_sha256", changed_plan_contract)
    monkeypatch.setattr(training, "_source_contract", changed_stage_contract)
    second = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    stage_after = training._fingerprint(paths, cfg)
    prepare_after = _prepare_fingerprint(paths, cfg)

    assert first.fingerprint != second.fingerprint
    assert stage_before != stage_after
    assert prepare_before == prepare_after
    assert first.family("irodori").fingerprint == second.family("irodori").fingerprint
    assert first.family("lfm").fingerprint == second.family("lfm").fingerprint


def test_plan_mappings_are_recursively_immutable(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    with pytest.raises(TypeError):
        plan.family("irodori").training["max_steps"] = 1  # type: ignore[index]


def test_plan_file_verification_detects_post_plan_mutation(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    verify_plan_files(plan, paths.root)

    manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract failed|checksum failed"):
        verify_plan_files(plan, paths.root)


def test_disabled_family_does_not_require_its_dataset(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    cfg.training.lfm.enabled = False
    (paths.dataset / "lfm_train.jsonl").unlink()

    plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    assert plan.family("lfm").enabled is False
    assert all(item.role != "lfm-conversations" for item in plan.files)


def test_canonical_plan_round_trips_across_executor_boundary(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    rebuilt = type(plan).from_bytes(canonical_plan_bytes(plan))

    assert rebuilt == plan
    assert rebuilt.fingerprint == plan.fingerprint
    with pytest.raises(TypeError):
        rebuilt.family("lfm").training["epochs"] = 99  # type: ignore[index]


def test_plan_parser_rejects_noncanonical_or_mutated_contract(tmp_path: Path) -> None:
    repo_root, paths, cfg, manifest = _prepared(tmp_path)
    plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)
    value = plan.as_dict()

    pretty = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="canonical"):
        type(plan).from_bytes(pretty)

    value["families"][0]["method"] = "lora" if value["families"][0]["method"] == "full" else "full"
    mutated = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    rebuilt = type(plan).from_bytes(mutated)
    assert rebuilt.fingerprint != plan.fingerprint
