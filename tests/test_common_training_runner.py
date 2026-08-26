from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import pytest

from personavoice import training
from personavoice.irodori import _write_irodori_lora_candidate_provenance
from personavoice.modal_transport import (
    CHECKPOINT_COMPLETION_NAME,
    DownloadedTrainingResult,
    ResultCandidate,
    ResultFamily,
    TrainingResultContract,
    latest_verified_family_checkpoint,
    verify_completed_directory,
)
from personavoice.project import PersonaPaths
from personavoice.training_plan import FamilyPlan, TrainingPlan


def _irodori_plan(
    *,
    method: str = "lora",
    auxiliary_speaker_inversion: bool = False,
) -> TrainingPlan:
    family = FamilyPlan(
        family="irodori",
        enabled=True,
        method=method,
        dataset_fingerprint="d" * 64,
        training={
            "auxiliary_speaker_inversion": auxiliary_speaker_inversion,
            "max_steps": 20,
            "speaker_inversion_max_steps": 10,
            "conditioning": "speaker",
            "validation_ratio": 0.1,
            "validation_every": 5,
            "checkpoint_best_n": 2,
        },
        model_contract={"revision": "pinned"},
        implementation_contract={"irodori.py": "a" * 64},
        checkpoint_policy={"resume_complete_only": True},
        evaluation_policy={"require_validation": True},
    )
    return TrainingPlan(persona="alice", files=(), families=(family,))


def _downloaded(
    plan: TrainingPlan,
    *,
    validation: dict[str, object],
) -> DownloadedTrainingResult:
    family = plan.family("irodori")
    result = ResultFamily(
        family="irodori",
        method=family.method,
        family_fingerprint=family.fingerprint,
        selected_artifact_path=f"models/irodori/{family.method}",
        candidates=(
            ResultCandidate(
                artifact_path=f"models/irodori/{family.method}",
                validation=validation,
            ),
        ),
    )
    return DownloadedTrainingResult(
        completion=None,  # type: ignore[arg-type]
        contract=TrainingResultContract(
            plan_fingerprint=plan.fingerprint,
            families=(result,),
        ),
    )


def _lora_artifact(root: Path, plan: TrainingPlan) -> Path:
    artifact = root / "models" / "irodori" / "lora"
    artifact.mkdir(parents=True)
    (artifact / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (artifact / "adapter_model.safetensors").write_bytes(b"adapter")
    return artifact


def _complete_lfm_checkpoint(path: Path, *, method: str) -> None:
    step = int(path.name.removeprefix("checkpoint-"))
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": step + 100}),
        encoding="utf-8",
    )
    for name in ("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"):
        (path / name).write_bytes(b"state")
    (path / ".personavoice-training-method").write_text(method + "\n", encoding="utf-8")
    if method == "full":
        (path / "config.json").write_text('{"model_type":"lfm2"}\n', encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"model")
    else:
        (path / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (path / "adapter_model.safetensors").write_bytes(b"adapter")
    files = []
    for candidate in sorted(path.iterdir()):
        payload = candidate.read_bytes()
        files.append(
            {
                "path": candidate.name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (path / ".personavoice-checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "step": step,
                "method": method,
                "precision": {"fp16": False, "bf16": False, "use_cpu": True},
                "files": files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_torch_step(path: Path, step: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trainer/data.pkl", pickle.dumps({"step": step}, protocol=4))
        archive.writestr("trainer/version", "3\n")


def test_remote_adoption_requires_family_bound_lora_provenance_and_matching_loss(
    tmp_path: Path,
) -> None:
    plan = _irodori_plan()
    paths = PersonaPaths(tmp_path / "persona")
    root = paths.models / ".remote" / plan.plan_id
    artifact = _lora_artifact(root, plan)
    downloaded = _downloaded(
        plan,
        validation={"passed": True, "validation_loss": 0.1, "step": 20},
    )

    with pytest.raises(RuntimeError, match="provenance|candidate"):
        training._remote_families(downloaded, plan=plan, paths=paths, root=root)

    _write_irodori_lora_candidate_provenance(
        artifact,
        plan_fingerprint=plan.family("irodori").fingerprint,
        best_validation_loss=0.2,
        best_step=20,
        selected_checkpoint="checkpoint_best_val_loss_0000020_0.200000",
    )
    with pytest.raises(RuntimeError, match="validation loss"):
        training._remote_families(downloaded, plan=plan, paths=paths, root=root)

    wrong_step = _downloaded(
        plan,
        validation={"passed": True, "validation_loss": 0.2, "step": 21},
    )
    with pytest.raises(RuntimeError, match="validation step"):
        training._remote_families(wrong_step, plan=plan, paths=paths, root=root)

    matching = _downloaded(
        plan,
        validation={"passed": True, "validation_loss": 0.2, "step": 20},
    )
    adopted = training._remote_families(matching, plan=plan, paths=paths, root=root)
    assert adopted["irodori"]["validation"] == {"loss": 0.2}


@pytest.mark.parametrize(
    ("requested", "supplied"),
    ((True, False), (False, True)),
)
def test_remote_adoption_requires_exact_auxiliary_speaker_contract(
    tmp_path: Path,
    requested: bool,
    supplied: bool,
) -> None:
    plan = _irodori_plan(auxiliary_speaker_inversion=requested)
    paths = PersonaPaths(tmp_path / "persona")
    root = paths.models / ".remote" / plan.plan_id
    artifact = _lora_artifact(root, plan)
    _write_irodori_lora_candidate_provenance(
        artifact,
        plan_fingerprint=plan.family("irodori").fingerprint,
        best_validation_loss=0.2,
        best_step=20,
        selected_checkpoint="checkpoint_best_val_loss_0000020_0.200000",
    )
    validation: dict[str, object] = {
        "passed": True,
        "validation_loss": 0.2,
        "step": 20,
    }
    if supplied:
        auxiliary = root / "auxiliary" / "speaker.speaker.safetensors"
        auxiliary.parent.mkdir(parents=True)
        auxiliary.write_bytes(b"speaker")
        validation["auxiliary_speaker_path"] = auxiliary.relative_to(root).as_posix()

    with pytest.raises(ValueError, match="auxiliary"):
        training._remote_families(
            _downloaded(plan, validation=validation),
            plan=plan,
            paths=paths,
            root=root,
        )


def test_speaker_inversion_remote_result_binds_upstream_best_checkpoint_loss(
    tmp_path: Path,
) -> None:
    plan = _irodori_plan(method="speaker-inversion")
    paths = PersonaPaths(tmp_path / "persona")
    root = paths.models / ".remote" / plan.plan_id
    artifact = root / "models" / "irodori" / "speaker-inversion"
    artifact.mkdir(parents=True)
    best = artifact / "checkpoint_best_val_loss_0000020_0.300000.speaker.safetensors"
    best.write_bytes(b"speaker")

    with pytest.raises(RuntimeError, match="validation loss"):
        training._remote_families(
            _downloaded(
                plan,
                validation={"passed": True, "validation_loss": 0.2, "step": 20},
            ),
            plan=plan,
            paths=paths,
            root=root,
        )

    adopted = training._remote_families(
        _downloaded(
            plan,
            validation={"passed": True, "validation_loss": 0.3, "step": 20},
        ),
        plan=plan,
        paths=paths,
        root=root,
    )
    assert adopted["irodori"]["validation"] == {"loss": 0.3}


def test_common_runner_resolves_only_tagged_candidates_inside_external_root(
    tmp_path: Path,
) -> None:
    paths = PersonaPaths(tmp_path / "workspace" / "persona")
    paths.root.mkdir(parents=True)
    candidate_root = tmp_path / "checkpoints" / "candidates"
    artifact = candidate_root / "irodori" / "family" / "models" / "checkpoint_final"
    artifact.mkdir(parents=True)

    value = "__candidate__/irodori/family/models/checkpoint_final"
    assert training._family_artifact_path(
        paths,
        value,
        external_candidate_root=candidate_root,
    ) == artifact
    with pytest.raises(RuntimeError, match="portable|escaped"):
        training._family_artifact_path(
            paths,
            "__candidate__/../escape",
            external_candidate_root=candidate_root,
        )
    with pytest.raises(RuntimeError, match="unexpected external"):
        training._family_artifact_path(paths, value)


def test_common_runner_rejects_root_symlink_before_resolving(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    original_resolve = Path.resolve

    def fake_is_symlink(path: Path) -> bool:
        return path == root

    def guarded_resolve(path: Path, *args, **kwargs):
        if path == root:
            raise AssertionError("symlink root was resolved before it was rejected")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(ValueError, match="real directory"):
        training._real_directory_root(root, label="result")


@pytest.mark.parametrize("method", ("full", "lora"))
def test_completed_lfm_checkpoint_is_native_complete_and_step_bound(
    tmp_path: Path,
    method: str,
) -> None:
    run_root = tmp_path / "lfm-run"
    checkpoint = run_root / "checkpoint-17"
    _complete_lfm_checkpoint(checkpoint, method=method)
    result = {
        "best_checkpoint": str(checkpoint),
        "best_validation_loss": 0.25,
        "reused": False,
    }

    assert training._completed_family_checkpoint(
        result,
        family="lfm",
        method=method,
        run_root=run_root,
    ) == (checkpoint, 17)

    result["checkpoint_step"] = 18
    with pytest.raises(RuntimeError, match="step is inconsistent"):
        training._completed_family_checkpoint(
            result,
            family="lfm",
            method=method,
            run_root=run_root,
        )


def test_completed_irodori_checkpoints_require_best_loss_and_native_resume_payload(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "irodori-run"
    checkpoint = run_root / "checkpoint_best_val_loss_0000025_0.125000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    _write_torch_step(checkpoint / "trainer_state.pt", 25)
    result = {
        "best_checkpoint": str(checkpoint),
        "checkpoint_step": 25,
        "best_validation_loss": 0.125,
        "reused": False,
    }

    assert training._completed_family_checkpoint(
        result,
        family="irodori",
        method="lora",
        run_root=run_root,
    ) == (checkpoint, 25)

    (checkpoint / "trainer_state.pt").unlink()
    with pytest.raises(RuntimeError, match="not exactly resumable"):
        training._completed_family_checkpoint(
            result,
            family="irodori",
            method="lora",
            run_root=run_root,
        )


def test_reused_candidate_is_not_claimed_as_a_new_completed_checkpoint(tmp_path: Path) -> None:
    assert (
        training._completed_family_checkpoint(
            {"reused": True},
            family="irodori",
            method="full",
            run_root=tmp_path,
        )
        is None
    )


def test_remote_artifact_copy_compares_source_and_destination_inventories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.speaker.safetensors"
    source.write_bytes(b"verified-speaker")
    destination = tmp_path / "result" / "candidate"

    def corrupt_copy(_source, target, *args, **kwargs):
        del args, kwargs
        Path(target).write_bytes(b"different-speaker")
        return str(target)

    monkeypatch.setattr(training.shutil, "copy2", corrupt_copy)
    with pytest.raises(RuntimeError, match="lossless|inventory|checksum"):
        training._copy_remote_artifact(
            source,
            destination,
            family="irodori",
            method="speaker-inversion",
            family_fingerprint="f" * 64,
        )


def test_remote_checkpoint_finalizer_attests_actual_native_file_and_rebinds_policy(
    tmp_path: Path,
) -> None:
    plan = _irodori_plan(method="speaker-inversion")
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    source = tmp_path / "native" / "checkpoint_best_val_loss_0000020_0.300000.speaker.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"native-speaker-resume-payload")

    destination = training._write_remote_checkpoint_marker(
        plan,
        "irodori",
        checkpoint_root,
        native_checkpoint=source,
        step=20,
        method="speaker-inversion",
    )

    assert (destination / source.name).read_bytes() == source.read_bytes()
    assert not (destination / "resume.json").exists()
    marker = verify_completed_directory(
        destination,
        expected_plan_fingerprint=plan.fingerprint,
        completion_name=CHECKPOINT_COMPLETION_NAME,
        expected_kind="checkpoint",
        expected_model="irodori",
    )
    assert marker.step == 20
    latest = latest_verified_family_checkpoint(
        destination.parent,
        plan_fingerprint=plan.fingerprint,
        family=plan.family("irodori"),
    )
    assert latest is not None and latest[0] == destination

    family = plan.family("irodori")
    policy_only_family = FamilyPlan(
        family=family.family,
        enabled=family.enabled,
        method=family.method,
        dataset_fingerprint=family.dataset_fingerprint,
        training=family.training,
        model_contract=family.model_contract,
        implementation_contract=family.implementation_contract,
        checkpoint_policy=family.checkpoint_policy,
        evaluation_policy={"changed_threshold_only": True},
    )
    policy_only_plan = TrainingPlan(
        persona=plan.persona,
        files=plan.files,
        families=(policy_only_family,),
    )
    assert policy_only_family.fingerprint == family.fingerprint
    assert policy_only_plan.fingerprint != plan.fingerprint
    rebound = training._write_remote_checkpoint_marker(
        policy_only_plan,
        "irodori",
        checkpoint_root,
        native_checkpoint=source,
        step=20,
        method="speaker-inversion",
    )
    assert rebound == destination
    verify_completed_directory(
        rebound,
        expected_plan_fingerprint=policy_only_plan.fingerprint,
        completion_name=CHECKPOINT_COMPLETION_NAME,
        expected_kind="checkpoint",
        expected_model="irodori",
    )


def test_remote_checkpoint_finalizer_marks_lfm_native_directory_in_place(
    tmp_path: Path,
) -> None:
    family = FamilyPlan(
        family="lfm",
        enabled=True,
        method="full",
        dataset_fingerprint="d" * 64,
        training={"epochs": 3.0},
        model_contract={"revision": "pinned"},
        implementation_contract={"train.py": "a" * 64},
        checkpoint_policy={"resume_complete_only": True},
        evaluation_policy={"require_validation": True},
    )
    plan = TrainingPlan(persona="alice", files=(), families=(family,))
    checkpoint_root = tmp_path / "checkpoints"
    source = checkpoint_root / "lfm" / family.fingerprint / "checkpoint-25"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"full-model-state")
    (source / "optimizer.pt").write_bytes(b"optimizer-state")

    destination = training._write_remote_checkpoint_marker(
        plan,
        "lfm",
        checkpoint_root,
        native_checkpoint=source,
        step=25,
        method="full",
    )

    assert destination == source
    assert (source / "model.safetensors").read_bytes() == b"full-model-state"
    assert (source / "optimizer.pt").read_bytes() == b"optimizer-state"
    latest = latest_verified_family_checkpoint(
        source.parent,
        plan_fingerprint=plan.fingerprint,
        family=family,
    )
    assert latest is not None and latest[0] == source and latest[1].step == 25


def test_remote_checkpoint_observer_commits_stable_periodic_checkpoint_during_run(
    tmp_path: Path,
) -> None:
    family = FamilyPlan(
        family="lfm",
        enabled=True,
        method="full",
        dataset_fingerprint="d" * 64,
        training={"epochs": 3.0},
        model_contract={"revision": "pinned"},
        implementation_contract={"train.py": "a" * 64},
        checkpoint_policy={"periodic": True, "resume_complete_only": True},
        evaluation_policy={"require_validation": True},
    )
    plan = TrainingPlan(persona="alice", files=(), families=(family,))
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "lfm" / family.fingerprint / "checkpoint-12"
    _complete_lfm_checkpoint(checkpoint, method="full")
    statuses: list[tuple[str, float, str]] = []
    observer = training._RemoteCheckpointObserver(
        plan,
        checkpoint_root=checkpoint_root,
        runtime_repo=tmp_path / "runtime",
        status_callback=lambda model, step, path: statuses.append((model, step, path)),
        interval_seconds=0.01,
    )

    observer.scan()
    assert not (checkpoint / CHECKPOINT_COMPLETION_NAME).exists()
    observer.scan()

    marker = verify_completed_directory(
        checkpoint,
        expected_plan_fingerprint=plan.fingerprint,
        completion_name=CHECKPOINT_COMPLETION_NAME,
        expected_kind="checkpoint",
        expected_model="lfm",
    )
    assert marker.step == 12
    assert statuses == [
        (
            "lfm",
            12.0,
            f"lfm/{family.fingerprint}/checkpoint-12",
        )
    ]


def test_remote_checkpoint_observer_ignores_incomplete_periodic_checkpoint(
    tmp_path: Path,
) -> None:
    family = FamilyPlan(
        family="lfm",
        enabled=True,
        method="full",
        dataset_fingerprint="d" * 64,
        training={"epochs": 3.0},
        model_contract={"revision": "pinned"},
        implementation_contract={"train.py": "a" * 64},
        checkpoint_policy={"periodic": True, "resume_complete_only": True},
        evaluation_policy={"require_validation": True},
    )
    plan = TrainingPlan(persona="alice", files=(), families=(family,))
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "lfm" / family.fingerprint / "checkpoint-13"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"partial")
    observer = training._RemoteCheckpointObserver(
        plan,
        checkpoint_root=checkpoint_root,
        runtime_repo=tmp_path / "runtime",
        status_callback=lambda *_args: (_ for _ in ()).throw(
            AssertionError("incomplete checkpoint must not be reported")
        ),
    )

    with pytest.raises(RuntimeError, match="could not be verified"):
        observer.scan(force=True)

    assert not (checkpoint / CHECKPOINT_COMPLETION_NAME).exists()


def test_remote_checkpoint_observer_fail_closes_partial_speaker_embedding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _irodori_plan(method="speaker-inversion")
    family = plan.family("irodori")
    candidate_root = tmp_path / "candidates"
    checkpoint = (
        candidate_root
        / "irodori"
        / family.fingerprint
        / "models"
        / "irodori"
        / "speaker"
        / "checkpoint_0000010.speaker.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"truncated")
    original = checkpoint.read_bytes()
    calls: list[Path] = []

    def reject(_vendor, path, *, env):
        del env
        calls.append(path)
        return False

    monkeypatch.setattr(training, "_verify_speaker_embedding_checkpoint", reject)
    monkeypatch.setattr(training, "local_model_env", lambda _root: {})
    observer = training._RemoteCheckpointObserver(
        plan,
        checkpoint_root=tmp_path / "checkpoints",
        candidate_root=candidate_root,
        runtime_repo=tmp_path / "runtime",
        status_callback=lambda *_args: (_ for _ in ()).throw(
            AssertionError("partial speaker checkpoint must not be reported")
        ),
    )

    with pytest.raises(RuntimeError, match="could not be verified"):
        observer.scan(force=True)

    assert calls == [checkpoint]
    assert checkpoint.read_bytes() == original
    assert not (checkpoint.parent / "checkpoint-10" / CHECKPOINT_COMPLETION_NAME).exists()


def test_local_checkpoint_observer_reports_native_step_without_remote_attestation(
    tmp_path: Path,
) -> None:
    family = FamilyPlan(
        family="lfm",
        enabled=True,
        method="lora",
        dataset_fingerprint="d" * 64,
        training={"epochs": 3.0},
        model_contract={"revision": "pinned"},
        implementation_contract={"train.py": "a" * 64},
        checkpoint_policy={"periodic": True, "resume_complete_only": True},
        evaluation_policy={"require_validation": True},
    )
    plan = TrainingPlan(persona="alice", files=(), families=(family,))
    persona = tmp_path / "persona"
    checkpoint_root = persona / "cache" / "training_runs"
    checkpoint = checkpoint_root / "lfm" / family.fingerprint / "checkpoint-14"
    _complete_lfm_checkpoint(checkpoint, method="lora")
    statuses: list[tuple[str, float, str]] = []
    observer = training._RemoteCheckpointObserver(
        plan,
        checkpoint_root=checkpoint_root,
        candidate_root=persona / "models" / ".candidates",
        runtime_repo=tmp_path,
        status_callback=lambda model, step, path: statuses.append((model, step, path)),
        attest=False,
        status_path_root=persona,
    )

    observer.scan()
    observer.scan()

    assert not (checkpoint / CHECKPOINT_COMPLETION_NAME).exists()
    assert statuses == [
        (
            "lfm",
            14.0,
            f"cache/training_runs/lfm/{family.fingerprint}/checkpoint-14",
        )
    ]
