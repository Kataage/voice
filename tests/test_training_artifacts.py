from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.artifacts import (
    PublicationItem,
    publish_artifact,
    publish_training_candidates,
    verify_portable_artifact,
    verify_publication,
    verify_training_candidate,
    write_artifact_contract,
)
from personavoice.config import PersonaConfig
from personavoice.irodori import _write_irodori_lora_candidate_provenance
from personavoice.model_assets import LFM_MODEL_REVISION
from personavoice.project import PersonaPaths
from personavoice.training_plan import TrainingPlan, build_training_plan


def _plan(tmp_path: Path) -> TrainingPlan:
    repo_root = Path(__file__).resolve().parents[1]
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text("{}\n", encoding="utf-8")
    (paths.dataset / "lfm_train.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = paths.dataset / "irodori_manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    return build_training_plan(
        repo_root,
        paths,
        PersonaConfig(name="alice"),
        irodori_manifest=manifest,
    )


def _irodori_candidate(root: Path, plan: TrainingPlan) -> Path:
    candidate = root / "candidate"
    (candidate / "tokenizer").mkdir(parents=True)
    (candidate / "model.safetensors").write_bytes(b"safe weights")
    (candidate / "tokenizer" / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    write_artifact_contract(
        candidate,
        family="irodori",
        method="full",
        plan=plan,
        family_fingerprint=plan.family("irodori").fingerprint,
        training=dict(plan.family("irodori").training),
        runtime={"backend": "local", "precision": "bf16"},
        source_checkpoint="checkpoints/best.pt",
    )
    return candidate


def _irodori_lora_candidate(root: Path, *, family_fingerprint: str) -> Path:
    candidate = root / "irodori-lora-candidate"
    candidate.mkdir()
    (candidate / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (candidate / "adapter_model.safetensors").write_bytes(b"irodori adapter")
    _write_irodori_lora_candidate_provenance(
        candidate,
        plan_fingerprint=family_fingerprint,
        best_validation_loss=0.25,
        best_step=25,
        selected_checkpoint="checkpoint_best_val_loss_0000025_0.250000",
    )
    return candidate


def _lfm_lora_candidate(root: Path, *, family_fingerprint: str) -> Path:
    candidate = root / "lfm-lora-candidate"
    candidate.mkdir()
    (candidate / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (candidate / "adapter_model.safetensors").write_bytes(b"lfm adapter")
    (candidate / ".personavoice-base-revision").write_text(
        LFM_MODEL_REVISION + "\n",
        encoding="utf-8",
    )
    (candidate / ".personavoice-training-method").write_text("lora\n", encoding="utf-8")
    (candidate / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "lfm",
                "method": "lora",
                "training_plan_fingerprint": family_fingerprint,
                "best_validation_loss": 0.125,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return candidate


def test_portable_contract_has_checksums_and_no_machine_paths(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    candidate = _irodori_candidate(tmp_path, plan)

    verified = verify_portable_artifact(
        candidate,
        expected_family="irodori",
        expected_plan_fingerprint=plan.fingerprint,
    )

    assert verified.published is False
    provenance = json.loads((candidate / "provenance.json").read_text(encoding="utf-8"))
    assert "C:\\" not in json.dumps(provenance)


def test_portable_contract_rejects_absolute_metadata_paths(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    candidate = tmp_path / "candidate"
    (candidate / "tokenizer").mkdir(parents=True)
    (candidate / "model.safetensors").write_bytes(b"safe weights")
    (candidate / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="absolute path"):
        write_artifact_contract(
            candidate,
            family="irodori",
            method="full",
            plan=plan,
            family_fingerprint=plan.family("irodori").fingerprint,
            training={},
            runtime={"working_directory": "C:\\private\\voice"},
            source_checkpoint="checkpoints/best.pt",
        )


def test_publish_is_transactional_and_requires_passing_gate(tmp_path: Path) -> None:
    candidate = _irodori_candidate(tmp_path, _plan(tmp_path))
    final = tmp_path / "published" / "full"

    with pytest.raises(RuntimeError, match="Quality gate did not pass"):
        publish_artifact(candidate, final, quality={"passed": False})
    assert not final.exists()

    publish_artifact(candidate, final, quality={"passed": True, "checks": []})
    assert verify_portable_artifact(final, require_published=True).published is True


def test_verification_detects_weight_tampering(tmp_path: Path) -> None:
    candidate = _irodori_candidate(tmp_path, _plan(tmp_path))
    (candidate / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="size mismatch|checksum mismatch"):
        verify_portable_artifact(candidate)


def test_verification_rejects_files_missing_from_portable_inventory(tmp_path: Path) -> None:
    candidate = _irodori_candidate(tmp_path, _plan(tmp_path))
    (candidate / "unlisted-private-note.txt").write_text("must not publish\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unlisted files"):
        verify_portable_artifact(candidate)


def test_irodori_lora_candidate_requires_exact_family_provenance(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fingerprint = plan.family("irodori").fingerprint
    legacy = tmp_path / "legacy-irodori-lora"
    legacy.mkdir()
    (legacy / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (legacy / "adapter_model.safetensors").write_bytes(b"legacy adapter")

    with pytest.raises(RuntimeError, match="provenance"):
        verify_training_candidate(
            legacy,
            family="irodori",
            method="lora",
            family_fingerprint=fingerprint,
        )

    candidate = _irodori_lora_candidate(tmp_path, family_fingerprint=fingerprint)
    verified = verify_training_candidate(
        candidate,
        family="irodori",
        method="lora",
        family_fingerprint=fingerprint,
    )
    assert verified.family_fingerprint == fingerprint

    provenance_path = candidate / ".personavoice-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["training_plan_fingerprint"] = ("0" if fingerprint[0] != "0" else "1") + fingerprint[
        1:
    ]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(RuntimeError, match="family plan"):
        verify_training_candidate(
            candidate,
            family="irodori",
            method="lora",
            family_fingerprint=fingerprint,
        )


def test_lfm_lora_candidate_requires_exact_family_provenance(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fingerprint = plan.family("lfm").fingerprint
    legacy = tmp_path / "legacy-lfm-lora"
    legacy.mkdir()
    (legacy / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (legacy / "adapter_model.safetensors").write_bytes(b"legacy adapter")
    (legacy / ".personavoice-base-revision").write_text(
        LFM_MODEL_REVISION + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="provenance"):
        verify_training_candidate(
            legacy,
            family="lfm",
            method="lora",
            family_fingerprint=fingerprint,
        )

    candidate = _lfm_lora_candidate(tmp_path, family_fingerprint=fingerprint)
    verified = verify_training_candidate(
        candidate,
        family="lfm",
        method="lora",
        family_fingerprint=fingerprint,
    )
    assert verified.family_fingerprint == fingerprint

    provenance_path = candidate / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    wrong_fingerprint = ("0" if fingerprint[0] != "0" else "1") + fingerprint[1:]
    for invalid_fingerprint in (None, wrong_fingerprint):
        provenance["training_plan_fingerprint"] = invalid_fingerprint
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with pytest.raises(RuntimeError, match="family plan"):
            verify_training_candidate(
                candidate,
                family="lfm",
                method="lora",
                family_fingerprint=fingerprint,
            )

    provenance["training_plan_fingerprint"] = fingerprint
    for field, wrong_value in (("family", "irodori"), ("method", "full")):
        expected_value = provenance[field]
        provenance[field] = wrong_value
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with pytest.raises(RuntimeError, match="family plan"):
            verify_training_candidate(
                candidate,
                family="lfm",
                method="lora",
                family_fingerprint=fingerprint,
            )
        provenance[field] = expected_value

    provenance["best_validation_loss"] = float("inf")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(RuntimeError, match="finite"):
        verify_training_candidate(
            candidate,
            family="lfm",
            method="lora",
            family_fingerprint=fingerprint,
        )


def test_candidate_set_publication_is_checksummed_and_tamper_evident(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    models = tmp_path / "persona" / "models"
    candidate = tmp_path / "candidate-cfm.pth"
    candidate.write_bytes(b"new-seed-vc-checkpoint")
    destination = models / "seed_vc" / "cfm.pth"
    item = PublicationItem(
        family="seed-vc",
        method="finetune",
        family_fingerprint=plan.family("seed-vc").fingerprint,
        candidate=candidate,
        destination=destination,
    )

    publication = publish_training_candidates(
        models,
        plan=plan,
        items=[item],
        quality={"passed": True, "checks": []},
    )

    assert publication["plan_fingerprint"] == plan.fingerprint
    expected = {
        "seed-vc": (
            "finetune",
            plan.family("seed-vc").fingerprint,
            "seed_vc/cfm.pth",
        )
    }
    verify_publication(
        models,
        expected_plan_fingerprint=plan.fingerprint,
        expected_families=expected,
    )
    destination.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest"):
        verify_publication(
            models,
            expected_plan_fingerprint=plan.fingerprint,
        )


def test_candidate_set_publication_retry_adopts_identical_committed_models(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    models = tmp_path / "persona" / "models"
    candidate = tmp_path / "candidate-cfm.pth"
    candidate.write_bytes(b"candidate-checkpoint")
    destination = models / "seed_vc" / "cfm.pth"
    item = PublicationItem(
        family="seed-vc",
        method="finetune",
        family_fingerprint=plan.family("seed-vc").fingerprint,
        candidate=candidate,
        destination=destination,
    )
    quality = {"passed": True, "checks": []}

    first = publish_training_candidates(models, plan=plan, items=[item], quality=quality)
    history = models / ".publication-history"
    history_before = sorted(path.name for path in history.iterdir())
    first_bytes = destination.read_bytes()

    second = publish_training_candidates(models, plan=plan, items=[item], quality=quality)

    assert second == first
    assert destination.read_bytes() == first_bytes
    assert sorted(path.name for path in history.iterdir()) == history_before


def test_candidate_set_publication_rolls_back_previous_model_on_marker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import personavoice.artifacts as artifacts

    plan = _plan(tmp_path)
    models = tmp_path / "persona" / "models"
    destination = models / "seed_vc" / "cfm.pth"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stable-v0.3-checkpoint")
    candidate = tmp_path / "candidate-cfm.pth"
    candidate.write_bytes(b"candidate-checkpoint")
    item = PublicationItem(
        family="seed-vc",
        method="finetune",
        family_fingerprint=plan.family("seed-vc").fingerprint,
        candidate=candidate,
        destination=destination,
    )

    def fail_marker(*args, **kwargs):
        del args, kwargs
        raise OSError("simulated marker write failure")

    monkeypatch.setattr(artifacts, "atomic_write_json", fail_marker)
    with pytest.raises(OSError, match="marker"):
        publish_training_candidates(
            models,
            plan=plan,
            items=[item],
            quality={"passed": True, "checks": []},
        )

    assert destination.read_bytes() == b"stable-v0.3-checkpoint"


def test_candidate_set_publication_restores_previous_model_when_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import personavoice.artifacts as artifacts

    plan = _plan(tmp_path)
    models = tmp_path / "persona" / "models"
    destination = models / "seed_vc" / "cfm.pth"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stable-v0.3-checkpoint")
    candidate = tmp_path / "candidate-cfm.pth"
    candidate.write_bytes(b"candidate-checkpoint")
    item = PublicationItem(
        family="seed-vc",
        method="finetune",
        family_fingerprint=plan.family("seed-vc").fingerprint,
        candidate=candidate,
        destination=destination,
    )
    real_replace = artifacts.os.replace
    failed = False

    def fail_first_staged_install(source, target):
        nonlocal failed
        source_path = Path(source)
        if not failed and ".publication-staging" in source_path.parts:
            failed = True
            raise OSError("simulated candidate install failure")
        return real_replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail_first_staged_install)
    with pytest.raises(OSError, match="candidate install"):
        publish_training_candidates(
            models,
            plan=plan,
            items=[item],
            quality={"passed": True, "checks": []},
        )

    assert failed is True
    assert destination.read_bytes() == b"stable-v0.3-checkpoint"
    assert candidate.read_bytes() == b"candidate-checkpoint"
    assert not (models / "publication.json").exists()
