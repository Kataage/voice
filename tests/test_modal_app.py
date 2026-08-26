from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import personavoice.modal_app as modal_app
from personavoice.modal_app import (
    ASSET_INDEX_NAME,
    MODAL_IMAGE_ROOT,
    MODAL_VOLUME_MOUNT,
    REMOTE_STATUS_NAME,
    RESUME_INDEX_NAME,
    AssetSpec,
    ModalAppContract,
    acquire_remote_family_claims,
    acquire_remote_training_claim,
    asset_specs_for_plan,
    create_modal_app,
    execute_claimed_remote_training,
    execute_remote_training,
    materialize_verified_assets,
    recover_remote_terminal_claim,
    release_remote_family_claims,
    release_remote_training_claim,
    training_plan_from_bytes,
)
from personavoice.modal_transport import (
    RESULT_COMPLETION_NAME,
    ResultCandidate,
    ResultFamily,
    verify_completed_directory,
    write_checkpoint_family_contract,
    write_completion_manifest,
)
from personavoice.model_assets import (
    LFM_MODEL_ASSET_SHA256,
    LFM_MODEL_REVISION,
    LFM_MODEL_WEIGHT_SHA256,
)
from personavoice.training_bundle import PLAN_PATH, build_training_bundle, canonical_plan_bytes
from personavoice.training_plan import (
    EXECUTOR_CONTRACT_FILES,
    FamilyPlan,
    FileContract,
    TrainingPlan,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _family(*, epochs: int = 3) -> FamilyPlan:
    return FamilyPlan(
        family="lfm",
        enabled=True,
        method="full",
        dataset_fingerprint="d" * 64,
        training={"epochs": epochs, "learning_rate": 0.00002},
        model_contract={
            "base_revision": LFM_MODEL_REVISION,
            "base_sha256": LFM_MODEL_WEIGHT_SHA256,
            "base_assets_sha256": LFM_MODEL_ASSET_SHA256,
        },
        implementation_contract={"workers/lfm/train.py": "a" * 64},
        checkpoint_policy={"resume_complete_only": True},
        evaluation_policy={"max_cer": 0.2},
    )


def _disabled_family(family: str, method: str) -> FamilyPlan:
    return FamilyPlan(
        family=family,
        enabled=False,
        method=method,
        dataset_fingerprint="0" * 64,
        training={},
        model_contract={},
        implementation_contract={},
        checkpoint_policy={},
        evaluation_policy={},
    )


def _plan(persona_root: Path, *, epochs: int = 3) -> TrainingPlan:
    dataset = persona_root / "dataset" / "lfm_train.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "こんにちは"}],
                "completion": [{"role": "assistant", "content": "やっほー"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainingPlan(
        persona="alice",
        files=(
            FileContract(
                path="dataset/lfm_train.jsonl",
                role="lfm-conversations",
                sha256=_sha(dataset.read_bytes()),
                size=dataset.stat().st_size,
                transfer=True,
            ),
        ),
        families=(
            _disabled_family("irodori", "full"),
            _family(epochs=epochs),
            _disabled_family("seed-vc", "finetune"),
        ),
        executor_contract={path: "e" * 64 for path in EXECUTOR_CONTRACT_FILES},
    )


def _remote_bundle(tmp_path: Path, volume: Path, plan: TrainingPlan, persona: Path):
    local = build_training_bundle(plan, persona, tmp_path / f"local-bundle-{plan.plan_id}")
    namespace = f"plans/{plan.plan_id}/bundles/{local.inventory.fingerprint}"
    destination = volume.joinpath(*namespace.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local.root, destination)
    payload = {
        "schema_version": 1,
        "plan_fingerprint": plan.fingerprint,
        "plan_path": f"/{namespace}/{PLAN_PATH}",
        "bundle_namespace": namespace,
        "bundle_fingerprint": local.inventory.fingerprint,
        "result_namespace": f"plans/{plan.plan_id}/results",
    }
    return payload


def test_remote_entrypoint_verifies_plan_invokes_shared_runner_and_finalizes(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    volume = tmp_path / "volume"
    volume.mkdir()
    plan = _plan(persona)
    payload = _remote_bundle(tmp_path, volume, plan, persona)
    calls = []
    commits = []

    def assets(received_plan, asset_root):
        assert received_plan.fingerprint == plan.fingerprint
        assert asset_root == volume / "assets"
        calls.append("assets")
        return {}

    def runner(
        plan_bytes,
        bundle_root,
        run_root,
        checkpoint_root,
        asset_root,
        *,
        status_callback,
    ):
        calls.append("runner")
        assert plan_bytes == canonical_plan_bytes(plan)
        assert checkpoint_root == volume / "checkpoints"
        resume_path = volume / "plans" / plan.plan_id / RESUME_INDEX_NAME
        assert resume_path.is_file()
        assert json.loads(resume_path.read_text(encoding="utf-8"))["families"][0][
            "latest_complete"
        ] is None
        artifact = run_root / "models" / "lfm" / "full"
        artifact.mkdir(parents=True)
        (artifact / "model.safetensors").write_bytes(b"full-model")
        checkpoint = (
            checkpoint_root
            / "lfm"
            / plan.family("lfm").fingerprint
            / "checkpoint-12"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer-state.json").write_text("{}", encoding="utf-8")
        write_checkpoint_family_contract(checkpoint, plan.family("lfm"))
        write_completion_manifest(
            checkpoint,
            kind="checkpoint",
            plan_fingerprint=plan.fingerprint,
            model="lfm",
            step=12,
            checkpoint="checkpoint-12",
            quality_gate_passed=False,
        )
        status_callback(
            "lfm",
            12.0,
            f"lfm/{plan.family('lfm').fingerprint}/checkpoint-12",
        )
        return (
            ResultFamily(
                family="lfm",
                method="full",
                family_fingerprint=plan.family("lfm").fingerprint,
                selected_artifact_path="models/lfm/full",
                candidates=(
                    ResultCandidate(
                        artifact_path="models/lfm/full",
                        validation={"passed": True, "validation_loss": 0.2, "step": 12},
                    ),
                ),
            ),
        )

    result = execute_remote_training(
        payload,
        volume_root=volume,
        runner=runner,
        asset_materializer=assets,
        volume_commit=lambda: commits.append("commit"),
    )

    assert calls == ["assets", "runner"]
    assert commits == ["commit", "commit", "commit", "commit"]
    assert result["plan_fingerprint"] == plan.fingerprint
    assert result["checkpoint"].endswith("checkpoint-12")
    marker = volume.joinpath(*result["completion_manifest_path"].lstrip("/").split("/"))
    assert marker.name == RESULT_COMPLETION_NAME
    completed = verify_completed_directory(
        marker.parent,
        expected_plan_fingerprint=plan.fingerprint,
        completion_name=RESULT_COMPLETION_NAME,
        expected_kind="result",
        expected_model="lfm",
        require_quality_gate=False,
    )
    assert completed.step == 12
    assert completed.quality_gate_passed is False
    contract = json.loads((marker.parent / "training-result.json").read_text(encoding="utf-8"))
    candidate = contract["families"][0]["candidates"][0]
    assert candidate["validation"]["validation_loss"] == 0.2
    assert candidate["files"][0]["sha256"] == _sha(b"full-model")
    status = json.loads(
        (volume / "plans" / plan.plan_id / REMOTE_STATUS_NAME).read_text(encoding="utf-8")
    )
    assert status == {
        "executor": "modal",
        "remote_state": "complete",
        "model": "lfm",
        "step": 12,
        "checkpoint": result["checkpoint"],
    }


def test_remote_entrypoint_hides_common_runner_secret_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf-runner-secret-sentinel"
    monkeypatch.setenv("HF_TOKEN", secret)
    persona = tmp_path / "persona"
    volume = tmp_path / "volume"
    volume.mkdir()
    plan = _plan(persona)
    payload = _remote_bundle(tmp_path, volume, plan, persona)

    def runner(*args, **kwargs):
        raise RuntimeError(f"third-party runner included {secret}")

    with pytest.raises(RuntimeError, match="common training runner failed") as caught:
        execute_remote_training(
            payload,
            volume_root=volume,
            runner=runner,
            asset_materializer=lambda plan, root: {},
        )

    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_remote_retry_exposes_only_latest_verified_complete_checkpoint(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    volume = tmp_path / "volume"
    volume.mkdir()
    plan = _plan(persona)
    payload = _remote_bundle(tmp_path, volume, plan, persona)
    checkpoint_root = volume / "checkpoints"
    namespace = checkpoint_root / "lfm" / plan.family("lfm").fingerprint
    complete = namespace / "checkpoint-5"
    complete.mkdir(parents=True)
    (complete / "trainer-state.json").write_text("complete", encoding="utf-8")
    write_checkpoint_family_contract(complete, plan.family("lfm"))
    write_completion_manifest(
        complete,
        kind="checkpoint",
        plan_fingerprint=plan.fingerprint,
        model="lfm",
        step=5,
        checkpoint="checkpoint-5",
        quality_gate_passed=False,
    )
    partial = namespace / "checkpoint-9"
    partial.mkdir()
    (partial / "trainer-state.json").write_text("partial", encoding="utf-8")

    def runner(plan_bytes, bundle_root, run_root, checkpoint_root, asset_root, *, status_callback):
        resume = json.loads(
            (
                volume / "plans" / plan.plan_id / RESUME_INDEX_NAME
            ).read_text(encoding="utf-8")
        )
        assert resume["families"][0]["latest_complete"]["step"] == 5
        assert "checkpoint-9" not in json.dumps(resume)
        artifact = run_root / "models" / "lfm" / "full"
        artifact.mkdir(parents=True)
        (artifact / "model.safetensors").write_bytes(b"resumed")
        return (
            ResultFamily(
                family="lfm",
                method="full",
                family_fingerprint=plan.family("lfm").fingerprint,
                selected_artifact_path="models/lfm/full",
                candidates=(
                    ResultCandidate(
                        "models/lfm/full",
                        {"passed": True, "validation_loss": 0.3, "step": 5},
                    ),
                ),
            ),
        )

    result = execute_remote_training(
        payload,
        volume_root=volume,
        runner=runner,
        asset_materializer=lambda plan, root: {},
    )

    assert result["checkpoint"].endswith("checkpoint-5")


def test_content_addressed_asset_cache_reuses_pins_across_plan_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_persona = tmp_path / "first"
    second_persona = tmp_path / "second"
    first = _plan(first_persona, epochs=3)
    second = _plan(second_persona, epochs=4)
    content = b"verified-asset"
    spec = AssetSpec(
        family="lfm",
        name="base",
        repo_id="owner/model",
        revision="1" * 40,
        required_files=("model.safetensors",),
        expected_sha256={"model.safetensors": _sha(content)},
    )
    monkeypatch.setattr(modal_app, "asset_specs_for_plan", lambda plan: (spec,))
    monkeypatch.setenv("HF_TOKEN", "modal-secret-value")

    class Downloader:
        def __init__(self):
            self.calls = 0
            self.token_seen = None

        def download(self, received, destination, *, token):
            self.calls += 1
            self.token_seen = token
            (destination / "model.safetensors").write_bytes(content)

    downloader = Downloader()
    root = tmp_path / "assets"
    first_paths = materialize_verified_assets(first, root, downloader=downloader)
    second_paths = materialize_verified_assets(second, root, downloader=downloader)

    assert downloader.calls == 1
    assert downloader.token_seen == "modal-secret-value"
    assert first_paths["lfm/base"] == second_paths["lfm/base"]
    assert (
        root
        / "plans"
        / first.plan_id
        / "lfm"
        / first.family("lfm").fingerprint
        / ASSET_INDEX_NAME
    ).is_file()
    assert (
        root
        / "plans"
        / second.plan_id
        / "lfm"
        / second.family("lfm").fingerprint
        / ASSET_INDEX_NAME
    ).is_file()
    assert "modal-secret-value" not in "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in root.rglob("*")
        if path.is_file()
    )


def test_asset_downloader_exception_never_exposes_modal_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf-downloader-secret-sentinel"
    plan = _plan(tmp_path / "persona")
    spec = AssetSpec(
        family="lfm",
        name="base",
        repo_id="owner/model",
        revision="1" * 40,
        required_files=("model.safetensors",),
        expected_sha256={"model.safetensors": "a" * 64},
    )
    monkeypatch.setattr(modal_app, "asset_specs_for_plan", lambda plan: (spec,))
    monkeypatch.setenv("HF_TOKEN", secret)

    class Downloader:
        def download(self, received, destination, *, token):
            raise RuntimeError(f"HTTP client included {token}")

    with pytest.raises(RuntimeError, match="asset download failed") as caught:
        materialize_verified_assets(plan, tmp_path / "assets", downloader=Downloader())

    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_asset_materialization_rejects_every_file_outside_the_allowlist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _plan(tmp_path / "persona")
    content = b"verified-asset"
    spec = AssetSpec(
        family="lfm",
        name="base",
        repo_id="owner/model",
        revision="1" * 40,
        required_files=("model.safetensors",),
        expected_sha256={"model.safetensors": _sha(content)},
    )
    monkeypatch.setattr(modal_app, "asset_specs_for_plan", lambda plan: (spec,))
    monkeypatch.setenv("HF_TOKEN", "modal-secret-value")

    class Downloader:
        def download(self, received, destination, *, token):
            assert token == "modal-secret-value"
            (destination / "model.safetensors").write_bytes(content)
            (destination / ".env").write_text("HF_TOKEN=leak", encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist"):
        materialize_verified_assets(plan, tmp_path / "assets", downloader=Downloader())
    assert not any(path.name == ".env" for path in (tmp_path / "assets").rglob("*"))


def test_asset_plan_pins_are_verified_before_materialization(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "persona")
    spec = asset_specs_for_plan(plan)[0]
    assert spec.revision == LFM_MODEL_REVISION
    assert spec.runtime_path == "models/lfm/base"
    assert spec.generated_files == (
        (".personavoice-revision", f"{LFM_MODEL_REVISION}\n"),
    )
    assert {
        name: spec.expected_sha256[name] for name in LFM_MODEL_ASSET_SHA256
    } == LFM_MODEL_ASSET_SHA256
    wrong = FamilyPlan(
        **{
            **plan.family("lfm").as_dict(),
            "model_contract": {
                "base_revision": "unpinned",
                "base_sha256": LFM_MODEL_WEIGHT_SHA256,
                "base_assets_sha256": LFM_MODEL_ASSET_SHA256,
            },
        }
    )
    wrong_plan = TrainingPlan(persona=plan.persona, files=plan.files, families=(wrong,))

    with pytest.raises(ValueError, match="disagrees"):
        asset_specs_for_plan(wrong_plan)


def test_training_plan_reconstruction_requires_exact_canonical_bytes(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "persona")
    restored = training_plan_from_bytes(
        canonical_plan_bytes(plan),
        expected_fingerprint=plan.fingerprint,
    )
    assert restored.as_dict() == plan.as_dict()

    pretty = json.dumps(plan.as_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="canonical"):
        training_plan_from_bytes(pretty, expected_fingerprint=plan.fingerprint)


def test_remote_claim_allows_same_call_retry_and_redirects_duplicate() -> None:
    payload = {
        "schema_version": 1,
        "plan_fingerprint": "a" * 64,
        "plan_path": "/plans/aaaaaaaaaaaaaaaaaaaaaaaa/bundles/b/contracts/training-plan.json",
        "bundle_namespace": "plans/aaaaaaaaaaaaaaaaaaaaaaaa/bundles/b",
        "bundle_fingerprint": "b" * 64,
        "result_namespace": "plans/aaaaaaaaaaaaaaaaaaaaaaaa/results",
    }

    class ClaimStore:
        def __init__(self) -> None:
            self.values = {}

        def put(self, key, value, *, skip_if_exists):
            assert skip_if_exists is True
            if key in self.values:
                return False
            self.values[key] = dict(value)
            return True

        def get(self, key):
            return self.values[key]

        def pop(self, key):
            return self.values.pop(key)

    claims = ClaimStore()
    assert (
        acquire_remote_training_claim(payload, claim_store=claims, call_id="fc-canonical")
        is None
    )
    # Modal retries/preemption retain the same FunctionCall and must be allowed
    # to resume its complete method-native checkpoint.
    assert (
        acquire_remote_training_claim(payload, claim_store=claims, call_id="fc-canonical")
        is None
    )
    assert acquire_remote_training_claim(
        payload,
        claim_store=claims,
        call_id="fc-duplicate",
    ) == {
        "schema_version": 1,
        "remote_state": "redirect",
        "plan_fingerprint": "a" * 64,
        "canonical_call_id": "fc-canonical",
    }

    claims.values["a" * 64]["bundle_fingerprint"] = "c" * 64
    with pytest.raises(RuntimeError, match="claim contract is invalid"):
        acquire_remote_training_claim(payload, claim_store=claims, call_id="fc-third")

    claims.values["a" * 64]["bundle_fingerprint"] = "b" * 64
    with pytest.raises(RuntimeError, match="ownership is invalid"):
        release_remote_training_claim(
            payload,
            claim_store=claims,
            call_id="fc-duplicate",
        )
    release_remote_training_claim(
        payload,
        claim_store=claims,
        call_id="fc-canonical",
    )
    assert claims.values == {}
    assert (
        acquire_remote_training_claim(payload, claim_store=claims, call_id="fc-recovery")
        is None
    )


def test_modal_resource_and_derived_claim_names_fit_official_sdk_limit() -> None:
    assert len(ModalAppContract().claim_dict_name) <= 64
    with pytest.raises(ValueError, match="claim Dict name"):
        ModalAppContract(app_name="a" * 64)


def test_failed_claimed_call_releases_plan_for_checkpoint_recovery() -> None:
    payload = {
        "schema_version": 1,
        "plan_fingerprint": "a" * 64,
        "plan_path": "/plans/aaaaaaaaaaaaaaaaaaaaaaaa/bundles/b/contracts/training-plan.json",
        "bundle_namespace": "plans/aaaaaaaaaaaaaaaaaaaaaaaa/bundles/b",
        "bundle_fingerprint": "b" * 64,
        "result_namespace": "plans/aaaaaaaaaaaaaaaaaaaaaaaa/results",
    }

    class ClaimStore:
        def __init__(self) -> None:
            self.values = {}

        def put(self, key, value, *, skip_if_exists):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = dict(value)
            return True

        def get(self, key):
            return self.values[key]

        def pop(self, key):
            return self.values.pop(key)

    claims = ClaimStore()

    def fail():
        raise ValueError("trainer failed")

    with pytest.raises(ValueError, match="trainer failed"):
        execute_claimed_remote_training(
            payload,
            claim_store=claims,
            call_id="fc-failed",
            executor=fail,
        )
    assert claims.values == {}
    assert execute_claimed_remote_training(
        payload,
        claim_store=claims,
        call_id="fc-recovery",
        executor=lambda: {"ok": True},
    ) == {"ok": True}
    # Successful calls remain canonical so a crash before local call-ID save
    # can reconnect through a duplicate redirect.
    assert claims.values["a" * 64]["call_id"] == "fc-recovery"


def test_terminal_recovery_independently_probes_and_releases_owned_plan_and_families() -> None:
    plan_fingerprint = "a" * 64
    bundle_fingerprint = "b" * 64
    family_fingerprint = "f" * 64
    call_id = "fc-terminal"
    payload = {
        "schema_version": 1,
        "plan_fingerprint": plan_fingerprint,
        "bundle_fingerprint": bundle_fingerprint,
        "call_id": call_id,
        "family_contracts": [
            {
                "family": "lfm",
                "fingerprint": family_fingerprint,
                "method": "full",
            }
        ],
    }

    class ClaimStore:
        def __init__(self) -> None:
            self.values = {
                plan_fingerprint: {
                    "schema_version": 1,
                    "plan_fingerprint": plan_fingerprint,
                    "bundle_fingerprint": bundle_fingerprint,
                    "call_id": call_id,
                },
                f"family:lfm:{family_fingerprint}": {
                    "schema_version": 1,
                    "family": "lfm",
                    "family_fingerprint": family_fingerprint,
                    "plan_fingerprint": plan_fingerprint,
                    "call_id": call_id,
                },
            }

        def get(self, key):
            return self.values.get(key)

        def pop(self, key):
            return self.values.pop(key)

    claims = ClaimStore()
    probes = []
    assert recover_remote_terminal_claim(
        payload,
        claim_store=claims,
        terminal_probe=lambda value: probes.append(value) or "failed",
    ) == {
        "schema_version": 1,
        "recovery_state": "released",
        "plan_fingerprint": plan_fingerprint,
        "call_id": call_id,
    }
    assert probes == [call_id]
    assert claims.values == {}
    # Recovery is idempotent if a previous invocation was interrupted after
    # completing the removals but before returning to the client.
    assert recover_remote_terminal_claim(
        payload,
        claim_store=claims,
        terminal_probe=lambda _value: "failed",
    )["recovery_state"] == "released"


def test_terminal_recovery_never_removes_running_or_superseding_owner() -> None:
    plan_fingerprint = "a" * 64
    payload = {
        "schema_version": 1,
        "plan_fingerprint": plan_fingerprint,
        "bundle_fingerprint": "b" * 64,
        "call_id": "fc-old",
        "family_contracts": [],
    }

    class ClaimStore:
        def __init__(self) -> None:
            self.values = {
                plan_fingerprint: {
                    "schema_version": 1,
                    "plan_fingerprint": plan_fingerprint,
                    "bundle_fingerprint": "b" * 64,
                    "call_id": "fc-old",
                }
            }

        def get(self, key):
            return self.values.get(key)

        def pop(self, key):
            return self.values.pop(key)

    claims = ClaimStore()
    assert recover_remote_terminal_claim(
        payload,
        claim_store=claims,
        terminal_probe=lambda _value: "running",
    )["recovery_state"] == "running"
    assert claims.values[plan_fingerprint]["call_id"] == "fc-old"

    claims.values[plan_fingerprint]["call_id"] = "fc-new"
    assert recover_remote_terminal_claim(
        payload,
        claim_store=claims,
        terminal_probe=lambda _value: "failed",
    )["recovery_state"] == "superseded"
    assert claims.values[plan_fingerprint]["call_id"] == "fc-new"


def test_family_claim_serializes_different_plans_with_reusable_checkpoint_namespace(
    tmp_path: Path,
) -> None:
    plan_a = _plan(tmp_path / "persona")
    plan_b = TrainingPlan(
        persona=plan_a.persona,
        files=plan_a.files,
        families=plan_a.families,
        executor_contract={path: "f" * 64 for path in EXECUTOR_CONTRACT_FILES},
    )
    assert plan_a.fingerprint != plan_b.fingerprint
    assert plan_a.family("lfm").fingerprint == plan_b.family("lfm").fingerprint

    class ClaimStore:
        def __init__(self) -> None:
            self.values = {}

        def put(self, key, value, *, skip_if_exists):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = dict(value)
            return True

        def get(self, key):
            return self.values[key]

        def pop(self, key):
            return self.values.pop(key)

    claims = ClaimStore()
    owned_a = acquire_remote_family_claims(
        plan_a,
        claim_store=claims,
        call_id="fc-plan-a",
        wait=lambda: pytest.fail("first owner must not wait"),
    )
    waits = []

    def finish_plan_a():
        waits.append(True)
        release_remote_family_claims(claim_store=claims, owned=owned_a)

    owned_b = acquire_remote_family_claims(
        plan_b,
        claim_store=claims,
        call_id="fc-plan-b",
        wait=finish_plan_a,
    )
    assert waits == [True]
    assert len(owned_b) == 1
    assert next(iter(claims.values.values()))["plan_fingerprint"] == plan_b.fingerprint
    release_remote_family_claims(claim_store=claims, owned=owned_b)
    assert claims.values == {}


def test_deployable_app_contract_uses_one_volume_gpu_secret_retry_and_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    class App:
        def __init__(self, name):
            captured["app_name"] = name

        def function(self, **kwargs):
            captured.setdefault("functions", []).append(kwargs)
            captured["function"] = kwargs
            return lambda function: function

    class VolumeObject:
        def commit(self):
            captured["committed"] = True

    volume = VolumeObject()

    class Volume:
        @staticmethod
        def from_name(name, *, create_if_missing):
            captured["volume"] = (name, create_if_missing)
            return volume

    class ClaimStore:
        pass

    claim_store = ClaimStore()

    class Dict:
        @staticmethod
        def from_name(name, *, create_if_missing):
            captured["claim_dict"] = (name, create_if_missing)
            return claim_store

    class Secret:
        @staticmethod
        def from_name(name, *, required_keys):
            captured["secret"] = (name, required_keys)
            return "secret-handle"

    class ImageObject:
        def apt_install(self, *packages):
            captured["apt"] = packages
            return self

        def uv_sync(self, **kwargs):
            captured["uv_sync"] = kwargs
            return self

        def add_local_dir(self, local_path, remote_path, *, copy, ignore):
            captured.setdefault("local_dirs", []).append(
                (Path(local_path), remote_path, copy, tuple(ignore))
            )
            return self

        def add_local_file(self, local_path, remote_path, *, copy):
            captured.setdefault("local_files", []).append(
                (Path(local_path), remote_path, copy)
            )
            return self

        def run_commands(self, *commands):
            captured["commands"] = commands
            return self

        def env(self, values):
            captured["image_env"] = values
            return self

        def workdir(self, path):
            captured["workdir"] = path
            return self

    class Image:
        @staticmethod
        def debian_slim(*, python_version):
            captured["python"] = python_version
            return ImageObject()

    class Retries:
        def __init__(self, **kwargs):
            captured["retries"] = kwargs

    fake_modal = SimpleNamespace(
        App=App,
        Volume=Volume,
        Dict=Dict,
        Secret=Secret,
        Image=Image,
        Retries=Retries,
        current_function_call_id=lambda: "fc-test",
    )
    contract = ModalAppContract(
        app_name="voice",
        volume_name="voice-volume",
        hf_secret_name="hf-secret",
        gpu="A100-40GB:1",
        timeout_seconds=3600,
        max_retries=3,
    )

    app = create_modal_app(modal_module=fake_modal, contract=contract)

    assert isinstance(app, App)
    assert captured["volume"] == ("voice-volume", True)
    assert captured["claim_dict"] == ("voice-claims", True)
    assert captured["secret"] == ("hf-secret", ["HF_TOKEN"])
    function = next(item for item in captured["functions"] if "gpu" in item)
    recovery_function = next(
        item for item in captured["functions"] if item.get("name") == "recover_terminal_claim"
    )
    assert function["gpu"] == "A100-40GB:1"
    assert function["volumes"] == {MODAL_VOLUME_MOUNT: volume}
    assert function["secrets"] == ["secret-handle"]
    assert function["timeout"] == 3600
    assert function["serialized"] is True
    assert recovery_function == {
        "image": function["image"],
        "timeout": 120,
        "retries": 2,
        "max_containers": 1,
        "name": "recover_terminal_claim",
        "serialized": True,
    }
    assert captured["retries"] == {
        "max_retries": 3,
        "initial_delay": 1.0,
        "backoff_coefficient": 2.0,
    }
    assert contract.as_dict()["gpu_count"] == 1
    assert contract.as_dict()["claim_dict_name"] == "voice-claims"
    assert captured["uv_sync"] == {
        "uv_project_dir": str(Path(__file__).resolve().parents[1]),
        "frozen": True,
        "extra_options": "--no-dev",
        "uv_version": "0.12.5",
    }
    assert {remote for _, remote, copy, _ in captured["local_dirs"] if copy} == {
        f"{MODAL_IMAGE_ROOT}/src",
        f"{MODAL_IMAGE_ROOT}/workers",
        f"{MODAL_IMAGE_ROOT}/locks",
        f"{MODAL_IMAGE_ROOT}/config",
    }
    assert {remote for _, remote, copy in captured["local_files"] if copy} == {
        f"{MODAL_IMAGE_ROOT}/pyproject.toml",
        f"{MODAL_IMAGE_ROOT}/uv.lock",
    }
    commands = "\n".join(captured["commands"])
    assert "8224dafb46d0aba89209a8f905f1cb7e3299d9c1" in commands
    assert f"uv sync --project {MODAL_IMAGE_ROOT} --frozen" in commands
    assert f"uv sync --project {MODAL_IMAGE_ROOT}/workers/lfm --frozen --extra cu128" in commands
    assert (
        f"uv sync --project {MODAL_IMAGE_ROOT}/vendor/Irodori-TTS --frozen --extra cu128"
        in commands
    )
    assert captured["image_env"] == {
        "PYTHONPATH": f"{MODAL_IMAGE_ROOT}/src",
        "PERSONAVOICE_IMAGE_ROOT": MODAL_IMAGE_ROOT,
    }
    assert captured["workdir"] == MODAL_IMAGE_ROOT

    deploy_root = tmp_path / "deploy-repository"
    for name in ("src", "workers", "locks", "config"):
        (deploy_root / name).mkdir(parents=True, exist_ok=True)
    (deploy_root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (deploy_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (deploy_root / ".env").write_text(
        "\n".join(
            (
                "HF_TOKEN=file-hf-secret",
                "MODAL_TOKEN_ID=file-modal-id",
                "MODAL_TOKEN_SECRET=file-modal-secret",
                "PERSONAVOICE_MODAL_APP=file-app",
                "PERSONAVOICE_MODAL_VOLUME=file-volume",
                "PERSONAVOICE_MODAL_GPU=H100",
                "PERSONAVOICE_MODAL_HF_SECRET=file-hf-handle",
                "PERSONAVOICE_MODAL_TIMEOUT_SECONDS=3600",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "PERSONAVOICE_MODAL_VOLUME",
        "PERSONAVOICE_MODAL_GPU",
        "PERSONAVOICE_MODAL_HF_SECRET",
        "PERSONAVOICE_MODAL_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN", "parent-hf-secret")
    monkeypatch.setenv("PERSONAVOICE_MODAL_APP", "process-app")
    captured.clear()

    create_modal_app(modal_module=fake_modal, repository_root=deploy_root)

    assert captured["app_name"] == "process-app"
    assert captured["volume"] == ("file-volume", True)
    assert captured["claim_dict"] == ("process-app-claims", True)
    deployed_train = next(item for item in captured["functions"] if "gpu" in item)
    assert deployed_train["gpu"] == "H100"
    assert deployed_train["timeout"] == 3600
    assert os.environ["HF_TOKEN"] == "parent-hf-secret"
    assert all(
        secret not in json.dumps(captured, default=str)
        for secret in (
            "parent-hf-secret",
            "file-hf-secret",
            "file-modal-id",
            "file-modal-secret",
        )
    )


def test_locked_modal_sdk_registers_stable_public_function_names() -> None:
    modal = pytest.importorskip("modal")
    app = create_modal_app(modal_module=modal)

    assert set(app.registered_functions) == {"train", "recover_terminal_claim"}
