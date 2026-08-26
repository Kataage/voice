from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from personavoice import training
from personavoice.modal_transport import (
    CHECKPOINT_COMPLETION_NAME,
    CHECKPOINT_FAMILY_NAME,
    RESULT_COMPLETION_NAME,
    ModalFunctionTerminalError,
    ModalSDKBackend,
    ModalSettings,
    ModalTerminalCallRecoveredError,
    ModalTransport,
    RemoteSubmission,
    ResultCandidate,
    ResultFamily,
    detect_modal_auth,
    latest_verified_checkpoint,
    latest_verified_family_checkpoint,
    verify_completed_directory,
    write_checkpoint_family_contract,
    write_completion_manifest,
    write_training_result_contract,
)
from personavoice.training_bundle import build_training_bundle, canonical_plan_bytes
from personavoice.training_plan import FamilyPlan, FileContract, TrainingPlan


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _family() -> FamilyPlan:
    return FamilyPlan(
        family="lfm",
        enabled=True,
        method="full",
        dataset_fingerprint="d" * 64,
        training={"epochs": 3, "learning_rate": 0.00002},
        model_contract={"revision": "pinned", "sha256": "b" * 64},
        implementation_contract={"train.py": "a" * 64},
        checkpoint_policy={"resume_complete_only": True},
        evaluation_policy={"max_cer": 0.2},
    )


def _plan_and_bundle(tmp_path: Path):
    persona = tmp_path / "persona"
    dataset = persona / "dataset" / "lfm_train.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "元気？"}],
                "completion": [{"role": "assistant", "content": "元気だよ"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plan = TrainingPlan(
        persona="alice",
        files=(
            FileContract(
                path="dataset/lfm_train.jsonl",
                role="lfm-conversations",
                sha256=_sha_bytes(dataset.read_bytes()),
                size=dataset.stat().st_size,
                transfer=True,
            ),
        ),
        families=(_family(),),
    )
    return plan, build_training_bundle(plan, persona, tmp_path / "bundle")


class FakeBackend:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.files: dict[str, bytes] = {}
        self.spawn_payload = None
        self.poll_value = None
        self.polled_call_ids: list[str] = []
        self.stream_reads = False

    def upload_payload(self, members):
        self.events.append("payload")
        for member in members:
            assert _sha_bytes(member.local_path.read_bytes()) == member.sha256
            self.files[member.remote_path] = member.local_path.read_bytes()

    def upload_completion(self, member):
        self.events.append("completion")
        self.files[member.remote_path] = member.local_path.read_bytes()

    def spawn(self, payload):
        self.events.append("spawn")
        self.spawn_payload = dict(payload)
        return "fc-123"

    def poll(self, call_id, *, timeout):
        self.polled_call_ids.append(call_id)
        return self.poll_value

    def read_file(self, remote_path):
        value = self.files[remote_path]
        if not self.stream_reads:
            return value
        return (value[index : index + 3] for index in range(0, len(value), 3))


def _remote_result(fake: FakeBackend, submission: RemoteSubmission, plan: TrainingPlan, tmp_path: Path):
    result_dir = tmp_path / "remote-result"
    artifact = result_dir / "models" / "lfm" / "full"
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"standalone-full-weights")
    write_training_result_contract(
        result_dir,
        plan=plan,
        families=(
            ResultFamily(
                family="lfm",
                method="full",
                family_fingerprint=plan.family("lfm").fingerprint,
                selected_artifact_path="models/lfm/full",
                candidates=(
                    ResultCandidate(
                        artifact_path="models/lfm/full",
                        validation={"passed": True, "validation_loss": 0.25},
                    ),
                ),
            ),
        ),
    )
    write_completion_manifest(
        result_dir,
        kind="result",
        plan_fingerprint=plan.fingerprint,
        model="lfm",
        step=30,
        checkpoint="checkpoints/checkpoint-30",
        quality_gate_passed=False,
    )
    for path in result_dir.rglob("*"):
        if path.is_file():
            relative = path.relative_to(result_dir).as_posix()
            fake.files[f"/{submission.result_namespace}/run-1/{relative}"] = path.read_bytes()
    marker_path = f"/{submission.result_namespace}/run-1/{RESULT_COMPLETION_NAME}"
    marker_sha = _sha_bytes(fake.files[marker_path])
    fake.poll_value = {
        "plan_fingerprint": plan.fingerprint,
        "remote_state": "complete",
        "model": "lfm",
        "step": 30,
        "checkpoint": "checkpoints/checkpoint-30",
        "completion_manifest_path": marker_path,
        "completion_manifest_sha256": marker_sha,
    }


def test_auth_probe_reports_source_only_and_requires_token_pair(tmp_path: Path) -> None:
    profile = tmp_path / ".modal.toml"
    profile.write_text(
        '[profiles.personavoice]\ntoken_id = "profile-id"\ntoken_secret = "profile-secret"\n',
        encoding="utf-8",
    )

    env_status = detect_modal_auth(
        env={"MODAL_TOKEN_ID": "env-id", "MODAL_TOKEN_SECRET": "env-secret"},
        profile_path=profile,
    )
    partial = detect_modal_auth(env={"MODAL_TOKEN_ID": "only-id"}, profile_path=profile)
    profile_status = detect_modal_auth(env={}, profile_path=profile)

    assert env_status.as_dict() == {"configured": True, "source": "environment"}
    assert partial.as_dict() == {"configured": False, "source": "environment-incomplete"}
    assert profile_status.as_dict() == {"configured": True, "source": "profile"}
    serialized = json.dumps([env_status.as_dict(), partial.as_dict(), profile_status.as_dict()])
    assert "env-id" not in serialized
    assert "profile-secret" not in serialized


def test_transport_uploads_completion_last_then_spawns_secret_free_payload(tmp_path: Path) -> None:
    plan, bundle = _plan_and_bundle(tmp_path)
    fake = FakeBackend()
    transport = ModalTransport(
        ModalSettings(app_name="voice", function_name="train", volume_name="cache"),
        backend=fake,
    )
    statuses = []
    original_upload_payload = fake.upload_payload

    def require_pre_dispatch_audit(members):
        assert statuses and statuses[0]["remote_state"] == "uploading"
        original_upload_payload(members)

    fake.upload_payload = require_pre_dispatch_audit

    submission = transport.submit(
        plan=plan,
        plan_bytes=json.dumps(
            plan.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode(),
        bundle=bundle,
        status_callback=statuses.append,
    )

    assert fake.events == ["payload", "completion", "spawn"]
    assert submission.call_id == "fc-123"
    assert submission.bundle_namespace.startswith(f"plans/{plan.plan_id}/bundles/")
    assert [status["remote_state"] for status in statuses] == ["uploading", "running"]
    audit = statuses[0]["bundle_audit"]
    assert audit == submission.bundle_audit.as_dict()
    assert audit["file_count"] == bundle.inventory.file_count + 1
    assert audit["total_bytes"] == (
        bundle.inventory.total_bytes + bundle.completion_path.stat().st_size
    )
    completion_records = [
        item for item in audit["files"] if item["role"] == "bundle-completion"
    ]
    assert len(completion_records) == 1
    assert statuses[1]["submission"] == submission.resume_dict()
    assert statuses[1]["submission"]["call_id"] == "fc-123"
    payload_text = json.dumps(fake.spawn_payload, sort_keys=True)
    assert str(tmp_path) not in payload_text
    assert "token" not in payload_text.casefold()
    assert set(fake.spawn_payload) == {
        "schema_version",
        "plan_fingerprint",
        "plan_path",
        "bundle_namespace",
        "bundle_fingerprint",
        "result_namespace",
    }


def test_transport_never_propagates_backend_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "modal-transport-secret-sentinel"
    monkeypatch.setenv("MODAL_TOKEN_SECRET", secret)
    plan, bundle = _plan_and_bundle(tmp_path)
    fake = FakeBackend()

    def leaking_spawn(payload):
        raise RuntimeError(f"backend accidentally included {secret}")

    fake.spawn = leaking_spawn
    transport = ModalTransport(ModalSettings("voice", "train", "cache"), backend=fake)

    with pytest.raises(RuntimeError, match="dispatch failed") as caught:
        transport.submit(
            plan=plan,
            plan_bytes=json.dumps(
                plan.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode(),
            bundle=bundle,
        )

    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_spawned_call_id_is_offered_for_durable_save_before_submit_returns(
    tmp_path: Path,
) -> None:
    plan, bundle = _plan_and_bundle(tmp_path)
    fake = FakeBackend()
    transport = ModalTransport(ModalSettings("voice", "train", "cache"), backend=fake)
    saved = []

    def save_then_interrupt(status):
        saved.append(status)
        if status["remote_state"] == "running":
            raise RuntimeError("simulated CLI interruption after durable state save")

    with pytest.raises(RuntimeError, match="simulated CLI interruption"):
        transport.submit(
            plan=plan,
            plan_bytes=_canonical_plan(plan),
            bundle=bundle,
            status_callback=save_then_interrupt,
        )

    assert fake.events == ["payload", "completion", "spawn"]
    assert saved[-1]["submission"]["call_id"] == "fc-123"
    restored = RemoteSubmission.from_resume_dict(saved[-1]["submission"])
    assert restored.call_id == "fc-123"
    assert restored.plan_fingerprint == plan.fingerprint


def test_transport_rejects_known_secret_values_in_remote_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf-remote-result-secret-sentinel"
    monkeypatch.setenv("HF_TOKEN", secret)
    plan, bundle = _plan_and_bundle(tmp_path)
    fake = FakeBackend()
    transport = ModalTransport(ModalSettings("voice", "train", "cache"), backend=fake)
    submission = transport.submit(
        plan=plan,
        plan_bytes=json.dumps(
            plan.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode(),
        bundle=bundle,
    )
    fake.poll_value = {
        "plan_fingerprint": plan.fingerprint,
        "remote_state": "complete",
        "model": "lfm",
        "step": 1,
        "checkpoint": f"checkpoint-{secret}",
        "completion_manifest_path": f"/{submission.result_namespace}/run/result-complete.json",
        "completion_manifest_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="configured secret value") as caught:
        transport.poll(submission)

    assert secret not in str(caught.value)


def test_poll_reconnect_and_checksummed_result_download(tmp_path: Path) -> None:
    plan, bundle = _plan_and_bundle(tmp_path)
    fake = FakeBackend()
    transport = ModalTransport(ModalSettings("voice", "train", "cache"), backend=fake)
    submission = transport.submit(
        plan=plan,
        plan_bytes=json.dumps(
            plan.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode(),
        bundle=bundle,
    )
    restored = RemoteSubmission.from_resume_dict(submission.resume_dict())
    assert restored == submission
    tampered_resume = submission.resume_dict()
    tampered_resume["bundle_audit"]["total_bytes"] += 1
    with pytest.raises(ValueError, match="audit totals|fingerprint"):
        RemoteSubmission.from_resume_dict(tampered_resume)
    assert transport.poll(restored) is None
    fake.files[f"/plans/{plan.plan_id}/remote-status.json"] = json.dumps(
        {
            "executor": "modal",
            "remote_state": "running",
            "model": "lfm",
            "step": 12,
            "checkpoint": f"lfm/{plan.family('lfm').fingerprint}/checkpoint-12",
        },
        sort_keys=True,
    ).encode()
    statuses = []
    assert transport.poll(restored, status_callback=statuses.append) is None
    assert statuses[-1] == {
        "executor": "modal",
        "remote_state": "running",
        "model": "lfm",
        "step": 12,
        "checkpoint": f"lfm/{plan.family('lfm').fingerprint}/checkpoint-12",
    }

    _remote_result(fake, restored, plan, tmp_path)
    result = transport.poll(restored, status_callback=statuses.append)
    assert result is not None
    assert fake.polled_call_ids == ["fc-123", "fc-123", "fc-123"]
    assert statuses[-1] == {
        "executor": "modal",
        "remote_state": "complete",
        "model": "lfm",
        "step": 30,
        "checkpoint": "checkpoints/checkpoint-30",
    }

    fake.stream_reads = True
    downloaded = transport.download_result(
        result,
        tmp_path / "downloaded",
        expected_plan_fingerprint=plan.fingerprint,
    )
    assert downloaded.completion.quality_gate_passed is False
    assert downloaded.contract.families[0].selected_artifact_path == "models/lfm/full"
    assert (tmp_path / "downloaded" / "models" / "lfm" / "full" / "model.safetensors").read_bytes() == b"standalone-full-weights"
    cached = training._load_downloaded_result(
        tmp_path / "downloaded",
        submission=restored,
        plan=plan,
    )
    assert cached.completion == downloaded.completion
    assert cached.contract == downloaded.contract
    with pytest.raises(ValueError, match="cannot be published"):
        verify_completed_directory(
            tmp_path / "downloaded",
            expected_plan_fingerprint=plan.fingerprint,
            completion_name=RESULT_COMPLETION_NAME,
            expected_kind="result",
            expected_model="lfm",
            require_quality_gate=True,
        )


def test_result_download_rejects_marker_or_payload_checksum_tamper(tmp_path: Path) -> None:
    plan, bundle = _plan_and_bundle(tmp_path)
    fake = FakeBackend()
    transport = ModalTransport(ModalSettings("voice", "train", "cache"), backend=fake)
    submission = transport.submit(plan=plan, plan_bytes=_canonical_plan(plan), bundle=bundle)
    _remote_result(fake, submission, plan, tmp_path)
    result = transport.poll(submission)
    assert result is not None
    fake.files[result.completion_manifest_path] += b"tamper"

    with pytest.raises(ValueError, match="completion marker checksum"):
        transport.download_result(
            result,
            tmp_path / "downloaded",
            expected_plan_fingerprint=plan.fingerprint,
        )


def _canonical_plan(plan: TrainingPlan) -> bytes:
    return json.dumps(
        plan.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_latest_checkpoint_ignores_partial_and_corrupted_newer_writes(tmp_path: Path) -> None:
    plan, _ = _plan_and_bundle(tmp_path)
    root = tmp_path / "checkpoints"
    valid = root / "checkpoint-10"
    valid.mkdir(parents=True)
    (valid / "trainer-state.json").write_text("{}", encoding="utf-8")
    write_completion_manifest(
        valid,
        kind="checkpoint",
        plan_fingerprint=plan.fingerprint,
        model="lfm",
        step=10,
        checkpoint="checkpoint-10",
        quality_gate_passed=False,
    )
    partial = root / "checkpoint-20"
    partial.mkdir()
    (partial / "trainer-state.json").write_text("partial", encoding="utf-8")
    corrupt = root / "checkpoint-30"
    corrupt.mkdir()
    (corrupt / "trainer-state.json").write_text("before", encoding="utf-8")
    write_completion_manifest(
        corrupt,
        kind="checkpoint",
        plan_fingerprint=plan.fingerprint,
        model="lfm",
        step=30,
        checkpoint="checkpoint-30",
        quality_gate_passed=False,
    )
    (corrupt / "trainer-state.json").write_text("after", encoding="utf-8")

    latest = latest_verified_checkpoint(root, plan_fingerprint=plan.fingerprint, model="lfm")

    assert latest is not None
    assert latest[0] == valid
    assert latest[1].step == 10
    verify_completed_directory(
        valid,
        expected_plan_fingerprint=plan.fingerprint,
        completion_name=CHECKPOINT_COMPLETION_NAME,
        expected_kind="checkpoint",
    )


def test_family_checkpoint_survives_evaluation_only_plan_change_and_rebinds_marker(
    tmp_path: Path,
) -> None:
    first, _ = _plan_and_bundle(tmp_path)
    first_family = first.family("lfm")
    second_family = FamilyPlan(
        **{
            **first_family.as_dict(),
            "evaluation_policy": {"max_cer": 0.1, "min_similarity": 0.8},
        }
    )
    second = TrainingPlan(
        persona=first.persona,
        files=first.files,
        families=(second_family,),
    )
    assert first_family.fingerprint == second_family.fingerprint
    assert first.fingerprint != second.fingerprint

    namespace = tmp_path / "family-checkpoints" / "lfm" / first_family.fingerprint
    checkpoint = namespace / "checkpoint-40"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer-state.json").write_text("complete", encoding="utf-8")
    write_checkpoint_family_contract(checkpoint, first_family)
    original = write_completion_manifest(
        checkpoint,
        kind="checkpoint",
        plan_fingerprint=first.fingerprint,
        model="lfm",
        step=40,
        checkpoint="checkpoint-40",
        quality_gate_passed=False,
    )
    assert CHECKPOINT_FAMILY_NAME in {item.path for item in original.files}

    latest = latest_verified_family_checkpoint(
        namespace,
        plan_fingerprint=second.fingerprint,
        family=second_family,
    )

    assert latest is not None
    assert latest[0] == checkpoint
    assert latest[1].plan_fingerprint == second.fingerprint
    rebound = verify_completed_directory(
        checkpoint,
        expected_plan_fingerprint=second.fingerprint,
        completion_name=CHECKPOINT_COMPLETION_NAME,
        expected_kind="checkpoint",
        expected_model="lfm",
    )
    assert rebound.step == 40


def test_family_checkpoint_read_only_probe_does_not_rebind_shared_marker(
    tmp_path: Path,
) -> None:
    plan, _ = _plan_and_bundle(tmp_path)
    family = plan.family("lfm")
    namespace = tmp_path / "checkpoints" / family.family / family.fingerprint
    checkpoint = namespace / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "native.bin").write_bytes(b"native")
    write_checkpoint_family_contract(checkpoint, family)
    original = write_completion_manifest(
        checkpoint,
        kind="checkpoint",
        plan_fingerprint=plan.fingerprint,
        model=family.family,
        step=7,
        checkpoint="checkpoint-7",
        quality_gate_passed=False,
        completion_name=CHECKPOINT_COMPLETION_NAME,
    )
    marker = checkpoint / CHECKPOINT_COMPLETION_NAME
    original_bytes = marker.read_bytes()

    latest = latest_verified_family_checkpoint(
        namespace,
        plan_fingerprint="f" * 64,
        family=family,
        rebind_plan_marker=False,
    )

    assert latest is not None
    assert latest[1] == original
    assert marker.read_bytes() == original_bytes


def test_training_result_rejects_failed_selected_candidate(tmp_path: Path) -> None:
    plan, _ = _plan_and_bundle(tmp_path)
    result_dir = tmp_path / "result"
    artifact = result_dir / "models" / "lfm" / "full"
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="did not pass"):
        write_training_result_contract(
            result_dir,
            plan=plan,
            families=(
                ResultFamily(
                    family="lfm",
                    method="full",
                    family_fingerprint=plan.family("lfm").fingerprint,
                    selected_artifact_path="models/lfm/full",
                    candidates=(
                        ResultCandidate(
                            artifact_path="models/lfm/full",
                            validation={"passed": False, "validation_loss": 9.0},
                        ),
                    ),
                ),
            ),
        )


def test_speaker_inversion_requires_finite_validation_loss(tmp_path: Path) -> None:
    def plan_for(method: str) -> TrainingPlan:
        family = FamilyPlan(
            family="irodori",
            enabled=True,
            method=method,
            dataset_fingerprint="d" * 64,
            training={"max_steps": 10},
            model_contract={"revision": "pinned"},
            implementation_contract={"train.py": "a" * 64},
            checkpoint_policy={"resume_complete_only": True},
            evaluation_policy={},
        )
        return TrainingPlan(persona="alice", files=(), families=(family,))

    result_dir = tmp_path / "speaker-result"
    artifact = result_dir / "models" / "irodori" / "speaker-inversion"
    artifact.mkdir(parents=True)
    (artifact / "speaker.safetensors").write_bytes(b"embedding")
    speaker_plan = plan_for("speaker-inversion")
    family = ResultFamily(
        family="irodori",
        method="speaker-inversion",
        family_fingerprint=speaker_plan.family("irodori").fingerprint,
        selected_artifact_path="models/irodori/speaker-inversion",
        candidates=(
            ResultCandidate(
                "models/irodori/speaker-inversion",
                {"passed": True, "validation_loss": None},
            ),
        ),
    )

    with pytest.raises(ValueError, match="finite validation loss"):
        write_training_result_contract(result_dir, plan=speaker_plan, families=(family,))


def test_optional_sdk_adapter_uses_official_volume_and_call_apis(tmp_path: Path) -> None:
    batches: list[list[tuple[str, str]]] = []

    class Batch:
        def __enter__(self):
            batches.append([])
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local, remote):
            batches[-1].append((local, remote))

    class VolumeObject:
        def batch_upload(self, *, force):
            assert force is True
            return Batch()

        def read_file(self, path):
            return iter((b"a", b"b"))

    volume = VolumeObject()

    class Volume:
        @staticmethod
        def from_name(name, *, create_if_missing, environment_name):
            assert (name, create_if_missing, environment_name) == ("cache", True, "prod")
            return volume

    class SpawnedFunction:
        def spawn(self, payload):
            assert payload == {"plan_fingerprint": "f" * 64}
            return SimpleNamespace(object_id="fc-official")

    class Function:
        @staticmethod
        def from_name(app, name, *, environment_name):
            assert (app, name, environment_name) == ("voice", "train", "prod")
            return SpawnedFunction()

    class Call:
        def get(self, *, timeout):
            assert timeout == 0
            return {"ok": True}

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            assert call_id == "fc-official"
            return Call()

    modal = SimpleNamespace(Volume=Volume, Function=Function, FunctionCall=FunctionCall)
    backend = ModalSDKBackend(
        ModalSettings("voice", "train", "cache", environment_name="prod"),
        modal_module=modal,
    )
    payload = tmp_path / "payload"
    payload.write_bytes(b"payload")
    member = SimpleNamespace(
        local_path=payload,
        remote_path="/plans/a/payload",
        sha256=_sha_bytes(b"payload"),
        size=7,
    )

    backend.upload_payload([member])
    backend.upload_completion(member)

    assert len(batches) == 2
    assert backend.spawn({"plan_fingerprint": "f" * 64}) == "fc-official"
    assert backend.poll("fc-official", timeout=0) == {"ok": True}
    assert b"".join(backend.read_file("/plans/a/payload")) == b"ab"


def test_official_sdk_poll_distinguishes_running_timeout_from_terminal_failure() -> None:
    class SDKError(Exception):
        pass

    class SDKTimeout(SDKError):
        pass

    class FunctionTimeout(SDKTimeout):
        pass

    class OutputExpired(SDKTimeout):
        pass

    class RemoteError(SDKError):
        pass

    class ExecutionError(SDKError):
        pass

    class InternalFailure(SDKError):
        pass

    exceptions = SimpleNamespace(
        Error=SDKError,
        TimeoutError=SDKTimeout,
        FunctionTimeoutError=FunctionTimeout,
        OutputExpiredError=OutputExpired,
        RemoteError=RemoteError,
        ExecutionError=ExecutionError,
        InternalFailure=InternalFailure,
    )
    raised = [SDKTimeout()]

    class Call:
        def get(self, *, timeout):
            assert timeout == 0
            raise raised[0]

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            assert call_id == "fc-official"
            return Call()

    backend = ModalSDKBackend(
        ModalSettings("voice", "train", "cache"),
        modal_module=SimpleNamespace(FunctionCall=FunctionCall, exception=exceptions),
    )
    # Modal's SDK timeout is not Python's built-in TimeoutError. It means the
    # durable call is still running and must retain all claims.
    assert backend.poll("fc-official", timeout=0) is None

    raised[0] = FunctionTimeout()
    with pytest.raises(ModalFunctionTerminalError) as terminal:
        backend.poll("fc-official", timeout=0)
    assert terminal.value.call_id == "fc-official"

    # Modal emits InternalFailure only after a durable output reports that all
    # platform retries were exhausted. It is terminal, unlike an SDK/service
    # exception raised before any output could be observed.
    raised[0] = InternalFailure()
    with pytest.raises(ModalFunctionTerminalError):
        backend.poll("fc-official", timeout=0)

    raised[0] = SDKError("transport/auth/service uncertainty")
    with pytest.raises(SDKError):
        backend.poll("fc-official", timeout=0)


def test_transport_recovers_only_terminal_call_then_requires_explicit_rerun(tmp_path: Path) -> None:
    plan, bundle = _plan_and_bundle(tmp_path)

    class TerminalBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.recovered = []

        def poll(self, call_id, *, timeout):
            raise ModalFunctionTerminalError("fc-canonical")

        def recover_terminal_call(self, submission, *, call_id):
            self.recovered.append((submission.plan_fingerprint, call_id))
            return "released"

    backend = TerminalBackend()
    transport = ModalTransport(ModalSettings("voice", "train", "cache"), backend=backend)
    submission = transport.submit(plan=plan, plan_bytes=canonical_plan_bytes(plan), bundle=bundle)

    with pytest.raises(ModalTerminalCallRecoveredError) as recovered:
        transport.poll(submission)
    assert recovered.value.call_id == "fc-canonical"
    assert backend.recovered == [(plan.fingerprint, "fc-canonical")]


def test_official_backend_uses_fixed_serialized_recovery_function_contract(tmp_path: Path) -> None:
    plan, bundle = _plan_and_bundle(tmp_path)
    submission = ModalTransport(
        ModalSettings("voice", "train", "cache"),
        backend=FakeBackend(),
    ).submit(plan=plan, plan_bytes=canonical_plan_bytes(plan), bundle=bundle)
    captured = {}

    class RecoveryFunction:
        def remote(self, payload):
            captured.update(payload)
            return {
                "schema_version": 1,
                "recovery_state": "released",
                "plan_fingerprint": plan.fingerprint,
                "call_id": "fc-canonical",
            }

    class Function:
        @staticmethod
        def from_name(app, name, **kwargs):
            assert (app, name, kwargs) == ("voice", "recover_terminal_claim", {})
            return RecoveryFunction()

    backend = ModalSDKBackend(
        ModalSettings("voice", "train", "cache"),
        modal_module=SimpleNamespace(Function=Function),
    )
    assert (
        backend.recover_terminal_call(submission, call_id="fc-canonical") == "released"
    )
    assert captured == {
        "schema_version": 1,
        "plan_fingerprint": plan.fingerprint,
        "bundle_fingerprint": submission.bundle_audit.bundle_fingerprint,
        "call_id": "fc-canonical",
        "family_contracts": [
            {
                "family": "lfm",
                "fingerprint": plan.family("lfm").fingerprint,
                "method": "full",
            }
        ],
    }


def test_official_backend_follows_atomic_duplicate_dispatch_redirect() -> None:
    calls = []
    results = {
        "fc-duplicate": {
            "schema_version": 1,
            "remote_state": "redirect",
            "plan_fingerprint": "f" * 64,
            "canonical_call_id": "fc-canonical",
        },
        "fc-canonical": {"remote_state": "complete", "sentinel": True},
    }

    class Call:
        def __init__(self, call_id):
            self.call_id = call_id

        def get(self, *, timeout):
            assert timeout == 0
            calls.append(self.call_id)
            return results[self.call_id]

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            return Call(call_id)

    backend = ModalSDKBackend(
        ModalSettings("voice", "train", "cache"),
        modal_module=SimpleNamespace(FunctionCall=FunctionCall),
    )

    assert backend.poll("fc-duplicate", timeout=0) == results["fc-canonical"]
    assert calls == ["fc-duplicate", "fc-canonical"]

    results["fc-canonical"] = {
        "schema_version": 1,
        "remote_state": "redirect",
        "plan_fingerprint": "f" * 64,
        "canonical_call_id": "fc-duplicate",
    }
    with pytest.raises(RuntimeError, match="redirect cycle"):
        backend.poll("fc-duplicate", timeout=0)
