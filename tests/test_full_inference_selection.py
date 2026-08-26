from __future__ import annotations

import json
import wave
from pathlib import Path

import personavoice.inference as inference
from personavoice.artifacts import PublicationItem, publish_training_candidates
from personavoice.config import ConsentConfig, PersonaConfig
from personavoice.irodori import _write_irodori_lora_candidate_provenance
from personavoice.model_assets import LFM_MODEL_REVISION
from personavoice.project import PersonaPaths
from personavoice.training_plan import FamilyPlan, TrainingPlan


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)


def _config() -> PersonaConfig:
    return PersonaConfig(name="alice", consent=ConsentConfig(authorized=True))


def _plan(family: str, method: str) -> TrainingPlan:
    return TrainingPlan(
        persona="alice",
        files=(),
        families=(
            FamilyPlan(
                family=family,
                enabled=True,
                method=method,
                dataset_fingerprint="d" * 64,
                training={},
                model_contract={},
                implementation_contract={},
                checkpoint_policy={},
                evaluation_policy={},
            ),
        ),
    )


def _publish(
    paths: PersonaPaths,
    *,
    family: str,
    method: str,
    candidate: Path,
    destination: Path,
) -> Path:
    plan = _plan(family, method)
    family_fingerprint = plan.family(family).fingerprint
    if family == "irodori" and method == "lora":
        _write_irodori_lora_candidate_provenance(
            candidate,
            plan_fingerprint=family_fingerprint,
            best_validation_loss=0.25,
            best_step=25,
            selected_checkpoint="checkpoint_best_val_loss_0000025_0.250000",
        )
    elif family == "lfm" and method == "lora":
        (candidate / ".personavoice-training-method").write_text(
            "lora\n",
            encoding="utf-8",
        )
        (candidate / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "family": "lfm",
                    "method": "lora",
                    "training_plan_fingerprint": family_fingerprint,
                    "best_validation_loss": 0.125,
                }
            ),
            encoding="utf-8",
        )
    publish_training_candidates(
        paths.models,
        plan=plan,
        items=[
            PublicationItem(
                family=family,
                method=method,
                family_fingerprint=family_fingerprint,
                candidate=candidate,
                destination=destination,
            )
        ],
        quality={"passed": True, "checks": []},
    )
    return destination


def test_full_irodori_candidate_defaults_to_direct_no_ref_without_lora(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    paths.outputs.mkdir(parents=True)
    artifact = paths.models / ".candidates" / "irodori" / "full"
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"full")
    base = tmp_path / "base.safetensors"
    codec = tmp_path / "codec.safetensors"
    base.write_bytes(b"base")
    codec.write_bytes(b"codec")
    captured: list[str] = []

    monkeypatch.setattr(inference, "vendor_dir", lambda root: tmp_path)
    monkeypatch.setattr(inference, "base_checkpoint", lambda root: base)
    monkeypatch.setattr(inference, "codec_checkpoint", lambda root: codec)
    monkeypatch.setattr(inference, "configured_backend", lambda root: "cpu")
    monkeypatch.setattr(inference, "backend_device", lambda backend: "cpu")
    monkeypatch.setattr(inference, "local_model_env", lambda root: {})
    monkeypatch.setattr(inference, "irodori_full_artifact_complete", lambda path: True)

    def fake_run(args, **kwargs):
        del kwargs
        captured.extend(str(value) for value in args)
        output = Path(captured[captured.index("--output-wav") + 1])
        _wav(output)

    monkeypatch.setattr(inference, "run", fake_run)
    output = paths.outputs / "candidate.wav"

    inference.synthesize(
        tmp_path,
        paths,
        _config(),
        "test",
        output=output,
        candidates=1,
        irodori_artifact=artifact,
        irodori_method="full",
    )

    assert captured[captured.index("--checkpoint") + 1] == str(artifact / "model.safetensors")
    assert "--lora-adapter" not in captured
    assert "--no-ref" in captured


def test_chat_routes_published_lfm_full_without_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    full = paths.models / "lfm" / "full"
    adapter = paths.models / "lfm" / "adapter"
    full.mkdir(parents=True)
    adapter.mkdir(parents=True)
    payloads = []

    class FakeWorker:
        def call(self, repo_root, command, payload):
            del repo_root
            assert command == "infer"
            payloads.append(payload)
            return {
                "text": '{"text":"ok","voice":{"caption":"natural","emotion":"NEUTRAL","events":[]}}'
            }

    monkeypatch.setattr(
        inference,
        "_published_primary_artifact",
        lambda paths, family: ("full", full),
    )
    monkeypatch.setattr(inference, "worker", lambda repo_root, name: FakeWorker())
    audio = paths.outputs / "reply.wav"
    _wav(audio)
    monkeypatch.setattr(inference, "synthesize", lambda *args, **kwargs: [audio])

    inference.chat_turn(tmp_path, paths, _config(), "hello")

    assert payloads[0]["full_model"] == str(full)
    assert payloads[0]["adapter"] is None


def test_tampered_irodori_publication_never_falls_back_to_fixed_lora(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    paths.outputs.mkdir(parents=True)
    candidate = tmp_path / "candidate-irodori-lora"
    candidate.mkdir()
    (candidate / "adapter_config.json").write_text("{}", encoding="utf-8")
    (candidate / "adapter_model.safetensors").write_bytes(b"published-weights")
    destination = _publish(
        paths,
        family="irodori",
        method="lora",
        candidate=candidate,
        destination=paths.models / "irodori" / "lora" / "checkpoint_final",
    )
    assert inference._published_artifact(paths, "irodori", "lora") == destination
    base = tmp_path / "base.safetensors"
    codec = tmp_path / "codec.safetensors"
    base.write_bytes(b"base")
    codec.write_bytes(b"codec")
    captured: list[str] = []
    monkeypatch.setattr(inference, "vendor_dir", lambda root: tmp_path)
    monkeypatch.setattr(inference, "base_checkpoint", lambda root: base)
    monkeypatch.setattr(inference, "codec_checkpoint", lambda root: codec)
    monkeypatch.setattr(inference, "configured_backend", lambda root: "cpu")
    monkeypatch.setattr(inference, "backend_device", lambda backend: "cpu")
    monkeypatch.setattr(inference, "local_model_env", lambda root: {})

    def fake_run(args, **kwargs):
        del kwargs
        captured.extend(str(value) for value in args)
        output = Path(captured[captured.index("--output-wav") + 1])
        _wav(output)

    monkeypatch.setattr(inference, "run", fake_run)
    inference.synthesize(
        tmp_path,
        paths,
        _config(),
        "test",
        output=paths.outputs / "published.wav",
        candidates=1,
    )
    assert captured[captured.index("--lora-adapter") + 1] == str(destination)

    # The fixed v0.3 path remains structurally complete after this change.  A
    # digest-blind legacy fallback would therefore select the tampered bytes.
    (destination / "adapter_model.safetensors").write_bytes(b"tampered-weights")
    assert inference._published_artifact(paths, "irodori", "lora") is None
    captured.clear()
    inference.synthesize(
        tmp_path,
        paths,
        _config(),
        "test",
        output=paths.outputs / "tampered.wav",
        candidates=1,
    )

    assert captured[captured.index("--checkpoint") + 1] == str(base)
    assert "--lora-adapter" not in captured


def test_tampered_lfm_publication_never_falls_back_to_fixed_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    candidate = tmp_path / "candidate-lfm-lora"
    candidate.mkdir()
    (candidate / "adapter_config.json").write_text("{}", encoding="utf-8")
    (candidate / "adapter_model.safetensors").write_bytes(b"published-weights")
    (candidate / ".personavoice-base-revision").write_text(
        LFM_MODEL_REVISION + "\n",
        encoding="utf-8",
    )
    destination = _publish(
        paths,
        family="lfm",
        method="lora",
        candidate=candidate,
        destination=paths.models / "lfm" / "adapter",
    )
    assert inference._published_artifact(paths, "lfm", "lora") == destination
    payloads = []

    class FakeWorker:
        def call(self, repo_root, command, payload):
            del repo_root
            assert command == "infer"
            payloads.append(payload)
            return {
                "text": (
                    '{"text":"ok","voice":{"caption":"natural","emotion":"NEUTRAL","events":[]}}'
                )
            }

    monkeypatch.setattr(inference, "worker", lambda repo_root, name: FakeWorker())
    audio = paths.outputs / "reply.wav"
    _wav(audio)
    monkeypatch.setattr(inference, "synthesize", lambda *args, **kwargs: [audio])

    inference.chat_turn(tmp_path, paths, _config(), "hello")
    assert payloads[0]["full_model"] is None
    assert payloads[0]["adapter"] == str(destination)

    # Keep the legacy adapter shape loadable while invalidating its publication
    # digest, reproducing the interrupted/tampered fixed-path hazard.
    (destination / "adapter_model.safetensors").write_bytes(b"tampered-weights")
    assert inference._published_artifact(paths, "lfm", "lora") is None
    inference.chat_turn(tmp_path, paths, _config(), "hello")

    assert payloads[1]["full_model"] is None
    assert payloads[1]["adapter"] is None
