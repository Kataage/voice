from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import personavoice.training_bundle as training_bundle
from personavoice.training_bundle import (
    COMPLETION_PATH,
    IRODORI_SANITIZED_MANIFEST_PATH,
    IRODORI_SOURCE_MANIFEST_PATH,
    PLAN_PATH,
    build_training_bundle,
    canonical_plan_bytes,
    verify_training_bundle,
)
from personavoice.training_plan import FamilyPlan, FileContract, TrainingPlan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(root: Path, relative: str, role: str, *, transfer: bool) -> FileContract:
    path = root.joinpath(*relative.split("/"))
    return FileContract(
        path=relative,
        role=role,
        sha256=_sha(path),
        size=path.stat().st_size,
        transfer=transfer,
    )


def _family(family: str, method: str = "full") -> FamilyPlan:
    return FamilyPlan(
        family=family,
        enabled=True,
        method=method,
        dataset_fingerprint="d" * 64,
        training={"seed": 42},
        model_contract={"revision": "pinned"},
        implementation_contract={"worker.py": "a" * 64},
        checkpoint_policy={"resume_complete_only": True},
        evaluation_policy={"max_cer": 0.2},
    )


def _prepared(tmp_path: Path) -> tuple[Path, TrainingPlan]:
    root = tmp_path / "persona"
    manifest = root / "dataset" / "irodori_manifest.jsonl"
    latent = root / "cache" / "irodori_latents" / "sample.pt"
    source = root / "dataset" / "irodori_source.jsonl"
    lfm = root / "dataset" / "lfm_train.jsonl"
    latent.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    latent.write_bytes(b"portable-dacvae-latent")
    manifest.write_text(
        json.dumps(
            {
                "text": "こんにちは",
                "caption": "明るく話す",
                "latent_path": "../cache/irodori_latents/sample.pt",
                "speaker_id": "alice",
                "num_frames": 12,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    source.write_text(
        '{"audio":"D:/private/identity.wav","text":"never upload"}\n',
        encoding="utf-8",
    )
    lfm.write_text(
        json.dumps(
            {
                "prompt": [
                    {"role": "system", "content": "persona contract"},
                    {"role": "user", "content": "元気？"},
                ],
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
            _contract(root, "dataset/irodori_source.jsonl", "irodori-source", transfer=False),
            _contract(
                root,
                "dataset/irodori_manifest.jsonl",
                "irodori-latent-manifest",
                transfer=True,
            ),
            _contract(root, "dataset/lfm_train.jsonl", "lfm-conversations", transfer=True),
        ),
        families=(_family("irodori"), _family("lfm")),
    )
    return root, plan


def test_bundle_is_minimal_deterministic_and_source_byte_identical(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    source_before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    first = build_training_bundle(plan, root, tmp_path / "bundle-one")
    second = build_training_bundle(plan, root, tmp_path / "bundle-two")

    assert first.inventory == second.inventory
    assert first.inventory.plan_fingerprint == plan.fingerprint
    assert first.inventory.file_count == 5
    inventory_paths = {item.path for item in first.inventory.files}
    assert {PLAN_PATH, IRODORI_SOURCE_MANIFEST_PATH, IRODORI_SANITIZED_MANIFEST_PATH} <= (
        inventory_paths
    )
    assert "dataset/lfm_train.jsonl" in inventory_paths
    latent_paths = {item.path for item in first.inventory.files if item.role == "irodori-latent"}
    assert len(latent_paths) == 1
    assert next(iter(latent_paths)).startswith("data/irodori/latents/")
    assert inventory_paths == {
        PLAN_PATH,
        IRODORI_SOURCE_MANIFEST_PATH,
        IRODORI_SANITIZED_MANIFEST_PATH,
        *latent_paths,
        "dataset/lfm_train.jsonl",
    }
    assert first.root.joinpath(*PLAN_PATH.split("/")).read_bytes() == canonical_plan_bytes(plan)
    source_manifest = first.root.joinpath(*IRODORI_SOURCE_MANIFEST_PATH.split("/"))
    sanitized_manifest = first.root.joinpath(*IRODORI_SANITIZED_MANIFEST_PATH.split("/"))
    assert (
        source_manifest.read_bytes() == (root / "dataset" / "irodori_manifest.jsonl").read_bytes()
    )
    sanitized = json.loads(sanitized_manifest.read_text(encoding="utf-8"))
    assert sanitized["latent_path"].startswith("latents/")
    assert ".." not in sanitized["latent_path"]
    assert not (first.root / "dataset" / "irodori_source.jsonl").exists()
    assert b"identity.wav" not in b"".join(
        path.read_bytes() for path in first.root.rglob("*") if path.is_file()
    )
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == source_before


def test_bundle_verification_rejects_tampering_and_unlisted_files(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    bundle = build_training_bundle(plan, root, tmp_path / "bundle")
    latent_relative = next(
        item.path for item in bundle.inventory.files if item.role == "irodori-latent"
    )
    latent = bundle.root.joinpath(*latent_relative.split("/"))
    latent.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum contract"):
        verify_training_bundle(bundle.root, expected_plan_fingerprint=plan.fingerprint)

    latent.write_bytes(b"portable-dacvae-latent")
    (bundle.root / ".env").write_text("MODAL_TOKEN_SECRET=leak", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        verify_training_bundle(bundle.root, expected_plan_fingerprint=plan.fingerprint)


@pytest.mark.parametrize(
    "latent_path",
    ["../../private.pt", "/private.pt", "C:/private.pt", "latents\\private.pt"],
)
def test_bundle_rejects_nonportable_manifest_latent_paths(
    tmp_path: Path,
    latent_path: str,
) -> None:
    root, plan = _prepared(tmp_path)
    manifest = root / "dataset" / "irodori_manifest.jsonl"
    manifest.write_text(
        json.dumps({"text": "test", "latent_path": latent_path}) + "\n",
        encoding="utf-8",
    )
    changed = _contract(
        root,
        "dataset/irodori_manifest.jsonl",
        "irodori-latent-manifest",
        transfer=True,
    )
    plan = TrainingPlan(
        persona=plan.persona,
        files=(plan.files[0], changed, plan.files[2]),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="portable|absolute|traversal|escapes"):
        build_training_bundle(plan, root, tmp_path / "bundle")


def test_bundle_rejects_lfm_path_or_audio_fields(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    lfm = root / "dataset" / "lfm_train.jsonl"
    lfm.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "test"}],
                "completion": [{"role": "assistant", "content": "ok"}],
                "source_path": "D:/private/raw.wav",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    changed = _contract(root, "dataset/lfm_train.jsonl", "lfm-conversations", transfer=True)
    plan = TrainingPlan(
        persona=plan.persona,
        files=(plan.files[0], plan.files[1], changed),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="path field"):
        build_training_bundle(plan, root, tmp_path / "bundle")


def test_bundle_rejects_audio_path_hidden_in_lfm_message_content(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    lfm = root / "dataset" / "lfm_train.jsonl"
    lfm.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "test"}],
                "completion": [
                    {"role": "assistant", "content": '{"audio_path":"D:/private/raw.wav"}'}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    changed = _contract(root, "dataset/lfm_train.jsonl", "lfm-conversations", transfer=True)
    changed_plan = TrainingPlan(
        persona=plan.persona,
        files=(plan.files[0], plan.files[1], changed),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="local or audio path"):
        build_training_bundle(changed_plan, root, tmp_path / "bundle")


def test_bundle_rejects_posix_path_hidden_in_lfm_message_content(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    lfm = root / "dataset" / "lfm_train.jsonl"
    lfm.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "test"}],
                "completion": [
                    {
                        "role": "assistant",
                        "content": "private source=(/home/alice/private/notes.txt)",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    changed = _contract(root, "dataset/lfm_train.jsonl", "lfm-conversations", transfer=True)
    changed_plan = TrainingPlan(
        persona=plan.persona,
        files=(plan.files[0], plan.files[1], changed),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="local or audio path"):
        build_training_bundle(changed_plan, root, tmp_path / "bundle")


def test_lfm_sensitive_path_detection_allows_https_urls() -> None:
    assert not training_bundle._contains_sensitive_lfm_string(
        {"content": "See https://example.com/docs/model-card for public documentation."}
    )


def test_bundle_rejects_non_assistant_lfm_completion(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    lfm = root / "dataset" / "lfm_train.jsonl"
    lfm.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "test"}],
                "completion": [{"role": "user", "content": "not an assistant reply"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    changed = _contract(root, "dataset/lfm_train.jsonl", "lfm-conversations", transfer=True)
    changed_plan = TrainingPlan(
        persona=plan.persona,
        files=(plan.files[0], plan.files[1], changed),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="completion messages require.*role assistant"):
        build_training_bundle(changed_plan, root, tmp_path / "bundle")


@pytest.mark.parametrize("field", ["text", "caption"])
def test_bundle_rejects_configured_secret_in_irodori_text_without_echoing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root, plan = _prepared(tmp_path)
    sentinel = "hf_secret_sentinel_that_must_never_leave_this_process"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    manifest = root / "dataset" / "irodori_manifest.jsonl"
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row[field] = f"safe prefix {sentinel} safe suffix"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    changed = _contract(
        root,
        "dataset/irodori_manifest.jsonl",
        "irodori-latent-manifest",
        transfer=True,
    )
    changed_plan = TrainingPlan(
        persona=plan.persona,
        files=(plan.files[0], changed, plan.files[2]),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="configured secret value") as exc_info:
        build_training_bundle(changed_plan, root, tmp_path / "bundle")
    assert sentinel not in str(exc_info.value)


def test_bundle_rejects_configured_secret_in_lfm_content_without_echoing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = _prepared(tmp_path)
    sentinel = "hf_secret_sentinel_that_must_never_leave_this_process"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    lfm = root / "dataset" / "lfm_train.jsonl"
    row = json.loads(lfm.read_text(encoding="utf-8"))
    row["completion"][0]["content"] = f"safe prefix {sentinel} safe suffix"
    lfm.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    changed = _contract(root, "dataset/lfm_train.jsonl", "lfm-conversations", transfer=True)
    changed_plan = TrainingPlan(
        persona=plan.persona,
        files=(plan.files[0], plan.files[1], changed),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="configured secret value") as exc_info:
        build_training_bundle(changed_plan, root, tmp_path / "bundle")
    assert sentinel not in str(exc_info.value)


def test_bundle_rejects_transfer_true_for_private_role(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    private = _contract(
        root,
        "dataset/irodori_source.jsonl",
        "irodori-source",
        transfer=True,
    )
    plan = TrainingPlan(
        persona=plan.persona,
        files=(private, plan.files[1], plan.files[2]),
        families=plan.families,
    )

    with pytest.raises(ValueError, match="not approved"):
        build_training_bundle(plan, root, tmp_path / "bundle")


def test_bundle_completion_marker_is_required(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    bundle = build_training_bundle(plan, root, tmp_path / "bundle")
    bundle.root.joinpath(*COMPLETION_PATH.split("/")).unlink()

    with pytest.raises(ValueError, match="completion manifest"):
        verify_training_bundle(bundle.root, expected_plan_fingerprint=plan.fingerprint)


def test_bundle_rejects_symlinked_latent_when_supported(tmp_path: Path) -> None:
    root, plan = _prepared(tmp_path)
    latent = root / "cache" / "irodori_latents" / "sample.pt"
    outside = tmp_path / "outside.pt"
    outside.write_bytes(latent.read_bytes())
    latent.unlink()
    try:
        latent.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available to this Windows test account")

    with pytest.raises(ValueError, match="link or junction"):
        build_training_bundle(plan, root, tmp_path / "bundle")


def test_bundle_rejects_junctioned_latent_directory(tmp_path: Path, monkeypatch) -> None:
    root, plan = _prepared(tmp_path)
    original = training_bundle._is_junction

    def junction(path: Path) -> bool:
        return path.name == "irodori_latents" or original(path)

    monkeypatch.setattr(training_bundle, "_is_junction", junction)

    with pytest.raises(ValueError, match="link or junction"):
        build_training_bundle(plan, root, tmp_path / "bundle")
