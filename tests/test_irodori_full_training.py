from __future__ import annotations

import json
import pickle
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest
import yaml

from personavoice import irodori
from personavoice.model_assets import IRODORI_TEXT_ENCODER_ID, IRODORI_TEXT_ENCODER_REVISION


def _standard_optimizer_state(step: int) -> dict:
    return {
        "state": {0: {"step": step}},
        "param_groups": [{"params": [0], "lr": 0.001}],
    }


def _write_torch_step(path: Path, step: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trainer/data.pkl", pickle.dumps({"step": step}, protocol=4))
        archive.writestr("trainer/version", "3\n")


def _native_trainer_payload(step: int, *, optimizer: str) -> dict:
    standard = _standard_optimizer_state(step)
    optimizer_state = {"muon": standard, "aux": None} if optimizer == "muon" else standard
    return {
        "step": step,
        "model": {"weight": [1]},
        "optimizer": optimizer_state,
        # Pinned upstream ScalarLRScheduler starts at -1 and is advanced
        # before the public training step is incremented.
        "scheduler": {"base_lrs": [0.001], "last_step": step - 1},
        "model_config": {"model_dim": 8},
        "train_config": {
            "optimizer": optimizer,
            "lr_scheduler": "wsd",
            "max_steps": step + 100,
        },
        "dataloader_state": {
            "version": 1,
            "world_size": 1,
            "rank_states": [{"_snapshot": {"_num_yielded": step}}],
        },
        "runtime_state": {"epoch": 2, "sampler_epoch": 1, "epoch_step": 3},
    }


def _execute_checkpoint_validation_script(
    monkeypatch,
    script: str,
    checkpoint: Path,
    payload: dict,
) -> None:
    torch = types.ModuleType("torch")

    def load(path, map_location=None, weights_only=None, mmap=None):
        assert map_location == "cpu" and weights_only is True
        del mmap
        if Path(path).name == "adapter_model.bin":
            return {"adapter.weight": [1]}
        return payload

    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    original_argv = sys.argv
    try:
        sys.argv = ["checkpoint-validator", str(checkpoint)]
        exec(compile(script, "<checkpoint-validator>", "exec"), {})
    finally:
        sys.argv = original_argv


def _execute_speaker_validation_script(
    monkeypatch,
    checkpoint: Path,
    *,
    metadata=None,
    keys=("speaker_embedding",),
    shape=(2, 8),
    dtype="F32",
    open_error: Exception | None = None,
) -> None:
    class Slice:
        def get_shape(self):
            return list(shape)

        def get_dtype(self):
            return dtype

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self):
            return metadata

        def keys(self):
            return list(keys)

        def get_slice(self, key):
            assert key == "speaker_embedding"
            return Slice()

    safetensors = types.ModuleType("safetensors")

    def safe_open(path, framework=None, device=None):
        assert Path(path) == checkpoint
        assert framework == "pt" and device == "cpu"
        if open_error is not None:
            raise open_error
        return Handle()

    safetensors.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    original_argv = sys.argv
    try:
        sys.argv = ["speaker-validator", str(checkpoint)]
        exec(compile(irodori._SPEAKER_EMBEDDING_VALIDATION_SCRIPT, "<speaker-validator>", "exec"), {})
    finally:
        sys.argv = original_argv


def _write_full_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "text_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                    "text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
                    "caption_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                },
                "train": {
                    "batch_size": 8,
                    "gradient_accumulation_steps": 1,
                    "num_workers": 4,
                    "max_steps": 4000,
                    "valid_ratio": 0.0005,
                    "valid_every": 1000,
                    "save_every": 1000,
                    "checkpoint_best_n": 5,
                    "precision": "bf16",
                    "allow_tf32": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_speaker_inversion_native_validation_requires_exact_upstream_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint_0000020.speaker.safetensors"
    checkpoint.write_bytes(b"native")
    _execute_speaker_validation_script(monkeypatch, checkpoint)

    with pytest.raises(ValueError, match="unexpected metadata"):
        _execute_speaker_validation_script(
            monkeypatch,
            checkpoint,
            metadata={"step": "20"},
        )
    with pytest.raises(ValueError, match="exactly speaker_embedding"):
        _execute_speaker_validation_script(
            monkeypatch,
            checkpoint,
            keys=("speaker_embedding", "unexpected"),
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        _execute_speaker_validation_script(monkeypatch, checkpoint, shape=(0, 8))
    with pytest.raises(ValueError, match="float32"):
        _execute_speaker_validation_script(monkeypatch, checkpoint, dtype="F16")
    with pytest.raises(ValueError, match="truncated"):
        _execute_speaker_validation_script(
            monkeypatch,
            checkpoint,
            open_error=ValueError("truncated safetensors"),
        )


def test_speaker_inversion_latest_checkpoint_falls_back_without_mutating_files(
    tmp_path: Path,
) -> None:
    older = tmp_path / "checkpoint_0000010.speaker.safetensors"
    newer = tmp_path / "checkpoint_0000020.speaker.safetensors"
    older.write_bytes(b"complete")
    newer.write_bytes(b"truncated")

    selected = irodori._latest_verified_speaker_embedding(
        tmp_path,
        verify=lambda path: path == older,
    )

    assert selected == older
    assert older.read_bytes() == b"complete"
    assert newer.read_bytes() == b"truncated"
    with pytest.raises(RuntimeError, match="none is a complete"):
        irodori._latest_verified_speaker_embedding(
            tmp_path,
            verify=lambda _path: False,
        )
    assert newer.read_bytes() == b"truncated"


def test_speaker_inversion_verifier_uses_pinned_isolated_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vendor = tmp_path / "vendor"
    checkpoint = tmp_path / "checkpoint_0000020.speaker.safetensors"
    checkpoint.write_bytes(b"native")
    calls: list[tuple[list[object], dict[str, object]]] = []

    def fake_run(args, **kwargs):
        calls.append((list(args), dict(kwargs)))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(irodori, "run", fake_run)
    assert irodori._verify_speaker_embedding_checkpoint(
        vendor,
        checkpoint,
        env={"HF_HUB_OFFLINE": "1"},
    )
    command, options = calls[0]
    assert command[:5] == ["uv", "run", "--project", vendor, "--no-sync"]
    assert command[-2] == irodori._SPEAKER_EMBEDDING_VALIDATION_SCRIPT
    assert command[-1] == checkpoint
    assert options == {
        "cwd": vendor,
        "env": {"HF_HUB_OFFLINE": "1"},
        "capture": True,
        "check": False,
    }

    checkpoint.write_bytes(b"")
    assert not irodori._verify_speaker_embedding_checkpoint(vendor, checkpoint, env={})
    assert len(calls) == 1


def test_irodori_full_native_validation_binds_all_resume_state_to_filename_step(
    tmp_path: Path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint_0000020.pt"
    checkpoint.write_bytes(b"native")
    payload = _native_trainer_payload(20, optimizer="muon")
    _execute_checkpoint_validation_script(
        monkeypatch,
        irodori._FULL_CHECKPOINT_VALIDATION_SCRIPT,
        checkpoint,
        payload,
    )

    payload["step"] = 19
    with pytest.raises(ValueError, match="filename step"):
        _execute_checkpoint_validation_script(
            monkeypatch,
            irodori._FULL_CHECKPOINT_VALIDATION_SCRIPT,
            checkpoint,
            payload,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("optimizer", "optimizer"),
        ("scheduler", "scheduler"),
        ("dataloader", "dataloader"),
        ("runtime", "runtime"),
    ],
)
def test_irodori_lora_native_validation_rejects_inexact_resume_state(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    message: str,
):
    checkpoint = tmp_path / "checkpoint_best_val_loss_0000020_0.300000"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}\n', encoding="utf-8"
    )
    (checkpoint / "adapter_model.bin").write_bytes(b"adapter")
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    payload = _native_trainer_payload(20, optimizer="adamw")
    payload.pop("model")
    _execute_checkpoint_validation_script(
        monkeypatch,
        irodori._LORA_CHECKPOINT_VALIDATION_SCRIPT,
        checkpoint,
        payload,
    )

    if mutation == "optimizer":
        payload["optimizer"]["state"] = {}
    elif mutation == "scheduler":
        payload["scheduler"]["last_step"] = 20
    elif mutation == "dataloader":
        payload["dataloader_state"]["rank_states"] = []
    else:
        payload["runtime_state"]["sampler_epoch"] = 0
    with pytest.raises(ValueError, match=message):
        _execute_checkpoint_validation_script(
            monkeypatch,
            irodori._LORA_CHECKPOINT_VALIDATION_SCRIPT,
            checkpoint,
            payload,
        )


def test_irodori_lora_latest_resume_skips_newer_native_step_mismatch(tmp_path: Path):
    older = tmp_path / "checkpoint_0000010"
    newer = tmp_path / "checkpoint_0000020"
    for checkpoint in (older, newer):
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        _write_torch_step(
            checkpoint / "trainer_state.pt",
            10 if checkpoint == older else 19,
        )

    assert not irodori.lora_resume_checkpoint_complete(newer)
    assert irodori._latest_resume(
        tmp_path,
        verify=lambda _checkpoint: True,
    ) == older
    with pytest.raises(RuntimeError, match="step-bound native trainer state"):
        irodori._latest_resume(tmp_path, verify=lambda _checkpoint: False)


def test_full_training_resumes_verified_checkpoint_and_exports_best_portably(
    tmp_path: Path,
    monkeypatch,
):
    vendor = tmp_path / "vendor" / "Irodori-TTS"
    _write_full_config(vendor / "configs" / "train_v4_small.yaml")
    (vendor / "train.py").write_text("", encoding="utf-8")
    (vendor / "convert_checkpoint_to_safetensors.py").write_text("", encoding="utf-8")
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    manifest = tmp_path / "dataset" / "irodori_manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(
        '{"text":"a","latent_path":"cache/a.pt"}\n'
        '{"text":"b","latent_path":"cache/b.pt"}\n',
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "irodori"
    run_dir.mkdir(parents=True)
    resume_10 = run_dir / "checkpoint_0000010.pt"
    resume_20_partial = run_dir / "checkpoint_0000020.pt"
    best_worse = run_dir / "checkpoint_best_val_loss_0000010_0.500000.pt"
    best = run_dir / "checkpoint_best_val_loss_0000020_0.300000.pt"
    for checkpoint in (resume_10, resume_20_partial, best_worse, best):
        checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(irodori, "vendor_dir", lambda _root: vendor)
    monkeypatch.setattr(irodori, "base_checkpoint", lambda _root: base)
    monkeypatch.setattr(irodori, "configured_backend", lambda _root: "cpu")
    monkeypatch.setattr(irodori, "local_model_env", lambda _root: {})
    monkeypatch.setattr(
        irodori,
        "_verify_full_training_checkpoint",
        lambda _vendor, path, env: path != resume_20_partial,
    )
    commands: list[list[str]] = []

    def fake_run(args, **_kwargs):
        command = [str(value) for value in args]
        commands.append(command)
        if "convert_checkpoint_to_safetensors.py" in " ".join(command):
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"standalone-model")
            tokenizer = output.parent / "tokenizer"
            tokenizer.mkdir()
            (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(irodori, "run", fake_run)
    artifact = tmp_path / "personas" / "alice" / "models" / "irodori" / "full"
    plan_fingerprint = "a" * 64
    result = irodori.train_irodori_full(
        tmp_path,
        manifest,
        artifact,
        run_dir,
        max_steps=4000,
        plan_fingerprint=plan_fingerprint,
        validation_ratio=0.1,
        validation_every=250,
        checkpoint_best_n=3,
    )

    train_command = next(
        command
        for command in commands
        if str(vendor / "train.py") in command
    )
    assert train_command[train_command.index("--resume") + 1] == str(resume_10)
    assert "--init-checkpoint" not in train_command
    assert train_command[train_command.index("--config") + 1] == str(
        run_dir / "personavoice_train_v4_small.yaml"
    )
    patched = yaml.safe_load((run_dir / "personavoice_train_v4_small.yaml").read_text(encoding="utf-8"))
    assert patched["train"]["valid_ratio"] == 0.1
    assert patched["train"]["valid_every"] == 250
    assert patched["train"]["save_every"] == 250
    assert patched["train"]["checkpoint_best_n"] == 3
    convert_command = next(
        command for command in commands if "convert_checkpoint_to_safetensors.py" in " ".join(command)
    )
    converter_index = convert_command.index(str(vendor / "convert_checkpoint_to_safetensors.py"))
    assert convert_command[converter_index + 1] == str(best)
    assert result["best_validation_loss"] == 0.3
    assert result["best_checkpoint"] == str(best)
    assert result["checkpoint_step"] == 20
    assert result["resumed_from"] == str(resume_10)
    assert artifact.joinpath("model.safetensors").read_bytes() == b"standalone-model"
    assert irodori.irodori_full_artifact_complete(
        artifact,
        plan_fingerprint=plan_fingerprint,
    )
    provenance_text = (artifact / "provenance.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in provenance_text
    assert json.loads(provenance_text)["selected_checkpoint"] == best.name

    command_count = len(commands)
    reused = irodori.train_irodori_full(
        tmp_path,
        manifest,
        artifact,
        run_dir,
        max_steps=4000,
        plan_fingerprint=plan_fingerprint,
        validation_ratio=0.1,
        validation_every=250,
        checkpoint_best_n=3,
    )
    assert reused["reused"] is True
    assert len(commands) == command_count


def test_full_resume_rejects_a_run_with_only_partial_numeric_checkpoints(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoint_0000100.pt").write_bytes(b"partial")
    (output / "checkpoint_0000200.pt").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="none have complete resumable state"):
        irodori._latest_verified_full_resume(output, verify=lambda _path: False)


def test_full_config_requires_the_pinned_full_yaml_and_validation(tmp_path: Path):
    source = tmp_path / "train_v4_small_lora.yaml"
    _write_full_config(source)
    with pytest.raises(ValueError, match="train_v4_small.yaml"):
        irodori._patched_full_config(
            source,
            tmp_path / "patched.yaml",
            max_steps=10,
            backend="cpu",
            validation_ratio=0.1,
            validation_every=5,
            checkpoint_best_n=1,
        )

    source = tmp_path / "train_v4_small.yaml"
    _write_full_config(source)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["train"]["valid_ratio"] = 0.0
    source.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_ratio"):
        irodori._patched_full_config(
            source,
            tmp_path / "patched.yaml",
            max_steps=10,
            backend="cpu",
            validation_ratio=0.0,
            validation_every=5,
            checkpoint_best_n=1,
        )


def test_method_dispatch_keeps_legacy_lora_and_speaker_apis(tmp_path: Path, monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_legacy(*_args, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(irodori, "train_irodori", fake_legacy)
    common = {
        "repo_root": tmp_path,
        "manifest": tmp_path / "manifest.jsonl",
        "models_dir": tmp_path / "models",
        "cache_dir": tmp_path / "cache",
        "max_steps": 25,
        "plan_fingerprint": "b" * 64,
        "validation_ratio": 0.1,
        "validation_every": 5,
        "checkpoint_best_n": 2,
    }
    irodori.train_irodori_method(**common, method="lora")
    irodori.train_irodori_method(**common, method="speaker-inversion")

    assert calls[0]["do_lora"] is True and calls[0]["do_speaker"] is False
    assert calls[0]["plan_fingerprint"] == "b" * 64
    assert calls[1]["do_speaker"] is True and calls[1]["do_lora"] is False
    assert calls[1]["plan_fingerprint"] == "b" * 64


def test_lora_candidate_provenance_is_new_path_only_and_fail_closed(
    tmp_path: Path,
    monkeypatch,
):
    vendor = tmp_path / "vendor"
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"text":"a","latent_path":"a.pt"}\n', encoding="utf-8")
    models = tmp_path / "models"
    cache = tmp_path / "cache"
    output = models / "irodori" / "lora"
    final = output / "checkpoint_final"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (final / "adapter_model.safetensors").write_bytes(b"adapter")
    (final / "trainer_state.pt").write_bytes(b"trainer")
    best = output / "checkpoint_best_val_loss_0000025_0.125000"
    best.mkdir()
    (best / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (best / "adapter_model.safetensors").write_bytes(b"best-adapter")
    (best / "trainer_state.pt").write_bytes(b"best-trainer")

    monkeypatch.setattr(irodori, "vendor_dir", lambda _root: vendor)
    monkeypatch.setattr(irodori, "base_checkpoint", lambda _root: base)
    monkeypatch.setattr(irodori, "configured_backend", lambda _root: "cpu")
    monkeypatch.setattr(irodori, "local_model_env", lambda _root: {})
    monkeypatch.setattr(
        irodori,
        "_verify_lora_training_checkpoint",
        lambda _vendor, _checkpoint, env: True,
    )

    # The direct v0.3 API has no family fingerprint and must not mutate an
    # already-complete legacy adapter.
    legacy = irodori.train_irodori(
        tmp_path,
        manifest,
        models,
        cache,
        speaker_steps=10,
        lora_steps=10,
        do_speaker=False,
        do_lora=True,
    )
    selected = output / "selected"
    sidecar = selected / irodori._LORA_PROVENANCE_NAME
    assert legacy["best_validation_loss"] == 0.125
    assert not sidecar.exists()

    fingerprint = "e" * 64
    candidate = irodori.train_irodori(
        tmp_path,
        manifest,
        models,
        cache,
        speaker_steps=10,
        lora_steps=10,
        do_speaker=False,
        do_lora=True,
        validation_ratio=0.1,
        validation_every=5,
        checkpoint_best_n=2,
        plan_fingerprint=fingerprint,
    )
    assert candidate["best_validation_loss"] == 0.125
    assert candidate["best_checkpoint"] == str(best)
    assert candidate["checkpoint_step"] == 25
    assert candidate["artifact"] == str(selected)
    assert candidate["lora_adapter"] == str(selected)
    assert candidate["provenance"] == str(sidecar)
    assert irodori.irodori_lora_candidate_complete(
        selected,
        plan_fingerprint=fingerprint,
    )
    assert (selected / "adapter_model.safetensors").read_bytes() == b"best-adapter"
    assert not (selected / "trainer_state.pt").exists()
    assert not (final / irodori._LORA_PROVENANCE_NAME).exists()
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["best_validation_loss"] == 0.125
    assert provenance["best_step"] == 25
    assert provenance["selected_checkpoint"] == best.name
    assert str(tmp_path) not in json.dumps(provenance)

    selected_weight = selected / "adapter_model.safetensors"
    original_weight = selected_weight.read_bytes()
    selected_weight.write_bytes(original_weight + b"tampered")
    assert not irodori.irodori_lora_candidate_complete(
        selected,
        plan_fingerprint=fingerprint,
    )
    selected_weight.write_bytes(original_weight)
    selected_config = selected / "adapter_config.json"
    original_config = selected_config.read_bytes()
    selected_config.write_text(
        json.dumps({"base_model_name_or_path": r"C:\private\base"}),
        encoding="utf-8",
    )
    assert not irodori.irodori_lora_candidate_complete(
        selected,
        plan_fingerprint=fingerprint,
    )
    selected_config.write_bytes(original_config)
    assert irodori.irodori_lora_candidate_complete(
        selected,
        plan_fingerprint=fingerprint,
    )

    # Once the sidecar has captured the finite family metric, the candidate is
    # reusable even if the retained training checkpoints are later archived.
    for path in best.iterdir():
        path.unlink()
    best.rmdir()
    reused = irodori.train_irodori(
        tmp_path,
        manifest,
        models,
        cache,
        speaker_steps=10,
        lora_steps=10,
        do_speaker=False,
        do_lora=True,
        plan_fingerprint=fingerprint,
    )
    assert reused["reused"] is True
    assert reused["best_validation_loss"] == 0.125
    assert reused["artifact"] == str(selected)

    provenance["best_validation_loss"] = float("nan")
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    assert not irodori.irodori_lora_candidate_complete(
        selected,
        plan_fingerprint=fingerprint,
    )


def test_speaker_inversion_selects_upstream_best_validation_embedding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vendor = tmp_path / "vendor"
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"text":"a","latent_path":"a.pt"}\n', encoding="utf-8")
    models = tmp_path / "models"
    output = models / "irodori" / "speaker"
    output.mkdir(parents=True)
    final = output / "checkpoint_final.speaker.safetensors"
    final.write_bytes(b"final-speaker")
    worse = output / "checkpoint_best_val_loss_0000010_0.500000.speaker.safetensors"
    best = output / "checkpoint_best_val_loss_0000020_0.300000.speaker.safetensors"
    incomplete = output / "checkpoint_best_val_loss_0000030_0.100000.speaker.safetensors"
    worse.write_bytes(b"worse-speaker")
    best.write_bytes(b"best-speaker")
    incomplete.touch()

    monkeypatch.setattr(irodori, "vendor_dir", lambda _root: vendor)
    monkeypatch.setattr(irodori, "base_checkpoint", lambda _root: base)
    monkeypatch.setattr(irodori, "configured_backend", lambda _root: "cpu")
    monkeypatch.setattr(irodori, "local_model_env", lambda _root: {})
    monkeypatch.setattr(
        irodori,
        "_verify_speaker_embedding_checkpoint",
        lambda _vendor, checkpoint, env: checkpoint != incomplete,
    )

    result = irodori.train_irodori(
        tmp_path,
        manifest,
        models,
        tmp_path / "cache",
        speaker_steps=25,
        lora_steps=25,
        do_speaker=True,
        do_lora=False,
        validation_ratio=0.1,
        validation_every=5,
        checkpoint_best_n=2,
        plan_fingerprint="f" * 64,
    )

    assert result["speaker_embedding"] == str(final)
    assert result["artifact"] == str(best)
    assert result["best_checkpoint"] == str(best)
    assert result["checkpoint_step"] == 20
    assert result["best_validation_loss"] == 0.3


def test_speaker_inversion_training_uses_older_verified_checkpoint_when_newer_is_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vendor = tmp_path / "vendor"
    _write_full_config(vendor / "configs" / "train_v4_small_speaker_inversion.yaml")
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"text":"a","latent_path":"a.pt"}\n', encoding="utf-8")
    models = tmp_path / "models"
    output = models / "irodori" / "speaker"
    output.mkdir(parents=True)
    older = output / "checkpoint_0000010.speaker.safetensors"
    newer = output / "checkpoint_0000020.speaker.safetensors"
    final = output / "checkpoint_final.speaker.safetensors"
    older.write_bytes(b"verified-old")
    newer.write_bytes(b"truncated-new")
    commands: list[list[object]] = []

    monkeypatch.setattr(irodori, "vendor_dir", lambda _root: vendor)
    monkeypatch.setattr(irodori, "base_checkpoint", lambda _root: base)
    monkeypatch.setattr(irodori, "configured_backend", lambda _root: "cpu")
    monkeypatch.setattr(irodori, "local_model_env", lambda _root: {})

    def verify(_vendor, checkpoint, *, env):
        del env
        return checkpoint in {older, final} and checkpoint.is_file()

    def fake_run(args, **_kwargs):
        commands.append(list(args))
        final.write_bytes(b"verified-final")
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(irodori, "_verify_speaker_embedding_checkpoint", verify)
    monkeypatch.setattr(irodori, "run", fake_run)

    result = irodori.train_irodori(
        tmp_path,
        manifest,
        models,
        tmp_path / "cache",
        speaker_steps=25,
        lora_steps=25,
        do_speaker=True,
        do_lora=False,
    )

    assert result["speaker_embedding"] == str(final)
    option = commands[0].index("--speaker-inversion-init-embedding")
    assert commands[0][option + 1] == older
    assert older.read_bytes() == b"verified-old"
    assert newer.read_bytes() == b"truncated-new"
