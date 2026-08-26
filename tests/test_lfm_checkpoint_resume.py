from __future__ import annotations

import contextlib
import importlib.util
import json
import pickle
import struct
import sys
import types
import zipfile
from pathlib import Path

import pytest

from personavoice.config import PersonaConfig
from personavoice.project import init_persona
from personavoice.training import _fingerprint, _lfm_native_checkpoint_complete


def _contract_module():
    path = Path(__file__).resolve().parents[1] / "workers" / "lfm" / "checkpoint_contract.py"
    spec = importlib.util.spec_from_file_location("lfm_checkpoint_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_safetensors(path: Path) -> None:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))


def _write_training_args(
    path: Path,
    *,
    fp16: bool,
    bf16: bool,
    use_cpu: bool,
) -> None:
    payload = pickle.dumps(
        {"fp16": fp16, "bf16": bf16, "use_cpu": use_cpu},
        protocol=4,
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("training_args/data.pkl", payload)
        archive.writestr("training_args/version", "3\n")
        archive.writestr("training_args/byteorder", "little")


def _write_state(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _install_safe_state_loader(contract, monkeypatch) -> None:
    def load(path: Path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(contract, "_safe_torch_load", load)


def _write_native_checkpoint(
    path: Path,
    *,
    method: str = "lora",
    fp16: bool = False,
    bf16: bool = False,
    use_cpu: bool = True,
) -> None:
    step = int(path.name.removeprefix("checkpoint-"))
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": step + 100}),
        encoding="utf-8",
    )
    _write_state(
        path / "optimizer.pt",
        {"state": {"0": {"step": step}}, "param_groups": [{"params": [0]}]},
    )
    _write_state(path / "scheduler.pt", {"last_epoch": step, "base_lrs": [0.001]})
    rng = {"python": [1], "numpy": [2], "cpu": [3]}
    if not use_cpu:
        rng["cuda"] = [4]
    _write_state(path / "rng_state.pth", rng)
    _write_training_args(path / "training_args.bin", fp16=fp16, bf16=bf16, use_cpu=use_cpu)
    if fp16:
        _write_state(
            path / "scaler.pt",
            {
                "scale": 65536.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 0,
            },
        )
    if method == "lora":
        (path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 16}),
            encoding="utf-8",
        )
        _write_safetensors(path / "adapter_model.safetensors")
    else:
        (path / "config.json").write_text(
            json.dumps({"model_type": "lfm2"}),
            encoding="utf-8",
        )
        _write_safetensors(path / "model.safetensors")


def _seal(
    contract,
    path: Path,
    *,
    method: str = "lora",
    fp16: bool = False,
    bf16: bool = False,
    use_cpu: bool = True,
) -> None:
    args = types.SimpleNamespace(fp16=fp16, bf16=bf16, use_cpu=use_cpu)
    contract.seal_checkpoint(path, method=method, training_args=args)


def test_lfm_resume_uses_highest_verified_checkpoint_without_deleting_rejected(
    tmp_path: Path,
    monkeypatch,
):
    contract = _contract_module()
    _install_safe_state_loader(contract, monkeypatch)
    older = tmp_path / "checkpoint-100"
    corrupt_newer = tmp_path / "checkpoint-200"
    newest_complete = tmp_path / "checkpoint-300"
    for path in (older, corrupt_newer, newest_complete):
        _write_native_checkpoint(path)
        _seal(contract, path)
    (corrupt_newer / "optimizer.pt").write_bytes(b"tampered")

    assert contract.latest_complete_checkpoint(tmp_path) == newest_complete
    rejected = contract.prune_incomplete_checkpoints(tmp_path)
    assert rejected == [corrupt_newer]
    assert corrupt_newer.exists()
    assert older.exists() and newest_complete.exists()


@pytest.mark.parametrize(
    "name",
    [
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "rng_state.pth",
        ".personavoice-training-method",
    ],
)
def test_lfm_attestation_rejects_missing_or_tampered_native_payload(
    tmp_path: Path,
    monkeypatch,
    name: str,
):
    contract = _contract_module()
    _install_safe_state_loader(contract, monkeypatch)
    checkpoint = tmp_path / "checkpoint-10"
    _write_native_checkpoint(checkpoint)
    _seal(contract, checkpoint)
    candidate = checkpoint / name
    original = candidate.read_bytes()

    candidate.unlink()
    assert not contract.checkpoint_complete(checkpoint)
    candidate.write_bytes(original)
    assert contract.checkpoint_complete(checkpoint)

    candidate.write_bytes(original + b"tampered")
    assert not contract.checkpoint_complete(checkpoint)
    # Verification failure never deletes or rewrites the checkpoint.
    assert candidate.read_bytes().endswith(b"tampered")


def test_lfm_fp16_requires_valid_scaler_but_bf16_and_cpu_do_not(
    tmp_path: Path,
    monkeypatch,
):
    contract = _contract_module()
    _install_safe_state_loader(contract, monkeypatch)
    fp16 = tmp_path / "checkpoint-10"
    _write_native_checkpoint(fp16, fp16=True, use_cpu=False)
    (fp16 / "scaler.pt").unlink()
    assert not contract.checkpoint_complete(fp16)
    with pytest.raises(ValueError, match="scaler"):
        _seal(contract, fp16, fp16=True, use_cpu=False)

    bf16 = tmp_path / "checkpoint-20"
    _write_native_checkpoint(bf16, bf16=True, use_cpu=False)
    _seal(contract, bf16, bf16=True, use_cpu=False)
    assert contract.checkpoint_complete(bf16)

    cpu = tmp_path / "checkpoint-30"
    _write_native_checkpoint(cpu)
    _seal(contract, cpu)
    assert contract.checkpoint_complete(cpu)


def test_lfm_rejects_inconsistent_trainer_scheduler_and_cuda_rng(
    tmp_path: Path,
    monkeypatch,
):
    contract = _contract_module()
    _install_safe_state_loader(contract, monkeypatch)
    trainer = tmp_path / "checkpoint-10"
    _write_native_checkpoint(trainer)
    (trainer / "trainer_state.json").write_text(
        json.dumps({"global_step": 9, "max_steps": 100}),
        encoding="utf-8",
    )
    assert not contract.checkpoint_complete(trainer)

    scheduler = tmp_path / "checkpoint-20"
    _write_native_checkpoint(scheduler)
    _write_state(scheduler / "scheduler.pt", {"last_epoch": 19})
    assert not contract.checkpoint_complete(scheduler)

    rng = tmp_path / "checkpoint-30"
    _write_native_checkpoint(rng, fp16=True, use_cpu=False)
    _write_state(rng / "rng_state.pth", {"python": [1], "numpy": [2], "cpu": [3]})
    assert not contract.checkpoint_complete(rng)


def test_lfm_resume_ignores_non_numeric_checkpoint_directories(tmp_path: Path):
    contract = _contract_module()
    unrelated = tmp_path / "checkpoint-final"
    unrelated.mkdir()
    assert contract.checkpoint_step(unrelated) is None
    assert contract.prune_incomplete_checkpoints(tmp_path) == []
    assert unrelated.exists()


def test_lfm_safe_loader_uses_weights_only_and_transformers_rng_allowlist(
    tmp_path: Path,
    monkeypatch,
):
    contract = _contract_module()
    calls = []
    checks = []

    torch = types.ModuleType("torch")

    def load(path, **kwargs):
        calls.append((Path(path), kwargs))
        return {"state": {}}

    torch.load = load
    transformers = types.ModuleType("transformers")
    trainer_utils = types.ModuleType("transformers.trainer_pt_utils")
    trainer_utils.safe_globals = contextlib.nullcontext
    utils = types.ModuleType("transformers.utils")
    utils.check_torch_load_is_safe = lambda: checks.append(True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.trainer_pt_utils", trainer_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)

    path = tmp_path / "state.pt"
    path.write_bytes(b"state")
    assert contract._safe_torch_load(path) == {"state": {}}
    assert checks == [True]
    assert calls == [
        (
            path,
            {"map_location": "cpu", "weights_only": True, "mmap": True},
        )
    ]


def test_modal_observer_native_contract_rejects_missing_attestation_and_tampering(
    tmp_path: Path,
    monkeypatch,
):
    contract = _contract_module()
    _install_safe_state_loader(contract, monkeypatch)
    checkpoint = tmp_path / "checkpoint-40"
    _write_native_checkpoint(checkpoint, method="full")

    # The root environment cannot safely load PyTorch state itself and must not
    # accept a merely non-empty worker directory.
    assert not _lfm_native_checkpoint_complete(checkpoint, method="full")
    _seal(contract, checkpoint, method="full")
    assert _lfm_native_checkpoint_complete(checkpoint, method="full")

    (checkpoint / "scheduler.pt").write_bytes(b"tampered")
    assert not _lfm_native_checkpoint_complete(checkpoint, method="full")


def test_training_fingerprint_includes_lfm_checkpoint_contract(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    (paths.dataset / "irodori_source.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    helper = tmp_path / "workers" / "lfm" / "checkpoint_contract.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("revision = 1\n", encoding="utf-8")

    first = _fingerprint(paths, cfg)
    helper.write_text("revision = 2\n", encoding="utf-8")
    second = _fingerprint(paths, cfg)
    assert first != second
