from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import struct
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_contract():
    path = ROOT / "workers" / "lfm" / "checkpoint_contract.py"
    spec = importlib.util.spec_from_file_location("checkpoint_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trainer_state(path: Path) -> None:
    step = int(path.name.removeprefix("checkpoint-"))
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": step + 100}),
        encoding="utf-8",
    )
    (path / "optimizer.pt").write_text(
        json.dumps({"state": {"0": {"step": step}}, "param_groups": [{"params": [0]}]}),
        encoding="utf-8",
    )
    (path / "scheduler.pt").write_text(
        json.dumps({"last_epoch": step}),
        encoding="utf-8",
    )
    (path / "rng_state.pth").write_text(
        json.dumps({"python": [1], "numpy": [2], "cpu": [3]}),
        encoding="utf-8",
    )
    payload = pickle.dumps({"fp16": False, "bf16": False, "use_cpu": True}, protocol=4)
    with zipfile.ZipFile(path / "training_args.bin", "w") as archive:
        archive.writestr("training_args/data.pkl", payload)
        archive.writestr("training_args/version", "3\n")


def _write_safetensors(path: Path) -> None:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))


def _install_safe_state_loader(contract, monkeypatch) -> None:
    monkeypatch.setattr(
        contract,
        "_safe_torch_load",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )


def test_checkpoint_contract_distinguishes_full_and_lora_without_cross_pruning(
    tmp_path: Path,
    monkeypatch,
):
    contract = _load_contract()
    _install_safe_state_loader(contract, monkeypatch)
    full = tmp_path / "checkpoint-10"
    lora = tmp_path / "checkpoint-20"
    partial = tmp_path / "checkpoint-30"
    _write_trainer_state(full)
    _write_trainer_state(lora)
    _write_trainer_state(partial)
    (full / "config.json").write_text('{"model_type":"lfm2"}\n', encoding="utf-8")
    _write_safetensors(full / "model.safetensors")
    (lora / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
    _write_safetensors(lora / "adapter_model.safetensors")
    (partial / "config.json").write_text('{"model_type":"lfm2"}\n', encoding="utf-8")
    (partial / "model.safetensors").write_bytes(b"")

    assert contract.checkpoint_complete(full, method="full")
    assert not contract.checkpoint_complete(full, method="lora")
    assert contract.checkpoint_complete(lora, method="lora")
    assert not contract.checkpoint_complete(lora, method="full")
    assert contract.latest_complete_checkpoint(tmp_path, method="full") == full
    assert contract.latest_complete_checkpoint(tmp_path, method="lora") == lora

    removed = contract.prune_incomplete_checkpoints(tmp_path, method="full")
    assert removed == [partial]
    assert full.exists() and lora.exists()

    (full / contract._TRAINING_METHOD_MARKER).write_text("lora\n", encoding="utf-8")
    assert not contract.checkpoint_complete(full, method="full")


class _FakeParameter:
    def __init__(self):
        self.requires_grad = False

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self


class _FakeModel:
    def __init__(self):
        self._parameters = [_FakeParameter(), _FakeParameter()]

    def parameters(self):
        return iter(self._parameters)

    def save_pretrained(self, directory: str):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text("{}\n", encoding="utf-8")
        (output / "model.safetensors").write_bytes(b"full-model")


class _FakeTokenizer:
    @staticmethod
    def apply_chat_template(
        messages,
        *,
        add_generation_prompt=False,
        return_dict=False,
        **_kwargs,
    ):
        if messages and messages[-1]["role"] == "assistant":
            input_ids = [10, 20, 30, 40]
        else:
            assert add_generation_prompt is True
            input_ids = [10, 20, 30]
        return {"input_ids": input_ids} if return_dict else input_ids

    @staticmethod
    def save_pretrained(directory: str):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (output / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (output / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")


class _FakeDataset:
    column_names = ["prompt", "completion"]

    def __init__(self, size: int = 10):
        self.size = size
        self.split_call = None

    def __len__(self):
        return self.size

    def train_test_split(self, **kwargs):
        self.split_call = kwargs
        valid_size = int(kwargs["test_size"])
        return {
            "train": _FakeSplitDataset(self.size - valid_size),
            "test": _FakeSplitDataset(valid_size),
        }


class _FakeSplitDataset:
    def __init__(self, size: int):
        self.rows = [
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": f"question-{index}"},
                ],
                "completion": [{"role": "assistant", "content": f"answer-{index}"}],
            }
            for index in range(size)
        ]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


def _load_train_module(monkeypatch, tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir(parents=True, exist_ok=True)
    (base / "special_tokens_map.json").write_text(
        '{"eos_token":"<|im_end|>"}\n', encoding="utf-8"
    )
    contract = _load_contract()
    _install_safe_state_loader(contract, monkeypatch)
    monkeypatch.setitem(sys.modules, "checkpoint_contract", contract)

    torch = types.ModuleType("torch")
    torch.dtype = object
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    dataset = _FakeDataset()
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    lora_target_calls = []
    model_contract = types.ModuleType("model_contract")
    model_contract.audited_attention_lora_targets = lambda model: lora_target_calls.append(model) or [
        "q_proj"
    ]
    model_contract.json_contains_absolute_local_path = lambda _value: False
    monkeypatch.setitem(sys.modules, "model_contract", model_contract)

    models: list[_FakeModel] = []
    tokenizers: list[_FakeTokenizer] = []

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            model = _FakeModel()
            models.append(model)
            return model

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            tokenizer = _FakeTokenizer()
            tokenizers.append(tokenizer)
            return tokenizer

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = AutoModel
    transformers.AutoTokenizer = AutoTokenizer
    transformers.TrainerCallback = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    peft_calls = []

    class LoraConfig:
        def __init__(self, **kwargs):
            peft_calls.append(kwargs)

    peft = types.ModuleType("peft")
    peft.LoraConfig = LoraConfig
    monkeypatch.setitem(sys.modules, "peft", peft)

    class FakeSFTConfig:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

    class FakeTrainer:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.args = kwargs["args"]
            self.method = "lora" if "peft_config" in kwargs else "full"
            self.state = types.SimpleNamespace(best_model_checkpoint=None, best_metric=0.25)
            self.train_dataset = [
                {"input_ids": [10, 20, 30, 40], "completion_mask": [0, 0, 0, 1]}
                for _row in kwargs["train_dataset"]
            ]
            self.eval_dataset = [
                {"input_ids": [10, 20, 30, 40], "completion_mask": [0, 0, 0, 1]}
                for _row in kwargs["eval_dataset"]
            ]
            self.resume = None
            self.instances.append(self)

        def train(self, *, resume_from_checkpoint):
            self.resume = resume_from_checkpoint
            run_dir = Path(self.args.kwargs["output_dir"])
            checkpoint = run_dir / "checkpoint-7"
            _write_trainer_state(checkpoint)
            if self.method == "full":
                (checkpoint / "config.json").write_text(
                    '{"model_type":"lfm2"}\n', encoding="utf-8"
                )
                _write_safetensors(checkpoint / "model.safetensors")
            else:
                (checkpoint / "adapter_config.json").write_text(
                    '{"peft_type":"LORA"}\n', encoding="utf-8"
                )
                _write_safetensors(checkpoint / "adapter_model.safetensors")
            self.state.best_model_checkpoint = str(checkpoint)

        def save_model(self, directory: str):
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (output / "adapter_model.safetensors").write_bytes(b"adapter-final")

    trl = types.ModuleType("trl")
    trl.SFTConfig = FakeSFTConfig
    trl.SFTTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)

    train_path = ROOT / "workers" / "lfm" / "train.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_lfm_full_train", train_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_verify_base", lambda _base: None)
    return module, {
        "dataset": dataset,
        "models": models,
        "tokenizers": tokenizers,
        "configs": FakeSFTConfig.instances,
        "trainers": FakeTrainer.instances,
        "peft_calls": peft_calls,
        "lora_target_calls": lora_target_calls,
    }


def _args(tmp_path: Path, *, method: str) -> argparse.Namespace:
    return argparse.Namespace(
        base=str(tmp_path / "base"),
        dataset=str(tmp_path / "dataset.jsonl"),
        output=str(tmp_path / "models" / method),
        run_dir=str(tmp_path / "runs" / method),
        method=method,
        plan_fingerprint="c" * 64 if method == "full" else None,
        epochs=3.0,
        learning_rate=None,
        validation_ratio=0.1,
        validation_seed=20260824,
        save_steps=25,
        lora_r=16,
        lora_alpha=32,
    )


def test_full_training_omits_peft_uses_validation_and_writes_portable_artifact(
    tmp_path: Path,
    monkeypatch,
):
    module, fakes = _load_train_module(monkeypatch, tmp_path)
    args = _args(tmp_path, method="full")
    result = module.run_training(args)

    trainer = fakes["trainers"][0]
    config = fakes["configs"][0].kwargs
    assert "peft_config" not in trainer.kwargs
    assert fakes["peft_calls"] == []
    assert fakes["lora_target_calls"] == []
    assert config["completion_only_loss"] is True
    assert config["eval_strategy"] == "steps"
    assert config["eval_steps"] == 25
    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == 25
    assert config["load_best_model_at_end"] is True
    assert config["metric_for_best_model"] == "eval_loss"
    assert config["learning_rate"] == 2e-5
    assert len(trainer.kwargs["callbacks"]) == 1
    assert trainer.kwargs["callbacks"][0].method == "full"
    assert fakes["dataset"].split_call == {
        "test_size": 1,
        "seed": 20260824,
        "shuffle": True,
    }
    assert all(parameter.requires_grad for parameter in fakes["models"][0]._parameters)

    artifact = Path(result["artifact"])
    assert artifact == Path(args.output).resolve()
    assert module.full_artifact_complete(artifact, plan_fingerprint="c" * 64)
    provenance = json.loads((artifact / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["best_checkpoint"] == "checkpoint-7"
    assert str(tmp_path) not in json.dumps(provenance)
    assert trainer.resume is None
    assert result["best_validation_loss"] == 0.25


def test_checkpoint_callback_marks_only_a_fully_resumable_periodic_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    module, _fakes = _load_train_module(monkeypatch, tmp_path)
    run_dir = tmp_path / "runs" / "full"
    checkpoint = run_dir / "checkpoint-9"
    _write_trainer_state(checkpoint)
    (checkpoint / "config.json").write_text('{"model_type":"lfm2"}\n', encoding="utf-8")
    _write_safetensors(checkpoint / "model.safetensors")
    callback = module._CheckpointMethodMarkerCallback("full")
    control = object()

    returned = callback.on_save(
        types.SimpleNamespace(
            output_dir=str(run_dir), fp16=False, bf16=False, use_cpu=True
        ),
        types.SimpleNamespace(global_step=9),
        control,
    )

    assert returned is control
    assert (checkpoint / module.TRAINING_METHOD_MARKER).read_text(
        encoding="utf-8"
    ).strip() == "full"
    assert module.checkpoint_complete(checkpoint, method="full")

    incomplete = run_dir / "checkpoint-10"
    incomplete.mkdir()
    with pytest.raises(RuntimeError, match="fully resumable"):
        callback.on_save(
            types.SimpleNamespace(
                output_dir=str(run_dir), fp16=False, bf16=False, use_cpu=True
            ),
            types.SimpleNamespace(global_step=10),
            control,
        )
    assert not (incomplete / module.TRAINING_METHOD_MARKER).exists()


def test_lora_path_keeps_peft_and_completion_only_validation(tmp_path: Path, monkeypatch):
    module, fakes = _load_train_module(monkeypatch, tmp_path)
    args = _args(tmp_path, method="lora")
    result = module.run_training(args)

    trainer = fakes["trainers"][0]
    assert "peft_config" in trainer.kwargs
    assert len(fakes["peft_calls"]) == 1
    assert fakes["lora_target_calls"] == [fakes["models"][0]]
    assert fakes["configs"][0].kwargs["completion_only_loss"] is True
    assert fakes["configs"][0].kwargs["learning_rate"] == 2e-4
    output = Path(result["artifact"])
    assert (output / "adapter_config.json").is_file()
    assert (output / module.ADAPTER_REVISION_MARKER).read_text(encoding="utf-8").strip() == (
        module.MODEL_REVISION
    )
    provenance = json.loads((output / module.PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert provenance["best_validation_loss"] == 0.25


def test_completion_preflight_rejects_truncation_and_tokenizer_prefix_drift(
    tmp_path: Path,
    monkeypatch,
):
    module, _fakes = _load_train_module(monkeypatch, tmp_path)
    rows = list(_FakeSplitDataset(1))

    class TooLongTokenizer(_FakeTokenizer):
        @staticmethod
        def apply_chat_template(messages, *, return_dict=False, **_kwargs):
            token_ids = (
                list(range(module.MAX_SEQUENCE_LENGTH + 1))
                if messages[-1]["role"] == "assistant"
                else list(range(module.MAX_SEQUENCE_LENGTH))
            )
            return {"input_ids": token_ids} if return_dict else token_ids

    with pytest.raises(RuntimeError, match="exceeding max_length"):
        module._raw_completion_shapes(
            rows,
            TooLongTokenizer(),
            dataset_name="train",
            max_length=module.MAX_SEQUENCE_LENGTH,
        )

    class PrefixDriftTokenizer(_FakeTokenizer):
        @staticmethod
        def apply_chat_template(messages, *, return_dict=False, **_kwargs):
            token_ids = [1, 9, 3] if messages[-1]["role"] == "assistant" else [1, 2]
            return {"input_ids": token_ids} if return_dict else token_ids

    with pytest.raises(RuntimeError, match="not a prefix"):
        module._raw_completion_shapes(
            rows,
            PrefixDriftTokenizer(),
            dataset_name="train",
            max_length=module.MAX_SEQUENCE_LENGTH,
        )


@pytest.mark.parametrize(
    ("field", "role", "match"),
    [
        ("prompt", "assistant", "prompt messages require.*role system/user"),
        ("completion", "user", "completion messages require.*role assistant"),
    ],
)
def test_completion_preflight_rejects_unsafe_conversation_roles(
    tmp_path: Path,
    monkeypatch,
    field: str,
    role: str,
    match: str,
):
    module, _fakes = _load_train_module(monkeypatch, tmp_path)
    row = list(_FakeSplitDataset(1))[0]
    row[field][-1]["role"] = role

    with pytest.raises(RuntimeError, match=match):
        module._raw_completion_shapes(
            [row],
            _FakeTokenizer(),
            dataset_name="train",
            max_length=module.MAX_SEQUENCE_LENGTH,
        )


def test_completion_preflight_rejects_missing_or_changed_processed_labels(
    tmp_path: Path,
    monkeypatch,
):
    module, _fakes = _load_train_module(monkeypatch, tmp_path)
    expected = module.Counter({(4, 1): 1})

    with pytest.raises(RuntimeError, match="invalid completion mask"):
        module._verify_processed_completion_shapes(
            [{"input_ids": [10, 20, 30, 40]}],
            dataset_name="train",
            expected=expected,
        )
    with pytest.raises(RuntimeError, match="no completion labels"):
        module._verify_processed_completion_shapes(
            [{"input_ids": [10, 20, 30, 40], "completion_mask": [0, 0, 0, 0]}],
            dataset_name="train",
            expected=expected,
        )
    with pytest.raises(RuntimeError, match="differ from the audited"):
        module._verify_processed_completion_shapes(
            [{"input_ids": [10, 20, 30, 40], "completion_mask": [0, 0, 1, 1]}],
            dataset_name="train",
            expected=expected,
        )


def _load_worker(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.dtype = object
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    contract = types.ModuleType("model_contract")
    contract.audited_attention_lora_targets = lambda _model: []
    contract.json_contains_absolute_local_path = lambda _value: False
    monkeypatch.setitem(sys.modules, "model_contract", contract)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    worker_path = ROOT / "workers" / "lfm" / "worker.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_lfm_full_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_worker_full_artifact(worker, output: Path) -> None:
    output.mkdir(parents=True)
    for name in worker.REQUIRED_MODEL_FILES:
        path = output / name
        path.write_text("{}\n", encoding="utf-8")
    (output / worker.ADAPTER_REVISION_MARKER).write_text(
        worker.MODEL_REVISION + "\n", encoding="utf-8"
    )
    (output / worker.TRAINING_METHOD_MARKER).write_text("full\n", encoding="utf-8")
    relative_paths = list(worker.REQUIRED_MODEL_FILES) + [
        worker.ADAPTER_REVISION_MARKER,
        worker.TRAINING_METHOD_MARKER,
    ]
    files = []
    for relative in relative_paths:
        path = output / relative
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": worker._sha256(path),
            }
        )
    (output / worker.PROVENANCE_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "lfm",
                "method": "full",
                "training_plan_fingerprint": "d" * 64,
                "base_model": {
                    "id": worker.MODEL_ID,
                    "revision": worker.MODEL_REVISION,
                    "model_weight_sha256": worker.MODEL_WEIGHT_SHA256,
                },
                "best_validation_loss": 0.25,
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_worker_verifies_and_uses_full_model_without_loading_peft(tmp_path: Path, monkeypatch):
    worker = _load_worker(monkeypatch)
    full = tmp_path / "full"
    _write_worker_full_artifact(worker, full)
    worker.verify_full_model(full)

    class InputIds:
        shape = (1, 2)

    class BatchEncoding(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Output:
        def __getitem__(self, index):
            assert index == (0, slice(2, None))
            return "generated"

    class Tokenizer:
        def apply_chat_template(self, *_args, **kwargs):
            assert kwargs["return_dict"] is True
            return BatchEncoding(input_ids=InputIds())

        def decode(self, _tokens, **_kwargs):
            return "full result"

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert isinstance(kwargs.pop("input_ids"), InputIds)
            return Output()
    monkeypatch.setattr(worker, "load_full_model", lambda path: (Tokenizer(), Model()))
    monkeypatch.setattr(
        worker,
        "load_base",
        lambda: (_ for _ in ()).throw(AssertionError("base must not load for full inference")),
    )
    result = worker.infer({"full_model": str(full), "messages": []})
    assert result == {"text": "full result"}

    (full / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.verify_full_model(full)
