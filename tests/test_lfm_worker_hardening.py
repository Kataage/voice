from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


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
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    contract = types.ModuleType("model_contract")
    contract.audited_attention_lora_targets = lambda _model: []
    monkeypatch.setitem(sys.modules, "model_contract", contract)

    peft = types.ModuleType("peft")
    peft.PeftModel = object
    monkeypatch.setitem(sys.modules, "peft", peft)

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    worker_path = Path(__file__).parents[1] / "workers" / "lfm" / "worker.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_lfm_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_complete_base(base: Path, worker) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for name in worker.REQUIRED_MODEL_FILES:
        path = base / name
        if name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        elif name.endswith(".jinja"):
            path.write_text("{{ messages }}\n", encoding="utf-8")
        else:
            path.write_bytes(b"weights")


def test_lfm_base_path_requires_complete_snapshot_and_pinned_revision(tmp_path: Path, monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    base = tmp_path / "models" / "lfm" / "base"
    _write_complete_base(base, worker)
    marker = base / worker.REVISION_MARKER

    (base / "model.safetensors").write_bytes(b"")
    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        worker.base_path()

    (base / "model.safetensors").write_bytes(b"weights")
    marker.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the audited revision"):
        worker.base_path()

    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.base_path()
    monkeypatch.setattr(worker, "_sha256", lambda _path: worker.MODEL_WEIGHT_SHA256)
    assert Path(worker.base_path()) == base


def test_lfm_adapter_requires_nonempty_config_weight_and_revision(tmp_path: Path, monkeypatch):
    worker = _load_worker(monkeypatch)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    weight = adapter / "adapter_model.safetensors"
    marker = adapter / worker.ADAPTER_REVISION_MARKER

    config.write_bytes(b"")
    weight.write_bytes(b"weights")
    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="adapter is incomplete"):
        worker.verify_adapter(adapter)

    config.write_text("{}\n", encoding="utf-8")
    weight.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="adapter is incomplete"):
        worker.verify_adapter(adapter)

    weight.write_bytes(b"weights")
    marker.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="JP-202606"):
        worker.verify_adapter(adapter)

    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    worker.verify_adapter(adapter)


def test_lfm_infer_passes_batch_encoding_as_keyword_inputs(monkeypatch):
    worker = _load_worker(monkeypatch)
    calls: dict[str, object] = {}

    class FakeTensor:
        shape = (1, 3)

    class FakeBatchEncoding(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class FakeOutput:
        def __getitem__(self, key):
            assert key == (0, slice(3, None))
            return [99]

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "test"}]
            assert kwargs == {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            return FakeBatchEncoding(
                input_ids=FakeTensor(),
                attention_mask=FakeTensor(),
            )

        def decode(self, generated, *, skip_special_tokens):
            assert generated == [99]
            assert skip_special_tokens is True
            return "ok"

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            calls.update(kwargs)
            assert isinstance(kwargs["input_ids"], FakeTensor)
            assert isinstance(kwargs["attention_mask"], FakeTensor)
            return FakeOutput()

    monkeypatch.setattr(worker, "load_base", lambda: (FakeTokenizer(), FakeModel()))
    result = worker.infer(
        {
            "messages": [{"role": "user", "content": "test"}],
            "adapter": None,
            "temperature": 0.1,
            "max_new_tokens": 32,
        }
    )

    assert result == {"text": "ok"}
    assert calls["do_sample"] is True
    assert calls["temperature"] == pytest.approx(0.1)
    assert calls["top_k"] == 50
    assert calls["repetition_penalty"] == pytest.approx(1.05)
    assert calls["max_new_tokens"] == 32


def test_lfm_download_refuses_incomplete_materialization(tmp_path: Path, monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    with pytest.raises(FileNotFoundError, match="required model files"):
        worker.download_model({})

    base = tmp_path / "models" / "lfm" / "base"
    _write_complete_base(base, worker)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.download_model({})
    assert not (base / worker.REVISION_MARKER).exists()
    monkeypatch.setattr(worker, "_sha256", lambda _path: worker.MODEL_WEIGHT_SHA256)
    result = worker.download_model({})
    assert result["revision"] == worker.MODEL_REVISION
    assert (base / worker.REVISION_MARKER).read_text(encoding="utf-8").strip() == worker.MODEL_REVISION
