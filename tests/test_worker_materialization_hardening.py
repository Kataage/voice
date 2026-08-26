from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from personavoice.doctor import report as doctor_report


def _load_asr_worker(monkeypatch):
    ctranslate2 = types.ModuleType("ctranslate2")
    ctranslate2.get_supported_compute_types = lambda _device: {"float32"}
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    runtime_policy = types.ModuleType("runtime_policy")
    runtime_policy.choose_compute_type = lambda _device: "float32"
    monkeypatch.setitem(sys.modules, "runtime_policy", runtime_policy)

    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    worker_path = Path(__file__).parents[1] / "workers" / "asr" / "worker.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_asr_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_diarization_worker(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda value: value
    monkeypatch.setitem(sys.modules, "torch", torch)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    pyannote = types.ModuleType("pyannote")
    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = object
    pyannote.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)

    worker_path = Path(__file__).parents[1] / "workers" / "diarization" / "worker.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_diarization_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_lfm_worker(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    torch.float32 = object()
    torch.float16 = object()
    torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    model_contract = types.ModuleType("model_contract")
    model_contract.audited_attention_lora_targets = lambda model: []
    model_contract.json_contains_absolute_local_path = lambda _value: False
    monkeypatch.setitem(sys.modules, "model_contract", model_contract)

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


def _write_complete_asr_files(local: Path, worker) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name in worker.REQUIRED_MODEL_FILES:
        if name.endswith(".json"):
            (local / name).write_text("{}\n", encoding="utf-8")
        else:
            (local / name).write_bytes(b"weights")


def _write_complete_pyannote_files(local: Path, worker) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name in worker.REQUIRED_MODEL_FILES:
        path = local / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".yaml"):
            path.write_text("pipeline: {}\n", encoding="utf-8")
        else:
            path.write_bytes(b"weights")


def _write_complete_lfm_files(local: Path, worker) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name in worker.REQUIRED_MODEL_FILES:
        path = local / name
        if name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        elif name.endswith(".jinja"):
            path.write_text("{{ messages }}\n", encoding="utf-8")
        else:
            path.write_bytes(b"weights")


def test_asr_model_path_requires_nonempty_required_files_and_revision(tmp_path: Path, monkeypatch):
    worker = _load_asr_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    local = tmp_path / "models" / "asr" / worker.PINNED_MODEL_NAME
    _write_complete_asr_files(local, worker)
    (local / "model.bin").write_bytes(b"")
    (local / worker.REVISION_MARKER).write_text(worker.PINNED_MODEL_REVISION + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        worker.model_path(worker.PINNED_MODEL_NAME)

    (local / "model.bin").write_bytes(b"weights")
    (local / worker.REVISION_MARKER).write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited revision"):
        worker.model_path(worker.PINNED_MODEL_NAME)

    (local / worker.REVISION_MARKER).write_text(worker.PINNED_MODEL_REVISION + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.model_path(worker.PINNED_MODEL_NAME)
    monkeypatch.setattr(worker, "_sha256", lambda _path: worker.PINNED_MODEL_WEIGHT_SHA256)
    assert Path(worker.model_path(worker.PINNED_MODEL_NAME)) == local


def test_asr_download_does_not_finalize_incomplete_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_asr_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    with pytest.raises(FileNotFoundError, match="required model files"):
        worker.download({})

    local = tmp_path / "models" / "asr" / worker.PINNED_MODEL_NAME
    _write_complete_asr_files(local, worker)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.download({})
    assert not (local / worker.REVISION_MARKER).exists()
    monkeypatch.setattr(worker, "_sha256", lambda _path: worker.PINNED_MODEL_WEIGHT_SHA256)
    result = worker.download({})
    assert result["revision"] == worker.PINNED_MODEL_REVISION
    assert (local / worker.REVISION_MARKER).read_text(encoding="utf-8").strip() == worker.PINNED_MODEL_REVISION


def test_doctor_asr_static_check_requires_full_offline_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_asr_worker(monkeypatch)
    local = tmp_path / "models" / "asr" / worker.PINNED_MODEL_NAME
    local.mkdir(parents=True)
    (local / "model.bin").write_bytes(b"weights")
    (local / "config.json").write_text("{}\n", encoding="utf-8")

    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["asr"] is False

    _write_complete_asr_files(local, worker)
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["asr"] is True

    (local / "tokenizer.json").unlink()
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["asr"] is False


def test_pyannote_local_source_requires_complete_snapshot_and_revision(tmp_path: Path, monkeypatch):
    worker = _load_diarization_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    local = tmp_path / "models" / "pyannote" / "community-1"
    _write_complete_pyannote_files(local, worker)
    marker = local / worker.REVISION_MARKER
    (local / "embedding" / "pytorch_model.bin").write_bytes(b"")
    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        worker.local_source()

    (local / "embedding" / "pytorch_model.bin").write_bytes(b"weights")
    marker.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited revision"):
        worker.local_source()

    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.local_source()
    monkeypatch.setattr(
        worker,
        "_sha256",
        lambda path: worker.MODEL_ASSET_SHA256[path.relative_to(local).as_posix()],
    )
    assert Path(worker.local_source()) == local


def test_pyannote_download_does_not_finalize_incomplete_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_diarization_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    with pytest.raises(FileNotFoundError, match="required model files"):
        worker.download({})

    local = tmp_path / "models" / "pyannote" / "community-1"
    _write_complete_pyannote_files(local, worker)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.download({})
    assert not (local / worker.REVISION_MARKER).exists()
    monkeypatch.setattr(
        worker,
        "_sha256",
        lambda path: worker.MODEL_ASSET_SHA256[path.relative_to(local).as_posix()],
    )
    result = worker.download({})
    assert result["revision"] == worker.MODEL_REVISION
    assert (local / worker.REVISION_MARKER).read_text(encoding="utf-8").strip() == worker.MODEL_REVISION


def test_doctor_pyannote_static_check_requires_full_offline_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_diarization_worker(monkeypatch)
    local = tmp_path / "models" / "pyannote" / "community-1"
    local.mkdir(parents=True)
    (local / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")

    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["pyannote"] is False

    _write_complete_pyannote_files(local, worker)
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["pyannote"] is True

    (local / "plda" / "plda.npz").unlink()
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["pyannote"] is False


def test_lfm_base_path_requires_complete_snapshot_and_revision(tmp_path: Path, monkeypatch):
    worker = _load_lfm_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    local = tmp_path / "models" / "lfm" / "base"
    _write_complete_lfm_files(local, worker)
    marker = local / worker.REVISION_MARKER
    (local / "tokenizer.json").write_bytes(b"")
    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        worker.base_path()

    (local / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    marker.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited revision"):
        worker.base_path()

    marker.write_text(worker.MODEL_REVISION + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.base_path()
    monkeypatch.setattr(worker, "_sha256", lambda path: worker.MODEL_ASSET_SHA256[path.name])
    assert Path(worker.base_path()) == local


def test_lfm_download_does_not_finalize_incomplete_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_lfm_worker(monkeypatch)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    with pytest.raises(FileNotFoundError, match="required model files"):
        worker.download_model({})

    local = tmp_path / "models" / "lfm" / "base"
    _write_complete_lfm_files(local, worker)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        worker.download_model({})
    assert not (local / worker.REVISION_MARKER).exists()
    monkeypatch.setattr(worker, "_sha256", lambda path: worker.MODEL_ASSET_SHA256[path.name])
    result = worker.download_model({})
    assert result["revision"] == worker.MODEL_REVISION
    assert (local / worker.REVISION_MARKER).read_text(encoding="utf-8").strip() == worker.MODEL_REVISION


def test_doctor_lfm_static_check_requires_full_offline_snapshot(tmp_path: Path, monkeypatch):
    worker = _load_lfm_worker(monkeypatch)
    local = tmp_path / "models" / "lfm" / "base"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}\n", encoding="utf-8")

    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["lfm"] is False

    _write_complete_lfm_files(local, worker)
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["lfm"] is True

    (local / "model.safetensors").unlink()
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["lfm"] is False


def test_doctor_sense_static_check_requires_assets_and_verified_marker(tmp_path: Path):
    local = tmp_path / "models" / "sense" / "SenseVoiceSmall"
    local.mkdir(parents=True)
    runtime = tmp_path / ".runtime"
    runtime.mkdir(parents=True)
    marker = runtime / "sense-model-ready"
    marker.write_text("verified\n", encoding="utf-8")

    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["sense"] is False

    for name in ("model.pt", "am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model"):
        (local / name).write_bytes(b"asset")
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["sense"] is True

    (local / "am.mvn").unlink()
    result = doctor_report(tmp_path, require_seed_vc=False)
    assert result["models"]["sense"] is False
