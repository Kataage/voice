from __future__ import annotations

from pathlib import Path

from personavoice import cuda_preflight, environment_contract, hardware, inference, irodori
from personavoice.model_assets import IRODORI_TEXT_ENCODER_ID, IRODORI_TEXT_ENCODER_REVISION


def _gpu(capability: str, *, memory: int = 16384, uuid: str = "GPU-aaaaaaaa") -> hardware.GpuInfo:
    return hardware.GpuInfo(
        index=0,
        name=f"GPU-{capability}",
        total_mib=memory,
        free_mib=max(1, memory - 1024),
        compute_capability=capability,
        uuid=uuid,
        pci_bus_id="00000000:01:00.0",
        driver_version="999.1",
    )


def test_irodori_precision_policy_covers_turing_ampere_and_unknown(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert hardware.irodori_training_precision("cu128", gpu=_gpu("7.5")) == {
        "precision": "fp32",
        "allow_tf32": False,
    }
    assert hardware.irodori_training_precision("cu128", gpu=_gpu("8.6")) == {
        "precision": "bf16",
        "allow_tf32": True,
    }
    assert hardware.irodori_training_precision("cu126", gpu=_gpu("8.6")) == {
        "precision": "fp32",
        "allow_tf32": False,
    }
    assert hardware.irodori_training_precision("cu128", gpu=_gpu("")) == {
        "precision": "fp32",
        "allow_tf32": False,
    }


def test_irodori_config_uses_turing_safe_precision(tmp_path: Path, monkeypatch):
    import yaml

    source = tmp_path / "source.yaml"
    destination = tmp_path / "patched.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "text_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                    "text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
                    "caption_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,
                },
                "train": {
                    "precision": "bf16",
                    "allow_tf32": True,
                    "dataloader_cuda_prefetch": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        irodori,
        "safe_batch_profile",
        lambda *, backend: {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_workers": 2,
            "gradient_checkpointing": True,
        },
    )
    monkeypatch.setattr(
        irodori,
        "irodori_training_precision",
        lambda backend: hardware.irodori_training_precision(backend, gpu=_gpu("7.5")),
    )
    irodori._patched_config(source, destination, max_steps=100, backend="cu128")
    value = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert value["train"]["precision"] == "fp32"
    assert value["train"]["allow_tf32"] is False


def test_uuid_selector_accepts_prefix_but_rejects_appended_garbage(monkeypatch):
    gpu = _gpu("8.6", uuid="GPU-aaaaaaaa-bbbb")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa")
    assert hardware.selected_nvidia_gpu([gpu]) == gpu
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaaaaaa-bbbb-extra")
    assert hardware.selected_nvidia_gpu([gpu]) is None


def test_candidate_count_uses_actual_logical_cuda_device(monkeypatch):
    monkeypatch.setattr(inference, "selected_nvidia_gpu", lambda: _gpu("8.6", memory=12000))
    assert inference._safe_candidate_count(4, backend="cu128") == 1
    monkeypatch.setattr(inference, "selected_nvidia_gpu", lambda: _gpu("8.6", memory=24576))
    assert inference._safe_candidate_count(8, backend="cu128") == 4
    assert inference._safe_candidate_count(4, backend="cu126") == 4
    assert inference._safe_candidate_count(4, backend="cpu") == 1


def test_environment_contract_hashes_irodori_and_inference_policy(tmp_path: Path):
    files = (
        "src/personavoice/hardware.py",
        "src/personavoice/irodori.py",
        "src/personavoice/inference.py",
        "src/personavoice/setup_env.py",
        "src/personavoice/runtime_dependencies.py",
        "src/personavoice/cuda_preflight.py",
        "src/personavoice/workers.py",
        "workers/asr/runtime_policy.py",
    )
    for relative in files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("v1", encoding="utf-8")
    recorded = environment_contract.environment_contract(tmp_path)
    assert "irodori_sha256" in recorded["runtime_policy"]
    assert "inference_sha256" in recorded["runtime_policy"]
    (tmp_path / "src/personavoice/irodori.py").write_text("v2", encoding="utf-8")
    assert not environment_contract.environment_contract_status(tmp_path, recorded)["ok"]


def test_cuda_preflight_exercises_runtime_fp16_and_bf16_kernels():
    assert "dtype=torch.float16" in cuda_preflight._TORCH_PROBE
    assert "dtype=torch.bfloat16" in cuda_preflight._TORCH_PROBE
    assert "capability >= (8, 0)" in cuda_preflight._TORCH_PROBE
