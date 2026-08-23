from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise RuntimeError(
            f"Expected exactly {count} patch anchor(s) in {path}, found {found}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


patch(
    "src/personavoice/irodori.py",
    "from personavoice.hardware import safe_batch_profile\n",
    "from personavoice.hardware import irodori_training_precision, safe_batch_profile\n",
)
patch(
    "src/personavoice/irodori.py",
    '''    if backend == "cu126":\n        train_cfg["precision"] = "fp32"\n        train_cfg["allow_tf32"] = False\n    elif backend == "cpu":\n        train_cfg["dataloader_cuda_prefetch"] = False\n        train_cfg["precision"] = "fp32"\n        train_cfg["allow_tf32"] = False\n''',
    '''    if backend in {"cu126", "cu128"}:\n        precision_policy = irodori_training_precision(backend)\n        train_cfg["precision"] = str(precision_policy["precision"])\n        train_cfg["allow_tf32"] = bool(precision_policy["allow_tf32"])\n    elif backend == "cpu":\n        train_cfg["dataloader_cuda_prefetch"] = False\n        train_cfg["precision"] = "fp32"\n        train_cfg["allow_tf32"] = False\n''',
)

patch(
    "src/personavoice/inference.py",
    "from personavoice.hardware import nvidia_gpus\n",
    "from personavoice.hardware import selected_nvidia_gpu\n",
)
patch(
    "src/personavoice/inference.py",
    '''def _nonempty_file(path: Path) -> bool:\n    try:\n        return path.is_file() and path.stat().st_size > 0\n    except OSError:\n        return False\n\n\ndef _caption''',
    '''def _nonempty_file(path: Path) -> bool:\n    try:\n        return path.is_file() and path.stat().st_size > 0\n    except OSError:\n        return False\n\n\ndef _safe_candidate_count(requested: int, *, backend: str) -> int:\n    """Clamp batched Irodori candidates to the actual logical CUDA device 0."""\n\n    gpu = selected_nvidia_gpu() if backend in {"cu126", "cu128"} else None\n    if gpu is None or gpu.total_mib < 16000:\n        return 1\n    return min(requested, 4)\n\n\ndef _caption''',
)
patch(
    "src/personavoice/inference.py",
    '''    gpus = nvidia_gpus() if backend == "cu128" else []\n    requested = 1 if not gpus or max(gpu.total_mib for gpu in gpus) < 16000 else min(requested, 4)\n''',
    '''    requested = _safe_candidate_count(requested, backend=backend)\n''',
)

patch(
    "src/personavoice/environment_contract.py",
    '''        "runtime_policy": {\n            "hardware_sha256": _sha256(repo_root / "src" / "personavoice" / "hardware.py"),\n            "setup_sha256": _sha256(repo_root / "src" / "personavoice" / "setup_env.py"),\n''',
    '''        "runtime_policy": {\n            "hardware_sha256": _sha256(repo_root / "src" / "personavoice" / "hardware.py"),\n            "irodori_sha256": _sha256(repo_root / "src" / "personavoice" / "irodori.py"),\n            "inference_sha256": _sha256(repo_root / "src" / "personavoice" / "inference.py"),\n            "setup_sha256": _sha256(repo_root / "src" / "personavoice" / "setup_env.py"),\n''',
)

patch(
    "src/personavoice/cuda_preflight.py",
    '''_TORCH_PROBE = r\'\'\'\nimport json\nimport torch\n\nif not torch.cuda.is_available():\n    raise RuntimeError("torch.cuda.is_available() is false in the synced CUDA environment")\ncapability = torch.cuda.get_device_capability(0)\nvalues = torch.arange(1, 1025, dtype=torch.float32, device="cuda")\nresult = (values.square() + 1.0).sum()\nif not torch.isfinite(result):\n    raise RuntimeError("CUDA smoke kernel produced a non-finite result")\ntorch.cuda.synchronize()\nprint(json.dumps({\n    "torch": str(torch.__version__),\n    "device_name": str(torch.cuda.get_device_name(0)),\n    "compute_capability": [int(capability[0]), int(capability[1])],\n    "arch_list": list(torch.cuda.get_arch_list()),\n    "smoke_value": float(result.item()),\n}))\n\'\'\'\n''',
    '''_TORCH_PROBE = r\'\'\'\nimport json\nimport torch\n\nif not torch.cuda.is_available():\n    raise RuntimeError("torch.cuda.is_available() is false in the synced CUDA environment")\ncapability = torch.cuda.get_device_capability(0)\n\ndef finite(value, label):\n    if not bool(torch.isfinite(value).all().item()):\n        raise RuntimeError(f"{label} CUDA smoke kernel produced a non-finite result")\n\nvalues = torch.arange(1, 1025, dtype=torch.float32, device="cuda")\nfp32_result = (values.square() + 1.0).sum()\nfinite(fp32_result, "float32")\n\nfp16_left = torch.randn((16, 16), dtype=torch.float16, device="cuda")\nfp16_right = torch.randn((16, 16), dtype=torch.float16, device="cuda")\nfp16_result = (fp16_left @ fp16_right).float().abs().mean()\nfinite(fp16_result, "float16")\n\nbf16_value = None\nif capability >= (8, 0):\n    bf16_left = torch.randn((16, 16), dtype=torch.bfloat16, device="cuda")\n    bf16_right = torch.randn((16, 16), dtype=torch.bfloat16, device="cuda")\n    bf16_result = (bf16_left @ bf16_right).float().abs().mean()\n    finite(bf16_result, "bfloat16")\n    bf16_value = float(bf16_result.item())\n\ntorch.cuda.synchronize()\nprint(json.dumps({\n    "torch": str(torch.__version__),\n    "device_name": str(torch.cuda.get_device_name(0)),\n    "compute_capability": [int(capability[0]), int(capability[1])],\n    "arch_list": list(torch.cuda.get_arch_list()),\n    "fp32_smoke": float(fp32_result.item()),\n    "fp16_smoke": float(fp16_result.item()),\n    "bf16_smoke": bf16_value,\n}))\n\'\'\'\n''',
)

# Focused regression coverage for the hardware gaps found after PR #14.
test = ROOT / "tests" / "test_gpu_portability_followup.py"
test.write_text(
    '''from __future__ import annotations\n\nfrom pathlib import Path\n\nfrom personavoice import cuda_preflight, environment_contract, hardware, inference, irodori\nfrom personavoice.model_assets import IRODORI_TEXT_ENCODER_ID, IRODORI_TEXT_ENCODER_REVISION\n\n\ndef _gpu(capability: str, *, memory: int = 16384, uuid: str = "GPU-aaaaaaaa") -> hardware.GpuInfo:\n    return hardware.GpuInfo(\n        index=0,\n        name=f"GPU-{capability}",\n        total_mib=memory,\n        free_mib=max(1, memory - 1024),\n        compute_capability=capability,\n        uuid=uuid,\n        pci_bus_id="00000000:01:00.0",\n        driver_version="999.1",\n    )\n\n\ndef test_irodori_precision_policy_covers_turing_ampere_and_unknown(monkeypatch):\n    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")\n    assert hardware.irodori_training_precision("cu128", gpu=_gpu("7.5")) == {\n        "precision": "fp32",\n        "allow_tf32": False,\n    }\n    assert hardware.irodori_training_precision("cu128", gpu=_gpu("8.6")) == {\n        "precision": "bf16",\n        "allow_tf32": True,\n    }\n    assert hardware.irodori_training_precision("cu126", gpu=_gpu("8.6")) == {\n        "precision": "fp32",\n        "allow_tf32": False,\n    }\n    assert hardware.irodori_training_precision("cu128", gpu=_gpu("")) == {\n        "precision": "fp32",\n        "allow_tf32": False,\n    }\n\n\ndef test_irodori_config_uses_turing_safe_precision(tmp_path: Path, monkeypatch):\n    import yaml\n\n    source = tmp_path / "source.yaml"\n    destination = tmp_path / "patched.yaml"\n    source.write_text(\n        yaml.safe_dump(\n            {\n                "model": {\n                    "text_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,\n                    "text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,\n                    "caption_tokenizer_repo": IRODORI_TEXT_ENCODER_ID,\n                },\n                "train": {\n                    "precision": "bf16",\n                    "allow_tf32": True,\n                    "dataloader_cuda_prefetch": True,\n                },\n            },\n            sort_keys=False,\n        ),\n        encoding="utf-8",\n    )\n    monkeypatch.setattr(\n        irodori,\n        "safe_batch_profile",\n        lambda *, backend: {\n            "batch_size": 1,\n            "gradient_accumulation_steps": 8,\n            "num_workers": 2,\n            "gradient_checkpointing": True,\n        },\n    )\n    monkeypatch.setattr(\n        irodori,\n        "irodori_training_precision",\n        lambda backend: hardware.irodori_training_precision(backend, gpu=_gpu("7.5")),\n    )\n    irodori._patched_config(source, destination, max_steps=100, backend="cu128")\n    value = yaml.safe_load(destination.read_text(encoding="utf-8"))\n    assert value["train"]["precision"] == "fp32"\n    assert value["train"]["allow_tf32"] is False\n\n\ndef test_uuid_selector_accepts_prefix_but_rejects_appended_garbage(monkeypatch):\n    gpu = _gpu("8.6", uuid="GPU-aaaaaaaa-bbbb")\n    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa")\n    assert hardware.selected_nvidia_gpu([gpu]) == gpu\n    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaaaaaa-bbbb-extra")\n    assert hardware.selected_nvidia_gpu([gpu]) is None\n\n\ndef test_candidate_count_uses_actual_logical_cuda_device(monkeypatch):\n    monkeypatch.setattr(inference, "selected_nvidia_gpu", lambda: _gpu("8.6", memory=12000))\n    assert inference._safe_candidate_count(4, backend="cu128") == 1\n    monkeypatch.setattr(inference, "selected_nvidia_gpu", lambda: _gpu("8.6", memory=24576))\n    assert inference._safe_candidate_count(8, backend="cu128") == 4\n    assert inference._safe_candidate_count(4, backend="cu126") == 4\n    assert inference._safe_candidate_count(4, backend="cpu") == 1\n\n\ndef test_environment_contract_hashes_irodori_and_inference_policy(tmp_path: Path):\n    files = (\n        "src/personavoice/hardware.py",\n        "src/personavoice/irodori.py",\n        "src/personavoice/inference.py",\n        "src/personavoice/setup_env.py",\n        "src/personavoice/runtime_dependencies.py",\n        "src/personavoice/cuda_preflight.py",\n        "src/personavoice/workers.py",\n        "workers/asr/runtime_policy.py",\n    )\n    for relative in files:\n        path = tmp_path / relative\n        path.parent.mkdir(parents=True, exist_ok=True)\n        path.write_text("v1", encoding="utf-8")\n    recorded = environment_contract.environment_contract(tmp_path)\n    assert "irodori_sha256" in recorded["runtime_policy"]\n    assert "inference_sha256" in recorded["runtime_policy"]\n    (tmp_path / "src/personavoice/irodori.py").write_text("v2", encoding="utf-8")\n    assert not environment_contract.environment_contract_status(tmp_path, recorded)["ok"]\n\n\ndef test_cuda_preflight_exercises_runtime_fp16_and_bf16_kernels():\n    assert "dtype=torch.float16" in cuda_preflight._TORCH_PROBE\n    assert "dtype=torch.bfloat16" in cuda_preflight._TORCH_PROBE\n    assert "capability >= (8, 0)" in cuda_preflight._TORCH_PROBE\n''',
    encoding="utf-8",
    newline="\n",
)

print("GPU portability follow-up patches applied")
