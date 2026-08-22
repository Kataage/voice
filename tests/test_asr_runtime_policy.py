from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _policy_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "workers" / "asr" / "runtime_policy.py"
    spec = importlib.util.spec_from_file_location("personavoice_asr_runtime_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_auto_prefers_float16_when_supported():
    policy = _policy_module()
    assert (
        policy.choose_compute_type(
            "cuda",
            {"float32", "int8_float32", "float16"},
        )
        == "float16"
    )


def test_cuda_auto_uses_int8_float32_when_pascal_has_no_float16():
    policy = _policy_module()
    assert (
        policy.choose_compute_type(
            "cuda",
            {"float32", "int8_float32"},
        )
        == "int8_float32"
    )


def test_explicit_unsupported_compute_type_fails_closed():
    policy = _policy_module()
    with pytest.raises(ValueError, match="not supported"):
        policy.choose_compute_type("cuda", {"float32"}, "float16")
