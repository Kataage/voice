from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from personavoice import cuda_preflight, process


def _asr_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "workers" / "asr"
    provider = tmp_path / "workers" / "diarization" / ".venv"
    project.mkdir(parents=True)
    provider.mkdir(parents=True)
    return project, provider


def _materialize_fake_cuda_runtime(provider: Path) -> Path:
    if os.name == "nt":
        directory = provider / "Lib" / "site-packages" / "torch" / "lib"
        directory.mkdir(parents=True)
        for name in process._WINDOWS_ASR_CUDA_DLLS:
            (directory / name).write_bytes(b"dll")
        return directory

    if sys.platform.startswith("linux"):
        site = provider / "lib" / "python3.12" / "site-packages"
        cublas = site / "nvidia" / "cublas" / "lib"
        cudnn = site / "nvidia" / "cudnn" / "lib"
        cublas.mkdir(parents=True)
        cudnn.mkdir(parents=True)
        (cublas / "libcublas.so.12").write_bytes(b"so")
        (cudnn / "libcudnn.so.9").write_bytes(b"so")
        return cublas

    pytest.skip("ASR CUDA native runtime is supported on Windows/Linux")


def _asr_argv(project: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(project),
        "--no-sync",
        "python",
        "worker.py",
        "health",
        "--request",
        "request.json",
    ]


def test_subprocess_environment_forces_utf8_output_contract():
    env = process._command_environment(
        [sys.executable, "-c", "print('ok')"],
        cwd=None,
        env={"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp932"},
    )
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_python_worker_json_round_trips_japanese_even_from_legacy_codepage_env():
    result = process.run_json(
        [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'text': '日本語'}, ensure_ascii=False))",
        ],
        env={"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp932"},
    )
    assert result == {"text": "日本語"}


def test_asr_cuda_runtime_is_injected_without_leaking_to_other_workers(tmp_path: Path):
    project, provider = _asr_project(tmp_path)
    expected = _materialize_fake_cuda_runtime(provider)

    env = process._command_environment(
        _asr_argv(project),
        cwd=tmp_path,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    assert env[key].split(os.pathsep)[0] == str(expected)
    assert str(expected) in env["PERSONAVOICE_ASR_CUDA_RUNTIME_DIRS"].split(os.pathsep)

    other = tmp_path / "workers" / "sense"
    other.mkdir(parents=True)
    other_env = process._command_environment(
        _asr_argv(other),
        cwd=tmp_path,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert "PERSONAVOICE_ASR_CUDA_RUNTIME_DIRS" not in other_env


def test_asr_cuda_runtime_missing_fails_closed_before_worker_launch(tmp_path: Path):
    if os.name != "nt" and not sys.platform.startswith("linux"):
        pytest.skip("ASR CUDA native runtime is supported on Windows/Linux")
    project, _provider = _asr_project(tmp_path)
    with pytest.raises(process.CommandError, match="ASR CUDA runtime is incomplete"):
        process._command_environment(
            _asr_argv(project),
            cwd=tmp_path,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )


def test_cpu_asr_does_not_require_cuda_native_runtime(tmp_path: Path):
    project, _provider = _asr_project(tmp_path)
    env = process._command_environment(
        _asr_argv(project),
        cwd=tmp_path,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "PERSONAVOICE_ASR_CUDA_RUNTIME_DIRS" not in env


def test_asr_preflight_requires_native_runtime_proof_for_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cuda_preflight, "local_model_env", lambda _root: {})
    monkeypatch.setattr(
        cuda_preflight,
        "run_json",
        lambda *_args, **_kwargs: {
            "selected_device": "cuda",
            "selected_compute_type": "float32",
            "native_cuda_runtime": {"ok": False},
        },
    )
    with pytest.raises(RuntimeError, match="did not prove the native cuBLAS/cuDNN runtime"):
        cuda_preflight._asr_preflight(tmp_path)


def test_asr_preflight_accepts_proven_native_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    expected = {
        "selected_device": "cuda",
        "selected_compute_type": "float32",
        "native_cuda_runtime": {"ok": True, "libraries": ["cublas"]},
    }
    monkeypatch.setattr(cuda_preflight, "local_model_env", lambda _root: {})
    monkeypatch.setattr(cuda_preflight, "run_json", lambda *_args, **_kwargs: expected)
    assert cuda_preflight._asr_preflight(tmp_path) == expected
