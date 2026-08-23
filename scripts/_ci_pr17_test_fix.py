from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise RuntimeError(
            f"Expected exactly {count} patch anchors in {path}, found {found}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


# Provenance tests must stub the raw resolver, not the public ffmpeg_runtime()
# wrapper. Stubbing the wrapper bypasses the very provenance guard under test.
patch(
    "tests/test_ffmpeg_setup_provenance.py",
    'monkeypatch.setattr(runtime_dependencies, "ffmpeg_runtime", lambda:',
    'monkeypatch.setattr(runtime_dependencies, "_discover_ffmpeg_runtime", lambda:',
    count=4,
)

# These tests exercise cache/env shaping and synthesis argument forwarding, not
# FFmpeg discovery. Isolate only the new runtime prerequisite locally.
patch(
    "tests/test_core.py",
    "from personavoice import inference, setup_env\n",
    "from personavoice import inference, setup_env, workers\n",
)
patch(
    "tests/test_core.py",
    "def test_huggingface_cache_layout_is_consistent(tmp_path: Path):\n"
    "    env = local_model_env(tmp_path, offline=True)\n",
    "def test_huggingface_cache_layout_is_consistent(tmp_path: Path, monkeypatch):\n"
    "    monkeypatch.setattr(workers, \"ffmpeg_environment\", lambda: {})\n"
    "    env = local_model_env(tmp_path, offline=True)\n",
)
patch(
    "tests/test_core.py",
    '    monkeypatch.setattr(inference, "selected_nvidia_gpu", lambda: None)\n',
    '    monkeypatch.setattr(inference, "selected_nvidia_gpu", lambda: None)\n'
    '    monkeypatch.setattr(inference, "local_model_env", lambda *_args, **_kwargs: {})\n',
)

# GPU ordering test is independent of FFmpeg availability.
patch(
    "tests/test_gpu_runtime_contract.py",
    "from personavoice import hardware, setup_env\n",
    "from personavoice import hardware, setup_env, workers\n",
)
patch(
    "tests/test_gpu_runtime_contract.py",
    "def test_local_model_env_forces_deterministic_cuda_order(tmp_path: Path):\n"
    "    env = local_model_env(tmp_path)\n",
    "def test_local_model_env_forces_deterministic_cuda_order(tmp_path: Path, monkeypatch):\n"
    "    monkeypatch.setattr(workers, \"ffmpeg_environment\", lambda: {})\n"
    "    env = local_model_env(tmp_path)\n",
)

# Irodori checkpoint integrity tests should reach the checksum/materialization
# logic they are intended to test, without depending on a host FFmpeg install.
patch(
    "tests/test_irodori_runtime_integrity.py",
    "def test_irodori_runtime_rejects_corrupt_pinned_assets(tmp_path: Path, monkeypatch):\n",
    "def test_irodori_runtime_rejects_corrupt_pinned_assets(tmp_path: Path, monkeypatch):\n"
    "    monkeypatch.setattr(irodori, \"local_model_env\", lambda *_args, **_kwargs: {})\n",
)
patch(
    "tests/test_irodori_runtime_integrity.py",
    "def test_online_base_materialization_replaces_corruption_and_rehashes(\n"
    "    tmp_path: Path,\n"
    "    monkeypatch,\n"
    "):\n",
    "def test_online_base_materialization_replaces_corruption_and_rehashes(\n"
    "    tmp_path: Path,\n"
    "    monkeypatch,\n"
    "):\n"
    "    cache = tmp_path / \"hf-cache\"\n"
    "    cache.mkdir()\n"
    "    monkeypatch.setattr(\n"
    "        irodori,\n"
    "        \"local_model_env\",\n"
    "        lambda *_args, **_kwargs: {\"HUGGINGFACE_HUB_CACHE\": str(cache)},\n"
    "    )\n",
)

# Manifest test already isolates provenance; isolate the subprocess environment
# construction too so it remains focused on codec/backend argument forwarding.
patch(
    "tests/test_model_assets.py",
    "def test_prepare_manifest_passes_local_codec_and_recorded_cpu_backend(\n"
    "    tmp_path: Path,\n"
    "    monkeypatch,\n"
    "):\n",
    "def test_prepare_manifest_passes_local_codec_and_recorded_cpu_backend(\n"
    "    tmp_path: Path,\n"
    "    monkeypatch,\n"
    "):\n"
    "    monkeypatch.setattr(irodori, \"local_model_env\", lambda *_args, **_kwargs: {})\n",
)

# Worker result-schema test must reach validate_worker_response instead of
# failing earlier on an unrelated FFmpeg prerequisite.
patch(
    "tests/test_worker_result_contracts.py",
    "def test_worker_call_rejects_invalid_subprocess_result_before_return(tmp_path: Path, monkeypatch):\n",
    "def test_worker_call_rejects_invalid_subprocess_result_before_return(tmp_path: Path, monkeypatch):\n"
    "    monkeypatch.setattr(workers, \"ffmpeg_environment\", lambda: {})\n",
)

print("PR17 test-contract fixes applied")
