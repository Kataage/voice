from __future__ import annotations

import ast
from pathlib import Path

from personavoice import doctor, setup_env


ROOT = Path(__file__).resolve().parents[1]


def _tuple_assignment(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple)
            return tuple(str(item) for item in value)
    raise AssertionError(f"{name} was not found in {path}")


def _dict_string_keys(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            keys = []
            for key in node.value.keys:
                assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                keys.append(key.value)
            return tuple(keys)
    raise AssertionError(f"{name} was not found in {path}")


def test_lfm_materialization_contract_is_identical_across_layers():
    worker_files = _tuple_assignment(ROOT / "workers" / "lfm" / "worker.py", "REQUIRED_MODEL_FILES")
    assert worker_files == setup_env._LFM_REQUIRED_FILES
    assert worker_files == doctor._LFM_REQUIRED_FILES


def test_asr_materialization_contract_is_identical_across_layers():
    worker_files = _tuple_assignment(ROOT / "workers" / "asr" / "worker.py", "REQUIRED_MODEL_FILES")
    assert worker_files == setup_env._ASR_REQUIRED_FILES
    assert worker_files == doctor._ASR_REQUIRED_FILES


def test_pyannote_materialization_contract_is_identical_across_layers():
    worker_files = _tuple_assignment(
        ROOT / "workers" / "diarization" / "worker.py",
        "REQUIRED_MODEL_FILES",
    )
    assert worker_files == setup_env._PYANNOTE_REQUIRED_FILES
    assert worker_files == doctor._PYANNOTE_REQUIRED_FILES


def test_sense_static_contract_matches_worker_verified_assets():
    worker_files = _dict_string_keys(ROOT / "workers" / "sense" / "worker.py", "MODEL_ASSETS")
    assert worker_files == doctor._SENSE_REQUIRED_FILES
