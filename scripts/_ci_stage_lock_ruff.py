from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tests" / "test_stage_runtime_lock.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    '    with setup_lock(tmp_path):\n        with pytest.raises(SetupLockError, match="already running"):\n            with setup_lock(tmp_path):\n                pass\n',
    '    with setup_lock(tmp_path), pytest.raises(SetupLockError, match="already running"):\n        with setup_lock(tmp_path):\n            pass\n',
)
text = text.replace(
    '        with pytest.raises(StageLockError, match="already running"):\n            with second.running("prepare", "second"):\n                pass\n',
    '        with pytest.raises(StageLockError, match="already running"), second.running(\n            "prepare", "second"\n        ):\n            pass\n',
)
path.write_text(text, encoding="utf-8", newline="\n")
