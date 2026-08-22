from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tests" / "test_environment_generation.py"
text = path.read_text(encoding="utf-8")
old = '''    paths = PersonaPaths(tmp_path / "personas" / "alice")\n    paths.root.mkdir(parents=True)\n\n    with pytest.raises(RuntimeError, match="different dependency contract"):\n        inference.synthesize(tmp_path, paths, "hello")\n'''
new = '''    with pytest.raises(RuntimeError, match="different dependency contract"):\n        inference.configured_backend(tmp_path)\n'''
if old not in text:
    raise RuntimeError("stale inference fixture anchor not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
