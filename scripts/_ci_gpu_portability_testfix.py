from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tests" / "test_core.py"
text = path.read_text(encoding="utf-8")
old = '    monkeypatch.setattr(inference, "nvidia_gpus", lambda: [])\n'
new = '    monkeypatch.setattr(inference, "selected_nvidia_gpu", lambda: None)\n'
if text.count(old) != 1:
    raise RuntimeError(f"Expected exactly one legacy inference GPU fixture, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
print("GPU portability test fixture aligned")
