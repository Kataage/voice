from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from personavoice import process


def _request(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "personas" / "alice" / "cache" / "asr" / ".checkpoints"
    checkpoint.mkdir(parents=True)
    request_dir = tmp_path / ".runtime" / "requests"
    request_dir.mkdir(parents=True)
    request = request_dir / "asr-test.json"
    request.write_text(
        json.dumps({"items": [], "checkpoint_dir": str(checkpoint.resolve())}),
        encoding="utf-8",
    )
    return request, checkpoint / "progress.json"


def test_run_json_terminates_asr_worker_after_progress_stalls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request, progress = _request(tmp_path)
    script = tmp_path / "stalled_worker.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import time
            from pathlib import Path

            progress = Path({str(progress)!r})
            progress.write_text(json.dumps({{
                "schema": 1,
                "worker": "asr",
                "command": "batch_transcribe",
                "phase": "transcribe",
                "completed": 0,
                "total": 1,
                "failed": 0,
                "current_id": "stuck-source",
                "state": "running",
                "device": "cuda",
                "compute_type": "int8_float32",
            }}), encoding="utf-8")
            time.sleep(30)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(process, "ASR_STALL_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr(process, "ASR_PROGRESS_POLL_SECONDS", 0.05)

    with pytest.raises(process.CommandError, match="stuck-source") as exc_info:
        process.run_json(
            [sys.executable, script, "batch_transcribe", "--request", request],
            cwd=tmp_path,
        )
    assert "terminated" in str(exc_info.value)
    assert "resume" in str(exc_info.value)


def test_run_json_accepts_regular_asr_heartbeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request, progress = _request(tmp_path)
    script = tmp_path / "progressing_worker.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import time
            from pathlib import Path

            progress = Path({str(progress)!r})
            for index in range(6):
                progress.write_text(json.dumps({{
                    "schema": 1,
                    "worker": "asr",
                    "command": "batch_transcribe",
                    "phase": "transcribe",
                    "completed": 0,
                    "total": 1,
                    "failed": 0,
                    "current_id": "active-source",
                    "state": "running",
                    "current_processed_seconds": index * 10.0,
                }}), encoding="utf-8")
                time.sleep(0.06)
            print(json.dumps({{"results": []}}))
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(process, "ASR_STALL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(process, "ASR_PROGRESS_POLL_SECONDS", 0.03)

    assert process.run_json(
        [sys.executable, script, "batch_transcribe", "--request", request],
        cwd=tmp_path,
    ) == {"results": []}


def test_asr_watchdog_rejects_checkpoint_path_outside_repo(tmp_path: Path):
    request_dir = tmp_path / ".runtime" / "requests"
    request_dir.mkdir(parents=True)
    request = request_dir / "asr-test.json"
    request.write_text(
        json.dumps({"checkpoint_dir": str((tmp_path.parent / "outside").resolve())}),
        encoding="utf-8",
    )
    argv = [sys.executable, "worker.py", "batch_transcribe", "--request", str(request)]
    assert process._asr_progress_path(argv, tmp_path) is None
