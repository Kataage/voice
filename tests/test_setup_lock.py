from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.setup_lock import SetupLockError, setup_lock


def test_setup_lock_rejects_concurrent_session_and_releases_cleanly(tmp_path: Path) -> None:
    with setup_lock(tmp_path) as path:
        assert path == tmp_path / ".runtime" / "setup.lock"
        assert path.is_file()
        with pytest.raises(SetupLockError, match="already running"), setup_lock(tmp_path):
            pass

    # The lock file intentionally persists, but the OS lock must be released so
    # a later setup can recover immediately after normal exit or process death.
    with setup_lock(tmp_path) as path:
        assert path.is_file()
