"""Ensure the project root is importable when running pytest from anywhere."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Point settings.json at a per-test temp file.

    App-level tests must never read the developer's real settings.json (a
    stored confidence would silently change test expectations), and they must
    never write one either. Default app config tests are hermetic because of
    this fixture; the settings module's own tests pass explicit paths where
    they need something special.
    """
    import blindassist.settings as settings_module

    monkeypatch.setattr(settings_module, "SETTINGS_PATH", tmp_path / "settings.json")
    # Same idea for the session CSV log: full-app tests must not write rows
    # into the project's real logs/session_YYYYMMDD.csv.
    import blindassist.sessionlog as sessionlog_module

    monkeypatch.setattr(sessionlog_module, "DEFAULT_ROOT", tmp_path)
    # ...and for the file log: several QA tests *deliberately* trigger errors
    # (no camera, corrupt frames, missing video). Their tracebacks must land
    # in a temp file, never in the production logs/blindassist.log - that file
    # is expected to hold only clean live-run entries (Manual_Check criterion).
    import blindassist.app as app_mod

    monkeypatch.setattr(app_mod, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(app_mod, "LOG_FILE", tmp_path / "logs" / "blindassist.log")
    yield
