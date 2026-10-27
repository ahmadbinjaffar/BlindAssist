"""Daily CSV log of every announcement (improvement feature 9).

Each spoken (or muted) announcement is appended as one row to
``logs/session_YYYYMMDD.csv``::

    timestamp,kind,message
    2026-10-25T14:03:11,speak,Chair on your left.

* ``timestamp`` - local time, ISO 8601, one-second resolution.
* ``kind``      - ``speak`` / ``muted`` / ``urgent`` / ``welcome`` /
                  ``describe`` so a reviewer can tell how a line was produced.
* ``message``   - the exact sentence (CSV-quoted when it contains commas).

Design rules:

* Appends across runs on the same day; a new day opens a new file.
* Writes never raise and never block the assistant for long: the speech +
  preview are the primary outputs, the log is a record for later review.
* Thread-safe: announcements originate from both the UI and the worker.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

from .config import SessionLogConfig

# Tests monkeypatch this to a tmp directory (same pattern as SETTINGS_PATH),
# so a pytest run never writes into the project's real logs/ folder.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent

HEADER = ("timestamp", "kind", "message")


class SessionLog:
    """Append announcement rows to ``<directory>/session_YYYYMMDD.csv``."""

    def __init__(self, cfg: SessionLogConfig, root: Path | None = None) -> None:
        self.cfg = cfg
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._day: str | None = None
        self.rows = 0  # rows written this session (for the exit summary)
        self.failed = False  # sticky: a broken log must not be retried forever
        self.path: Path | None = None  # file currently open (None = nothing yet)

    # -- internals ---------------------------------------------------------
    def _directory(self) -> Path:
        directory = Path(self.cfg.directory)
        return directory if directory.is_absolute() else self.root / directory

    def _ensure(self, day: str) -> bool:
        """Open the CSV for ``day`` (rolling over from a previous day)."""
        if not self.cfg.enabled or self.failed:
            return False
        if self._writer is not None and self._day == day:
            return True
        try:
            self._close_handle()
            directory = self._directory()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{self.cfg.prefix}_{day}.csv"
            is_new = not path.exists()
            handle = path.open("a", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(HEADER)
            self._file, self._writer, self._day, self.path = handle, writer, day, path
            return True
        except Exception:
            # A full disk / locked folder / nonsense path: log once, disable
            # the log for the rest of the run, keep the assistant alive.
            self.failed = True
            return False

    def _close_handle(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self._writer = None
        self._day = None

    # -- public API --------------------------------------------------------
    def write(self, kind: str, message: str) -> bool:
        """Append one announcement row. Returns False when nothing was logged."""
        if not self.cfg.enabled:
            return False
        now = datetime.now()
        with self._lock:
            if not self._ensure(now.strftime("%Y%m%d")):
                return False
            try:
                self._writer.writerow(
                    [now.isoformat(timespec="seconds"), str(kind), str(message)]
                )
                self._file.flush()
                self.rows += 1
                return True
            except Exception:
                self.failed = True
                return False

    def close(self) -> None:
        """Flush and close the current file (safe to call twice)."""
        with self._lock:
            self._close_handle()
