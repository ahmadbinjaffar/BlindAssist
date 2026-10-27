"""Tests for the daily session CSV log (improvement feature 9).

The conftest autouse fixture points ``sessionlog.DEFAULT_ROOT`` at a tmp
directory, so even app-level tests never write into the project's logs/.
These tests pass an explicit ``root`` where they need one.
"""

from __future__ import annotations

import csv
from datetime import datetime
from types import SimpleNamespace

from blindassist.app import BlindAssistApp, build_arg_parser
from blindassist.config import SessionLogConfig
from blindassist.sessionlog import HEADER, SessionLog


def make_log(tmp_path, **kwargs) -> SessionLog:
    return SessionLog(SessionLogConfig(**kwargs), root=tmp_path)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_write_creates_daily_file_with_header_and_row(tmp_path):
    log = make_log(tmp_path)
    assert log.write("speak", "Chair on your left.") is True
    log.close()

    day = datetime.now().strftime("%Y%m%d")
    path = tmp_path / "logs" / f"session_{day}.csv"
    assert path.exists()
    rows = read_rows(path)
    assert rows[0] == list(HEADER)
    assert len(rows) == 2
    timestamp, kind, message = rows[1]
    datetime.fromisoformat(timestamp)  # a real ISO timestamp, parseable back
    assert kind == "speak"
    assert message == "Chair on your left."
    assert log.rows == 1


def test_message_with_commas_and_quotes_is_csv_safe(tmp_path):
    log = make_log(tmp_path)
    tricky = 'Stop! "Person" ahead, near the door.'
    log.write("urgent", tricky)
    log.close()

    rows = read_rows(log.path)
    assert rows[1][2] == tricky  # round-trips exactly despite , and "


def test_same_day_appends_without_a_second_header(tmp_path):
    log = make_log(tmp_path)
    log.write("speak", "One.")
    log.close()
    log2 = make_log(tmp_path)  # a second run on the same day
    log2.write("speak", "Two.")
    log2.close()

    rows = read_rows(log2.path)
    assert rows[0] == list(HEADER)  # header appears exactly once
    assert [r[2] for r in rows[1:]] == ["One.", "Two."]


def test_disabled_log_writes_nothing(tmp_path):
    log = make_log(tmp_path, enabled=False)
    assert log.write("speak", "Nope.") is False
    log.close()
    assert log.rows == 0
    assert not (tmp_path / "logs").exists()


def test_rollover_opens_a_new_file_for_a_new_day(tmp_path):
    log = make_log(tmp_path)
    assert log._ensure("20260101") is True
    first = log.path
    assert log._ensure("20260101") is True  # same day: same handle
    assert log.path is first
    assert log._ensure("20260102") is True  # next day: rolled over
    assert log.path != first
    assert first.exists() and log.path.exists()
    log.close()


def test_unwritable_directory_disables_the_log_without_raising(tmp_path):
    # A *file* where the directory should be: mkdir must fail.
    blocker = tmp_path / "logs"
    blocker.write_text("not a directory", encoding="utf-8")
    log = make_log(tmp_path)
    assert log.write("speak", "Never written.") is False  # no exception
    assert log.failed is True
    # Sticky: further writes keep failing quietly.
    assert log.write("speak", "Also never.") is False
    log.close()


def test_close_is_idempotent(tmp_path):
    log = make_log(tmp_path)
    log.write("speak", "One.")
    log.close()
    log.close()  # second close must not raise
    assert log._file is None


# -- integration with the app ----------------------------------------------
def test_announcements_land_in_the_session_log(tmp_path, monkeypatch):
    monkeypatch.setattr("blindassist.sessionlog.DEFAULT_ROOT", tmp_path / "approot")
    app = BlindAssistApp(build_arg_parser().parse_args(["--no-speech"]))
    app.speaker = SimpleNamespace(speak=lambda *a, **k: None)  # pipeline not built
    app._announce("Person on your right.")
    app.muted = True
    app._announce("Chair ahead.")
    app._announce("Stop! Car ahead.", urgent=True)
    app.sessionlog.close()

    rows = read_rows(app.sessionlog.path)
    kinds = [r[1] for r in rows[1:]]
    messages = [r[2] for r in rows[1:]]
    assert kinds == ["speak", "muted", "muted"]
    assert messages == ["Person on your right.", "Chair ahead.", "Stop! Car ahead."]
