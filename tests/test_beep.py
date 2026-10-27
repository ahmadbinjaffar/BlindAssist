"""Proximity beep: pace mapping, controller thread, null fallbacks."""

from __future__ import annotations

import time

from blindassist.beep import BeepController, _NullController, make_beep_controller, pace
from blindassist.config import BeepConfig


def fast_cfg() -> BeepConfig:
    """Small intervals so a thread test finishes in well under a second."""
    return BeepConfig(
        duration_ms=10,
        interval_by_rank={0: 30, 1: 60, 2: 100},
        frequency_by_rank={0: 1200, 1: 900, 2: 600},
    )


# ---------------------------------------------------------------------------
# pace() - pure mapping
# ---------------------------------------------------------------------------
def test_pace_for_each_rank():
    cfg = BeepConfig()
    assert pace(cfg, 0) == (1500, 450)
    assert pace(cfg, 1) == (1000, 800)
    assert pace(cfg, 2) == (700, 1200)


def test_pace_none_when_silent_or_no_object():
    cfg = BeepConfig()
    assert pace(cfg, None) is None
    # A rank missing from the tables means "stay silent for that rank".
    assert pace(BeepConfig(interval_by_rank={}, frequency_by_rank={}), 0) is None


def test_higher_rank_slowest():
    assert (
        pace(BeepConfig(), 0)[1] < pace(BeepConfig(), 1)[1] < pace(BeepConfig(), 2)[1]
    )


# ---------------------------------------------------------------------------
# Controller thread
# ---------------------------------------------------------------------------
def test_controller_beeps_and_stops():
    cfg = fast_cfg()
    calls: list[tuple[int, int]] = []
    ctl = BeepController(cfg, beeper=lambda f, d: calls.append((f, d)))
    ctl.start()
    ctl.set_rank(0)
    deadline = time.time() + 3.0
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert len(calls) >= 3  # beeping happened on its own thread
    assert all(f == 1200 and d == 10 for f, d in calls)
    ctl.set_rank(None)  # nothing in view -> silent
    n = len(calls)
    time.sleep(0.15)
    assert len(calls) == n
    ctl.stop()
    assert not (ctl._thread and ctl._thread.is_alive())


def test_set_enabled_pauses_without_losing_the_rank():
    calls: list[tuple[int, int]] = []
    ctl = BeepController(fast_cfg(), beeper=lambda f, d: calls.append((f, d)))
    ctl.start()
    ctl.set_rank(1)
    deadline = time.time() + 3.0
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.01)
    ctl.set_enabled(False)
    n = len(calls)
    time.sleep(0.2)
    assert len(calls) == n  # muted while disabled...
    before = len(calls)
    ctl.set_enabled(True)  # ...but resumes at the same proximity rank
    deadline = time.time() + 3.0
    while len(calls) <= before and time.time() < deadline:
        time.sleep(0.01)
    assert len(calls) > before
    ctl.stop()


def test_beeping_stops_after_stop_called():
    calls: list[tuple[int, int]] = []
    ctl = BeepController(fast_cfg(), beeper=lambda f, d: calls.append((f, d)))
    ctl.start()
    ctl.set_rank(0)
    time.sleep(0.15)
    ctl.stop()
    ctl.set_rank(0)  # ignored: the worker is gone
    n = len(calls)
    time.sleep(0.2)
    assert len(calls) == n
    assert ctl.beeps >= 1


def test_no_object_never_beeps():
    calls: list[tuple[int, int]] = []
    ctl = BeepController(fast_cfg(), beeper=lambda f, d: calls.append((f, d)))
    ctl.start()
    time.sleep(0.25)
    assert calls == []
    ctl.stop()


def test_beeper_failure_sets_error_and_exits():
    def broken(freq: int, dur: int) -> None:
        raise OSError("no audio device")

    ctl = BeepController(fast_cfg(), beeper=broken)
    ctl.start()
    ctl.set_rank(2)
    time.sleep(0.2)
    assert ctl.last_error is not None
    assert "OSError" in ctl.last_error
    ctl.stop()


def test_make_controller_disabled_or_headless_is_null():
    assert isinstance(make_beep_controller(BeepConfig(enabled=False)), _NullController)
    assert isinstance(
        make_beep_controller(BeepConfig(), headless=True), _NullController
    )


def test_null_controller_is_harmless():
    null = _NullController()
    null.start()
    null.set_rank(0)
    null.set_enabled(True)
    null.stop()
    assert null.available is False


def test_start_is_idempotent():
    calls: list[tuple[int, int]] = []
    ctl = BeepController(fast_cfg(), beeper=lambda f, d: calls.append((f, d)))
    ctl.start()
    thread = ctl._thread
    ctl.start()  # second start must not spawn a second worker
    assert ctl._thread is thread
    ctl.stop()


# ---------------------------------------------------------------------------
# Real Windows beeper path (short and quiet, but it proves winsound works)
# ---------------------------------------------------------------------------
def test_real_winsound_beeps_without_error():
    try:
        import winsound
    except ImportError:
        import pytest

        pytest.skip("winsound not available on this platform")
    cfg = BeepConfig(
        duration_ms=40, interval_by_rank={0: 80}, frequency_by_rank={0: 1000}
    )
    real_beeps: list[tuple[int, int]] = []

    def counting_beep(freq: int, dur: int) -> None:
        real_beeps.append((freq, dur))
        winsound.Beep(freq, dur)  # the actual system call

    ctl = BeepController(cfg, beeper=counting_beep)
    ctl.start()
    ctl.set_rank(0)
    deadline = time.time() + 3.0
    while len(real_beeps) < 2 and time.time() < deadline:
        time.sleep(0.01)
    ctl.stop()
    assert len(real_beeps) >= 2  # two real audible beeps, ~0.08 s total
    assert ctl.last_error is None
