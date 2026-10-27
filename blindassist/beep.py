"""Non-blocking proximity beeper (winsound on Windows).

A detection loop must never block on audio, so the caller only ever calls
``set_rank()`` — an O(1) function that stores the nearest object's distance
rank. A dedicated daemon thread beeps at a pace governed by that rank, so the
beeping gets **faster as an object gets closer**:

==============  ============  ============
distance rank   interval      frequency
==============  ============  ============
very close (0)  450 ms        1500 Hz
near (1)        800 ms        1000 Hz
far (2)         1200 ms       700 Hz
none            silent        silent
==============  ============  ============

(The actual numbers live in :class:`blindassist.config.BeepConfig`.)

``winsound.Beep`` takes frequency + duration and blocks the *calling* thread
for the duration, which is why the beeping owns a thread of its own. Anything
unavailable (not Windows, beep disabled, no winsound) degrades to a harmless
null controller, mirroring how the speech layer degrades.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from .config import BeepConfig

# Sleep between polls while there is nothing to beep (no object / disabled).
_IDLE_POLL_S = 0.1


def pace(cfg: BeepConfig, rank: int | None) -> tuple[int, int] | None:
    """Map a distance rank to ``(frequency_hz, interval_ms)`` or None (silent)."""
    interval = cfg.interval_by_rank.get(rank) if rank is not None else None
    frequency = cfg.frequency_by_rank.get(rank) if rank is not None else None
    if interval is None or frequency is None:
        return None
    return int(frequency), max(1, int(interval))


class _NullController:
    """No-op stand-in for every "beeping not possible" situation."""

    available = False

    def start(self) -> None:
        return

    def set_enabled(self, enabled: bool) -> None:
        return

    def set_rank(self, rank: int | None) -> None:
        return

    def stop(self) -> None:
        return


class BeepController:
    """Owns the beeping thread. Cheap to drive, safe to share across threads."""

    available = True

    def __init__(
        self,
        cfg: BeepConfig,
        beeper: Callable[[int, int], None],
        enabled: bool = True,
    ) -> None:
        self.cfg = cfg
        self._beep = beeper
        self._enabled = bool(enabled)
        self._rank: int | None = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self.beeps: int = 0  # diagnostic counter (tests / shutdown log)
        self.last_error: str | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="blindassist-beep", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_evt.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    # -- drive ---------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def set_rank(self, rank: int | None) -> None:
        """Publish the nearest object's distance rank (None = nothing in view)."""
        with self._lock:
            self._rank = None if rank is None else int(rank)

    # -- worker --------------------------------------------------------------
    def _current(self) -> tuple[int, int] | None:
        with self._lock:
            if not self._enabled:
                return None
            return pace(self.cfg, self._rank)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            current = self._current()
            if current is None:
                if self._stop_evt.wait(_IDLE_POLL_S):
                    break
                continue
            frequency, interval_ms = current
            duration = max(10, int(self.cfg.duration_ms))
            try:
                self._beep(frequency, duration)
            except Exception as exc:  # no audio device / os error: stop quietly
                self.last_error = f"{type(exc).__name__}: {exc}"
                break
            self.beeps += 1
            rest = max(0.05, interval_ms / 1000.0 - duration / 1000.0)
            if self._stop_evt.wait(rest):
                break


def make_beep_controller(
    cfg: BeepConfig, headless: bool = False
) -> BeepController | _NullController:
    """Real beeper when possible, a silent null controller otherwise.

    Headless/automated runs are silent on purpose so a test machine never
    fills the room with beeps; the rank wiring is still exercised via the
    controller's public API by the unit tests.
    """
    if not cfg.enabled or headless:
        return _NullController()
    if not _winsound_available():
        return _NullController()
    import winsound

    return BeepController(cfg, winsound.Beep, enabled=True)


def _winsound_available() -> bool:
    try:
        import winsound  # noqa: F401

        return True
    except ImportError:
        return False


# Re-exported so callers (and tests) can construct with a custom beeper.
__all__ = ["BeepController", "_NullController", "make_beep_controller", "pace"]
