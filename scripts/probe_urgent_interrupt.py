"""Probe the urgent-interrupt path - the COM threading suspect.

speech.py's ``speak(..., urgent=True)`` calls ``engine.stop()`` from the CALLING
thread while the worker thread is inside ``runAndWait()``.  pyttsx3's SAPI5
engine is a COM object created inside the worker, so that is a cross-thread
call.  The worker swallows every exception (``except Exception: pass``), so any
resulting failure is invisible.

This script wraps the engine in a proxy that RECORDS every exception, then
measures the real output peak for:

  1. a plain utterance                      (control)
  2. an urgent interrupt mid-utterance
  3. a plain utterance AFTER the interrupt
  4. 8 more urgent interrupts in a row      (the app's normal danger traffic)
  5. one final plain utterance              (does speech still come out?)

Run:  .venv\\Scripts\\python.exe scripts\\probe_urgent_interrupt.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_peak_probe import PeakSampler  # noqa: E402


class RecordingProxy:
    """Forwards everything to the real pyttsx3 engine, logging any exception."""

    def __init__(self, real) -> None:
        self._real = real
        self.errors: list[str] = []

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except Exception as exc:
                self.errors.append(f"{name}{args!r}: {type(exc).__name__}: {exc}")
                raise

        return wrapper


def peak(label: str, fn, settle: float = 0.25, hold: float = 6.0) -> str:
    """Sample the endpoint for settle + fn() + hold seconds.

    ``hold`` must cover the whole utterance (a sentence takes ~1-3 s to reach
    the engine and play), otherwise the window closes before any audio starts.
    """
    with PeakSampler() as p:
        time.sleep(settle)
        fn()
        time.sleep(hold)
    return f"{label:<38} {p.report()}"


def main() -> int:
    from blindassist.speech import Pyttsx3Speaker

    print("Urgent-interrupt probe (cross-thread engine.stop)\n")

    sp = Pyttsx3Speaker(rate_wpm=175, volume=1.0)
    sp.start()
    time.sleep(0.5)
    proxy = RecordingProxy(sp._engine)
    sp._engine = proxy  # the worker reads self._engine on every utterance

    def say(text: str, urgent: bool = False) -> None:
        sp.speak(text, urgent=urgent)

    print(peak("1 control utterance", lambda: say("Control sentence one.")))
    time.sleep(3.5)

    def urgent_mid() -> None:
        say("This sentence should be interrupted half way through.")
        time.sleep(0.8)
        say("URGENT warning", urgent=True)  # cross-thread engine.stop()

    print(peak("2 urgent interrupt", urgent_mid))
    time.sleep(3.0)

    print(
        peak(
            "3 utterance after interrupt", lambda: say("Sentence after the interrupt.")
        )
    )
    time.sleep(3.5)

    def burst() -> None:
        for i in range(8):
            say(f"Number {i} normal announcement.")
            time.sleep(0.35)
            say(f"URGENT {i}", urgent=True)
            time.sleep(0.35)

    print(peak("4 eight urgent interrupts", burst))
    time.sleep(3.0)

    print(peak("5 final utterance", lambda: say("Final sentence after the burst.")))
    time.sleep(4.0)

    print(f"\nengine exceptions recorded: {len(proxy.errors)}")
    for e in proxy.errors[:12]:
        print("   ", e)
    print(f"spoken_log: {list(sp.spoken_log)}")
    sp.shutdown(timeout=8)
    print(
        f"worker alive after shutdown: {sp._thread is not None and sp._thread.is_alive()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
