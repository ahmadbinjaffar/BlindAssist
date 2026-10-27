"""Validate the repair design: fresh engine per utterance, destroy the old one
BEFORE creating the next one.

Measured per cycle (each window = real endpoint peak):

  MAIN thread : 5 speak cycles
  WORKER thread (pythoncom.CoInitialize, like speech.py): 5 speak cycles
  URGENT cut  : stop() the doomed engine mid-sentence, then speak again

A cycle must show peak > 0 every time, otherwise the design is not a fix.

Run:  .venv\\Scripts\\python.exe scripts\\probe_cycles.py
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_peak_probe import PeakSampler  # noqa: E402


class FreshEngineSpeaker:
    """The proposed repair: one engine per utterance, old engine destroyed first."""

    def __init__(self, rate: int = 175) -> None:
        self.rate = rate
        self._eng = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        with self._lock:
            self._discard()
            import pyttsx3

            eng = pyttsx3.init()
            eng.setProperty("rate", self.rate)
            self._eng = eng
            eng.say(text)
            eng.runAndWait()

    def cut(self) -> None:
        """Urgent interrupt: stop the CURRENT engine (it is discarded next)."""
        eng = self._eng
        if eng is not None:
            try:
                eng.stop()
            except Exception as exc:
                print(f"      cut() raised {type(exc).__name__}: {exc}", flush=True)

    def _discard(self) -> None:
        if self._eng is None:
            return
        try:
            self._eng.stop()
        except Exception:
            pass
        self._eng = None
        gc.collect()

    def shutdown(self) -> None:
        with self._lock:
            self._discard()


def cycle(label: str, sp: FreshEngineSpeaker, text: str) -> None:
    with PeakSampler() as p:
        time.sleep(0.3)
        sp.speak(text)
        time.sleep(3.5)
    print(f"  {label:<52} {p.report()}", flush=True)


def main() -> int:
    print("Repair-design validation: fresh engine per utterance\n")

    print("MAIN thread")
    sp = FreshEngineSpeaker()
    for i in range(5):
        cycle(f"main cycle {i + 1}", sp, f"Main thread cycle number {i + 1}.")
        time.sleep(0.5)
    sp.shutdown()

    print("\nWORKER thread (pythoncom.CoInitialize, like speech.py)")
    out: list[str] = []

    def worker() -> None:
        import pythoncom

        pythoncom.CoInitialize()
        wsp = FreshEngineSpeaker()
        for i in range(5):
            with PeakSampler() as p:
                time.sleep(0.3)
                wsp.speak(f"Worker thread cycle number {i + 1}.")
                time.sleep(3.5)
            out.append(f"  {'worker cycle ' + str(i + 1):<52} {p.report()}")
        wsp.shutdown()
        pythoncom.CoUninitialize()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(120)
    for line in out:
        print(line, flush=True)
    print(f"  worker finished: {not t.is_alive()}")

    print("\nURGENT cut then continue (same worker-style object in main thread)")
    sp2 = FreshEngineSpeaker()
    with PeakSampler() as p:
        time.sleep(0.3)
        threading.Timer(1.2, sp2.cut).start()
        sp2.speak("This long sentence gets interrupted by an urgent warning.")
        time.sleep(4.0)
    print(f"  {'urgent window (cut at 1.2s)':<52} {p.report()}")
    time.sleep(1.0)
    cycle(
        "utterance AFTER the urgent cut",
        sp2,
        "This sentence comes after the urgent cut.",
    )
    sp2.shutdown()

    print("\nVerdict rule: every cycle peak must be > 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
