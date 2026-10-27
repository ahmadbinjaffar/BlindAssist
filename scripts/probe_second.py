"""Why is the 2nd pyttsx3 utterance silent? Measure what actually happens.

For every attempt we print:
  * how long runAndWait() took (0.05 s = the driver never spoke)
  * the real endpoint peak for that window

Scenarios (each in a FRESH process is not needed - we want in-process history):

  S1  engine #1, utterance 1                    (known good)
  S2  engine #1, utterance 2  (same engine)     -> does re-use work?
  S3  engine #2 after deleting engine #1        -> does discarding help?
  S4  engine #3, utterance after stop() on #2

Run:  .venv\\Scripts\\python.exe scripts\\probe_second.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_peak_probe import PeakSampler  # noqa: E402


def timed_say(engine, text: str) -> float:
    engine.say(text)
    t0 = time.time()
    engine.runAndWait()
    return time.time() - t0


def measured(label: str, fn) -> None:
    durations: list[float] = []
    with PeakSampler() as p:
        time.sleep(0.3)
        durations.append(fn())
        time.sleep(4.0)
    print(f"  {label:<52} runAndWait={durations[0]:6.2f}s  {p.report()}", flush=True)


def main() -> int:
    import pyttsx3

    print("Second-utterance isolation (all in ONE process)\n")

    def new_engine():
        e = pyttsx3.init()
        e.setProperty("rate", 175)
        return e

    e1 = new_engine()

    measured(
        "S1 engine #1 utterance 1",
        lambda: timed_say(e1, "First utterance of engine one."),
    )
    time.sleep(1.0)
    measured(
        "S2 engine #1 utterance 2 (same engine)",
        lambda: timed_say(e1, "Second utterance of engine one."),
    )

    # Drop the reference so engine #1 is really gone before engine #2 starts.
    e1 = None
    gc.collect()
    time.sleep(1.0)

    e2 = new_engine()
    measured(
        "S3 engine #2 after deleting engine #1",
        lambda: timed_say(e2, "First utterance of engine two."),
    )
    time.sleep(1.0)
    measured(
        "S4 engine #2 utterance 2",
        lambda: timed_say(e2, "Second utterance of engine two."),
    )

    print("\nInterpretation:")
    print("  runAndWait ~= 0.05s -> the driver never started speaking")
    print("  runAndWait ~= 2-4s and peak 0 -> it spoke into a dead endpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
