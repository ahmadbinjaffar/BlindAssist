"""Definitive root-cause experiment: per-phase peak measurement.

Each variant is measured in TWO separate windows:

    window 1 : utterance BEFORE the urgent interrupt   (expect sound)
    interrupt: speak(..., urgent=True) -> engine.stop()
    window 2 : utterance AFTER  the urgent interrupt   (sound? = verdict)

Variant A = today's blindassist.speech.Pyttsx3Speaker (engine reused)
Variant B = candidate fix: FRESH engine for every utterance, urgent stop()
            only ever applied to the engine that is about to be discarded

Run:  .venv\\Scripts\\python.exe scripts\\probe_repair.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_peak_probe import PeakSampler  # noqa: E402


def window(label: str, fn) -> str:
    with PeakSampler() as p:
        time.sleep(0.3)
        fn()
        time.sleep(6.0)
    return f"    {label:<40} {p.report()}"


# --------------------------------------------------------------------- A (bug)
def variant_a() -> None:
    from blindassist.speech import Pyttsx3Speaker

    print("A. current design: one engine reused, stop() from the main thread")
    sp = Pyttsx3Speaker(rate_wpm=175, volume=1.0)
    sp.start()
    time.sleep(0.6)

    print(
        window(
            "A1 before interrupt",
            lambda: sp.speak("This sentence is spoken before the interrupt."),
        )
    )
    time.sleep(1.0)
    sp.speak("URGENT stop right now", urgent=True)  # <-- the bug trigger
    time.sleep(2.0)
    print(
        window(
            "A2 AFTER interrupt",
            lambda: sp.speak("This sentence is spoken after the interrupt."),
        )
    )
    print(
        f"    spoken_log entries: {len(sp.spoken_log)} "
        f"(queue accepted the text: yes -> audio: see A2 peak)"
    )
    sp.shutdown(timeout=8)


# --------------------------------------------------------------- B (candidate)
def variant_b() -> None:
    import pyttsx3

    print("B. candidate fix: fresh engine per utterance, discarded after use")
    box: dict = {"eng": None}

    def speak_fresh(text: str) -> None:
        old = box["eng"]
        if old is not None:
            try:
                old.stop()  # killing a doomed engine is safe
            except Exception:
                pass
        eng = pyttsx3.init()
        eng.setProperty("rate", 175)
        box["eng"] = eng
        eng.say(text)
        eng.runAndWait()

    print(
        window(
            "B1 before interrupt",
            lambda: speak_fresh("This sentence is spoken before the interrupt."),
        )
    )
    time.sleep(1.0)
    try:
        box["eng"].stop()  # urgent cut-in on a doomed engine
    except Exception:
        pass
    time.sleep(2.0)
    print(
        window(
            "B2 AFTER interrupt",
            lambda: speak_fresh("This sentence is spoken after the interrupt."),
        )
    )
    try:
        if box["eng"]:
            box["eng"].stop()
    except Exception:
        pass


def main() -> int:
    print("Root-cause isolation: does speech survive an urgent interrupt?\n")
    variant_a()
    print()
    variant_b()
    print()
    print("Verdict rule: A2 peak == 0 while B2 peak > 0 proves that reusing the")
    print("SAPI5 engine after a cross-thread stop() is the cause of the silence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
