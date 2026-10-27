"""Which repair actually restores speech after an urgent interrupt?

Experiment matrix (each variant measures the real endpoint peak):

  V1  cross-thread engine.stop()            -> baseline (the bug)
  V2  interrupt flag honoured INSIDE worker -> no cross-thread call
  V3  fresh engine per utterance            -> engine recreated after interrupt
  V4  same as V1 but stop() called from the worker thread itself

Also records how many utterances the app's own code sends with urgent=True,
because that is how quickly a real session goes silent.

Run:  .venv\\Scripts\\python.exe scripts\\probe_fix_variants.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_peak_probe import PeakSampler  # noqa: E402


def report(label: str, fn) -> str:
    with PeakSampler() as p:
        time.sleep(0.3)
        fn()
        time.sleep(6.0)
    return f"{label:<46} {p.report()}"


def v1_cross_thread_stop() -> str:
    """Current speech.py behaviour."""
    from blindassist.speech import Pyttsx3Speaker

    sp = Pyttsx3Speaker(rate_wpm=175, volume=1.0)
    sp.start()
    time.sleep(0.5)

    def go() -> None:
        sp.speak("Before the urgent interrupt this sentence is audible.")
        time.sleep(4.0)
        sp.speak("URGENT", urgent=True)  # -> engine.stop() from main thread
        time.sleep(1.5)
        sp.speak("After the urgent interrupt this sentence should be heard.")

    out = report("V1 cross-thread stop (current app)", go)
    sp.shutdown(timeout=8)
    return out


def v2_interrupt_flag() -> str:
    """Urgent sets a flag; the worker checks it and re-issues stop() itself."""
    from blindassist.speech import Pyttsx3Speaker

    sp = Pyttsx3Speaker(rate_wpm=175, volume=1.0)
    sp.start()
    time.sleep(0.5)
    flag = threading.Event()
    original_stop = sp._stop_current
    sp._stop_current = flag.set  # urgent now only sets an event

    def worker_watch() -> None:
        while not flag.is_set():
            time.sleep(0.05)
        original_stop()  # stop() runs ON the worker's COM apartment
        flag.clear()

    threading.Thread(target=worker_watch, daemon=True).start()

    def go() -> None:
        sp.speak("Before the urgent interrupt this sentence is audible.")
        time.sleep(4.0)
        sp.speak("URGENT", urgent=True)
        time.sleep(1.5)
        sp.speak("After the urgent interrupt this sentence should be heard.")

    out = report("V2 flag -> stop() inside worker", go)
    sp.shutdown(timeout=8)
    return out


def v3_fresh_engine() -> str:
    """Engine is recreated for every utterance (Phase 2 design)."""
    import pyttsx3

    state = {"eng": None}

    def speak_fresh(text: str) -> None:
        if state["eng"] is not None:
            try:
                state["eng"].stop()
            except Exception:
                pass
        eng = pyttsx3.init()
        eng.setProperty("rate", 175)
        state["eng"] = eng
        eng.say(text)
        eng.runAndWait()

    def go() -> None:
        speak_fresh("Before the urgent interrupt this sentence is audible.")
        time.sleep(4.0)
        try:
            state["eng"].stop()  # the "urgent" cut-in, engine still ours
        except Exception:
            pass
        time.sleep(1.5)
        speak_fresh("After the urgent interrupt this sentence should be heard.")

    out = report("V3 fresh engine per utterance", go)
    try:
        if state["eng"]:
            state["eng"].stop()
    except Exception:
        pass
    return out


def v4_stop_from_worker() -> str:
    """stop() called on the worker thread (its own COM apartment)."""
    import pyttsx3

    box: dict = {"eng": None, "ready": threading.Event()}

    def worker() -> None:
        eng = pyttsx3.init()
        eng.setProperty("rate", 175)
        box["eng"] = eng
        box["ready"].set()
        # Keep an event loop alive like runAndWait would.
        while True:
            time.sleep(0.1)

    threading.Thread(target=worker, daemon=True).start()
    box["ready"].wait(5)
    eng = box["eng"]

    def go() -> None:
        eng.say("Before the urgent interrupt this sentence is audible.")
        eng.runAndWait()
        time.sleep(1.0)
        eng.stop()  # from THIS thread (still not the worker)
        time.sleep(1.0)
        eng.say("After the urgent interrupt this sentence should be heard.")
        eng.runAndWait()

    # Run go() on the worker thread so stop() and runAndWait() share the apartment
    result: dict = {}

    def on_worker() -> None:
        try:
            result["out"] = report("V4 stop()+say() on one thread", go)
        except Exception as exc:
            result["out"] = f"V4 failed: {type(exc).__name__}: {exc}"

    t = threading.Thread(target=on_worker, daemon=True)
    t.start()
    t.join(40)
    return result.get("out", "V4 timed out")


def app_urgent_usage() -> str:
    """Where does the real app send urgent=True?"""
    src = (ROOT / "blindassist" / "app.py").read_text("utf-8", errors="replace")
    hits = [ln.strip() for ln in src.splitlines() if "urgent=True" in ln]
    return (
        "app.py urgent=True call sites: "
        + str(len(hits))
        + "\n    "
        + "\n    ".join(hits)
    )


def main() -> int:
    print("Repair-variant experiment\n")
    print(report.__doc__ or "")
    for fn in (
        v1_cross_thread_stop,
        v2_interrupt_flag,
        v3_fresh_engine,
        v4_stop_from_worker,
    ):
        try:
            print(fn())
        except Exception as exc:
            import traceback

            print(f"{fn.__name__:<46} ERROR {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
        time.sleep(1.0)
    print()
    print(app_urgent_usage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
