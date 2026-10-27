"""Capture the exception pyttsx3 swallows on the 2nd utterance.

pyttsx3's DriverProxy._pump() does:

    except Exception as e:
        self.notify("error", exception=e)      # -> engine._notify
        if self._debug: traceback.print_exc()   # debug is OFF by default

so with debug=False every driver error vanishes.  This script connects to the
"error" topic and prints exactly what fails on utterance #1 and #2.

Run:  .venv\\Scripts\\python.exe scripts\\probe_driver_error.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import pyttsx3

    errors: list[str] = []
    events: list[str] = []

    eng = pyttsx3.init()
    eng.connect(
        "error",
        lambda exception, name=None, **kw: errors.append(
            f"{type(exception).__name__}: {exception}"
        ),
    )
    for topic in ("started-utterance", "finished-utterance"):
        eng.connect(
            topic,
            lambda name=None, completed=None, topic=topic, **kw: events.append(
                f"{topic} completed={completed}"
            ),
        )

    for i in (1, 2, 3):
        t0 = time.time()
        eng.say(f"Utterance number {i} in this test.")
        eng.runAndWait()
        print(
            f"utterance {i}: runAndWait={time.time() - t0:.2f}s "
            f"events={events} errors={errors}",
            flush=True,
        )
        events.clear()
        time.sleep(0.5)

    print(f"\ntotal swallowed driver errors: {len(errors)}")
    for e in errors:
        print("  ", e)
    try:
        eng.stop()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
