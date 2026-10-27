"""Which call purges utterance #2? Instrument pyttsx3's stop path.

Wraps SAPI5Driver.stop and DriverProxy.endLoop with stack-trace recorders so
the purge of utterance 2 can be attributed to a specific call site.

Run:  .venv\\Scripts\\python.exe scripts\\probe_purge_origin.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import pyttsx3
    import pyttsx3.driver as drv
    from pyttsx3.drivers import sapi5

    records: list[str] = []

    real_stop = sapi5.SAPI5Driver.stop
    real_endloop = drv.DriverProxy.endLoop
    real_say = sapi5.SAPI5Driver.say

    def stop_wrapper(self):
        if self._speaking:
            records.append(
                "SAPI5Driver.stop() while speaking (PURGES audio), called from:\n"
                + traceback.format_stack()[-6:-1]
            )
        return real_stop(self)

    def endloop_wrapper(self, useDriverLoop):
        records.append(
            "DriverProxy.endLoop() called from:\n" + traceback.format_stack()[-6:-1]
        )
        return real_endloop(self, useDriverLoop)

    def say_wrapper(self, text):
        records.append("---- SAPI5Driver.say() for a new utterance ----")
        return real_say(self, text)

    sapi5.SAPI5Driver.stop = stop_wrapper
    drv.DriverProxy.endLoop = endloop_wrapper
    sapi5.SAPI5Driver.say = say_wrapper

    eng = pyttsx3.init()
    for i in (1, 2):
        records.clear()
        t0 = time.time()
        eng.say(f"Utterance number {i}.")
        eng.runAndWait()
        print(f"\n=== utterance {i} (runAndWait={time.time() - t0:.2f}s) ===")
        for r in records:
            print(r)
        time.sleep(0.5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
