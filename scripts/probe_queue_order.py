"""Prove the purge mechanism: log DriverProxy's command queue order.

Hypothesis being tested (derived from engine.py / driver.py / sapi5.py):

  * Engine.runAndWait() ends with ``proxy.setBusy(False)``  -> busy = False
  * so on utterance N+1 ``engine.say()`` executes IMMEDIATELY (queue empty),
    and only afterwards ``runAndWait()`` pushes the ``endLoop`` command.
  * at startLoop the lone ``endLoop`` command is popped FIRST and
    DriverProxy.endLoop() calls ``driver.stop()`` while the new utterance is
    already speaking  ->  SAPI5 Speak("", 3) purges it  ->  silence.

Utterance 1 has busy=True (fresh proxy) so ``say`` is queued BEFORE ``endLoop``
and is spoken normally.

Run:  .venv\\Scripts\\python.exe scripts\\probe_queue_order.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import pyttsx3
    import pyttsx3.driver as drv

    def label(mtd) -> str:
        owner = getattr(mtd, "__self__", None)
        name = getattr(mtd, "__name__", str(mtd))
        cls = type(owner).__name__ if owner is not None else ""
        return f"{cls}.{name}" if cls else name

    orig_push = drv.DriverProxy._push
    orig_stop = drv.DriverProxy.stop

    def push(self, mtd, args, name=None):
        queue = [label(m) for m, _a, _n in self._queue]
        print(
            f"   PUSH {label(mtd):28s} busy={self._busy!s:5s} queue={queue}", flush=True
        )
        return orig_push(self, mtd, args, name)

    def stop(self):
        print(
            f"   DriverProxy.stop() -> driver.stop() (PURGE if speaking) "
            f"speaking={getattr(self._driver, '_speaking', '?')} "
            f"queue={[label(m) for m, _a, _n in self._queue]}",
            flush=True,
        )
        return orig_stop(self)

    drv.DriverProxy._push = push
    drv.DriverProxy.stop = stop

    eng = pyttsx3.init()
    for i in (1, 2, 3):
        print(f"\n=== utterance {i} ===", flush=True)
        t0 = time.time()
        eng.say(f"Utterance number {i}.")
        eng.runAndWait()
        print(f"   runAndWait={time.time() - t0:.2f}s", flush=True)
        time.sleep(0.4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
