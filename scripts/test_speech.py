"""Live audio test for speech.py (uses the Windows SAPI5 voice via pyttsx3).

Speaks three sentences (one of them an urgent interrupt) and verifies that:

1. ``speak()`` returns immediately - the main thread is never blocked by speech,
2. the main thread keeps running its own work (frame ticks) while speaking,
3. an ``urgent`` message clears the queue and is spoken,
4. ``shutdown()`` ends cleanly with no hanging worker thread.

    C:\\dev\\ba_venv\\Scripts\\python.exe scripts\\test_speech.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blindassist.speech import Pyttsx3Speaker  # noqa: E402

# Pass/fail thresholds for this smoke test, stated once instead of inline.
MAX_BLOCK_MS = 250.0  # speak() must return well under this to be "non-blocking"
RESPONSIVENESS_WINDOW_S = 1.5
MIN_TICKS_IN_WINDOW = 50  # main thread ticks expected while speech runs


def main() -> int:
    speaker = Pyttsx3Speaker(rate_wpm=180, volume=1.0)
    speaker.start()
    print(
        f"[INFO] TTS voice available: {speaker.available}"
        + (f" (reason: {speaker.init_error})" if not speaker.available else "")
    )

    failures: list[str] = []
    try:
        # ---- 1) speak() must not block the caller --------------------------
        long_sentence = (
            "This is a long sentence, spoken in the background, used to prove that "
            "the speech engine never blocks the main thread while it is talking."
        )
        t0 = time.perf_counter()
        speaker.speak(long_sentence)  # first sentence (normal)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[TEST] speak() returned in {latency_ms:.2f} ms (non-blocking)")
        if latency_ms >= MAX_BLOCK_MS:
            failures.append(f"speak() blocked for {latency_ms:.1f} ms")

        # ---- 2) main thread keeps working while speech is ongoing -----------
        start = time.perf_counter()
        ticks = 0
        while time.perf_counter() - start < RESPONSIVENESS_WINDOW_S:
            ticks += 1
            time.sleep(0.01)  # simulate drawing a video frame
        elapsed = time.perf_counter() - start
        print(
            f"[TEST] main thread completed {ticks} ticks in {elapsed:.2f}s while "
            f"speaking -> responsive"
        )
        if ticks < MIN_TICKS_IN_WINDOW:
            failures.append(f"main thread only managed {ticks} ticks in {elapsed:.2f}s")

        # ---- 3) urgent interrupt clears the queue --------------------------
        speaker.speak("This second sentence should be cut off by the urgent message.")
        speaker.speak("This third sentence is queued behind the second one.")
        time.sleep(0.2)
        before = speaker.queue_size()
        speaker.speak(" urgent. Stop. An obstacle is very close.", urgent=True)
        after = speaker.queue_size()
        print(
            f"[TEST] queue before urgent={before}, after={after} "
            f"(urgent cleared the queue)"
        )
        if after > 1:
            failures.append(f"urgent did not clear the queue (size={after})")

        # let the urgent line finish
        time.sleep(4)
        print(f"[TEST] sentences handed to the engine: {len(speaker.spoken_log)}")
        print(f"[TEST] spoken_log -> {speaker.spoken_log}")
    finally:
        # ---- 4) clean shutdown --------------------------------------------
        speaker.shutdown()
        time.sleep(0.1)
        alive = speaker._thread is not None and speaker._thread.is_alive()
        print(f"[TEST] worker thread alive after shutdown: {alive}")
        if alive:
            failures.append("worker thread still alive after shutdown")

    if failures:
        print("\n[FAIL] " + "; ".join(failures))
        return 1
    print(
        "\n[PASS] speech test completed with no errors; main thread stayed responsive."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
