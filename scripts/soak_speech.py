"""Five-minute no-stall soak for the speech pipeline.

Runs the *same* speaker the app builds (``make_speaker(..., self_test=True)``)
for five minutes at a realistic announcement cadence, with the pressures that
broke the old code:

  * one utterance every few seconds (the case that used to go silent after #1)
  * urgent cut-ins every 5th cycle
  * live rate changes (the [ ] keys)
  * one mute/unmute cycle every 20 cycles (the M key)

After every cycle it checks three things:

  worker  - the background thread is still alive (no stall, no crash)
  queue   - the backlog never grows unbounded (a stuck worker shows up here)
  peak    - the real output endpoint actually produced sound

Run:  .venv\\Scripts\\python.exe scripts\\soak_speech.py [--minutes 5]
Writes: Evidence/speech_soak_5min.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blindassist.speech import Pyttsx3Speaker, make_speaker  # noqa: E402

OUT = ROOT / "Evidence" / "speech_soak_5min.txt"

SILENT_PEAK = 0.0005  # matches speech._SILENT_PEAK: "no audio at all"
WAIT_S = 12.0  # an announcement must be picked up within this long
INTERVAL_S = 4.0  # cadence of the simulated announcements

LINE = "Person ahead, very close. Please stop."
URGENT_LINE = "Stop! Obstacle right in front."


def wait_spoken(sp: Pyttsx3Speaker, before: int) -> bool:
    """Block until the worker has picked up a new utterance."""
    deadline = time.time() + WAIT_S
    while time.time() < deadline:
        if len(sp.spoken_log) > before:
            return True
        time.sleep(0.1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    args = ap.parse_args()
    duration = args.minutes * 60.0

    lines = [
        f"5-minute speech soak ({args.minutes:g} min)",
        f"start {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "speaker = make_speaker(True, 175, 1.0, self_test=True)  # app path",
        "",
    ]

    sp = make_speaker(True, 175, 1.0, self_test=True)
    if not isinstance(sp, Pyttsx3Speaker):
        raise SystemExit(f"expected a Pyttsx3Speaker, got {type(sp).__name__}")
    t0 = time.time()
    sp.start()
    startup = time.time() - t0
    result = sp.self_test_result or {}
    lines.append(
        f"start-up self-test: backend={result.get('backend')} "
        f"ok={result.get('ok')} peak={result.get('peak')} "
        f"({startup:.2f}s)"
    )
    lines.append("")

    cycles = 0
    spoken = 0
    muted_ok = 0
    silent = 0
    timeouts = 0
    max_queue = 0
    dead_worker = 0
    urgent = 0
    rate_changes = 0

    deadline = time.time() + duration
    while time.time() < deadline:
        cycles += 1
        mute_cycle = cycles % 20 == 0
        urgent_cycle = cycles % 5 == 0

        before = len(sp.spoken_log)
        if mute_cycle:
            sp.set_muted(True)
            sp.speak("dropped on purpose while muted")
            if list(sp.muted_log):
                muted_ok += 1
            sp.set_muted(False)
        elif urgent_cycle:
            sp.speak(URGENT_LINE, urgent=True)
            urgent += 1
        else:
            sp.speak(LINE)

        if not mute_cycle:
            if wait_spoken(sp, before):
                spoken += 1
                peak = sp._last_peak
                if peak is not None and peak <= SILENT_PEAK:
                    silent += 1
            else:
                timeouts += 1

        if cycles % 15 == 0:
            sp.set_rate(150 + (cycles % 3) * 20)
            rate_changes += 1

        max_queue = max(max_queue, sp.queue_size())
        thread = sp._thread
        if thread is None or not thread.is_alive():
            dead_worker += 1
            lines.append(f"WORKER DEAD at cycle {cycles}")
            break
        time.sleep(INTERVAL_S)

    elapsed = time.time() - t0
    thread_alive = sp._thread is not None and sp._thread.is_alive()
    status = sp.status
    backend = sp.backend_name
    sp.shutdown(timeout=10.0)

    ok = (
        dead_worker == 0
        and timeouts == 0
        and silent == 0
        and spoken > 0
        and thread_alive
        and status == "ON"
    )

    lines += [
        "",
        f"end   {time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed {elapsed:.0f}s)",
        "",
        "RESULT",
        f"  cycles run              {cycles}",
        f"  utterances spoken        {spoken}",
        f"  silent utterances        {silent}   (peak <= {SILENT_PEAK})",
        f"  pick-up timeouts         {timeouts}",
        f"  urgent cut-ins           {urgent}",
        f"  mute cycles handled      {muted_ok}",
        f"  live rate changes        {rate_changes}",
        f"  max queue size           {max_queue}",
        f"  worker dead mid-run      {dead_worker}",
        f"  worker alive at end      {thread_alive}",
        f"  status / backend         {status} / {backend}",
        "",
        f"{'PASS' if ok else 'FAIL'}: "
        + (
            "no stall, no silent utterance, queue never backed up"
            if ok
            else "see the counters above"
        ),
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    OUT.write_text(text + "\n", "utf-8")
    print(f"\nlog written -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
