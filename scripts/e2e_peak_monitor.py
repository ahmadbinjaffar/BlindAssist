"""Endpoint monitor for end-to-end proof.

Samples the default output device's peak meter and prints the timeline of
AUDIO BURSTS (contiguous periods with peak > 0.001) with start/stop times.
Run it in a second process while the real app is speaking; a burst with no
matching announcement = sound, an announcement with no burst = silence.

Run:  .venv\\Scripts\\python.exe scripts\\e2e_peak_monitor.py --seconds 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_peak_probe import make_meter  # noqa: E402

# Peak above this = the endpoint is actually producing sound.
_AUDIBLE = 0.001
# Silence longer than this inside a burst means the utterance has finished.
_GAP_S = 0.35


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--out", default=str(ROOT / "Evidence" / "e2e_peak_timeline.txt"))
    args = ap.parse_args()

    meter = make_meter()
    t0 = time.time()
    samples: list[tuple[float, float]] = []
    print(f"monitoring endpoint for {args.seconds:g}s ...", flush=True)
    while time.time() - t0 < args.seconds:
        try:
            v = float(meter.GetPeakValue())
        except Exception:
            v = -1.0
        samples.append((time.time() - t0, v))
        time.sleep(args.step)

    # burst detection
    bursts: list[list[float]] = []
    in_burst = False
    for t, v in samples:
        loud = v > _AUDIBLE
        if loud and not in_burst:
            bursts.append([t, t])
            in_burst = True
        elif loud and in_burst:
            bursts[-1][1] = t
        elif not loud and in_burst:
            if t - bursts[-1][1] > _GAP_S:  # gap => burst finished
                in_burst = False

    lines = [
        f"monitor run start={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}"
        f" end={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"samples={len(samples)} duration={args.seconds:g}s",
        f"overall max peak={max((v for _, v in samples if v >= 0), default=-1):.4f}",
        "bursts (start s, end s, seconds, max peak):",
    ]
    for b in bursts:
        seg = [v for t, v in samples if b[0] <= t <= b[1] and v >= 0]
        lines.append(
            f"  {b[0]:7.2f} - {b[1]:7.2f}  "
            f"({b[1] - b[0]:5.2f}s)  peak={max(seg):.4f}"
        )
    lines.append(f"total bursts = {len(bursts)}")
    text = "\n".join(lines)
    print(text, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
