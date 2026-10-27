"""Record a fixed-length clip from the webcam into assets/.

Used to produce the shared 60-second video that the before/after stability
comparison replays in both modes, so both runs see *identical* frames.

Usage:  python scripts/record_clip.py --seconds 60 --fps 10 --out assets/scene60.avi
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Record a clip from the webcam")
    p.add_argument("--out", default=str(ROOT / "assets" / "scene60.avi"))
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] camera {args.camera} not available", file=sys.stderr)
        return 2

    # Request the size from the camera; fall back to whatever it delivers.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] camera opened but no frame", file=sys.stderr)
        cap.release()
        return 2
    height, width = frame.shape[:2]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (width, height)
    )
    if not writer.isOpened():
        print("[ERROR] VideoWriter failed to open", file=sys.stderr)
        cap.release()
        return 2

    target = round(args.seconds * args.fps)
    print(
        f"[INFO] recording {target} frames ({args.seconds:.0f} s @ "
        f"{args.fps:.0f} fps) at {width}x{height} ...",
        flush=True,
    )
    t0 = time.time()
    written = 0
    while written < target:
        if written:  # first frame already captured above
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] frame grab failed mid-recording", flush=True)
                break
        writer.write(frame)
        written += 1
        if written % 60 == 0:
            print(f"  {written}/{target}", flush=True)
        # Pace to the requested fps (the camera may deliver faster).
        expected = t0 + written / args.fps
        delay = expected - time.time()
        if delay > 0:
            time.sleep(delay)

    cap.release()
    writer.release()
    print(
        f"[OK] {out_path}: {written} frames, "
        f"{written / args.fps:.1f} s of video, recorded in {time.time() - t0:.1f} s"
    )
    if written < target:
        print("[WARN] recording stopped early", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
