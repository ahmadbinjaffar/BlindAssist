"""Probe the webcam without loading any ML model.

Useful to verify camera access quickly:

    C:\\dev\\ba_venv\\Scripts\\python.exe scripts\\probe_webcam.py
    ... --camera 1 --frames 5 --save Evidence\\probe.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Probe a webcam with OpenCV")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--save", type=str, default="", help="optional path to save a frame")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args(argv)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FAIL] Could not open camera {args.camera}")
        return 2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, first = cap.read()
    if not ok or first is None:
        print(f"[FAIL] Camera {args.camera} opened but could not read a frame.")
        cap.release()
        return 2

    h, w = first.shape[:2]
    print(
        f"[OK] Camera {args.camera} opened. Requested {args.width}x{args.height}, "
        f"got {w}x{h} (mean pixel {first.mean():.1f})"
    )

    frames_read = 1
    start = time.time()
    last = first
    for _ in range(max(0, args.frames - 1)):
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] read() failed mid-stream")
            break
        frames_read += 1
        last = frame
    elapsed = time.time() - start
    fps = frames_read / elapsed if elapsed > 0 else 0.0
    print(
        f"[OK] Read {frames_read} frames in {elapsed:.2f}s (~{fps:.1f} FPS). "
        f"Last frame mean pixel {last.mean():.1f}."
    )

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), last)
        print(f"[OK] Saved last frame to {out.resolve()} ({out.stat().st_size} bytes)")

    cap.release()
    print("[OK] Camera released cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
