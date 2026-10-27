"""Extract still frames from a recorded video for use as evaluation scenes.

Used to rebuild the position/distance test set from the shared 60 s webcam
clip (``assets/scene60.avi``): real camera frames, taken at spread-out
timestamps so the scenes are not all identical.

Usage:
    python scripts/extract_frames.py --video assets/scene60.avi \
        --out assets/eval_frames_v2 --times 2 10 18 26 34 42 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract frames from a video clip")
    p.add_argument("--video", default=str(ROOT / "assets" / "scene60.avi"))
    p.add_argument("--out", default=str(ROOT / "assets" / "eval_frames_v2"))
    p.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[2, 10, 18, 26, 34, 42, 50],
        help="timestamps in seconds",
    )
    args = p.parse_args(argv)

    video = Path(args.video)
    if not video.exists():
        print(f"[ERROR] no such video: {video}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open {video}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps else 0.0
    print(f"[INFO] {video.name}: {total} frames @ {fps:.0f} fps = {duration:.0f} s")

    written = 0
    for t in args.times:
        if t >= duration:
            print(f"[WARN] t={t:g}s is past the end, skipping", file=sys.stderr)
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] t={t:g}s: read failed", file=sys.stderr)
            continue
        name = f"vid_{round(t):02d}s.jpg"
        path = out_dir / name
        if not cv2.imwrite(str(path), frame):
            print(f"[WARN] could not write {path}", file=sys.stderr)
            continue
        written += 1
        mean_luma = float(frame.mean())
        print(f"[OK] {path.name} (t={t:g}s, luma={mean_luma:.0f})")

    cap.release()
    print(f"[RESULT] extracted {written} frames into {out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
