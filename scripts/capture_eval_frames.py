"""Capture a handful of webcam frames for the position/distance accuracy test.

Frames are taken ~1.5 s apart so they are genuinely different (people move),
and saved to ``assets/eval_frames/``. Ground truth for those frames is recorded
by hand in ``assets/eval_frames/ground_truth.json``.

    .\\.venv\\Scripts\\python.exe scripts\\capture_eval_frames.py --count 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "assets" / "eval_frames"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def open_camera(index: int, width: int, height: int):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap, frame
        cap.release()
    return None, None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--count", type=int, default=7)
    p.add_argument("--gap", type=float, default=1.5, help="seconds between frames")
    p.add_argument(
        "--warmup", type=int, default=6, help="discard this many first frames"
    )
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap, frame = open_camera(args.camera, 1280, 720)
    if cap is None:
        print("[ERROR] could not open camera", file=sys.stderr)
        return 2

    for _ in range(args.warmup):  # let auto-exposure settle
        ok, frame = cap.read()
        if not ok:
            break

    saved = 0
    for i in range(1, args.count + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] frame {i} failed", file=sys.stderr)
            time.sleep(args.gap)
            continue
        out = OUT_DIR / f"cam_{i:02d}.jpg"
        cv2.imwrite(str(out), frame)
        saved += 1
        print(f"[OK] {out.name}  ({frame.shape[1]}x{frame.shape[0]})")
        if i < args.count:
            time.sleep(args.gap)

    cap.release()
    # Copy the three sample images in, so all 10 evaluation frames live together.
    samples = ROOT / "assets" / "samples"
    extra = 0
    # The evaluation set is 10 frames: up to 7 from the webcam plus these.
    max_sample_images = 3
    for src in sorted(samples.glob("*")):
        if src.suffix.lower() in IMAGE_EXTENSIONS and extra < max_sample_images:
            dst = OUT_DIR / f"sample_{src.stem}{src.suffix}"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
                extra += 1
    print(
        f"[OK] captured {saved} webcam frame(s) + {extra} sample image(s) -> {OUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
