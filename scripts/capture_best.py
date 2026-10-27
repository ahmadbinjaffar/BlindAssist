"""Sample the webcam for a few seconds and save the frame with the most detections.

C:\\dev\\ba_venv\\Scripts\\python.exe scripts\\capture_best.py
    --camera 0 --frames 60 --out Evidence\\detection.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blindassist.config import load_config  # noqa: E402
from blindassist.detector import Detector  # noqa: E402
from blindassist.position import Positioner  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--out", type=str, default="Evidence/detection.jpg")
    p.add_argument("--conf", type=float, default=0.40)
    args = p.parse_args(argv)

    cfg = load_config()
    cfg.vision.confidence = args.conf

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[FAIL] camera")
        return 2

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        print("[FAIL] read")
        return 2
    h, w = frame.shape[:2]

    det = Detector(cfg.vision, cfg.model_path, cfg.class_whitelist)
    det.load()
    pos = Positioner(cfg.position, w, h)

    best_frame = None
    best_dets: list = []
    # Early-exit as soon as this many objects are on screen; enough to show the
    # pipeline working without burning the whole frame budget.
    target_objects = 2
    for _ in range(args.frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        detections = det.detect(frame)
        if len(detections) > len(best_dets):
            best_dets = detections
            best_frame = frame.copy()
        if len(best_dets) >= target_objects:
            break
    cap.release()

    if best_frame is None:
        print("[FAIL] no usable frame")
        return 2

    for d in best_dets:
        placed = pos.place(d.label, d.confidence, d.box)
        b = placed.box
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(best_frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            best_frame,
            placed.describe(),
            (p1[0], max(20, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        print(f"[DETECT] {placed.describe()} (conf {placed.confidence:.2f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), best_frame)
    print(
        f"[OK] best frame had {len(best_dets)} detections -> {out.resolve()} ({out.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
