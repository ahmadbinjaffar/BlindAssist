"""Capture one annotated frame as visual evidence.

C:\\dev\\ba_venv\\Scripts\\python.exe scripts\\capture_detection.py
    --camera 0 --out Evidence\\detection.jpg
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
    p.add_argument("--out", type=str, default="Evidence/detection.jpg")
    p.add_argument("--conf", type=float, default=0.45)
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
    cap.release()
    if not ok or frame is None:
        print("[FAIL] read")
        return 2

    h, w = frame.shape[:2]
    det = Detector(cfg.vision, cfg.model_path, cfg.class_whitelist)
    det.load()
    detections = det.detect(frame)

    pos = Positioner(cfg.position, w, h)
    for d in detections:
        placed = pos.place(d.label, d.confidence, d.box)
        b = placed.box
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            frame,
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
    cv2.imwrite(str(out), frame)
    print(
        f"[OK] Saved {len(detections)} detections -> {out.resolve()} ({out.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
