"""Validate the YOLOv8n detector on three free sample images.

Downloads one image each for **person**, **chair** and **bottle**, runs
:class:`blindassist.detector.Detector`, saves an annotated copy to
``Evidence/model_validation/``, and prints the detections plus a PASS/FAIL line
for the expected class.

    C:\\dev\\ba_venv\\Scripts\\python.exe scripts\\validate_detector.py

Sources (all free / openly licensed, direct download):
* person  - ultralytics sample bus.jpg      (CC-BY, ultralytics assets)
* chair   - COCO val2017 000000000139.jpg   (COCO, CC-BY 4.0)
* bottle  - COCO val2017 000000000632.jpg   (COCO, CC-BY 4.0)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blindassist.config import load_config  # noqa: E402
from blindassist.detector import Detector  # noqa: E402

OUT_DIR = ROOT / "Evidence" / "model_validation"

SOURCES = {
    "person": {
        "url": "https://ultralytics.com/images/bus.jpg",
        "expect": "person",
    },
    "chair": {
        "url": "http://images.cocodataset.org/val2017/000000000139.jpg",
        "expect": "chair",
    },
    "bottle": {
        "url": "http://images.cocodataset.org/val2017/000000000632.jpg",
        "expect": "bottle",
    },
}

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"
}


def download(url: str):
    r = requests.get(url, timeout=60, headers=_UA)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not decode image from {url}")
    return img


def annotate(img, detections, expect: str):
    out = img.copy()
    for d in detections:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        color = (0, 255, 0) if d.label == expect else (0, 200, 255)
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(
            out,
            f"{d.label} {d.confidence:.2f}",
            (p1[0], max(18, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out


def main(argv=None) -> int:
    cfg = load_config()
    # Keep ALL 80 COCO classes for validation (no whitelist filtering).
    det = Detector(cfg.vision, cfg.model_path, class_whitelist=None)
    det.load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[model] {det.model_path.name}  device={det.device}  classes={len(det.names)}  "
        f"conf={cfg.vision.confidence}  iou={cfg.vision.iou}  imgsz={cfg.vision.inference_size}\n"
    )

    results = []
    for name, meta in SOURCES.items():
        img = download(meta["url"])
        detections = det.detect(img)
        annotated = annotate(img, detections, meta["expect"])
        out_path = OUT_DIR / f"{name}.jpg"
        cv2.imwrite(str(out_path), annotated)

        labels = [f"{d.label}:{d.confidence:.2f}" for d in detections]
        found = any(d.label == meta["expect"] for d in detections)
        results.append((name, found, out_path, labels))
        print(f"[{name}] {meta['url']}")
        print(f"    -> {len(detections)} detections: {labels}")
        print(f"    -> saved: {out_path}")
        print(f"    -> expected '{meta['expect']}': {'PASS' if found else 'FAIL'}\n")

    passed = sum(1 for _, found, _, _ in results if found)
    print(f"SUMMARY: {passed}/{len(results)} expected classes detected.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
