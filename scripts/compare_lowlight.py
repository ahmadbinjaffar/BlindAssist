"""Measure detection on the dark sample images: raw vs v2 CLAHE enhancement.

For every image this reports:

* mean luminance (proof the scene really is dark),
* detections the **v1** app would have produced (raw frame, no CLAHE existed),
* detections the **v2** app feeds to the detector (CLAHE-brightened frame),
* the labels found in each case.

The detector, confidence and class whitelist are identical in both cases, so
any difference comes from the enhancement alone. A JSON result is written for
the comparison chart.

Usage:
    python scripts/compare_lowlight.py --json Evidence/final/lowlight.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from blindassist.config import (  # noqa: E402
    DEFAULT_CLASS_WHITELIST,
    LightConfig,
    VisionConfig,
)
from blindassist.detector import Detector  # noqa: E402
from blindassist.lighting import enhance, mean_luma  # noqa: E402

DARK_IMAGES = (
    "dark_person.jpg",
    "dark_chair.jpg",
    "dark_bottle.jpg",
)


def labels_of(detections) -> list[str]:
    return sorted(d.label for d in detections)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Low-light detections before/after CLAHE")
    p.add_argument("--samples", default=str(ROOT / "assets" / "samples"))
    p.add_argument("--json", default=None, help="where to write the result JSON")
    args = p.parse_args(argv)

    samples = Path(args.samples)
    light = LightConfig()
    detector = Detector(
        VisionConfig(), ROOT / "assets" / "yolov8n.pt", set(DEFAULT_CLASS_WHITELIST)
    )
    detector.load()

    images = []
    for name in DARK_IMAGES:
        path = samples / name
        if not path.exists():
            print(f"[WARN] missing {path}", file=sys.stderr)
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] unreadable {path}", file=sys.stderr)
            continue

        luma = mean_luma(image)
        raw = detector.detect(image)
        bright = enhance(image, light)
        lit = detector.detect(bright)

        row = {
            "image": name,
            "mean_luma": round(luma, 1),
            "before_raw": len(raw),
            "before_labels": labels_of(raw),
            "after_clahe": len(lit),
            "after_labels": labels_of(lit),
        }
        images.append(row)
        print(
            f"[OK] {name}: luma={row['mean_luma']}  "
            f"before={row['before_raw']} {row['before_labels']}  "
            f"after={row['after_clahe']} {row['after_labels']}"
        )

    if not images:
        print("[ERROR] no dark images found", file=sys.stderr)
        return 2

    totals = {
        "before_raw": sum(r["before_raw"] for r in images),
        "after_clahe": sum(r["after_clahe"] for r in images),
        "images": len(images),
    }
    result = {
        "threshold_luma": light.dark_mean_threshold,
        "clahe_clip": light.clip_limit,
        "totals": totals,
        "images": images,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"\n[RESULT] total detections on {totals['images']} dark images: "
        f"before={totals['before_raw']}  after={totals['after_clahe']}",
        flush=True,
    )
    if args.json:
        print(f"[OK] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
