"""Measure how accurate the positioner's direction + distance band really are.

Ten images (7 live webcam frames + 3 sample photos) are annotated by hand in
``assets/eval_frames/ground_truth.json``: for each object a human recorded where
they perceive it to be, independently of what the detector outputs. The script
runs the app's real pipeline (``Detector`` -> ``Positioner``) and compares.

    .\\.venv\\Scripts\\python.exe scripts\\evaluate_position.py

Outputs
-------
* ``Evidence/position_accuracy.md``   - the report
* ``Evidence/position_accuracy.json`` - raw counts (for the notebook)
* ``Evidence/eval_position/*.jpg``    - annotated frames showing what was found
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; force UTF-8 so report characters never
# break a run under PowerShell / cmd.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2  # noqa: E402

from blindassist.config import (  # noqa: E402
    DEFAULT_CLASS_WHITELIST,
    PositionConfig,
    TrackerConfig,
    VisionConfig,
)
from blindassist.detector import Detector  # noqa: E402
from blindassist.position import Positioner  # noqa: E402
from blindassist.tracker import Tracker, confirmed_tracks  # noqa: E402

FRAMES_DIR = ROOT / "assets" / "eval_frames"
GT_PATH = FRAMES_DIR / "ground_truth.json"
OUT_DIR = ROOT / "Evidence" / "eval_position"
REPORT_MD = ROOT / "Evidence" / "position_accuracy.md"
REPORT_JSON = ROOT / "Evidence" / "position_accuracy.json"

BAND_COLORS = {"very close": (0, 0, 255), "near": (0, 200, 255), "far": (90, 200, 90)}

# How many times each still image is fed through the v2 pipeline before
# scoring (the app loops a still source the same way while the user listens).
TRACK_PASSES = 6


def match(expected: list[dict], positions, frame_width: float) -> list[dict]:
    """Pair each expected object with the closest same-label detection.

    Pairing uses only the object's **horizontal position** (the human's estimated
    centre-x versus the detector's box centre-x), never the direction word or the
    distance band being scored. Without this, several objects of the same class
    in one frame get cross-matched and correct answers are counted as failures.
    """
    pool = list(positions)
    rows: list[dict] = []
    for exp in expected:
        label = exp["label"]
        candidates = [p for p in pool if p.label == label]
        if not candidates:
            rows.append({**exp, "detected": False})
            continue
        best = min(
            candidates,
            key=lambda p: abs(p.box.center[0] / frame_width - exp["cx_frac"]),
        )
        pool.remove(best)
        # Sanity check: does the hand-written direction agree with the hand-written
        # centre-x under the app's own rule? Reported so annotation slips show up.
        rule_dir = (
            "left"
            if exp["cx_frac"] < 1 / 3
            else "ahead" if exp["cx_frac"] < 2 / 3 else "right"
        )
        rows.append(
            {
                **exp,
                "detected": True,
                "gt_consistent": rule_dir == exp["horizontal"],
                "predicted_horizontal": best.horizontal,
                "predicted_band": best.distance_band,
                "predicted_cx_frac": round(best.box.center[0] / frame_width, 3),
                "confidence": round(best.confidence, 3),
                "area_ratio": round(best.area_ratio, 4),
                "direction_ok": best.horizontal == exp["horizontal"],
                "band_ok": best.distance_band == exp["distance_band"],
            }
        )
    return rows


def annotate(image, positions) -> object:
    """Draw what the detector actually saw, so the report shows real evidence."""
    out = image.copy()
    for p in positions:
        b = p.box
        # round() already returns an int when called without ndigits.
        x1, y1, x2, y2 = (round(v) for v in (b.x1, b.y1, b.x2, b.y2))
        colour = BAND_COLORS.get(p.distance_band, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{p.label} {p.confidence:.2f} {p.horizontal}/{p.distance_band}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), colour, -1)
        cv2.putText(
            out,
            text,
            (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def write_report(
    per_image: list[dict],
    totals: dict,
    conf: float,
    report_md: Path = REPORT_MD,
    pipeline: str = "detector",
) -> None:
    L: list[str] = []
    a = L.append
    exp = totals["expected"]
    det = totals["detected"]

    a("# BlindAssist — Position & Distance Accuracy")
    a("")
    a(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    a("")
    a("## What was tested")
    a("")
    a(f"- **{totals['images']} images**: 7 live webcam frames from this machine's")
    a("  camera plus 3 sample photos.")
    a(f"- **{exp} annotated objects** in `assets/eval_frames/ground_truth.json`.")
    a("- Ground truth was written **by looking at each image**, independently of")
    a("  the detector, using the same three-way split the app implements.")
    if pipeline == "app":
        a(f"- Pipeline under test: the app's own `Detector` (conf={conf}, default")
        a("  class whitelist) feeding the **v2 speech path**: IoU tracker with")
        a("  5-frame box smoothing and the 3-frame confirmation gate, then the")
        a("  `Positioner` — i.e. exactly what a user hears announced.")
    else:
        a(f"- Pipeline under test: the app's own `Detector` (conf={conf}, default")
        a("  class whitelist) feeding the app's own `Positioner`.")
    a("")
    a("## Results")
    a("")
    a("| Measure | Count | Rate |")
    a("|---------|------:|-----:|")
    a(f"| Objects expected | {exp} | — |")
    a(f"| Correctly detected | {det} | **{det / exp:.1%}** |")
    dir_n = totals["direction_correct"]
    band_n = totals["band_correct"]
    both_n = totals["both_correct"]
    a(f"| Direction correct (of those detected) | {dir_n} | **{dir_n / det:.1%}** |")
    a(
        f"| Distance band correct (of those detected) | {band_n} | **{band_n / det:.1%}** |"
    )
    a(f"| Both correct (of those detected) | {both_n} | **{both_n / det:.1%}** |")
    a(
        f"| Both correct (end-to-end over all {exp}) | {both_n} | **{both_n / exp:.1%}** |"
    )
    a("")
    a(f"**Direction: {dir_n}/{det} right. Distance band: {band_n}/{det} right.**")
    a("")

    a("## Per-image detail")
    a("")
    a("| Image | Expected | Found | Direction OK | Band OK | Notes |")
    a("|-------|---------:|------:|-------------:|--------:|-------|")
    for img in per_image:
        rows = img["rows"]
        n = len(rows)
        found = sum(1 for r in rows if r["detected"])
        dok = sum(1 for r in rows if r.get("direction_ok"))
        bok = sum(1 for r in rows if r.get("band_ok"))
        misses = [r["label"] for r in rows if not r["detected"]]
        wrong_d = [
            r["label"] for r in rows if r["detected"] and not r.get("direction_ok")
        ]
        note = []
        if misses:
            note.append(f"missed: {', '.join(misses)}")
        if wrong_d:
            note.append(f"wrong direction: {', '.join(wrong_d)}")
        a(
            f"| {img['image']} | {n} | {found} | {dok} | {bok} | "
            f"{'; '.join(note) if note else '—'} |"
        )
    a("")

    a("## Where it went wrong")
    a("")
    misses = [
        (img["image"], r["label"])
        for img in per_image
        for r in img["rows"]
        if not r["detected"]
    ]
    wrong_dir = [
        (img["image"], r["label"], r["horizontal"], r["predicted_horizontal"])
        for img in per_image
        for r in img["rows"]
        if r["detected"] and not r.get("direction_ok")
    ]
    wrong_band = [
        (img["image"], r["label"], r["distance_band"], r["predicted_band"])
        for img in per_image
        for r in img["rows"]
        if r["detected"] and not r.get("band_ok")
    ]
    a(
        f"- **Missed detections ({len(misses)})**: "
        + (", ".join(f"`{i}` {lb}" for i, lb in misses) if misses else "none")
        + "."
    )
    a(
        f"- **Wrong direction ({len(wrong_dir)})**: "
        + (
            ", ".join(f"`{i}` {lb}: said {pr}, is {ex}" for i, lb, ex, pr in wrong_dir)
            if wrong_dir
            else "none"
        )
        + "."
    )
    a(
        f"- **Wrong distance band ({len(wrong_band)})**: "
        + (
            ", ".join(f"`{i}` {lb}: said {pr}, is {ex}" for i, lb, ex, pr in wrong_band)
            if wrong_band
            else "none"
        )
        + "."
    )

    inconsistent = []
    for img in per_image:
        for r in img["rows"]:
            cf = r["cx_frac"]
            rule = "left" if cf < 1 / 3 else ("ahead" if cf < 2 / 3 else "right")
            if rule != r["horizontal"]:
                inconsistent.append(
                    (img["image"], r["label"], cf, r["horizontal"], rule)
                )
    if inconsistent:
        detail = " — " + ", ".join(
            f"`{i}` {lb} (cx={cf:.2f}, wrote {w}, rule says {ru})"
            for i, lb, cf, w, ru in inconsistent
        )
    else:
        detail = "."
    a(f"- **Ground-truth self-consistency**: {len(inconsistent)} of {exp} hand-written")
    a("  directions disagree with the hand-written centre-x under the app's own")
    a(f"  three-way split{detail}")
    if not inconsistent:
        a("  That check confirms the annotations are internally consistent.")
    a("")

    a("## Annotated frames (what the detector actually saw)")
    a("")
    for img in per_image:
        a(f"**{img['image']}** — {img['scene']}")
        a("")
        a(f"![{img['image']}](eval_position/{img['image']})")
        a("")

    a("## Honest limits on this evaluation")
    a("")
    a("- **Direction is objective; distance is not.** There is no depth sensor in")
    a('  this project, so "distance" is only a bounding-box-area heuristic. The')
    a("  ground truth for it is a human judging apparent size, so the distance")
    a("  column measures self-consistency of the heuristic, **not** metres.")
    a("- **The images come from the same scene.** Six of the seven webcam frames")
    a("  are one room with the same three people, so they are not independent")
    a("  samples; they mainly test the close/face and small/background cases.")
    a("- **Pairing is by position, not by outcome.** When a frame contains several")
    a("  objects of the same class, each expectation is paired with the detection")
    a("  whose box centre is closest to the human's estimated centre-x. The")
    a("  direction word and distance band being scored play no part in the pairing.")
    a('  "Detected" simply means a box of the right class was produced somewhere in')
    a("  the frame — it is not an IoU test, so a loosely placed box counts as found")
    a("  and is then judged on its direction.")
    a("- **Small, dim, cropped objects are the failure mode.** Objects that are")
    a("  partly outside the frame or badly lit are the ones that get missed, which")
    a("  is the worst case for a blind user who needs to know about them.")
    a("- **One operating point.** This runs at the app's default `conf=0.45`.")
    a("  Lowering it would recover more objects but add false alarms.")
    a("")
    report_md.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate direction/distance accuracy")
    p.add_argument("--conf", type=float, default=VisionConfig.confidence)
    p.add_argument("--frames-dir", default=str(FRAMES_DIR))
    p.add_argument("--gt", default=str(GT_PATH))
    p.add_argument(
        "--pipeline",
        choices=("detector", "app"),
        default="detector",
        help="'detector' = Detector->Positioner (v1 path); 'app' = the v2 speech "
        "path with tracker smoothing + 3-frame confirmation",
    )
    p.add_argument("--out", default=None, help="output directory (default: Evidence/)")
    args = p.parse_args(argv)

    frames_dir = Path(args.frames_dir)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    base = Path(args.out) if args.out else OUT_DIR.parent
    out_dir = base / OUT_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = base / REPORT_MD.name
    report_json = base / REPORT_JSON.name

    vision = VisionConfig()
    vision.confidence = args.conf
    detector = Detector(
        vision,
        model_path=ROOT / "assets" / "yolov8n.pt",
        class_whitelist=set(DEFAULT_CLASS_WHITELIST),
    )
    detector.load()
    pos_cfg = PositionConfig()
    tracker_cfg = TrackerConfig()

    per_image: list[dict] = []
    totals = {
        "images": 0,
        "expected": 0,
        "detected": 0,
        "direction_correct": 0,
        "band_correct": 0,
        "both_correct": 0,
    }

    for entry in gt["images"]:
        path = frames_dir / entry["image"]
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] unreadable: {path}", file=sys.stderr)
            continue
        h, w = image.shape[:2]
        if args.pipeline == "app":
            # v2 speech path: the still image loops like the app's video source,
            # the tracker smooths the box over 5 frames and only tracks that
            # passed the 3-frame confirmation gate are announced at all.
            tracker = Tracker(tracker_cfg)
            positioner = Positioner(pos_cfg, w, h)
            for _ in range(TRACK_PASSES):
                tracks = tracker.update(detector.detect(image))
            confirmed = confirmed_tracks(tracks, tracker_cfg.min_confirm_frames)
            positions = [
                positioner.place(t.label, t.confidence, t.box) for t in confirmed
            ]
        else:
            positions = [
                Positioner(pos_cfg, w, h).place(d.label, d.confidence, d.box)
                for d in detector.detect(image)
            ]
        rows = match(entry["expected"], positions, w)

        totals["images"] += 1
        totals["expected"] += len(rows)
        totals["detected"] += sum(1 for r in rows if r["detected"])
        totals["direction_correct"] += sum(1 for r in rows if r.get("direction_ok"))
        totals["band_correct"] += sum(1 for r in rows if r.get("band_ok"))
        totals["both_correct"] += sum(
            1 for r in rows if r.get("direction_ok") and r.get("band_ok")
        )

        cv2.imwrite(str(out_dir / entry["image"]), annotate(image, positions))
        per_image.append(
            {"image": entry["image"], "scene": entry["scene"], "rows": rows}
        )
        print(
            f"[OK] {entry['image']}: {sum(1 for r in rows if r['detected'])}"
            f"/{len(rows)} objects found, {len(positions)} detections in frame"
        )

    if totals["expected"] == 0:
        print("[ERROR] no ground truth rows evaluated", file=sys.stderr)
        return 2

    write_report(per_image, totals, args.conf, report_md, args.pipeline)
    report_json.write_text(
        json.dumps(
            {"config": {"conf": args.conf}, "totals": totals, "images": per_image},
            indent=2,
        ),
        encoding="utf-8",
    )

    det, exp = totals["detected"], totals["expected"]
    print("\n=== POSITION ACCURACY ===")
    print(f"  Expected objects      : {exp}")
    print(f"  Detected              : {det} ({det / exp:.1%})")
    print(
        f"  Direction correct     : {totals['direction_correct']}/{det} "
        f"({totals['direction_correct'] / det:.1%})"
    )
    print(
        f"  Distance band correct : {totals['band_correct']}/{det} "
        f"({totals['band_correct'] / det:.1%})"
    )
    print(
        f"  Both correct (e2e)    : {totals['both_correct']}/{exp} "
        f"({totals['both_correct'] / exp:.1%})"
    )
    print(f"\n[OK] {report_md}")
    print(f"[OK] {report_json}")
    print(f"[OK] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
