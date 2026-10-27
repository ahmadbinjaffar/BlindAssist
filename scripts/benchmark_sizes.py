"""Detector input-size sweep (improvement feature: selectable 320/480/640).

Times BlindAssist's real ``Detector`` pipeline (preprocess + inference +
postprocess + whitelist filter) on the bundled sample images at every
supported input size and writes ``Evidence/benchmark_sizes.md``.

Unlike ``scripts/benchmark.py`` (which times raw ``model.predict`` and owns
``Evidence/benchmark.md``), this script only ever writes its own file, so the
original benchmark evidence stays untouched.

Usage:  python scripts/benchmark_sizes.py [--frames N] [--warmup N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from blindassist.config import (  # noqa: E402
    DEFAULT_CLASS_WHITELIST,
    INPUT_SIZES,
    load_config,
)
from blindassist.detector import Detector  # noqa: E402

SAMPLE_NAMES = ("person.jpg", "chair.jpg", "bottle.jpg")
MD_PATH = ROOT / "Evidence" / "benchmark_sizes.md"


def load_frames() -> list:
    frames = []
    for name in SAMPLE_NAMES:
        img = cv2.imread(str(ROOT / "assets" / "samples" / name))
        if img is not None:
            frames.append(img)
    if not frames:
        raise SystemExit(f"no sample images found in {ROOT / 'assets' / 'samples'}")
    return frames


def bench(
    detector: Detector, frames: list, size: int, n_frames: int, warmup: int
) -> dict:
    """Time ``n_frames`` sequential detections at ``size`` input pixels."""
    detector.cfg.inference_size = size
    for i in range(warmup):
        detector.detect(frames[i % len(frames)])
    times: list[float] = []
    for i in range(n_frames):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        detector.detect(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    avg = statistics.fmean(times)
    return {
        "size": size,
        "avg": avg,
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "fps": 1000.0 / avg,
    }


def write_markdown(results: list[dict], n_frames: int, warmup: int) -> None:
    import platform

    L: list[str] = []
    L.append("# Benchmark: detector input size (320 / 480 / 640 px)")
    L.append("")
    L.append("Evidence for improvement feature 8 (selectable input size).")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append("- Full BlindAssist `Detector.detect()` pipeline (YOLOv8n, CPU).")
    L.append(f"- Sample images: {', '.join(SAMPLE_NAMES)} cycled.")
    L.append(f"- {n_frames} timed detections per size after {warmup} warm-up calls.")
    L.append(f"- Machine: {platform.machine()}, {platform.processor() or 'CPU'}.")
    L.append(
        "- `--size` in the app selects the same values "
        f"(`INPUT_SIZES = {INPUT_SIZES}`)."
    )
    L.append("")
    L.append("## Results")
    L.append("")
    L.append(
        "| Input size | Avg (ms) | Median (ms) | Min (ms) | Max (ms) | Detections/s |"
    )
    L.append("|---|---|---|---|---|---|")
    for r in results:
        L.append(
            f"| {r['size']} px | {r['avg']:.1f} | {r['median']:.1f} | "
            f"{r['min']:.1f} | {r['max']:.1f} | **{r['fps']:.1f}** |"
        )
    L.append("")
    best, worst = min(results, key=lambda r: r["avg"]), max(
        results, key=lambda r: r["avg"]
    )
    L.append(
        f"- {best['size']} px is **{worst['avg'] / best['avg']:.2f}x** faster per "
        f"detection than {worst['size']} px on this CPU."
    )
    L.append(
        "- In the app the effect shows as *more analyzed frames per second* "
        "(the preview loop itself is paced separately) and, with `--skip N`, "
        "as fewer frames dropped by the detection worker."
    )
    L.append("")
    MD_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"[INFO] wrote {MD_PATH}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BlindAssist detector input-size benchmark")
    p.add_argument("--frames", type=int, default=40, help="timed detections per size")
    p.add_argument("--warmup", type=int, default=5, help="warm-up calls per size")
    p.add_argument("--sizes", type=int, nargs="+", default=list(INPUT_SIZES))
    args = p.parse_args(argv)

    cfg = load_config()
    frames = load_frames()
    detector = Detector(cfg.vision, cfg.model_path, DEFAULT_CLASS_WHITELIST)
    detector.load()
    print(
        f"[INFO] model loaded: {detector.model_path} on {detector.device}", flush=True
    )

    results = []
    for size in args.sizes:
        print(f"[INFO] benchmarking imgsz={size} ...", flush=True)
        r = bench(detector, frames, size, args.frames, args.warmup)
        results.append(r)
        print(
            f"      {r['size']:>4} px : {r['avg']:7.1f} ms  |  {r['fps']:5.1f} FPS",
            flush=True,
        )

    write_markdown(results, args.frames, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
