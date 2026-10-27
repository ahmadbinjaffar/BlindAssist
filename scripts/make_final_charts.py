"""Build the two comparison charts in Evidence/final/ from result JSONs.

Chart 1  chart_fps_before_after.png
    Inference latency / throughput at 320 and 640 px, BEFORE (the original
    v1 benchmark run, Evidence/benchmark.json) vs AFTER (the v2 re-run,
    Evidence/final/benchmark.json).

Chart 2  chart_stability_lowlight.png
    Two panels: spoken announcements on the shared 60 s video before vs after
    the stability filter, and total detections on the dark sample images
    before (raw) vs after (CLAHE).

Every number is read from a JSON written by an actual run - nothing is
hard-coded. Missing inputs abort with a clear error instead of guessing.

Usage:  python scripts/make_final_charts.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "Evidence" / "final"

BEFORE_BENCH = ROOT / "Evidence" / "benchmark.json"
AFTER_BENCH = FINAL / "benchmark.json"
STABILITY_BEFORE = FINAL / "stability_before.json"
STABILITY_AFTER = FINAL / "stability_after.json"
LOWLIGHT = FINAL / "lowlight.json"

OUT_CHART1 = FINAL / "chart_fps_before_after.png"
OUT_CHART2 = FINAL / "chart_stability_lowlight.png"

BAR_BEFORE = "#4C72B0"
BAR_AFTER = "#DD8452"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[ERROR] missing required result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def bench_map(data: dict) -> dict[int, dict]:
    return {int(r["imgsz"]): r for r in data["results"]}


def chart_fps(before: dict, after: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(set(before) & set(after))
    if not sizes:
        raise SystemExit("[ERROR] no input sizes in common between the two runs")

    x = range(len(sizes))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle(
        "Detector speed BEFORE (v1 benchmark) vs AFTER (v2 re-run), CPU, batch=1",
        fontsize=12,
    )

    ms_b = [before[s]["avg_ms"] for s in sizes]
    ms_a = [after[s]["avg_ms"] for s in sizes]
    b1 = axes[0].bar(
        [i - width / 2 for i in x],
        ms_b,
        width,
        label="before",
        color=BAR_BEFORE,
        edgecolor="black",
    )
    b2 = axes[0].bar(
        [i + width / 2 for i in x],
        ms_a,
        width,
        label="after",
        color=BAR_AFTER,
        edgecolor="black",
    )
    axes[0].set_title("Average latency per frame")
    axes[0].set_ylabel("milliseconds (lower is better)")
    axes[0].bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
    axes[0].bar_label(b2, fmt="%.1f", padding=2, fontsize=9)

    fps_b = [before[s]["fps"] for s in sizes]
    fps_a = [after[s]["fps"] for s in sizes]
    b3 = axes[1].bar(
        [i - width / 2 for i in x],
        fps_b,
        width,
        label="before",
        color=BAR_BEFORE,
        edgecolor="black",
    )
    b4 = axes[1].bar(
        [i + width / 2 for i in x],
        fps_a,
        width,
        label="after",
        color=BAR_AFTER,
        edgecolor="black",
    )
    axes[1].set_title("Throughput")
    axes[1].set_ylabel("frames per second (higher is better)")
    axes[1].bar_label(b3, fmt="%.1f", padding=2, fontsize=9)
    axes[1].bar_label(b4, fmt="%.1f", padding=2, fontsize=9)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{s} px" for s in sizes])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_CHART1, dpi=130)
    plt.close(fig)


def chart_behaviour(stab_before: dict, stab_after: dict, lowlight: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle("Behaviour BEFORE vs AFTER the v2 improvements", fontsize=12)

    # Left: spoken announcements on the shared 60 s video.
    values = [stab_before["announcements"], stab_after["announcements"]]
    bars = axes[0].bar(
        ["before\n(no gate) ", "after\n(3-frame gate)"],
        values,
        color=[BAR_BEFORE, BAR_AFTER],
        edgecolor="black",
    )
    axes[0].set_title("Spoken announcements in one 60 s video")
    axes[0].set_ylabel("announcement count (lower = fewer false alarms)")
    axes[0].bar_label(bars, fmt="%d", padding=3, fontsize=11)
    axes[0].set_ylim(bottom=0)

    # Right: detections on the dark sample images.
    lt = lowlight["totals"]
    bars2 = axes[1].bar(
        ["before\n(raw dark frame)", "after\n(CLAHE)"],
        [lt["before_raw"], lt["after_clahe"]],
        color=[BAR_BEFORE, BAR_AFTER],
        edgecolor="black",
    )
    axes[1].set_title(f"Detections on {lt['images']} dark sample images")
    axes[1].set_ylabel("objects detected (higher = better in the dark)")
    axes[1].bar_label(bars2, fmt="%d", padding=3, fontsize=11)
    axes[1].set_ylim(bottom=0)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_CHART2, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Evidence/final comparison charts")
    p.add_argument("--before-bench", default=str(BEFORE_BENCH))
    p.add_argument("--after-bench", default=str(AFTER_BENCH))
    args = p.parse_args(argv)

    before = bench_map(load(Path(args.before_bench)))
    after = bench_map(load(Path(args.after_bench)))
    stab_before = load(STABILITY_BEFORE)
    stab_after = load(STABILITY_AFTER)
    lowlight = load(LOWLIGHT)

    FINAL.mkdir(parents=True, exist_ok=True)
    chart_fps(before, after)
    chart_behaviour(stab_before, stab_after, lowlight)
    print(f"[OK] {OUT_CHART1}")
    print(f"[OK] {OUT_CHART2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
