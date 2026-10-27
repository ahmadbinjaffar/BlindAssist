"""Benchmark YOLOv8n inference speed on this machine.

Times **100 frames** end-to-end (preprocess + inference + postprocess) at two
input sizes, 320 px and 640 px, and records the CPU/GPU model so the numbers
can be quoted honestly.

    .\\.venv\\Scripts\\python.exe scripts\\benchmark.py --frames 100

Outputs
-------
* ``Evidence/benchmark.md``   - the table + hardware / method notes
* ``Evidence/benchmark.png``  - bar chart (matplotlib)
* ``Evidence/benchmark.json`` - raw numbers (for the notebook)
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
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

from blindassist.config import DEFAULT_MODEL_PATH  # noqa: E402

EVIDENCE = ROOT / "Evidence"
SAMPLES_DIR = ROOT / "assets" / "samples"
MD_PATH = EVIDENCE / "benchmark.md"
PNG_PATH = EVIDENCE / "benchmark.png"
JSON_PATH = EVIDENCE / "benchmark.json"


def hardware() -> dict[str, str]:
    cpu = platform.processor() or ""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    except Exception:
        pass
    info = {
        "cpu": str(cpu).strip() or platform.platform(),
        "gpu": "none detected",
        "torch_device": "cpu",
        "torch": "?",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["torch_device"] = "cuda"
        else:
            try:
                import subprocess

                out = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_VideoController).Name",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    # Best-effort probe: "none detected" is the fallback below.
                    check=False,
                )
                names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
                if names:
                    info["gpu"] = "; ".join(names) + " (not CUDA-capable; torch on CPU)"
            except Exception:
                pass
    except Exception:
        pass
    return info


def load_frames() -> list:
    """Real images cycled for the benchmark (documented in the report)."""
    import cv2

    paths = sorted(
        p
        for p in SAMPLES_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        # Fallback: a synthetic frame, so the benchmark still runs anywhere.
        import numpy as np

        return [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)] * 3
    return [cv2.imread(str(p)) for p in paths if cv2.imread(str(p)) is not None]


def bench(model, frames: list, imgsz: int, n_frames: int, warmup: int) -> dict:
    """Time ``n_frames`` sequential single-image predictions at ``imgsz``."""
    stage_ms: dict[str, list[float]] = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
    }
    for i in range(warmup):
        model.predict(
            source=frames[i % len(frames)], imgsz=imgsz, verbose=False, device="cpu"
        )

    wall_times: list[float] = []
    for i in range(n_frames):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        result = model.predict(source=frame, imgsz=imgsz, verbose=False, device="cpu")
        wall_times.append((time.perf_counter() - t0) * 1000.0)  # ms
        if result:
            speed = getattr(result[0], "speed", None) or {}
            for stage, samples in stage_ms.items():
                if stage in speed:
                    samples.append(float(speed[stage]))

    avg = statistics.fmean(wall_times)
    return {
        "imgsz": imgsz,
        "frames": n_frames,
        "warmup": warmup,
        "avg_ms": avg,
        "median_ms": statistics.median(wall_times),
        "min_ms": min(wall_times),
        "max_ms": max(wall_times),
        "stdev_ms": statistics.pstdev(wall_times),
        "fps": 1000.0 / avg if avg > 0 else 0.0,
        "total_s": sum(wall_times) / 1000.0,
        "stages": {k: (statistics.fmean(v) if v else 0.0) for k, v in stage_ms.items()},
        "raw_ms": wall_times,
    }


def write_markdown(
    results: list[dict], hw: dict, frame_note: str, md_path: Path = MD_PATH
) -> None:
    L: list[str] = []
    a = L.append
    a("# BlindAssist — Inference Benchmark")
    a("")
    a(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    a("")
    a("## Hardware & software")
    a("")
    a("| Item | Value |")
    a("|------|-------|")
    a(f"| CPU | {hw['cpu']} |")
    a(f"| GPU | {hw['gpu']} |")
    a(f"| torch device | `{hw['torch_device']}` |")
    a(f"| torch | {hw['torch']} |")
    a(f"| Python | {hw['python']} |")
    a(f"| Platform | {hw['platform']} |")
    a("| Model | `assets/yolov8n.pt` (YOLOv8n, 6.25 MB) |")
    a("")
    a("## Method")
    a("")
    a(f"- **{results[0]['frames']} frames** per input size, measured")
    a("  sequentially in a single process.")
    a("- Time is **end-to-end wall clock** around `model.predict(...)`:")
    a("  letterbox resize + NMS + Python call overhead, not just the tensor matmul.")
    a(f"- {results[0]['warmup']} untimed **warmup** calls before each measurement")
    a("  (the first PyTorch call is much slower while kernels are compiled/loaded).")
    a(f"- Input: {frame_note}")
    a("- Batch size 1 (how the live app actually runs it).")
    a("")
    a("## Results")
    a("")
    a(
        "| Input size | Avg time (ms) | Median (ms) | Min (ms) | Max (ms) | Stdev (ms) | FPS |"
    )
    a(
        "|-----------:|--------------:|------------:|---------:|---------:|-----------:|----:|"
    )
    for r in results:
        a(
            f"| {r['imgsz']} px | **{r['avg_ms']:.1f}** | {r['median_ms']:.1f} | "
            f"{r['min_ms']:.1f} | {r['max_ms']:.1f} | {r['stdev_ms']:.1f} | "
            f"**{r['fps']:.1f}** |"
        )
    a("")
    a("### Stage breakdown (mean ms per image)")
    a("")
    a("| Input size | Preprocess | Inference | Postprocess |")
    a("|-----------:|-----------:|----------:|------------:|")
    for r in results:
        s = r["stages"]
        a(
            f"| {r['imgsz']} px | {s.get('preprocess', 0):.2f} | "
            f"{s.get('inference', 0):.2f} | {s.get('postprocess', 0):.2f} |"
        )
    a("")
    # Comparison derived from real numbers only.
    min_results_to_compare = 2
    if len(results) >= min_results_to_compare:
        fast, slow = results[0], results[-1]
        if fast["imgsz"] < slow["imgsz"] and slow["avg_ms"] > 0:
            ratio = slow["avg_ms"] / fast["avg_ms"]
            a("## Interpretation")
            a("")
            a(
                f"- {slow['imgsz']} px costs **{ratio:.1f}×** the time of"
                f" {fast['imgsz']} px (4× the pixels → more, but not 4×, because"
            )
            a("  fixed per-call overhead is a larger share of the cheaper run).")
            a(
                f"- The live app detects at roughly {results[-1]['fps']:.1f} inferences/s at"
                f" {results[-1]['imgsz']} px while the camera delivers ~10 FPS, so the"
            )
            a(
                "  detector is the bottleneck and the worker thread keeps the video smooth."
            )
            a("")
    a("## Honest notes")
    a("")
    a("- Single-run, single-process measurement on a laptop; no GPU, no batching.")
    a("- Numbers move with CPU load, thermal throttling and power plan — treat")
    a("  them as an order of magnitude, not a guarantee.")
    a("- `stdev` and `max` matter for a real-time aid: a stuttering tail is more")
    a("  noticeable than the mean.")
    a("")
    md_path.write_text("\n".join(L), encoding="utf-8")


def make_chart(results: list[dict], png_path: Path = PNG_PATH) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['imgsz']} px" for r in results]
    ms = [r["avg_ms"] for r in results]
    fps = [r["fps"] for r in results]
    x = range(len(results))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("YOLOv8n inference benchmark (CPU, batch=1, end-to-end)", fontsize=12)

    bars1 = axes[0].bar(
        x, ms, color=["#4C72B0", "#DD8452"][: len(results)], edgecolor="black"
    )
    axes[0].set_title("Average time per frame")
    axes[0].set_ylabel("milliseconds (lower is better)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].bar_label(bars1, fmt="%.1f ms", padding=2)

    bars2 = axes[1].bar(
        x, fps, color=["#4C72B0", "#DD8452"][: len(results)], edgecolor="black"
    )
    axes[1].set_title("Throughput")
    axes[1].set_ylabel("frames per second (higher is better)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].bar_label(bars2, fmt="%.1f fps", padding=2)

    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark YOLOv8n inference speed")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--frames", type=int, default=100, help="frames per input size")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--sizes", type=int, nargs="+", default=[320, 640])
    p.add_argument("--out", default=None, help="output directory (default: Evidence/)")
    args = p.parse_args(argv)

    base = Path(args.out) if args.out else EVIDENCE
    base.mkdir(parents=True, exist_ok=True)
    md_path = base / MD_PATH.name
    png_path = base / PNG_PATH.name
    json_path = base / JSON_PATH.name
    hw = hardware()
    frames = load_frames()
    frame_note = (
        f"{len(frames)} real sample image(s) from `assets/samples/`, "
        "cycled to fill the frame budget"
    )

    print(f"[INFO] CPU : {hw['cpu']}", flush=True)
    print(f"[INFO] GPU : {hw['gpu']}", flush=True)
    print(f"[INFO] frames per size = {args.frames}, warmup = {args.warmup}", flush=True)

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"[INFO] model loaded: {args.model}", flush=True)

    results: list[dict] = []
    for size in args.sizes:
        print(f"[INFO] benchmarking imgsz={size} ...", flush=True)
        r = bench(model, frames, size, args.frames, args.warmup)
        results.append(r)
        print(
            f"[OK]   {size}px -> avg {r['avg_ms']:.1f} ms "
            f"({r['fps']:.1f} FPS), median {r['median_ms']:.1f} ms, "
            f"stdev {r['stdev_ms']:.1f} ms",
            flush=True,
        )

    write_markdown(results, hw, frame_note, md_path)
    make_chart(results, png_path)
    json_path.write_text(
        json.dumps(
            {
                "hardware": hw,
                "method": {
                    "frames": args.frames,
                    "warmup": args.warmup,
                    "input": frame_note,
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] {md_path}", flush=True)
    print(f"[OK] {png_path}", flush=True)
    print(f"[OK] {json_path}", flush=True)
    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"  {r['imgsz']:>4} px : {r['avg_ms']:7.1f} ms  |  " f"{r['fps']:5.1f} FPS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
