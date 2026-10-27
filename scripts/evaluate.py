"""Evaluate YOLOv8n on a free COCO validation subset and write Evidence/evaluation.md.

Uses ultralytics' bundled **coco128** dataset (128 images sampled from COCO
val2017, openly licensed). It is downloaded once and cached under ``datasets/``,
so after the first run this is fully offline.

Standard COCO-style evaluation settings are used (``conf=0.001``, ``iou=0.6``)
because mAP must be computed over *all* detections, not just the confident ones
the app would actually announce.

    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py

Outputs
-------
* ``Evidence/evaluation.md``     - the human-readable report
* ``Evidence/eval_metrics.json`` - overall metrics + config (for the notebook)
* ``Evidence/eval_per_class.json``- per-class AP / support (for the notebook)
* ``Evidence/eval/``             - PR / F1 / confusion-matrix plots
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
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
EVAL_DIR = EVIDENCE / "eval"
METRICS_JSON = EVIDENCE / "eval_metrics.json"
PER_CLASS_JSON = EVIDENCE / "eval_per_class.json"
REPORT_MD = EVIDENCE / "evaluation.md"

# Plots ultralytics writes when plots=True; copied into Evidence/eval/.
# (8.4.x prefixes the curve plots with the task name, e.g. "BoxPR_curve.png".)
PLOT_FILES = (
    "confusion_matrix_normalized.png",
    "confusion_matrix.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    # Ground-truth vs prediction panels: used as detection examples in the notebook.
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_labels.jpg",
    "val_batch2_pred.jpg",
)


def hardware() -> dict[str, str]:
    """CPU / GPU description for the report header."""
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
    cpu = cpu or platform.platform()

    info = {"cpu": str(cpu).strip(), "gpu": "none detected", "torch_device": "cpu"}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["torch_device"] = "cuda"
        else:
            # Integrated GPU may exist even though torch cannot use it.
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
                    # Best-effort probe: a missing/gone CIM class is expected on
                    # non-Windows hosts, so the GPU name just stays "none detected".
                    check=False,
                )
                names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
                if names:
                    info["gpu"] = (
                        "; ".join(names) + " (not CUDA-capable; torch runs on CPU)"
                    )
            except Exception:
                pass
    except Exception:
        pass
    return info


def _metrics_of(results):
    """YOLO.val() returns DetMetrics (sometimes wrapped in a list)."""
    if isinstance(results, (list, tuple)):
        results = results[0]
    return results


def collect(results) -> tuple[dict, list[dict]]:
    """Split ultralytics' DetMetrics into (overall, per-class) dictionaries."""
    m = _metrics_of(results)
    names: dict[int, str] = dict(getattr(m, "names", {}) or {})

    mp, mr, map50, map5095 = [float(v) for v in m.mean_results()]
    overall = {
        "precision": mp,
        "recall": mr,
        "map50": float(map50),
        "map5095": float(map5095),
        "speed_ms": {k: float(v) for k, v in getattr(m, "speed", {}).items()},
        "results_dict": {
            k: float(v)
            for k, v in m.results_dict.items()
            if isinstance(v, (int, float))
        },
    }

    box = m.box
    # These three arrays are positionally aligned (see Metric.class_result).
    ap_class_index = [int(c) for c in box.ap_class_index]
    ap50 = [float(a) for a in box.ap50]
    ap = [float(a) for a in box.ap]
    p_arr = [float(x) for x in getattr(box, "p", [])]
    r_arr = [float(x) for x in getattr(box, "r", [])]

    nt = getattr(m, "nt_per_class", None)  # ground-truth instances/class
    nt = [int(x) for x in nt] if nt is not None else None

    per_class: list[dict] = []
    for pos, cid in enumerate(ap_class_index):
        cls_id = int(cid)
        per_class.append(
            {
                "class_id": cls_id,
                "name": str(names.get(cls_id, cls_id)),
                "instances": (nt[cls_id] if nt is not None and cls_id < len(nt) else 0),
                "precision": p_arr[pos] if pos < len(p_arr) else None,
                "recall": r_arr[pos] if pos < len(r_arr) else None,
                "ap50": ap50[pos],
                "ap5095": ap[pos],
            }
        )

    # Lengths must match for class_result() to be trustworthy; report honestly.
    overall["aligned_p_r"] = len(p_arr) == len(ap50) == len(ap)
    overall["classes_evaluated"] = len(per_class)
    overall["classes_in_model"] = len(names)

    # Confusion matrix (rows = ground truth, cols = prediction; last = no pred).
    cm = getattr(m, "confusion_matrix", None)
    if cm is not None and getattr(cm, "matrix", None) is not None:
        overall["confusion_matrix"] = cm.matrix.astype(int).tolist()
        overall["confusion_names"] = [str(n) for n in cm.names.values()]
    return overall, per_class


def write_report(
    overall: dict,
    per_class: list[dict],
    cfg: dict,
    hw: dict,
    wall_s: float,
    dataset_note: str,
    report_md: Path = REPORT_MD,
) -> Path:
    lines: list[str] = []
    a = lines.append

    a("# BlindAssist — Model Evaluation Report")
    a("")
    a(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    a("")
    a("## 1. Configuration")
    a("")
    a("| Item | Value |")
    a("|------|-------|")
    a(f"| Model | `{cfg['model']}` (YOLOv8n, COCO-pretrained, no fine-tuning) |")
    a(f"| Dataset | `{cfg['data']}` — {dataset_note} |")
    a(f"| Input size | {cfg['imgsz']} px |")
    a(f"| Confidence threshold | {cfg['conf']} (standard for mAP) |")
    a(f"| IoU / NMS threshold | {cfg['iou']} |")
    a(f"| CPU | {hw['cpu']} |")
    a(f"| GPU | {hw['gpu']} |")
    a(f"| torch device | `{hw['torch_device']}` |")
    a(f"| Wall time | {wall_s:.1f} s |")
    a("")
    a("> The app itself announces detections at `conf=0.45` (chosen so speech is")
    a("> precise rather than chatty). mAP is computed at `conf=0.001`, because an")
    a("> AP metric must consider **every** prediction, not just the confident ones.")
    a("")
    a("## 2. Headline metrics")
    a("")
    a("| Metric | Value |")
    a("|--------|-------|")
    a(f"| Precision (P) | **{overall['precision']:.4f}** |")
    a(f"| Recall (R) | **{overall['recall']:.4f}** |")
    a(f"| mAP@0.5 | **{overall['map50']:.4f}** |")
    a(f"| mAP@0.5:0.95 | **{overall['map5095']:.4f}** |")
    a("")
    a(f"_Averages over {overall['classes_evaluated']} classes that occur in this")
    a(f"subset (model knows {overall['classes_in_model']} COCO classes in total)._")
    a("")
    sp = overall.get("speed_ms", {})
    a("## 3. Inference speed measured during validation")
    a("")
    a("| Stage | ms / image |")
    a("|-------|-----------|")
    for k in ("preprocess", "inference", "postprocess"):
        if k in sp:
            a(f"| {k} | {sp[k]:.2f} |")
    total = sum(v for k, v in sp.items() if k != "loss")
    if total > 0:
        a(f"| **total** | **{total:.2f}** |")
        a(f"| **FPS (single-stream)** | **{1000.0 / total:.1f}** |")
    a("")
    a("_See `benchmark.md` / `benchmark.png` for a controlled 100-frame")
    a("benchmark at 320 px and 640 px._")
    a("")
    a("## 4. Per-class AP (10 most common classes in this subset)")
    a("")
    top = sorted(per_class, key=lambda c: (-c["instances"], -c["ap50"]))[:10]
    a("| Class | GT instances | P | R | AP@0.5 | AP@0.5:0.95 |")
    a("|-------|-------------|---|---|--------|-------------|")
    for c in top:
        p = f"{c['precision']:.3f}" if c["precision"] is not None else "n/a"
        r = f"{c['recall']:.3f}" if c["recall"] is not None else "n/a"
        a(
            f"| {c['name']} | {c['instances']} | {p} | {r} | "
            f"**{c['ap50']:.3f}** | {c['ap5095']:.3f} |"
        )
    a("")
    worst = sorted(per_class, key=lambda c: c["ap50"])[:5]
    a("### Weakest classes (lowest AP@0.5)")
    a("")
    a("| Class | GT instances | AP@0.5 | AP@0.5:0.95 |")
    a("|-------|-------------|--------|-------------|")
    for c in worst:
        a(f"| {c['name']} | {c['instances']} | {c['ap50']:.3f} | {c['ap5095']:.3f} |")
    a("")
    if EVAL_DIR.exists():
        a("## 5. Curves")
        a("")
        for fname, title in (
            ("BoxPR_curve.png", "Precision–Recall curve"),
            ("PR_curve.png", "Precision–Recall curve"),
            ("BoxF1_curve.png", "F1–confidence curve"),
            ("F1_curve.png", "F1–confidence curve"),
            ("confusion_matrix_normalized.png", "Confusion matrix (normalised)"),
        ):
            if (EVAL_DIR / fname).exists():
                a(f"![{title}]({'eval/' + fname})")
                a("")
    a("### Detection examples (ground truth vs prediction)")
    a("")
    a("Ground truth labels:")
    a("")
    a("![ground truth](eval/val_batch0_labels.jpg)")
    a("")
    a("YOLOv8n predictions on the same batch:")
    a("")
    a("![predictions](eval/val_batch0_pred.jpg)")
    a("")
    a("## 6. Honest limitations of these numbers")
    a("")
    a("- **The subset is small (128 images).** COCO's official eval uses 5,000")
    a("  images; on 128 images each mAP point is noisy, and many of the 80 COCO")
    a("  classes never appear at all (so they contribute nothing to the average).")
    a("- **Micro-averages hide small objects.** AP@0.5:0.95 is dragged down by")
    a("  small objects, because IoU must be matched at thresholds up to 0.95 — a")
    a("  few pixels of box error on a 20 px object destroys AP there while barely")
    a("  moving AP@0.5.")
    a("- **Not the app's operating point.** These numbers use `conf=0.001`; the")
    a("  running app uses `conf=0.45`, which raises precision and lowers recall")
    a("  relative to what is reported here.")
    a("- **No fine-tuning.** This is off-the-shelf COCO performance on a laptop")
    a("  CPU. Real-world webcam scenes (low light, motion blur, unusual objects)")
    a("  will score lower than a curated validation set.")
    a("")
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report_md


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate YOLOv8n on a COCO subset")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--data", default="coco128.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--name", default="coco128")
    p.add_argument("--out", default=None, help="output directory (default: Evidence/)")
    args = p.parse_args(argv)

    base = Path(args.out) if args.out else EVIDENCE
    base.mkdir(parents=True, exist_ok=True)
    report_md = base / REPORT_MD.name
    metrics_json = base / METRICS_JSON.name
    per_class_json = base / PER_CLASS_JSON.name
    eval_dir = base / EVAL_DIR.name

    print(f"[INFO] model  : {args.model}", flush=True)
    print(f"[INFO] data   : {args.data}", flush=True)
    print("[INFO] loading YOLO...", flush=True)
    from ultralytics import YOLO

    model = YOLO(args.model)

    print("[INFO] running validation (first run downloads the dataset)...", flush=True)
    t0 = time.perf_counter()
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        verbose=False,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )
    wall = time.perf_counter() - t0
    print(f"[INFO] validation finished in {wall:.1f} s", flush=True)

    overall, per_class = collect(results)
    cfg = {
        "model": str(args.model),
        "data": args.data,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
    }
    hw = hardware()

    # Copy plots next to the report so the markdown/notebook can embed them.
    run_dir = ROOT / "runs" / args.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in PLOT_FILES:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, eval_dir / fname)
            copied.append(fname)

    metrics_json.write_text(
        json.dumps(
            {
                "config": cfg,
                "hardware": hw,
                "wall_seconds": round(wall, 2),
                "overall": overall,
                "dataset_note": "coco128 - 128 images from COCO val2017",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    per_class_json.write_text(json.dumps(per_class, indent=2), encoding="utf-8")

    report = write_report(
        overall, per_class, cfg, hw, wall, "128 images from COCO val2017", report_md
    )
    print(f"[OK] {report}", flush=True)
    print(f"[OK] {metrics_json}", flush=True)
    print(f"[OK] {per_class_json}", flush=True)
    if copied:
        print(f"[OK] {eval_dir} ({', '.join(copied)})", flush=True)

    print("\n=== HEADLINE ===")
    print(f"  Precision        : {overall['precision']:.4f}")
    print(f"  Recall           : {overall['recall']:.4f}")
    print(f"  mAP@0.5          : {overall['map50']:.4f}")
    print(f"  mAP@0.5:0.95     : {overall['map5095']:.4f}")
    print(
        f"  Classes evaluated: {overall['classes_evaluated']} / "
        f"{overall['classes_in_model']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
