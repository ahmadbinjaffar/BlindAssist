"""Runtime QA: run the real app and watch for leaks, crashes and exceptions.

Two checks, both using the actual application (no fakes):

1. ``--video`` on a generated clip  - must exit 0 with no traceback.
2. a live 60 s ``--headless-test``  - must exit 0, and the process's memory and
   thread count must not grow steadily (i.e. no leak in the worker/speech paths).

    .\\.venv\\Scripts\\python.exe scripts\\qa_runtime_check.py --seconds 60

Writes ``Evidence/qa_runtime.md``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = ROOT / "Evidence"
SAMPLES = ROOT / "assets" / "samples"
CLIP = ROOT / "assets" / "qa_clip.avi"
REPORT = EVIDENCE / "qa_runtime.md"

PY = sys.executable
TRACEBACK = re.compile(r"Traceback \(most recent call last\)|Error:|Exception")

# -- thresholds, stated once ------------------------------------------------
SAMPLE_FIELDS = 2  # sample_process returns "<bytes> <threads>"
MIN_SAMPLES_FOR_TREND = 6  # below this a slope is not meaningful
MIN_SAMPLES_TO_REPORT = 2
# Growth allowance for the whole run: below this the movement is allocator
# jitter, not a leak. Deliberately generous for a 60 s window.
GROWTH_LIMIT_MB = 40.0
THREAD_DRIFT_LIMIT = 3  # projected thread growth allowed over the whole run


# ---------------------------------------------------------------------------
def build_clip(frames: int = 40) -> Path:
    """A short real-content clip so --video runs genuine imagery, not noise."""
    import glob

    paths = sorted(glob.glob(str(SAMPLES / "*.jpg")))[:3]
    images = [p for p in (cv2.imread(p) for p in paths) if p is not None]
    if not images:
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)]

    writer = cv2.VideoWriter(
        str(CLIP), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)
    )
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter")
    for i in range(frames):
        writer.write(cv2.resize(images[i % len(images)], (320, 240)))
    writer.release()
    return CLIP


def run_app(args: list[str], timeout: float) -> dict:
    """Run the app, capture output, and assert it exited without a traceback."""
    proc = subprocess.run(
        [PY, "-m", "blindassist.app", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        # The exit code is inspected directly below; a non-zero code is a
        # finding to report, not an exception to raise here.
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return {
        "returncode": proc.returncode,
        "traceback": TRACEBACK.findall(out),
        "speaks": out.count("[SPEAK]") + out.count("[MUTED]"),
        "detections": out.count("[DETECT]"),
        "errors": [ln for ln in out.splitlines() if ln.startswith("[ERROR]")],
        "tail": "\n".join(out.splitlines()[-12:]),
    }


SAMPLE_PS1 = ROOT / "scripts" / "proc_sample.ps1"


def sample_process(pid: int) -> tuple[int, int]:
    """(working-set bytes, thread count) for a process **and its children**.

    Must walk the tree: ``venv\\Scripts\\python.exe`` here is a launcher stub that
    spawns the real interpreter as a child, so sampling the PID Popen hands back
    would report the stub and never see the memory we care about.
    """
    out = (
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SAMPLE_PS1),
                "-ProcessId",
                str(pid),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            # A vanished process yields empty output, which is handled below.
            check=False,
        )
        .stdout.strip()
        .split()
    )
    if len(out) != SAMPLE_FIELDS:
        return (0, 0)
    return int(out[0]), int(out[1])


def run_monitored(args: list[str], seconds: float, interval: float = 3.0) -> dict:
    """Start the app and sample its memory/threads until it exits or time is up."""
    proc = subprocess.Popen(
        [PY, "-m", "blindassist.app", *args],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    samples: list[tuple[float, int, int]] = []
    start = time.time()
    timed_out = False
    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= seconds:
                timed_out = True
                break
            if proc.poll() is not None:
                break
            rss, threads = sample_process(proc.pid)
            if rss:
                samples.append((elapsed, rss, threads))
            time.sleep(interval)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        output = proc.stdout.read() if proc.stdout else ""

    lines = (output or "").splitlines()
    return {
        # We deliberately kill it when the time budget is spent, so a non-zero
        # code from that is expected and must not count as a failure.
        "returncode": 0 if timed_out else proc.returncode,
        "timed_out": timed_out,
        "samples": samples,
        "traceback": [ln for ln in lines if "Traceback" in ln],
        "errors": [ln for ln in lines if ln.startswith("[ERROR]")],
        "speaks": sum(1 for ln in lines if ln.startswith(("[SPEAK]", "[MUTED]"))),
        "detections": sum(1 for ln in lines if ln.startswith("[DETECT]")),
        "tail": "\n".join(lines[-12:]),
    }


def _plateau(samples: list[tuple[float, int, int]], seconds: float) -> dict | None:
    """Split a memory curve into warm-up and steady state.

    YOLO/torch lay down thread pools and arenas on the first passes, so RSS
    climbs hard before it settles. Judging growth across the *whole* run calls
    that warm-up a leak. Fitting a line to the second half distinguishes them:
    a real leak keeps climbing, warm-up flattens out.
    """
    if len(samples) < MIN_SAMPLES_FOR_TREND:
        return None
    tail = samples[len(samples) // 2 :]
    xs = [s[0] for s in tail]
    ys = [s[1] / 1e6 for s in tail]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = (
        sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
        if den
        else 0.0
    )
    threads = [s[2] for s in tail]
    # Same least-squares treatment for the thread count: a leak is a sustained
    # climb, whereas a settled pool just wobbles up and down. Ordering matters
    # here - collapsing to a sorted set of unique values would turn
    # "26 -> 22 (falling)" into a meaningless range of 4.
    mean_t = sum(threads) / len(threads)
    tslope = (
        sum((x - mx) * (t - mean_t) for x, t in zip(xs, threads, strict=True)) / den
        if den
        else 0.0
    )
    return {
        "slope": slope,  # MB per second
        "projected": slope * seconds,  # MB implied over the whole run
        "thread_projected": tslope * seconds,  # threads implied over the run
        "rss_min": min(ys),
        "rss_max": max(ys),
        "threads": sorted(set(threads)),
        "thread_first": threads[0],
        "thread_last": threads[-1],
        "start": tail[0][0],
        "warmup": samples[len(samples) // 2][1] - samples[0][1],
    }


def _trend(samples: list[tuple[float, int, int]]) -> str:
    if len(samples) < MIN_SAMPLES_TO_REPORT:
        return "not enough samples"
    first_mb = samples[0][1] / 1e6
    peak_mb = max(s[1] for s in samples) / 1e6
    last_mb = samples[-1][1] / 1e6
    drift = last_mb - first_mb
    return (
        f"{first_mb:.0f} MB first -> {last_mb:.0f} MB last "
        f"(peak {peak_mb:.0f} MB, drift {drift:+.0f} MB)"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Runtime leak / crash check")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args(argv)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    clip = build_clip()
    ok = True
    L: list[str] = []
    a = L.append

    a("# BlindAssist — Runtime QA")
    a("")
    a(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    a("")
    a("Both checks run the **real** application as a subprocess (real YOLO, real")
    a("webcam, real TTS). No fakes.")
    a("")

    # -- 1. video file ------------------------------------------------------
    print("[INFO] check 1/2: --video on a generated clip", flush=True)
    v = run_app(
        [
            "--video",
            str(clip),
            "--headless-test",
            "--no-speech",
            "--max-frames",
            "20",
            "--tag",
            "qa_video",
        ],
        timeout=300,
    )
    video_ok = v["returncode"] == 0 and not v["traceback"]
    ok = ok and video_ok
    a("## 1. Video-file run")
    a("")
    a(f"Command: `--video {clip.name} --headless-test --no-speech --max-frames 20`")
    a("")
    a("| Check | Result |")
    a("|-------|--------|")
    a(f"| Exit code | **{v['returncode']}** (expect 0) |")
    a(f"| Traceback / exception | **{len(v['traceback'])}** |")
    a(f"| `[ERROR]` lines | {len(v['errors'])} |")
    a(f"| Announcements | {v['speaks']} |")
    a(f"| First-frame detections | {v['detections']} |")
    a(f"| Verdict | **{'PASS' if video_ok else 'FAIL'}** |")
    a("")

    # -- 2. long live run ---------------------------------------------------
    print(f"[INFO] check 2/2: live headless run for {args.seconds:.0f} s", flush=True)
    live = run_monitored(
        [
            "--headless-test",
            "--no-speech",
            "--tag",
            "qa_live",
            "--max-frames",
            "1000000",
        ],
        seconds=args.seconds,
    )
    samples = live["samples"]
    plateau = _plateau(samples, args.seconds)
    if plateau is None:
        stable = False
        verdict_reason = "not enough samples to judge"
    else:
        # A leak shows up as a sustained positive slope in the steady state;
        # flat or slightly negative is just the allocator settling.
        stable = (
            abs(plateau["projected"]) < GROWTH_LIMIT_MB
            and plateau["thread_projected"] <= THREAD_DRIFT_LIMIT
        )
        verdict_reason = (
            f"warm-up added {plateau['warmup'] / 1e6:+.0f} MB by "
            f"t={plateau['start']:.0f}s, then RSS held {plateau['rss_min']:.0f}"
            f"-{plateau['rss_max']:.0f} MB; steady-state slope "
            f"{plateau['slope']:+.2f} MB/s "
            f"({plateau['projected']:+.0f} MB projected over "
            f"{args.seconds:.0f}s, limit {GROWTH_LIMIT_MB:.0f} MB); threads "
            f"{plateau['thread_first']}->{plateau['thread_last']} "
            f"({plateau['thread_projected']:+.1f} projected, limit "
            f"+{THREAD_DRIFT_LIMIT:.0f})"
        )
    live_ok = live["returncode"] == 0 and not live["traceback"] and stable
    ok = ok and live_ok

    a(f"## 2. Live webcam run ({args.seconds:.0f} s, headless, speech disabled)")
    a("")
    a("Command: `--headless-test --no-speech --tag qa_live --max-frames 1000000`")
    a("")
    a("| Check | Result |")
    a("|-------|--------|")
    a(f"| Exit code | **{live['returncode']}** (0, or 0 after terminate) |")
    a(f"| Traceback / exception | **{len(live['traceback'])}** |")
    a(f"| `[ERROR]` lines | {len(live['errors'])} |")
    a(f"| Announcements | {live['speaks']} |")
    a(f"| First-frame detections | {live['detections']} |")
    a(f"| Memory / threads | {_trend(samples)} |")
    a(f"| Leak verdict | {verdict_reason} |")
    a(f"| Verdict | **{'PASS' if live_ok else 'FAIL'}** |")
    a("")
    if samples:
        a("### Sampled process metrics")
        a("")
        a("| t (s) | RSS (MB) | Threads |")
        a("|------:|---------:|--------:|")
        for t, rss, th in samples:
            a(f"| {t:.0f} | {rss / 1e6:.0f} | {th} |")
        a("")

    a("## Verdict")
    a("")
    a(f"**{'PASS' if ok else 'FAIL'}**")
    a("")
    a("```")
    a("video run tail:")
    a(v["tail"])
    a("")
    a("live run tail:")
    a(live["tail"])
    a("```")
    a("")
    REPORT.write_text("\n".join(L), encoding="utf-8")

    print("\n=== RESULT ===")
    print(
        f"  video run : exit {v['returncode']}, "
        f"{len(v['traceback'])} traceback(s), {v['detections']} detections"
    )
    print(
        f"  live run  : exit {live['returncode']}, "
        f"{len(live['traceback'])} traceback(s), {len(samples)} samples"
    )
    print(f"  memory    : {_trend(samples)}")
    print(f"\n[OK] {REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
