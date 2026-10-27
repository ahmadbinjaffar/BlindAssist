"""Count spoken announcements on one video: v1 behaviour vs the v2 filter.

The two modes differ in **exactly one place** - the tracker knobs:

* ``--mode before``: ``min_confirm_frames=1, smoothing_window=1``
  (the original app: announce the instant a detection appears).
* ``--mode after``: the v2 defaults - 5-frame box smoothing and announcements
  only after 3 consecutive detections.

Everything else (detector, confidence, cooldowns, danger levels) is
identical, so the difference in announcement count is attributable to the
stability filter alone. The same video file is replayed in both modes.

The real app is run headless with speech disabled; its stdout is captured and
``[SPEAK]`` lines are counted (the welcome message is excluded). A JSON result
is written for the comparison chart.

Usage:
    python scripts/compare_stability.py --mode before --video assets/scene60.avi
        --json Evidence/final/stability_before.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blindassist import app as app_mod  # noqa: E402
from blindassist.config import load_config  # noqa: E402

WELCOME_PREFIX = "[SPEAK] Assistant started."


def build_config(mode: str):
    """Return a load_config wrapper pinned to one stability mode."""

    def patched():
        cfg = load_config()
        if mode == "before":
            cfg.tracker.min_confirm_frames = 1  # v1: announce on first frame
            cfg.tracker.smoothing_window = 1  # v1: raw, jittery box
        return cfg

    return patched


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Announcement count before/after stability")
    p.add_argument("--mode", choices=("before", "after"), required=True)
    p.add_argument("--video", required=True, help="video file replayed for the run")
    p.add_argument("--json", default=None, help="where to write the result JSON")
    args = p.parse_args(argv)

    video = Path(args.video)
    if not video.exists():
        print(f"[ERROR] no such video: {video}", file=sys.stderr)
        return 2

    # The app imports load_config by name; patch it before the app is built.
    app_mod.load_config = build_config(args.mode)

    argv_app = [
        "--video",
        str(video),
        "--no-speech",
        "--headless-test",
        "--max-frames",
        "100000",  # the video's own EOF ends the run
        "--tag",
        f"stability_{args.mode}",
    ]
    print(f"[INFO] mode={args.mode} video={video}", flush=True)
    buffer = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buffer):
            app = app_mod.BlindAssistApp(
                app_mod.build_arg_parser().parse_args(argv_app)
            )
            code = app.run()
    finally:
        app_mod.load_config = load_config  # never leak the patch

    out = buffer.getvalue()
    sys.stdout.write(out)

    spoken = [
        line
        for line in out.splitlines()
        if line.startswith("[SPEAK]") and not line.startswith(WELCOME_PREFIX)
    ]
    muted = [line for line in out.splitlines() if line.startswith("[MUTED]")]

    # Frames analyzed by the detector, taken from the app's own exit line.
    processed = None
    for line in out.splitlines():
        if "detected on" in line:
            try:
                processed = int(line.split("detected on")[1].split(",")[0].strip())
            except ValueError:
                processed = None

    result = {
        "mode": args.mode,
        "video": str(video),
        "exit_code": code,
        "announcements": len(spoken),
        "muted_lines": len(muted),
        "welcome_excluded": 1,
        "frames_processed_by_detector": processed,
        "utterances": spoken,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"\n[RESULT] mode={args.mode}  announcements={len(spoken)}  "
        f"processed={processed}  exit={code}",
        flush=True,
    )
    if args.json:
        print(f"[OK] {args.json}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
