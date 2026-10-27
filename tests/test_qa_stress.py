"""QA / stress tests: the failure modes a blind user would actually hit.

These exercise the application layer without a camera, a model or an audio
device (fakes keep them fast and hermetic), plus the frame sources which are
tested against real OpenCV objects:

* missing / undecodable media and an unplugged camera,
* an explicitly requested model file that does not exist,
* a very dark frame and a frame crowded with 30 objects,
* 50 rapid pause/resume cycles,
* no leaked threads and no growing backlog while paused.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import blindassist.app as app_mod
from blindassist.announcer import Announcer
from blindassist.app import (
    BlindAssistApp,
    CameraSource,
    FrameSource,
    VideoSource,
    _safe_tag,
    build_arg_parser,
)
from blindassist.config import PositionConfig, SpeechConfig, TrackerConfig, load_config
from blindassist.detector import Detection
from blindassist.position import Positioner
from blindassist.tracker import Tracker

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeDetector:
    """Stands in for the YOLO wrapper: no weights, no inference."""

    def __init__(self, cfg, model_path, class_whitelist=None, device=None):
        self.cfg = cfg
        self.model_path = model_path
        self.class_whitelist = class_whitelist
        self.device = device or "cpu"

    def load(self) -> None:
        return

    def detect(self, frame):
        return []


class _DyingCamera(FrameSource):
    """A camera that delivers a few frames and is then unplugged for good."""

    frames_to_deliver = 2

    def __init__(self, index: int, width: int, height: int) -> None:
        self.index = index
        self.ok = index == 0
        self._n = 0
        self._released = False

    @property
    def description(self) -> str:
        return f"camera {self.index}"

    @property
    def is_live(self) -> bool:
        return True

    def read(self):
        if self._released or self._n >= self.frames_to_deliver:
            return None
        self._n += 1
        return FRAME.copy()

    def release(self) -> None:
        self._released = True

    def reopen(self) -> bool:
        return False


class _DeadCamera(FrameSource):
    """A camera index that never opens."""

    def __init__(self, index: int, width: int, height: int) -> None:
        self.index = index
        self.ok = False

    def release(self) -> None:
        return


def _run(args, monkeypatch, tmp_path, detector=_FakeDetector, camera=None):
    """Drive BlindAssistApp.run() with the model/camera faked out.

    Returns ``(exit_code, app)``.
    """
    monkeypatch.setattr(app_mod, "Detector", detector)
    if camera is not None:
        monkeypatch.setattr(app_mod, "CameraSource", camera)
    monkeypatch.setattr(app_mod, "SCREENSHOT_DIR", tmp_path / "shots")
    app = BlindAssistApp(build_arg_parser().parse_args(args))
    return app.run(), app


def _write_video(path, frames: int = 4) -> str:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)
    )
    assert writer.isOpened(), "VideoWriter failed to open"
    for i in range(frames):
        writer.write(np.full((240, 320, 3), (i * 30) % 255, dtype=np.uint8))
    writer.release()
    return str(path)


def _write_dark_video(path, frames: int = 4) -> str:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)
    )
    assert writer.isOpened(), "VideoWriter failed to open"
    for _ in range(frames):
        writer.write(FRAME)
    writer.release()
    return str(path)


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
def test_frame_source_base_is_abstract():
    src = FrameSource()
    assert src.reopen() is False
    assert src.is_live is False
    assert src.description == "source"
    with pytest.raises(NotImplementedError):
        src.read()


def test_video_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "definitely-not-here.mp4")


def test_video_source_undecodable_image_raises(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not really a jpeg")
    with pytest.raises(ValueError):
        VideoSource(bad)


def test_video_source_unopenable_video_raises(tmp_path):
    """A file with a video extension that is not a video must not be silently OK."""
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"garbage")
    with pytest.raises(ValueError):
        VideoSource(fake)


def test_video_source_still_image_loops_and_copies(tmp_path):
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), FRAME)
    src = VideoSource(img)
    assert src.description.startswith("image")
    assert src.is_live is False
    a, b = src.read(), src.read()
    assert a is not None and b is not None
    a[0, 0] = 255  # mutating one read must not corrupt the next
    assert b[0, 0].max() == 0
    src.release()


def test_video_source_video_reaches_end_of_file(tmp_path):
    path = _write_video(tmp_path / "clip.avi", frames=3)
    src = VideoSource(path)
    assert src.description.startswith("video")
    assert [src.read() is not None for _ in range(4)] == [True, True, True, False]
    src.release()


def test_camera_source_reports_unavailable_index():
    cam = CameraSource(99, 64, 48)
    if cam.ok:  # a machine that genuinely has camera 99
        cam.release()
        pytest.skip("camera index 99 exists on this machine")
    assert cam.read() is None
    assert cam.is_live is True
    assert cam.description == "camera 99"
    assert cam.reopen() is False
    cam.release()


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------
def test_conf_above_maximum_is_clamped():
    cfg = app_mod._apply_overrides(
        load_config(), build_arg_parser().parse_args(["--conf", "5.0"])
    )
    assert cfg.vision.confidence == app_mod.CONF_MAX


def test_conf_below_minimum_is_clamped():
    cfg = app_mod._apply_overrides(
        load_config(), build_arg_parser().parse_args(["--conf", "-3"])
    )
    assert cfg.vision.confidence == app_mod.CONF_MIN


def test_rate_out_of_range_is_clamped():
    cfg = app_mod._apply_overrides(
        load_config(), build_arg_parser().parse_args(["--rate", "100000"])
    )
    assert cfg.speech.rate_wpm == app_mod.RATE_MAX


def test_negative_camera_index_is_clamped():
    cfg = app_mod._apply_overrides(
        load_config(), build_arg_parser().parse_args(["--camera", "-2"])
    )
    assert cfg.vision.camera_index == 0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("normal", "normal"),
        ("../../evil", "evil"),
        ("a/b\\c", "abc"),
        ("has spaces", "has spaces"),
        ("...", "frame"),
        ("", "frame"),
    ],
)
def test_safe_tag_strips_path_separators(raw, expected):
    """--tag must never be able to write outside Evidence/screenshots/."""
    assert _safe_tag(raw) == expected


# ---------------------------------------------------------------------------
# Startup failures
# ---------------------------------------------------------------------------
def test_no_camera_gives_a_clear_error(tmp_path, monkeypatch):
    code, _app = _run(["--no-speech"], monkeypatch, tmp_path, camera=_DeadCamera)
    assert code == 2


def test_missing_video_gives_a_clear_error(tmp_path, monkeypatch):
    code, _app = _run(
        ["--video", str(tmp_path / "nope.mp4"), "--no-speech"], monkeypatch, tmp_path
    )
    assert code == 2


def test_missing_model_fails_fast_instead_of_downloading(tmp_path, monkeypatch):
    """An explicit --model that does not exist must not trigger a download."""
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), FRAME)

    def _boom(self):  # pragma: no cover - must never run
        raise AssertionError("weights were downloaded for an explicit --model")

    monkeypatch.setattr(_FakeDetector, "load", _boom)
    code, _app = _run(
        ["--video", str(img), "--model", str(tmp_path / "missing.pt"), "--no-speech"],
        monkeypatch,
        tmp_path,
    )
    assert code == 3


def test_end_of_video_exits_cleanly(tmp_path, monkeypatch):
    path = _write_video(tmp_path / "clip.avi", frames=4)
    code, app = _run(
        ["--video", path, "--headless-test", "--no-speech", "--max-frames", "2"],
        monkeypatch,
        tmp_path,
    )
    assert code == 0
    assert app._worker is None or not app._worker.is_alive()


def test_camera_unplugged_mid_run_shuts_down_cleanly(tmp_path, monkeypatch):
    """Losing the camera must exit 0 with a clean shutdown, never hang or crash."""
    code, app = _run(
        ["--headless-test", "--no-speech", "--max-frames", "100000"],
        monkeypatch,
        tmp_path,
        camera=_DyingCamera,
    )
    assert code == 0
    assert app._consecutive_failures > 60


# ---------------------------------------------------------------------------
# Runtime robustness
# ---------------------------------------------------------------------------
def test_very_dark_frame_produces_no_crash(tmp_path, monkeypatch):
    class _DarkDetector(_FakeDetector):
        def detect(self, frame):
            assert frame.shape == FRAME.shape
            return []

    path = _write_dark_video(tmp_path / "dark.avi")
    code, _app = _run(
        ["--video", path, "--headless-test", "--no-speech", "--max-frames", "4"],
        monkeypatch,
        tmp_path,
        detector=_DarkDetector,
    )
    assert code == 0


def test_frame_crowded_with_thirty_objects(tmp_path):
    """30 simultaneous detections must not break tracking, wording or drawing."""
    crowd = [
        Detection("chair", 0.6 + (i % 4) / 10.0, 10 + i * 8, 60, 58 + i * 8, 200)
        for i in range(30)
    ]
    tracker = Tracker(TrackerConfig())
    tracks = tracker.update(crowd)
    assert len(tracks) == 30

    positioner = Positioner(PositionConfig(), 320, 240)
    placements = [positioner.place(t.label, t.confidence, t.box) for t in tracks]
    message = Announcer(SpeechConfig()).update(placements)
    assert message  # something must be said about 30 objects

    app = BlindAssistApp(build_arg_parser().parse_args([]))
    app.fps = 10.0
    out = app._draw(FRAME.copy(), placements, message)
    assert out.shape == (240, 320, 3)


def test_fifty_rapid_pause_resume_cycles(tmp_path, monkeypatch):
    """50 quick SPACE presses must not desync state or leak anything."""
    before = {t.name for t in threading.enumerate()}
    path = _write_video(tmp_path / "clip.avi", frames=4)
    args = build_arg_parser().parse_args(
        ["--video", path, "--headless-test", "--no-speech", "--max-frames", "3"]
    )
    monkeypatch.setattr(app_mod, "Detector", _FakeDetector)
    monkeypatch.setattr(app_mod, "SCREENSHOT_DIR", tmp_path / "shots")
    app = BlindAssistApp(args)

    for _ in range(50):
        app._handle_key(ord(" "))
    assert app.paused is False  # even number of presses -> back to running

    assert app.run() == 0
    assert app._worker is None or not app._worker.is_alive()
    leaked = {t.name for t in threading.enumerate()} - before
    assert not leaked, f"threads leaked: {leaked}"


def test_no_thread_leak_after_a_normal_run(tmp_path, monkeypatch):
    before = {t.name for t in threading.enumerate()}
    path = _write_video(tmp_path / "clip.avi", frames=4)
    code, _app = _run(
        ["--video", path, "--headless-test", "--no-speech", "--max-frames", "3"],
        monkeypatch,
        tmp_path,
    )
    assert code == 0
    leaked = {t.name for t in threading.enumerate()} - before
    assert not leaked, f"threads leaked: {leaked}"


def test_backlog_stays_bounded_while_paused():
    """Paused frames must not accumulate: the slot holds at most one."""
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    for _ in range(200):
        app._submit(object())
    with app._pending_lock:
        held = 0 if app._pending_frame is None else 1
    assert held <= 1
    assert app.dropped == 199


def test_worker_survives_an_exception_in_detection(tmp_path, monkeypatch):
    """One bad frame must not kill the detection thread."""

    class _ExplodingDetector(_FakeDetector):
        def detect(self, frame):
            raise RuntimeError("corrupt frame")

    path = _write_video(tmp_path / "clip.avi", frames=4)
    code, app = _run(
        ["--video", path, "--headless-test", "--no-speech", "--max-frames", "4"],
        monkeypatch,
        tmp_path,
        detector=_ExplodingDetector,
    )
    assert code == 0
    assert app.processed == 0


def test_keys_are_safe_before_the_pipeline_exists():
    """Regression: +/- and [ ] used to dereference detector/speaker = None."""
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    assert app.detector is None and app.speaker is None
    for key in (ord("+"), ord("-"), ord("]"), ord("["), ord("m"), ord(" ")):
        app._handle_key(key)  # must not raise
    app._handle_key(ord("s"))  # no frame captured yet -> no-op, must not raise
    assert app._last_frame is None


def test_screenshot_key_without_a_frame_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "SCREENSHOT_DIR", tmp_path)
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    app._save_current_screenshot()
    assert list(tmp_path.glob("*.png")) == []


def test_muted_announces_are_logged_but_not_spoken(tmp_path, monkeypatch, capsys):
    """Muting must suppress speech, not the record of what would have been said."""
    spoken: list[str] = []
    monkeypatch.setattr(app_mod, "SCREENSHOT_DIR", tmp_path)
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    app.speaker = SimpleNamespace(speak=spoken.append)

    app.muted = True
    app._announce("chair on your left.")
    assert spoken == []
    assert "MUTED" in capsys.readouterr().out

    app.muted = False
    app._announce("chair on your left.")
    assert spoken == ["chair on your left."]
    assert "SPEAK" in capsys.readouterr().out


def test_demo_key_q_quits_a_headless_run(tmp_path, monkeypatch, capsys):
    """Regression: scripted presses must honour Q exactly like real key presses.

    The headless branch used to ignore ``_handle_key``'s return value, so a
    ``--demo-keys "5:Q"`` press could never end the run early and automated
    proofs had to rely on --max-frames instead.
    """
    img = tmp_path / "loop.png"
    cv2.imwrite(str(img), FRAME)
    code, _app = _run(
        [
            "--video",
            str(img),
            "--no-speech",
            "--headless-test",
            "--max-frames",
            "400",
            "--demo-keys",
            "5:Q",
        ],
        monkeypatch,
        tmp_path,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Stopped (quit)" in out  # the Q press ended the run...
    assert "Reached --max-frames" not in out  # ...not the frame limit


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
