"""Tests for the real-time application layer (no camera / no model required).

The app's distinguishing behaviours are covered here:

* the documented command-line flags,
* the "drop, never queue" frame hand-off to the detection worker,
* the overlay renderer (status bar, urgency colours, caption).
"""

from __future__ import annotations

import numpy as np

from blindassist.app import (
    BlindAssistApp,
    _apply_overrides,
    _parse_demo_keys,
    _skip_detection,
    build_arg_parser,
)
from blindassist.config import load_config
from blindassist.position import Box, Position


# -- command line ----------------------------------------------------------
def test_parser_accepts_documented_flags():
    args = build_arg_parser().parse_args(
        [
            "--camera",
            "1",
            "--conf",
            "0.30",
            "--video",
            "clip.mp4",
            "--no-speech",
            "--headless-test",
            "--tag",
            "demo",
        ]
    )
    assert args.camera == 1
    assert args.conf == 0.30
    assert args.video == "clip.mp4"
    assert args.no_speech is True
    assert args.headless_test is True
    assert args.tag == "demo"


def test_headless_flag_accepts_both_spellings():
    assert build_arg_parser().parse_args(["--headless-test"]).headless_test is True
    assert build_arg_parser().parse_args(["--headless"]).headless_test is True


def test_overrides_reach_the_config():
    args = build_arg_parser().parse_args(
        ["--camera", "2", "--conf", "0.60", "--rate", "220", "--no-speech"]
    )
    cfg = _apply_overrides(load_config(), args)
    assert cfg.vision.camera_index == 2
    assert cfg.vision.confidence == 0.60
    assert cfg.speech.rate_wpm == 220
    assert cfg.speech.enabled is False


def test_language_override_reaches_the_config():
    args = build_arg_parser().parse_args(["--language", "ur"])
    cfg = _apply_overrides(load_config(), args)
    assert cfg.language.language == "ur"
    # Default stays English:
    assert (
        _apply_overrides(
            load_config(), build_arg_parser().parse_args([])
        ).language.language
        == "en"
    )


# -- frame hand-off --------------------------------------------------------
def test_frames_are_dropped_not_queued_while_worker_busy():
    """The core real-time guarantee: a busy worker must never build a backlog."""
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    frame = object()
    app._submit(frame)  # accepted (worker slot is free)
    for _ in range(5):
        app._submit(frame)  # worker hasn't consumed it -> dropped
    with app._pending_lock:
        assert app._pending_frame is frame  # still exactly one held
    assert app.dropped == 5  # nothing was ever queued behind it


# -- overlay rendering -----------------------------------------------------
def _app_with_state() -> BlindAssistApp:
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    app.fps = 27.0
    app.confidence = 0.45
    app.rate_wpm = 175
    return app


def test_draw_renders_status_bar_help_and_caption():
    app = _app_with_state()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = app._draw(frame, [], "Chair on your left.")

    assert out.shape == frame.shape
    # Top status bar background is painted (not pure black).
    assert out[20, 300].max() > 0
    # The green "RUNNING" state text really rendered.
    assert out[10:32, 8:120, 1].max() > 200
    # On-screen keyboard help rendered as bright pixels.
    assert out[50:75, 8:420].max() > 200
    # Bottom caption bar + white sentence text.
    assert out[455:478, 70:600].max() > 200


def test_draw_colours_box_by_urgency():
    """'very close' must be red, 'far' must be green."""
    app = _app_with_state()

    close = np.zeros((480, 640, 3), dtype=np.uint8)
    p = Position("person", 0.9, Box(100, 100, 300, 400), "ahead", "very close", 0.25)
    app._draw(close, [p], "")
    # Top border of the box: red => BGR (0,0,255).
    b, g, r = close[100, 200]
    assert r > 200 and g < 80 and b < 80

    far = np.zeros((480, 640, 3), dtype=np.uint8)
    p2 = Position("person", 0.9, Box(100, 100, 300, 400), "ahead", "far", 0.01)
    app._draw(far, [p2], "")
    b, g, r = far[100, 200]
    assert g > 150 and r < 80 and b < 80


def test_draw_stays_inside_frame_for_edge_boxes():
    """A box hugging the right edge must not push its label off screen."""
    app = _app_with_state()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    p = Position("chair", 0.8, Box(600, 200, 640, 460), "right", "far", 0.05)
    app._draw(frame, [p], "")
    # No pixel of the label may be written past the image border (OpenCV would
    # silently clip, so instead assert the label background exists near the edge).
    assert frame[170:200, 630:640].max() > 0


# -- keyboard controls -----------------------------------------------------
def test_keyboard_controls(monkeypatch, tmp_path):
    """Every documented key must do what the on-screen help promises."""
    from types import SimpleNamespace

    import blindassist.app as app_mod

    app = BlindAssistApp(build_arg_parser().parse_args([]))
    # Stand in for the parts that need a camera / model / audio device.
    app.detector = SimpleNamespace(cfg=SimpleNamespace(confidence=0.45))
    rates: list[int] = []
    app.speaker = SimpleNamespace(set_rate=rates.append)
    monkeypatch.setattr(app_mod, "SCREENSHOT_DIR", tmp_path)
    app._last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    app._last_message = "Chair ahead."

    # SPACE toggles pause
    assert app.paused is False
    app._handle_key(ord(" "))
    assert app.paused is True
    app._handle_key(ord(" "))
    assert app.paused is False

    # + / - move the confidence threshold and reach the detector
    app._handle_key(ord("+"))
    assert app.confidence == 0.50
    app._handle_key(ord("-"))
    app._handle_key(ord("-"))
    assert app.confidence == 0.40
    assert app.detector.cfg.confidence == 0.40

    # ] / [ move the speech rate and are pushed into the speaker
    app._handle_key(ord("]"))
    app._handle_key(ord("["))
    assert app.rate_wpm == 175
    assert rates == [185, 175]

    # M toggles mute
    app._handle_key(ord("M"))
    assert app.muted is True
    app._handle_key(ord("M"))
    assert app.muted is False

    # S writes a screenshot to Evidence/screenshots (redirected to tmp).
    # The interactive S key uses a timestamped name so repeated presses never
    # overwrite each other (--tag is only for headless runs).
    app._handle_key(ord("S"))
    assert len(list(tmp_path.glob("*.png"))) == 1
    app._handle_key(ord("S"))
    assert len(list(tmp_path.glob("*.png"))) == 2

    # Q and ESC quit; unrelated keys do nothing
    assert app._handle_key(ord("Q")) is True
    assert app._handle_key(27) is True
    assert app._handle_key(ord("x")) is False


# ---------------------------------------------------------------------------
# --demo-keys (headless testing aid, feature: on-demand scene summary demo)
# ---------------------------------------------------------------------------
def test_parser_accepts_demo_keys():
    args = build_arg_parser().parse_args(["--demo-keys", "12:D,30:SPACE"])
    assert args.demo_keys == "12:D,30:SPACE"


def test_parse_demo_keys_maps_frames_to_keys():
    presses = _parse_demo_keys("12:D, 30:SPACE,7:S")
    assert presses == {12: ord("D"), 30: 32, 7: ord("S")}


def test_parse_demo_keys_accepts_control_keys():
    presses = _parse_demo_keys("5:+,10:-,20:[,30:]")
    assert presses == {5: ord("+"), 10: ord("-"), 20: ord("["), 30: ord("]")}


def test_parse_demo_keys_skips_junk_items():
    presses = _parse_demo_keys("abc:D,12:D,30:SHIFT,,:")
    assert presses == {12: ord("D")}


# ---------------------------------------------------------------------------
# Improved overlay (feature: cleaner UI)
# ---------------------------------------------------------------------------
def test_panel_alpha_zero_is_identity():
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    frame = np.full((100, 100, 3), (10, 200, 90), dtype=np.uint8)
    app._panel(frame, 10, 10, 60, 40, (35, 35, 35), 0.0)
    assert frame[20, 20].tolist() == [10, 200, 90]  # untouched


def test_panel_alpha_one_is_solid_colour():
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    frame = np.full((100, 100, 3), (10, 200, 90), dtype=np.uint8)
    app._panel(frame, 10, 10, 60, 40, (35, 35, 35), 1.0)
    assert frame[20, 20].tolist() == [35, 35, 35]  # solid
    assert frame[80, 80].tolist() == [10, 200, 90]  # outside untouched


def test_panel_blends_in_between():
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    app._panel(frame, 0, 0, 100, 100, (100, 100, 100), 0.5)
    value = int(frame[50, 50, 0])
    assert 45 <= value <= 55  # half-way blend


def test_panel_clamps_out_of_bounds_and_invalid_rects():
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    app._panel(frame, -20, -20, 200, 200, (7, 7, 7), 1.0)  # fully covers
    assert frame[0, 0, 0] == 7 and frame[49, 49, 0] == 7
    before = frame.copy()
    app._panel(frame, 30, 30, 30, 10, (9, 9, 9), 1.0)  # empty rect: no-op
    assert np.array_equal(before, frame)


def test_h_key_toggles_help_and_keeps_running():
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    assert app.show_help is False
    assert app._handle_key(ord("H")) is False
    assert app.show_help is True
    assert app._handle_key(ord("h")) is False
    assert app.show_help is False


def test_draw_help_screen_changes_the_center_of_the_frame():
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    frame = np.full((480, 640, 3), (60, 60, 60), dtype=np.uint8)
    without = app._draw(frame.copy(), [], "")
    app.show_help = True
    with_help = app._draw(frame.copy(), [], "")
    assert without.shape == with_help.shape  # same dimensions
    center = with_help[200:280, 200:440]
    assert not np.array_equal(center, without[200:280, 200:440])  # panel drawn
    # The help text is yellow-titled: some yellow appears now.
    assert (with_help[:, :, 2] > 200).sum() > (without[:, :, 2] > 200).sum()


def test_legend_and_zone_guides_toggle():
    app = BlindAssistApp(build_arg_parser().parse_args([]))
    frame = np.full((480, 640, 3), (60, 60, 60), dtype=np.uint8)
    app.cfg.ui.show_legend = False
    app.cfg.ui.show_zone_guides = False
    app._draw(frame.copy(), [], "")  # must not raise with everything off
    app.cfg.ui.show_legend = True
    app.cfg.ui.show_zone_guides = True
    app._draw(frame.copy(), [], "")  # ...and with everything on again


# ---------------------------------------------------------------------------
# Performance controls (feature: frame skipping + input size)
# ---------------------------------------------------------------------------
def test_size_flag_sets_detector_input_size():
    cfg = _apply_overrides(
        load_config(), build_arg_parser().parse_args(["--size", "320"])
    )
    assert cfg.vision.inference_size == 320


def test_size_flag_snaps_odd_values_to_nearest_supported():
    cfg = _apply_overrides(
        load_config(), build_arg_parser().parse_args(["--size", "450"])
    )
    assert cfg.vision.inference_size == 480  # nearest of {320, 480, 640}


def test_skip_flag_and_clamping():
    assert (
        _apply_overrides(
            load_config(), build_arg_parser().parse_args(["--skip", "3"])
        ).vision.frame_skip
        == 3
    )
    assert (
        _apply_overrides(
            load_config(), build_arg_parser().parse_args(["--skip", "0"])
        ).vision.frame_skip
        == 1
    )  # 0 would freeze the app: clamped to "every frame"
    assert (
        _apply_overrides(
            load_config(), build_arg_parser().parse_args(["--skip", "999"])
        ).vision.frame_skip
        == 30
    )  # capped so the app cannot feel frozen


def test_skip_detection_pattern():
    # skip=1: analyze everything.
    assert [_skip_detection(n, 1) for n in range(1, 7)] == [False] * 6
    # skip=2: analyze frames 1, 3, 5.
    assert [_skip_detection(n, 2) for n in range(1, 7)] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    # skip=3: analyze frames 1, 4.
    assert [_skip_detection(n, 3) for n in range(1, 7)] == [
        False,
        True,
        True,
        False,
        True,
        True,
    ]


def test_status_bar_shows_size_and_skip_when_non_default():
    app = BlindAssistApp(
        build_arg_parser().parse_args(["--size", "320", "--skip", "3"])
    )
    app.fps = 27.0
    app.confidence = 0.45
    app.rate_wpm = 175
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = app._draw(frame, [], "")
    # The "SIZE 320  SKIP 3" part of the status line rendered as bright pixels
    # in the middle of the bar (left of the legend panel).
    assert out[8:34, 430:640].max() > 200
