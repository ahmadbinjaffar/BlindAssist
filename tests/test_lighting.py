"""Low-light helper: darkness measurement, CLAHE enhancement, watcher logic."""

from __future__ import annotations

import time

import cv2
import numpy as np

from blindassist.config import LightConfig
from blindassist.lighting import LightWatcher, enhance, is_dark, mean_luma


def dark_frame() -> np.ndarray:
    """A synthetic very dark indoor frame (mean luma ~10)."""
    return np.full((240, 320, 3), (12, 10, 8), dtype=np.uint8)


def bright_frame() -> np.ndarray:
    """A synthetic normal-illumination frame (mean luma ~150)."""
    return np.full((240, 320, 3), (150, 150, 150), dtype=np.uint8)


# ---------------------------------------------------------------------------
# mean_luma / is_dark
# ---------------------------------------------------------------------------
def test_mean_luma_of_dark_and_bright_frames():
    assert mean_luma(dark_frame()) < 20.0
    assert mean_luma(bright_frame()) > 120.0


def test_mean_luma_matches_full_resolution_mean():
    img = bright_frame()
    img[0:40, 0:40] = (0, 0, 0)
    # Downsampled mean must track the global mean within a few units.
    downscaled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    assert abs(mean_luma(img) - downscaled.mean()) < 20.0


def test_is_dark_uses_the_configured_threshold():
    cfg = LightConfig(dark_mean_threshold=60.0)
    assert is_dark(59.9, cfg)
    assert not is_dark(60.0, cfg)
    assert not is_dark(120.0, cfg)


# ---------------------------------------------------------------------------
# enhance (CLAHE)
# ---------------------------------------------------------------------------
def test_enhance_raises_dynamic_range_on_a_dark_frame():
    src = dark_frame()
    # Add a modest gradient so CLAHE has something to stretch locally.
    ramp = (np.arange(320, dtype=np.uint8) // 16)[None, :, None]
    src = np.maximum(src, ramp.astype(np.uint8))
    out = enhance(src, LightConfig())
    assert out.shape == src.shape
    assert out.std() > src.std()  # local contrast increased


def test_enhance_never_crashes_on_a_flat_or_odd_image():
    flat = np.full((100, 120, 3), 40, dtype=np.uint8)
    out = enhance(flat, LightConfig())
    assert out.shape == flat.shape
    odd = np.full((7, 9, 3), 40, dtype=np.uint8)
    assert enhance(odd, LightConfig()).shape == odd.shape  # tile < 8 px grid


def test_enhance_preserves_colours_reasonably():
    """Atom a saturated red square: after CLAHE it must still be reddish."""
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    img[40:120, 40:120] = (0, 0, 200)
    out = enhance(img, LightConfig())
    centre = out[80, 80]
    assert centre[2] > centre[1] and centre[2] > centre[0]  # BGR: red channel


# ---------------------------------------------------------------------------
# LightWatcher decision logic
# ---------------------------------------------------------------------------
def test_watcher_waits_for_the_configured_streak():
    cfg = LightConfig(dark_streak_frames=3, warn_cooldown_s=60.0)
    w = LightWatcher(cfg)
    assert w.update(30.0) == (False, False)  # frame 1 dark, not yet
    assert w.update(30.0) == (False, False)  # frame 2
    assert w.update(30.0) == (True, True)  # frame 3: enhance and warn
    assert w.update(30.0) == (True, False)  # still dark, warning on cooldown


def test_watcher_resets_streak_on_a_bright_frame():
    cfg = LightConfig(dark_streak_frames=3)
    w = LightWatcher(cfg)
    w.update(30.0)
    w.update(30.0)
    assert w.update(150.0) == (False, False)  # bright frame cancels the streak
    assert w.update(30.0) == (False, False)  # streak restarts from zero


def test_watcher_honours_the_event_clock_cooldown():
    cfg = LightConfig(dark_streak_frames=2, warn_cooldown_s=10.0)
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    w = LightWatcher(cfg, clock=clock)
    w.update(20.0)
    _, warn1 = w.update(20.0)
    assert warn1 is True
    now["t"] = 5.0
    w.update(20.0)
    w.reset()  # (simulate alert) then go dark again quickly
    w.update(20.0)
    _, warn_soon = w.update(20.0)
    assert warn_soon is False  # inside the 10 s cooldown
    now["t"] = 20.0
    _, warn_later = w.update(20.0)
    assert warn_later is True  # cooldown elapsed


def test_disabled_watcher_never_enhances_or_warns():
    w = LightWatcher(LightConfig(enabled=False))
    for _ in range(10):
        assert w.update(5.0) == (False, False)


def test_cooldown_is_bounded_by_time_not_frames():
    cfg = LightConfig(dark_streak_frames=1, warn_cooldown_s=0.2)
    w = LightWatcher(cfg)
    assert w.update(10.0) == (True, True)
    time.sleep(0.25)
    assert w.update(10.0) == (True, True)  # cooldown elapsed in real time
