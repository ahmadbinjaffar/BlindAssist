"""Low-light help: measure frame darkness, enhance contrast, warn the user.

Two parts:

* **Pure CV** - :func:`mean_luma` (downsampled mean luminance, cheap enough to
  run on every processed frame) and :func:`enhance` (CLAHE on the L channel of
  the LAB colour space, so local contrast rises for YOLO without washing out
  the whole frame).
* **Decision logic** - :class:`LightWatcher`, a small stateful monitor that
  decides, frame by frame, whether to enhance the frame sent to the detector
  and whether to speak the "too dark" warning. It requires the darkness to
  persist (:attr:`LightConfig.dark_streak_frames`) so a single unlucky frame
  (camera auto-exposure dip, a TV scene) does not trigger anything, and rate-
  limits the warning to once per :attr:`LightConfig.warn_cooldown_s`.

The preview window always shows the *true* camera image; only the frame
handed to YOLO is enhanced, and the overlay prints "DARK MODE" while lazy
enhancement is active.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import LightConfig

_DOWNSAMPLE = (64, 64)  # mean_luma resolution - plenty for a brightness score


def mean_luma(frame: np.ndarray) -> float:
    """Mean 0-255 luminance of a BGR frame (downsampled for speed)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, _DOWNSAMPLE, interpolation=cv2.INTER_AREA)
    return float(small.mean())


def is_dark(mean_luma: float, cfg: LightConfig) -> bool:
    """Simple threshold check used by tests and the watcher alike."""
    return mean_luma < cfg.dark_mean_threshold


def enhance(frame: np.ndarray, cfg: LightConfig) -> np.ndarray:
    """CLAHE contrast enhancement on the LAB L channel (BGR in, BGR out)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cfg.clip_limit, tileGridSize=cfg.tile_grid)
    enhanced = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((enhanced, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


class LightWatcher:
    """Frame-by-frame dark-mode decisions (streak + cooldown state machine)."""

    def __init__(self, cfg: LightConfig, clock=time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self._dark_streak = 0
        self._last_warn = -cfg.warn_cooldown_s  # a warning is possible at once

    def reset(self) -> None:
        self._dark_streak = 0

    def update(self, luma: float) -> tuple[bool, bool]:
        """Feed one frame's mean luminance.

        Returns ``(enhance_now, warn_now)``:

        * ``enhance_now`` - hand a CLAHE-enhanced copy to the detector.
        * ``warn_now`` - speak the "too dark" warning (honours the cooldown).
        """
        if not self.cfg.enabled:
            self._dark_streak = 0
            return False, False
        if luma >= self.cfg.dark_mean_threshold:
            self._dark_streak = 0
            return False, False
        self._dark_streak += 1
        if self._dark_streak < self.cfg.dark_streak_frames:
            return False, False
        now = self._clock()
        warn = (now - self._last_warn) >= self.cfg.warn_cooldown_s
        if warn:
            self._last_warn = now
        return True, warn
