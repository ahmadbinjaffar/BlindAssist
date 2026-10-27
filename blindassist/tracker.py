"""Lightweight IoU tracker.

Assigns a stable integer id to each object across frames so the announcer can
avoid repeating itself and can add motion hints ("approaching"). This is a
deliberately simple tracker (greedy IoU matching) - no external dependencies,
no Kalman filter. It is enough for a fixed, forward-facing webcam.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import TrackerConfig
from .position import Box

# Minimum number of observed bounding boxes before a motion trend is meaningful.
TREND_MIN_HISTORY = 3


def _iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class Track:
    """A tracked object."""

    track_id: int
    label: str
    box: Box
    confidence: float
    hits: int = 1  # total matched frames
    time_since_update: int = 0  # consecutive unmatched frames
    history: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(self.box)

    @property
    def age(self) -> int:
        return self.hits

    def smoothed_box(self, window: int) -> Box:
        """Box averaged over the last ``window`` observations.

        Smooths bounding-box jitter so the reported *direction* and
        *approximate distance* do not flicker frame to frame. The raw boxes
        stay in :attr:`history`, so :meth:`trend` is untouched by this.
        ``window <= 1`` reproduces the unsmoothed behaviour exactly.
        """
        recent = list(self.history)[-max(1, window) :]
        n = len(recent)
        return Box(
            x1=sum(b.x1 for b in recent) / n,
            y1=sum(b.y1 for b in recent) / n,
            x2=sum(b.x2 for b in recent) / n,
            y2=sum(b.y2 for b in recent) / n,
        )

    def trend(self, min_delta: float = 0.06) -> str:
        """Rough motion hint from recent bounding-box height changes.

        Growing box => object getting closer ("approaching").
        Shrinking box => object moving away ("receding").
        """
        if len(self.history) < TREND_MIN_HISTORY:
            return "steady"
        heights = [b.height for b in self.history]
        first = sum(heights[: len(heights) // 2]) / max(1, len(heights) // 2)
        second = sum(heights[len(heights) // 2 :]) / max(
            1, len(heights) - len(heights) // 2
        )
        if first <= 0:
            return "steady"
        delta = (second - first) / first
        if delta > min_delta:
            return "approaching"
        if delta < -min_delta:
            return "receding"
        return "steady"


def confirmed_tracks(tracks: list[Track], min_frames: int) -> list[Track]:
    """Tracks that have persisted long enough to be announced or drawn.

    The app applies this gate right after :meth:`Tracker.update`: a detection
    must survive ``min_frames`` frames before the user hears about it, which
    filters one-frame glitches (false alarms).
    """
    return [t for t in tracks if t.age >= min_frames]


class Tracker:
    """Greedy IoU multi-object tracker."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, detections) -> list[Track]:
        """Advance the tracker with this frame's detections.

        `detections` is any iterable of objects exposing `.label`,
        `.confidence` and `.box` (see :class:`blindassist.detector.Detection`).
        Returns the list of tracks that were matched this frame.
        """
        dets = list(detections)
        matched_ids: set[int] = set()
        matched_det_idx: set[int] = set()

        # Greedy match: repeatedly take the globally best IoU pair.
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks.values()):
            for di, det in enumerate(dets):
                if det.label != track.label:
                    continue  # never merge two different object classes
                score = _iou(track.box, det.box)
                if score >= self.cfg.iou_match_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True, key=lambda p: p[0])

        track_list = list(self._tracks.values())
        for _score, ti, di in pairs:
            track = track_list[ti]
            if track.track_id in matched_ids or di in matched_det_idx:
                continue
            det = dets[di]
            track.confidence = det.confidence
            track.hits += 1
            track.time_since_update = 0
            # Raw boxes go into the history (the motion trend reads raw
            # heights); the published box is the average of the last few,
            # which smooths jitter in both box position and distance.
            track.history.append(det.box)
            while len(track.history) > self.cfg.history_len:
                track.history.popleft()
            track.box = track.smoothed_box(self.cfg.smoothing_window)
            matched_ids.add(track.track_id)
            matched_det_idx.add(di)

        # Unmatched detections become new tracks.
        for di, det in enumerate(dets):
            if di in matched_det_idx:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                label=det.label,
                box=det.box,
                confidence=det.confidence,
            )
            matched_ids.add(tid)

        # Age unmatched tracks; drop stale ones.
        for tid in list(self._tracks.keys()):
            if tid in matched_ids:
                continue
            t = self._tracks[tid]
            t.time_since_update += 1
            if t.time_since_update > self.cfg.max_lost_frames:
                del self._tracks[tid]

        return [self._tracks[tid] for tid in matched_ids if tid in self._tracks]
