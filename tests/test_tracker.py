"""Tests for the IoU tracker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from blindassist.config import TrackerConfig
from blindassist.position import Box
from blindassist.tracker import Tracker, _iou, confirmed_tracks


@dataclass
class Det:
    label: str
    confidence: float
    box: Box


def test_iou_identical_is_one():
    b = Box(0, 0, 100, 100)
    assert abs(_iou(b, b) - 1.0) < 1e-9


def test_iou_disjoint_is_zero():
    a = Box(0, 0, 100, 100)
    b = Box(200, 200, 300, 300)
    assert _iou(a, b) == 0.0


def test_track_id_stable_across_frames():
    tr = Tracker(TrackerConfig())
    box1 = Box(100, 100, 200, 300)
    box2 = Box(105, 102, 205, 302)  # slight shift
    t1 = tr.update([Det("person", 0.9, box1)])
    t2 = tr.update([Det("person", 0.9, box2)])
    assert len(t1) == 1 and len(t2) == 1
    assert t1[0].track_id == t2[0].track_id
    assert t2[0].hits == 2


def test_different_labels_never_merge():
    tr = Tracker(TrackerConfig())
    a = tr.update(
        [Det("person", 0.9, Box(0, 0, 100, 100)), Det("dog", 0.9, Box(0, 0, 100, 100))]
    )
    assert {t.label for t in a} == {"person", "dog"}
    assert len({t.track_id for t in a}) == 2


def test_lost_track_dropped_after_max_frames():
    cfg = TrackerConfig(max_lost_frames=3)
    tr = Tracker(cfg)
    tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])
    for _ in range(3):
        tr.update([])  # unmatched frames
    assert len(tr.tracks) == 1  # still alive at exactly max_lost
    tr.update([])  # one more => dropped
    assert len(tr.tracks) == 0


# ---------------------------------------------------------------------------
# Box smoothing (feature: stable detection)
# ---------------------------------------------------------------------------
def test_first_frame_box_is_exactly_the_detection():
    tr = Tracker(TrackerConfig(smoothing_window=5))
    t = tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])
    assert t[0].box == Box(100, 100, 200, 300)


def test_box_is_mean_of_last_three_frames():
    cfg = TrackerConfig(smoothing_window=3)
    tr = Tracker(cfg)
    tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])
    t2 = tr.update([Det("person", 0.9, Box(110, 100, 210, 300))])
    assert t2[0].box.x1 == pytest.approx(105.0)
    t3 = tr.update([Det("person", 0.9, Box(120, 100, 220, 300))])
    assert t3[0].box == Box(110.0, 100.0, 210.0, 300.0)
    # Only the last 3 raw boxes are averaged from here on.
    t4 = tr.update([Det("person", 0.9, Box(130, 100, 230, 300))])
    assert t4[0].box.x1 == pytest.approx(120.0)


def test_smoothing_window_one_is_exact():
    cfg = TrackerConfig(smoothing_window=1)
    tr = Tracker(cfg)
    tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])
    t2 = tr.update([Det("person", 0.9, Box(150, 100, 250, 300))])
    assert t2[0].box == Box(150, 100, 250, 300)


def test_history_keeps_raw_boxes_for_the_trend():
    """The published box is smoothed but the trend must still read raw heights."""
    cfg = TrackerConfig(smoothing_window=5, history_len=8)
    tr = Tracker(cfg)
    last = None
    for i in range(6):
        raw = Box(100, 100, 200, 200 + 20 * i)  # growing object -> approaching
        last = tr.update([Det("person", 0.9, raw)])
    assert last[0].history[-1] == Box(100, 100, 200, 300)
    assert not math.isclose(last[0].box.y2, 300.0)  # smoothed, not raw
    assert tr.tracks[0].trend() == "approaching"


def test_smoothed_box_method_window_zero_or_negative_is_safe():
    tr = Tracker(TrackerConfig())
    t = tr.update([Det("person", 0.9, Box(0, 0, 10, 10))])
    b = t[0].smoothed_box(0)
    assert b == Box(0, 0, 10, 10)


# ---------------------------------------------------------------------------
# Confirmation gate (feature: stable detection)
# ---------------------------------------------------------------------------
def test_confirmed_tracks_filters_young_tracks():
    tr = Tracker(TrackerConfig())
    young = tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])  # hits = 1
    assert confirmed_tracks(young, 3) == []
    tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])  # hits = 2
    two = tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])  # hits = 3
    assert confirmed_tracks(two, 3) and two[0].age == 3


def test_confirmation_gate_disabled_with_min_frames_one():
    tr = Tracker(TrackerConfig())
    first = tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])
    assert confirmed_tracks(first, 1) == first  # instant confirmation


def test_trend_approaching_when_growing():
    tr = Tracker(TrackerConfig())
    tr.update([Det("person", 0.9, Box(100, 100, 200, 300))])  # h=200
    tr.update([Det("person", 0.9, Box(100, 90, 200, 320))])  # h=230
    t = tr.update([Det("person", 0.9, Box(100, 80, 200, 340))])[0]  # h=260
    assert t.trend() == "approaching"


def test_new_tracks_have_distinct_ids():
    tr = Tracker(TrackerConfig())
    t1 = tr.update([Det("person", 0.9, Box(0, 0, 50, 50))])
    # Move far away so it is a new object (old track not matched).
    t2 = tr.update([Det("person", 0.9, Box(900, 600, 950, 650))])
    assert t1[0].track_id != t2[0].track_id
