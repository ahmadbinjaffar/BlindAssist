"""Tests for direction (frame thirds) + approximate distance (area ratio)."""

from __future__ import annotations

from blindassist.config import PositionConfig
from blindassist.position import Box, Positioner

W = H = 10  # tiny frame => frame_area = 100, easy exact ratios
FRAME_AREA = W * H


def make_poser(cfg: PositionConfig | None = None) -> Positioner:
    return Positioner(cfg or PositionConfig(), image_width=W, image_height=H)


# ---------------------------------------------------------------------------
# Direction (horizontal thirds)
# ---------------------------------------------------------------------------
def test_horizontal_thirds():
    p = make_poser()
    assert p.horizontal_zone(1.0) == "left"  # < W/3
    assert p.horizontal_zone(5.0) == "ahead"  # W/3 .. 2W/3
    assert p.horizontal_zone(9.0) == "right"  # >= 2W/3


def test_boxes_at_frame_edges():
    p = make_poser()
    left = p.place("person", 0.9, Box(0, 0, 2, 2))  # flush left edge
    right = p.place("person", 0.9, Box(8, 0, 10, 2))  # flush right edge
    full = p.place("person", 0.9, Box(0, 0, 10, 2))  # spans full width
    assert left.horizontal == "left"
    assert right.horizontal == "right"
    assert full.horizontal == "ahead"  # centre of full-width box is the middle


def test_horizontal_boundary_is_inclusive_on_the_right_side():
    p = make_poser()
    # centre exactly at W/3 (3.333..) must be "ahead", not "left".
    assert p.horizontal_zone(W / 3.0) == "ahead"
    # centre just below W/3 is still "left".
    assert p.horizontal_zone(W / 3.0 - 0.01) == "left"
    # centre exactly at 2W/3 must be "right".
    assert p.horizontal_zone(2.0 * W / 3.0) == "right"


# ---------------------------------------------------------------------------
# Distance (area-ratio approximation)
# ---------------------------------------------------------------------------
def test_distance_bands_from_area_ratio():
    p = make_poser()
    very_close = p.place("car", 0.9, Box(0, 0, 5, 5))  # area 25 => ratio .25
    near = p.place("car", 0.9, Box(0, 0, 3, 5))  # area 15 => ratio .15
    far = p.place("car", 0.9, Box(0, 0, 2, 2))  # area  4 => ratio .04
    assert very_close.distance_band == "very close"
    assert near.distance_band == "near"
    assert far.distance_band == "far"


def test_distance_threshold_boundaries_inclusive():
    p = make_poser()
    # Exactly at the very_close threshold (0.20) => very close.
    assert p.distance_band(0.20) == "very close"
    assert p.distance_band(0.1999) == "near"
    # Exactly at the near threshold (0.05) => near.
    assert p.distance_band(0.05) == "near"
    assert p.distance_band(0.0499) == "far"


def test_area_ratio_computed_from_frame():
    p = make_poser()
    pos = p.place("car", 0.9, Box(0, 0, 5, 5))
    assert abs(pos.area_ratio - (25 / FRAME_AREA)) < 1e-9


# ---------------------------------------------------------------------------
# Phrasing
# ---------------------------------------------------------------------------
def test_describe_includes_distance_only_when_very_close():
    p = make_poser()
    close = p.place(
        "person", 0.9, Box(2, 2, 7, 7)
    )  # centred => ahead; area 25 => very close
    assert close.describe() == "person ahead, very close"


def test_describe_omits_distance_when_not_close():
    p = make_poser()
    chair = p.place("chair", 0.9, Box(0, 0, 2, 2))  # left, far
    assert chair.describe() == "chair on your left"


def test_horizontal_phrases_for_all_zones():
    p = make_poser()
    assert p.place("x", 0.9, Box(0, 0, 2, 2)).horizontal_phrase == "on your left"
    assert p.place("x", 0.9, Box(4, 0, 6, 2)).horizontal_phrase == "ahead"
    assert p.place("x", 0.9, Box(8, 0, 10, 2)).horizontal_phrase == "on your right"
