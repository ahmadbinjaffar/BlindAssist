"""Turn a bounding box into a human-readable direction + approximate distance.

Direction
---------
The frame is split into three equal vertical bands. The horizontal centre of the
box maps to "on your left" / "ahead" / "on your right".

Distance (APPROXIMATION - no depth sensor)
------------------------------------------
Distance is estimated only from the fraction of the frame the box covers:

    area_ratio = box.area / (frame_width * frame_height)

A larger box is assumed to be closer. The thresholds that split the ratio into
"very close" / "near" / "far" live in :class:`blindassist.config.PositionConfig`.
This is a coarse heuristic, NOT a measurement - there is no depth sensor, so it
cannot tell a small nearby object from a large distant one. It is only meant to
separate "right in front of you" from "across the room".
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PositionConfig

# Public words for each horizontal third.
HORIZONTAL_PHRASES: dict[str, str] = {
    "left": "on your left",
    "ahead": "ahead",
    "right": "on your right",
}

# How urgent each distance band is (lower = closer = more urgent).
DISTANCE_RANK: dict[str, int] = {"very close": 0, "near": 1, "far": 2}


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in pixel coords (x1,y1 top-left; x2,y2 bottom-right)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Position:
    """A detection annotated with a direction and an approximate distance band."""

    label: str
    confidence: float
    box: Box
    horizontal: str  # "left" | "ahead" | "right"
    distance_band: str  # "very close" | "near" | "far"
    area_ratio: float

    @property
    def horizontal_phrase(self) -> str:
        return HORIZONTAL_PHRASES[self.horizontal]

    @property
    def distance_rank(self) -> int:
        return DISTANCE_RANK[self.distance_band]

    @property
    def is_very_close(self) -> bool:
        return self.distance_band == "very close"

    def describe(self) -> str:
        """Short phrase, e.g. 'person ahead, very close' or 'chair on your left'.

        The distance band is only spoken when the object is "very close" (the
        urgent case); otherwise the direction alone reads more naturally.
        """
        base = f"{self.label} {self.horizontal_phrase}"
        if self.is_very_close:
            base += f", {self.distance_band}"
        return base


class Positioner:
    """Stateless helper turning boxes into :class:`Position` objects."""

    def __init__(
        self, cfg: PositionConfig, image_width: int, image_height: int
    ) -> None:
        self.cfg = cfg
        self.image_width = float(image_width)
        self.image_height = float(image_height)
        self.frame_area = self.image_width * self.image_height

    # -- direction ---------------------------------------------------------
    def horizontal_zone(self, center_x: float) -> str:
        if center_x < self.image_width / 3.0:
            return "left"
        if center_x < 2.0 * self.image_width / 3.0:
            return "ahead"
        return "right"

    # -- distance ----------------------------------------------------------
    def distance_band(self, area_ratio: float) -> str:
        if area_ratio >= self.cfg.very_close_area_ratio:
            return "very close"
        if area_ratio >= self.cfg.near_area_ratio:
            return "near"
        return "far"

    # -- public ------------------------------------------------------------
    def place(self, label: str, confidence: float, box: Box) -> Position:
        cx, _cy = box.center
        ratio = (box.area / self.frame_area) if self.frame_area > 0 else 0.0
        return Position(
            label=label,
            confidence=confidence,
            box=box,
            horizontal=self.horizontal_zone(cx),
            distance_band=self.distance_band(ratio),
            area_ratio=ratio,
        )
