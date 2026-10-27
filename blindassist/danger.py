"""Three-level alert priority: CRITICAL / HIGH / NORMAL.

Pure logic - no OpenCV, no speech, no threads - so it is cheap to test and can
be reused by the announcer, the CSV session log and the overlay.

Levels
------
CRITICAL - a dangerous class (person or a vehicle) is "very close". The
    announcement interrupts speech and uses the short "Stop! ..." phrase.
HIGH - a dangerous class is "near", or *any* class is "very close" without
    belonging to the critical set. Spoken normally, but sorted first.
NORMAL - everything else. Exactly the behaviour the app had before this
    feature existed; with :attr:`DangerConfig.enabled = False` every object
    classifies as NORMAL.
"""

from __future__ import annotations

from .config import DangerConfig
from .position import Position

LEVEL_CRITICAL = "CRITICAL"
LEVEL_HIGH = "HIGH"
LEVEL_NORMAL = "NORMAL"


def classify_level(label: str, distance_band: str, danger: DangerConfig) -> str:
    """Rate a single object's urgency from its class and distance band."""
    if not danger.enabled:
        return LEVEL_NORMAL
    critical_class = label in danger.critical_labels
    if critical_class and distance_band == "very close":
        return LEVEL_CRITICAL
    if critical_class and distance_band == "near":
        return LEVEL_HIGH
    if not critical_class and distance_band == "very close":
        return LEVEL_HIGH
    return LEVEL_NORMAL


def highest_level(placements: list[Position], danger: DangerConfig) -> str:
    """The most urgent level among this frame's placements ("NORMAL" if none)."""
    level = LEVEL_NORMAL
    for p in placements:
        lv = classify_level(p.label, p.distance_band, danger)
        if lv == LEVEL_CRITICAL:
            return LEVEL_CRITICAL
        if lv == LEVEL_HIGH:
            level = LEVEL_HIGH
    return level
