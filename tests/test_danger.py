"""Danger priority levels: classification and effect on announcements."""

from __future__ import annotations

from blindassist.announcer import Announcer
from blindassist.config import DangerConfig, SpeechConfig
from blindassist.danger import (
    LEVEL_CRITICAL,
    LEVEL_HIGH,
    LEVEL_NORMAL,
    classify_level,
    highest_level,
)
from blindassist.position import Box, Position


def mk(
    label: str, horizontal: str = "ahead", band: str = "far", confidence: float = 0.9
) -> Position:
    return Position(
        label=label,
        confidence=confidence,
        box=Box(0, 0, 1, 1),
        horizontal=horizontal,
        distance_band=band,
        area_ratio=0.01,
    )


# ---------------------------------------------------------------------------
# Pure classification
# ---------------------------------------------------------------------------
def test_critical_label_very_close_is_critical():
    assert classify_level("car", "very close", DangerConfig()) == LEVEL_CRITICAL
    assert classify_level("person", "very close", DangerConfig()) == LEVEL_CRITICAL


def test_critical_label_near_is_high():
    assert classify_level("car", "near", DangerConfig()) == LEVEL_HIGH
    assert classify_level("person", "near", DangerConfig()) == LEVEL_HIGH


def test_harmless_class_very_close_is_high_not_critical():
    # A chair right in front of you is urgent, but it is not "Stop!" urgent.
    assert classify_level("chair", "very close", DangerConfig()) == LEVEL_HIGH
    assert classify_level("chair", "far", DangerConfig()) == LEVEL_NORMAL


def test_disabled_config_makes_everything_normal():
    cfg = DangerConfig(enabled=False)
    assert classify_level("car", "very close", cfg) == LEVEL_NORMAL
    assert classify_level("person", "near", cfg) == LEVEL_NORMAL


def test_critical_labels_are_configurable():
    cfg = DangerConfig(critical_labels=("dog",))
    assert classify_level("dog", "very close", cfg) == LEVEL_CRITICAL
    assert classify_level("car", "very close", cfg) == LEVEL_HIGH  # still very close


def test_highest_level_of_multiple_placements():
    danger = DangerConfig()
    assert highest_level([mk("chair"), mk("dog")], danger) == LEVEL_NORMAL
    assert highest_level([mk("chair"), mk("car", band="near")], danger) == LEVEL_HIGH
    # Critical wins even when it is not the first item in the list.
    assert (
        highest_level(
            [mk("chair", band="very close"), mk("person", band="very close")], danger
        )
        == LEVEL_CRITICAL
    )
    assert highest_level([], danger) == LEVEL_NORMAL


# ---------------------------------------------------------------------------
# Announcer integration
# ---------------------------------------------------------------------------
def make_announcer(danger: DangerConfig | None = None) -> Announcer:
    return Announcer(SpeechConfig(), danger=danger)


def test_critical_person_uses_short_stop_phrase():
    ann = make_announcer(DangerConfig())
    msg = ann.update([mk("person", "ahead", "very close")], now=0.0)
    assert msg == "Stop! Person ahead."
    assert ann.last_level == LEVEL_CRITICAL


def test_critical_group_of_two_stays_grammatical():
    ann = make_announcer(DangerConfig())
    msg = ann.update(
        [mk("person", "ahead", "very close"), mk("person", "ahead", "very close")],
        now=0.0,
    )
    assert msg == "Stop! 2 people ahead."


def test_custom_warning_template():
    danger = DangerConfig(warning_template="Danger! {label} {direction}!")
    ann = make_announcer(danger)
    msg = ann.update([mk("car", "left", "very close")], now=0.0)
    assert msg == "Danger! Car on your left!"


def test_danger_off_keeps_the_original_sentence():
    ann = make_announcer(DangerConfig(enabled=False))
    msg = ann.update([mk("person", "ahead", "very close")], now=0.0)
    assert msg == "Person ahead, very close."
    assert ann.last_level == LEVEL_NORMAL


def test_no_danger_config_keeps_the_original_sentence():
    # Announcer per-object callers (older embedders) pass no danger config at
    # all; that must behave exactly like the pre-danger app.
    ann = make_announcer(None)
    assert ann.update([mk("car", "ahead", "very close")], now=0.0) == (
        "Car ahead, very close."
    )


def test_high_level_is_spoken_normally_but_first():
    ann = make_announcer(DangerConfig())
    # A near car (HIGH) comes before a further chair (NORMAL) in one message.
    msg = ann.update([mk("chair", "ahead"), mk("car", "ahead", "near")], now=0.0)
    assert msg == "Car ahead. Chair ahead."
    assert ann.last_level == LEVEL_NORMAL


def test_critical_beats_high_in_the_same_frame():
    ann = make_announcer(DangerConfig())
    msg = ann.update([mk("car", "ahead", "near"), mk("person", "left", "very close")])
    assert msg == "Stop! Person on your left."


def test_critical_alert_repeats_sooner_than_normal_cooldown():
    ann = make_announcer(DangerConfig(critical_cooldown_s=2.0))
    first = ann.update([mk("person", "ahead", "very close")], now=0.0)
    assert first == "Stop! Person ahead."
    assert ann.update([mk("person", "ahead", "very close")], now=1.0) is None
    again = ann.update([mk("person", "ahead", "very close")], now=2.0)
    assert again == "Stop! Person ahead."  # 2 s < default 3 s cooldown


def test_critical_message_suppresses_other_items():
    ann = make_announcer(DangerConfig())
    msg = ann.update(
        [mk("person", "ahead", "very close"), mk("chair", "right")], now=0.0
    )
    assert msg == "Stop! Person ahead."  # the chair is not spoken this frame
    # ... but it is still remembered and can be spoken next frame.
    next_msg = ann.update([mk("chair", "right")], now=0.1)
    assert next_msg == "Chair on your right."
