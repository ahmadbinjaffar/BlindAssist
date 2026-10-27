"""Tests for the announcer: grouping, priority, cooldown and phrasing."""

from __future__ import annotations

from blindassist.announcer import Announcer, _pluralize
from blindassist.config import SpeechConfig
from blindassist.position import Box, Position


def mk(
    label: str, horizontal: str = "ahead", band: str = "far", confidence: float = 0.9
) -> Position:
    """Build a Position; area_ratio is irrelevant to the announcer."""
    return Position(
        label=label,
        confidence=confidence,
        box=Box(0, 0, 1, 1),
        horizontal=horizontal,
        distance_band=band,
        area_ratio=0.01,
    )


def make_announcer(**kwargs) -> Announcer:
    return Announcer(SpeechConfig(**kwargs))


# ---------------------------------------------------------------------------
# Grouping + singular/plural
# ---------------------------------------------------------------------------
def test_pluralise_person():
    assert _pluralize("person") == "people"
    assert _pluralize("car") == "cars"
    assert _pluralize("box") == "boxes"


def test_grouping_two_people_into_one_phrase():
    ann = make_announcer()
    msg = ann.update([mk("person", "ahead"), mk("person", "ahead")], now=0.0)
    assert msg == "2 people ahead."


def test_single_person_very_close_sentence():
    ann = make_announcer()
    msg = ann.update([mk("person", "ahead", "very close")], now=0.0)
    assert msg == "Person ahead, very close."


def test_direction_phrases_in_sentence():
    ann = make_announcer()
    assert ann.update([mk("chair", "left")], now=0.0) == "Chair on your left."
    ann = make_announcer()
    assert ann.update([mk("car", "right")], now=0.0) == "Car on your right."


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------
def test_person_prioritised_over_closer_chair():
    ann = make_announcer()
    msg = ann.update(
        [mk("chair", "left", "very close"), mk("person", "ahead", "far")],
        now=0.0,
    )
    # person (priority) must be spoken before the closer but less urgent chair.
    assert msg.index("Person") < msg.index("Chair")


def test_closest_first_within_same_class():
    ann = make_announcer(max_items_per_message=1)
    msg = ann.update(
        [mk("car", "ahead", "far"), mk("car", "right", "very close")],
        now=0.0,
    )
    # Two separate groups (ahead vs right); the very-close one wins top slot.
    assert "very close" in msg


def test_max_items_per_message_capped():
    ann = make_announcer(max_items_per_message=3)
    msg = ann.update(
        [mk("person"), mk("car"), mk("dog"), mk("chair")],  # chair = non-priority
        now=0.0,
    )
    low = msg.lower()
    assert low.count(".") == 3  # three sentences
    assert "person" in low and "car" in low and "dog" in low
    assert "chair" not in low  # lowest priority got dropped


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
def test_same_object_not_repeated_within_cooldown():
    ann = make_announcer(cooldown_s=3.0)
    first = ann.update([mk("person", "ahead")], now=0.0)
    assert first is not None
    # Same object+position, within 3s => suppressed.
    assert ann.update([mk("person", "ahead")], now=1.0) is None
    assert ann.update([mk("person", "ahead")], now=2.9) is None
    # After the cooldown it may repeat.
    assert ann.update([mk("person", "ahead")], now=3.1) is not None


def test_new_object_announced_immediately_despite_cooldown():
    ann = make_announcer(cooldown_s=3.0)
    ann.update([mk("person", "ahead")], now=0.0)
    # A brand-new object (car) appears while person is still in cooldown.
    msg = ann.update([mk("person", "ahead"), mk("car", "right")], now=1.0)
    assert msg is not None and "Car" in msg


def test_much_closer_object_announced_immediately():
    ann = make_announcer(cooldown_s=3.0)
    ann.update([mk("person", "ahead", "far")], now=0.0)
    # Same object but now MUCH closer (far -> very close), within cooldown.
    msg = ann.update([mk("person", "ahead", "very close")], now=0.5)
    assert msg is not None and "very close" in msg


def test_farther_object_still_respects_cooldown():
    ann = make_announcer(cooldown_s=3.0)
    ann.update([mk("person", "ahead", "very close")], now=0.0)
    # Moves farther (very close -> far): not urgent, still in cooldown => None.
    assert ann.update([mk("person", "ahead", "far")], now=0.5) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_detections_returns_none():
    ann = make_announcer()
    assert ann.update([], now=0.0) is None


def test_many_objects_grouped_and_capped():
    ann = make_announcer(max_items_per_message=3)
    placements = (
        [mk("person", "ahead")] * 3
        + [mk("car", "right")] * 2
        + [mk("dog", "left")]
        + [mk("zebra", "ahead")]  # 4th, non-priority => should be dropped
    )
    msg = ann.update(placements, now=0.0)
    low = msg.lower()
    assert "3 people ahead" in low
    assert "2 cars on your right" in low
    assert low.count(". ") == 2  # three sentences joined
    assert "dog" in low  # kept (3rd by priority)
    assert "zebra" not in low  # dropped by the 3-item cap


def test_message_is_capitalised_and_punctuated():
    ann = make_announcer()
    msg = ann.update([mk("chair", "left"), mk("person", "ahead")], now=0.0)
    assert msg[0].isupper()
    assert msg.endswith(".")
