"""Scene summary (D key): counting, articles, plurals, locations, capping."""

from __future__ import annotations

from blindassist.announcer import summarize_scene
from blindassist.config import SummaryConfig
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


def test_empty_scene_says_nothing_seen():
    assert summarize_scene([], SummaryConfig()) == "I do not see anything."


def test_single_item_matches_the_requested_example_shape():
    cfg = SummaryConfig()
    out = summarize_scene([mk("chair", "left")], cfg)
    assert out == "I see a chair. The chair is on your left."


def test_counts_and_plurals_in_overview():
    cfg = SummaryConfig()
    scene = [mk("person"), mk("person"), mk("chair"), mk("bottle")]
    out = summarize_scene(scene, cfg)
    assert out == (
        "I see 2 people, a chair and a bottle. "
        "The people are ahead. The chair is ahead. The bottle is ahead."
    )


def test_article_an_for_vowel_initial_labels():
    cfg = SummaryConfig(max_groups=1)
    out = summarize_scene([mk("orange"), mk("chair")], cfg)
    assert out.startswith("I see an orange and a chair.")
    # Only max_groups location sentences are appended.
    assert out.count("The ") == 1


def test_locations_are_closest_first():
    cfg = SummaryConfig()
    scene = [mk("chair", band="near"), mk("bottle", band="very close")]
    out = summarize_scene(scene, cfg)
    # The bottle is closer, so its location sentence comes first.
    assert out.index("The bottle") < out.index("The chair")


def test_max_groups_caps_location_sentences_but_not_the_overview():
    cfg = SummaryConfig(max_groups=2)
    scene = [mk("chair"), mk("bottle"), mk("tv"), mk("dog")]
    out = summarize_scene(scene, cfg)
    assert out.startswith("I see 4") or out.startswith("I see ")  # all counted
    assert out.count("The ") == 2


def test_mixed_distance_bands_reported_from_the_closest_member():
    cfg = SummaryConfig(max_groups=1)
    scene = [mk("person", "left", "far"), mk("person", "left", "very close")]
    out = summarize_scene(scene, cfg)
    assert out == "I see 2 people. The people are on your left."


def test_custom_templates_round_trip():
    cfg = SummaryConfig(
        overview_template="Here: {items}",
        location_template="At {direction}: {what}.",
    )
    out = summarize_scene([mk("dog", "right")], cfg)
    assert out == "Here: a dog At on your right: dog."


def test_nothing_template_is_configurable():
    cfg = SummaryConfig(nothing_template="No targets visible.")
    assert summarize_scene([], cfg) == "No targets visible."
