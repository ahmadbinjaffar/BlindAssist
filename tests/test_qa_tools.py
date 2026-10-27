"""Tests for the QA harness itself.

``scripts/qa_runtime_check.py`` decides whether the app leaks memory. That
verdict was wrong twice during QA (it first measured the venv launcher stub
instead of the real interpreter, then read thread counts as an unordered set),
so the analysis function is pinned down here with recorded real-world curves
instead of being trusted by eye.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "qa_runtime_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("qa_runtime_check", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qa = _load()

# A real 60 s run recorded during QA: RSS climbs while torch/ultralytics warm
# up, then sits flat (and finishes slightly lower), with threads settling at 25.
WARMUP_THEN_FLAT = [
    (0, 44_000_000, 14),
    (4, 194_000_000, 20),
    (8, 241_000_000, 20),
    (13, 358_000_000, 24),
    (17, 443_000_000, 26),
    (22, 443_000_000, 26),
    (26, 444_000_000, 26),
    (30, 446_000_000, 25),
    (34, 444_000_000, 25),
    (39, 444_000_000, 25),
    (43, 431_000_000, 25),
    (47, 424_000_000, 25),
    (52, 433_000_000, 25),
    (56, 429_000_000, 25),
    (60, 431_000_000, 22),
]

# A genuinely leaking process: both resources climb for the whole run.
LEAKING = [(i * 4.0, (100 + i * 20) * 1e6, 5 + i) for i in range(16)]


def test_steady_state_curve_passes():
    p = qa._plateau(WARMUP_THEN_FLAT, 60.0)
    assert p is not None
    assert abs(p["projected"]) < qa.GROWTH_LIMIT_MB
    assert p["thread_projected"] <= qa.THREAD_DRIFT_LIMIT


def test_warmup_is_reported_rather_than_counted_as_leak():
    p = qa._plateau(WARMUP_THEN_FLAT, 60.0)
    # The honest reading: a big early climb, then a flat tail.
    assert p["warmup"] / 1e6 > 300
    assert p["start"] >= 25
    assert p["rss_max"] - p["rss_min"] < 40


def test_falling_thread_count_is_not_treated_as_growth():
    """Regression: threads 26 -> 22 must not read as "drift of 4".

    The first implementation stored ``sorted(set(...))``, which discards time
    order, so a *falling* count produced a range of 4 and a false FAIL.
    """
    p = qa._plateau(WARMUP_THEN_FLAT, 60.0)
    assert p["thread_first"] >= p["thread_last"]
    assert p["thread_projected"] <= 0.0


def test_a_real_leak_is_still_detected():
    """The same function must flag memory AND thread growth."""
    p = qa._plateau(LEAKING, 60.0)
    assert p is not None
    assert p["projected"] > qa.GROWTH_LIMIT_MB
    assert p["thread_projected"] > qa.THREAD_DRIFT_LIMIT


def test_too_few_samples_returns_none():
    assert qa._plateau(WARMUP_THEN_FLAT[:5], 60.0) is None
    assert qa._plateau([], 60.0) is None


def test_trend_reports_rather_than_crashes_on_short_input():
    assert qa._trend(WARMUP_THEN_FLAT[:1]) == "not enough samples"
    assert "MB" in qa._trend(WARMUP_THEN_FLAT)
