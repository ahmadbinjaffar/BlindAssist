"""Turn this frame's placements into one short spoken message.

Responsibilities:

* **Grouping** - objects of the same class in the same direction are merged
  ("2 people ahead" instead of "person ahead, person ahead").
* **Prioritisation** - urgent classes (person, car, bus, ...) and closer objects
  come first; at most :attr:`SpeechConfig.max_items_per_message` items are spoken.
* **Cooldown** - the same class+direction is not repeated within
  :attr:`SpeechConfig.cooldown_s` seconds, but a brand-new object, or one that
  has come *much closer*, is announced immediately.
* **Natural sentences** - e.g. ``"Person ahead, very close. Chair on your left."``
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from .config import DangerConfig, SpeechConfig, SummaryConfig
from .danger import LEVEL_CRITICAL, LEVEL_NORMAL, classify_level
from .position import HORIZONTAL_PHRASES, Position

# Irregular English plurals for the classes we most often see.
_PLURALS = {"person": "people"}

# Cap for the announcement log. Only its length is reported at shutdown, and a
# long unattended run would otherwise grow this without bound.
_HISTORY_LIMIT = 1000


def _pluralize(label: str) -> str:
    if label in _PLURALS:
        return _PLURALS[label]
    if label.endswith(("s", "sh", "ch", "x", "z")):
        return label + "es"
    if label.endswith("y") and label[-2:-1] not in ("a", "e", "i", "o", "u"):
        return label[:-1] + "ies"
    return label + "s"


def _capitalise(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _article(label: str) -> str:
    """ "a bottle" / "an orange" - correct English article for a class name."""
    return "an" if label[0].lower() in "aeiou" else "a"


@dataclass
class _Group:
    """All objects of one class in one direction for this frame."""

    label: str
    horizontal: str
    count: int
    best_rank: int  # closest distance rank among the members
    best_band: str  # the closest member's distance band


class Announcer:
    """Groups, prioritises and rate-limits what gets spoken each frame."""

    def __init__(
        self,
        cfg: SpeechConfig,
        clock=time.monotonic,
        danger: DangerConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.danger = danger
        self._clock = clock
        # (label, horizontal) -> (last_spoken_time, band_rank_at_that_time)
        self._memory: dict[tuple[str, str], tuple[float, int]] = {}
        self.history: deque[str] = deque(maxlen=_HISTORY_LIMIT)
        # Level of the most recent update(); "CRITICAL" tells the caller to
        # speak the returned message with urgent=True (interrupting speech).
        self.last_level: str = LEVEL_NORMAL

    # -- helpers -----------------------------------------------------------
    def _priority_index(self, label: str) -> int:
        try:
            return self.cfg.priority_labels.index(label)
        except ValueError:
            return len(self.cfg.priority_labels)  # non-priority classes sort last

    def _group(self, placements: list[Position]) -> list[_Group]:
        groups: dict[tuple[str, str], _Group] = {}
        for p in placements:
            key = (p.label, p.horizontal)
            g = groups.get(key)
            if g is None:
                groups[key] = _Group(
                    p.label, p.horizontal, 1, p.distance_rank, p.distance_band
                )
            else:
                g.count += 1
                if p.distance_rank < g.best_rank:
                    g.best_rank = p.distance_rank
                    g.best_band = p.distance_band
        return list(groups.values())

    def _phrase(self, g: _Group) -> str:
        direction = HORIZONTAL_PHRASES[g.horizontal]
        if g.count > 1:
            return f"{g.count} {_pluralize(g.label)} {direction}"
        base = f"{g.label} {direction}"
        if g.best_band == "very close":
            base += f", {g.best_band}"
        return base

    def _is_critical(self, g: _Group) -> bool:
        """True when danger priorities are on and this group is CRITICAL."""
        if self.danger is None or not self.danger.enabled:
            return False
        return classify_level(g.label, g.best_band, self.danger) == LEVEL_CRITICAL

    def _should_announce(self, g: _Group, now: float) -> bool:
        key = (g.label, g.horizontal)
        mem = self._memory.get(key)
        if mem is None:
            return True  # newly appearing object -> announce now
        last_time, last_rank = mem
        much_closer = g.best_rank < last_rank  # moved to a closer band
        cooldown = self.cfg.cooldown_s
        if self._is_critical(g) and self.danger is not None:
            # An object you can walk into deserves a sooner repeat warning.
            cooldown = self.danger.critical_cooldown_s
        cooled = (now - last_time) >= cooldown
        return much_closer or cooled

    # -- main --------------------------------------------------------------
    def update(
        self, placements: list[Position], now: float | None = None
    ) -> str | None:
        """Return a spoken message for this frame, or None if nothing is new."""
        now = self._clock() if now is None else now
        if not placements:
            self.last_level = LEVEL_NORMAL
            return None

        eligible = [g for g in self._group(placements) if self._should_announce(g, now)]
        if not eligible:
            self.last_level = LEVEL_NORMAL
            return None

        # CRITICAL alerts speak alone, interrupt speech and use the short
        # warning phrase ("Stop! Person ahead."). Everything else keeps the
        # pre-danger behaviour exactly.
        critical = [g for g in eligible if self._is_critical(g)]
        if critical:
            critical.sort(key=lambda g: (self._priority_index(g.label), g.best_rank))
            message = self._critical_phrase(critical[0])
            self._memory[(critical[0].label, critical[0].horizontal)] = (
                now,
                critical[0].best_rank,
            )
            self.history.append(message)
            self.last_level = LEVEL_CRITICAL
            return message

        # Most important class first; within a class, the closest first.
        eligible.sort(key=lambda g: (self._priority_index(g.label), g.best_rank))
        chosen = eligible[: self.cfg.max_items_per_message]

        message = ". ".join(_capitalise(self._phrase(g)) for g in chosen) + "."

        for g in chosen:
            self._memory[(g.label, g.horizontal)] = (now, g.best_rank)
        self.history.append(message)
        self.last_level = LEVEL_NORMAL
        return message

    def _critical_phrase(self, g: _Group) -> str:
        """Short warning sentence from the configured template."""
        direction = HORIZONTAL_PHRASES[g.horizontal]
        if g.count > 1:
            what = f"{g.count} {_pluralize(g.label)} {direction}"
        else:
            what = f"{g.label} {direction}"
        if self.danger is None:  # unreachable from update(); defensive only
            return f"Stop! {what}."
        # {label}/{direction} are offered for custom templates; {what} keeps
        # the default wording grammatical for both singular and plural groups.
        # The body is sentence-capitalised so "Stop! Person ahead." reads like
        # the app's other spoken sentences.
        return self.danger.warning_template.format(
            what=_capitalise(what), label=_capitalise(g.label), direction=direction
        )


# ---------------------------------------------------------------------------
# Scene summary (the D key "describe" command)
# ---------------------------------------------------------------------------
def summarize_scene(placements: list[Position], cfg: SummaryConfig) -> str:
    """Describe the whole scene at once, e.g.

    ``"I see 2 people, a chair and a bottle. The chair is on your left."``

    Items are counted (plurals included), then up to :attr:`max_groups`
    locations are added, most-important / closest first. Unlike the automatic
    announcements this is spoken on demand and is not rate-limited, because
    the user explicitly pressed the key.
    """
    if not placements:
        return cfg.nothing_template

    groups: dict[str, list[Position]] = {}
    for p in placements:
        groups.setdefault(p.label, []).append(p)

    # Overview: "2 people, a chair and an orange" (most important classes first).
    counts: list[tuple[int, str]] = []  # (count, label)
    for g in groups.values():
        counts.append((g[0].distance_rank, g[0].label))
    labels_in_rank = [lab for _, lab in sorted(counts, key=lambda c: c[0])]

    items: list[str] = []
    for label in labels_in_rank:
        count = len(groups[label])
        items.append(
            f"{count} {_pluralize(label)}"
            if count > 1
            else f"{_article(label)} {label}"
        )
    if len(items) > 1:
        listed = ", ".join(items[:-1]) + f" and {items[-1]}"
    else:
        listed = items[0]
    parts: list[str] = [cfg.overview_template.format(items=listed)]

    # Locations, closest/most important last (like the overview list flow).
    positioner = []
    for label in labels_in_rank:
        members = groups[label]
        best = min(members, key=lambda p: p.distance_rank)
        positioner.append((best.distance_rank, label, best.horizontal))
    for _, label, horizontal in sorted(positioner, key=lambda r: r[0])[
        -cfg.max_groups :
    ]:
        count = len(groups[label])
        what = _pluralize(label) if count > 1 else label
        parts.append(
            cfg.location_template.format(
                what=what,
                be=cfg.plural_be if count > 1 else cfg.singular_be,
                direction=HORIZONTAL_PHRASES[horizontal],
            )
        )
    return " ".join(parts)
