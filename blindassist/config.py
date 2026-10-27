"""Central configuration for BlindAssist.

All tunables live here so the rest of the codebase imports a single source of
truth. Values are deliberately conservative so that a first run "just works"
with a standard integrated laptop webcam.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repository root = parent of the `blindassist` package.
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
EVIDENCE_DIR = ROOT / "Evidence"

# Local YOLO weights. During SETUP (online) ultralytics caches the weights;
# afterwards everything runs fully OFFLINE from this path.
DEFAULT_MODEL_PATH = ASSETS_DIR / "yolov8n.pt"


def _env_default(name: str, fallback: str) -> str:
    """Config default overridable from the environment (TTS_BACKEND, ...)."""
    value = os.environ.get(name, "").strip().lower()
    return value or fallback


# ---------------------------------------------------------------------------
# Detection / vision
# ---------------------------------------------------------------------------
# YOLO's default input size: the accuracy/FPS middle ground of INPUT_SIZES.
DEFAULT_INFERENCE_SIZE = 640


@dataclass
class VisionConfig:
    """Camera + YOLO inference settings."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    confidence: float = 0.45
    iou: float = 0.45
    inference_size: int = DEFAULT_INFERENCE_SIZE  # YOLO input size in px
    # Frame skipping: run the detector on every Nth captured frame (1 = every
    # frame). The overlay keeps showing the last known positions in between,
    # which roughly multiplies the UI FPS on slow CPUs.
    frame_skip: int = 1
    show_preview: bool = False


# Supported detector input sizes. 320 = fastest, 640 = most accurate,
# 480 = the middle ground; anything else is snapped to the nearest entry.
INPUT_SIZES: tuple[int, ...] = (320, 480, 640)
# Upper bound for --skip; skipping more than this would make the app feel
# frozen rather than fast.
MAX_FRAME_SKIP = 30


# COCO class ids we surface to the user by default. Empty set => keep all.
# This is an *indoor-navigation* whitelist: vehicles and animals, plus the
# furniture / objects a blind user can actually walk into. For the complete
# 80-class COCO set run the app with --all-classes.
#   0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck, 13 bench,
#  15 cat, 16 dog, 24 backpack, 25 umbrella, 28 suitcase, 39 bottle, 41 cup,
#  56 chair, 57 couch, 58 potted plant, 59 bed, 60 dining table, 62 tv,
#  63 laptop, 67 cell phone
DEFAULT_CLASS_WHITELIST: set[int] = {
    0,
    1,
    2,
    3,
    5,
    7,
    13,
    15,
    16,
    24,
    25,
    28,
    39,
    41,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    67,
}


# ---------------------------------------------------------------------------
# Positioning: direction + approximate distance
# ---------------------------------------------------------------------------
@dataclass
class PositionConfig:
    """How an object's location is turned into words.

    Direction
    ---------
    The frame is split into three equal vertical bands. The horizontal centre of
    the bounding box decides "on your left" / "ahead" / "on your right".

    Distance (APPROXIMATION - NO DEPTH SENSOR)
    ------------------------------------------
    There is no depth camera / stereo rig, so distance is *estimated* purely from
    how much of the frame the bounding box covers (its area ratio). A bigger box
    is assumed to be closer. This is a coarse heuristic, not a measurement: it
    assumes objects are roughly the same real size, so a small object close by and
    a large object far away can look similar. It is good enough to distinguish
    "very close" (right in front of you) from "far" (across the room).
    """

    # Bounding-box area / full-frame area thresholds.
    very_close_area_ratio: float = 0.20  # >= 20% of the frame => "very close"
    near_area_ratio: float = 0.05  # >=  5% of the frame => "near", else "far"


# ---------------------------------------------------------------------------
# Speech / announcement pacing + prioritisation
# ---------------------------------------------------------------------------
# COCO has no "stairs" class, so stairs-like obstacles cannot be prioritised by
# label (a known limitation). The classes below are ordered most-important-first.
DEFAULT_PRIORITY_LABELS: tuple[str, ...] = (
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "dog",
)


@dataclass
class SpeechConfig:
    """Text-to-speech pacing, grouping and prioritisation."""

    enabled: bool = True
    rate_wpm: int = 175  # words per minute
    volume: float = 1.0  # 0.0 .. 1.0
    # Voice selection: None = system default; int = index; str = name/id substring.
    voice: str | int | None = None
    # TTS backend chain: "auto" tries pyttsx3 -> PowerShell -> beep, verifying
    # each with the startup self-test; pin one with TTS_BACKEND or --tts-backend.
    backend: str = _env_default("TTS_BACKEND", "auto")
    # Don't repeat the same object+position within this many seconds.
    cooldown_s: float = 3.0
    # At most this many distinct items are packed into one spoken message.
    max_items_per_message: int = 3
    # Ordered most-important-first (see DEFAULT_PRIORITY_LABELS).
    priority_labels: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_PRIORITY_LABELS
    )


# ---------------------------------------------------------------------------
# Danger priority levels (feature: smarter alerts)
# ---------------------------------------------------------------------------
# Labels whose very-close presence is a CRITICAL alert (interrupts speech and
# uses the short "Stop! ..." warning phrase). Vehicles + person by default.
DEFAULT_CRITICAL_LABELS: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)


@dataclass
class DangerConfig:
    """Three-level alert priority: CRITICAL / HIGH / NORMAL.

    CRITICAL - a dangerous class (person/vehicle) is "very close"; the
        announcement interrupts current speech and uses the short warning
        phrase in :attr:`warning_template`. Placeholders: ``{what}`` (the
        "person ahead" / "2 people ahead" body), ``{label}``, ``{direction}``.
    HIGH - a dangerous class is "near", or *any* other class is "very close".
        Spoken as a normal sentence, but sorted ahead of NORMAL items.
    NORMAL - everything else.
    """

    enabled: bool = True
    critical_labels: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_CRITICAL_LABELS
    )
    # A critical alert may repeat sooner than the normal cooldown; walking into
    # a car that is still there deserves a second warning sooner than a chair.
    critical_cooldown_s: float = 2.0
    warning_template: str = "Stop! {what}."


# ---------------------------------------------------------------------------
# Proximity beep (winsound, non-blocking background thread)
# ---------------------------------------------------------------------------
@dataclass
class BeepConfig:
    """Geiger-style beeping that gets faster as the nearest object gets closer.

    ``interval_by_rank`` / ``frequency_by_rank`` are keyed by distance rank
    (0 = "very close", 1 = "near", 2 = "far"). A rank missing from the tables
    means "stay silent for that rank"; no object at all is always silent.
    """

    enabled: bool = True
    duration_ms: int = 90
    interval_by_rank: dict[int, int] = field(
        default_factory=lambda: {0: 450, 1: 800, 2: 1200}
    )
    frequency_by_rank: dict[int, int] = field(
        default_factory=lambda: {0: 1500, 1: 1000, 2: 700}
    )


# ---------------------------------------------------------------------------
# On-screen UI (feature: cleaner overlay, legend, H help screen)
# ---------------------------------------------------------------------------
@dataclass
class UIConfig:
    """What the OpenCV overlay draws and how transparent the panels are.

    ``panel_alpha`` is the blend weight of the status/caption panels over the
    camera image (0 = invisible panels, 1 = fully opaque as in the original
    overlay). The help screen (H key) is always semi-transparent so the live
    scene stays visible behind it.
    """

    panel_alpha: float = 0.6  # 0..1
    show_legend: bool = True  # colour legend: red = very close, ...
    show_zone_guides: bool = True  # dashed left/ahead/right guides


# ---------------------------------------------------------------------------
# Session announcement log (logs/session_YYYYMMDD.csv)
# ---------------------------------------------------------------------------
@dataclass
class SessionLogConfig:
    """Every spoken/muted announcement is appended to a daily CSV file.

    The file lives at ``<directory>/session_YYYYMMDD.csv`` (local date). Runs
    on the same day append to the same file; midnight starts a new one.
    """

    enabled: bool = True
    directory: str = "logs"  # relative to the project root unless absolute
    prefix: str = "session"


# ---------------------------------------------------------------------------
# Persistent user settings (settings.json)
# ---------------------------------------------------------------------------
@dataclass
class SettingsConfig:
    """settings.json behaviour (see :mod:`blindassist.settings`).

    ``enabled`` switches the whole feature off (nothing is read or written).
    ``autosave`` writes the file after every keyboard-initiated change; with
    it off, values only persist when the app shuts down cleanly.
    """

    enabled: bool = True
    autosave: bool = True


# ---------------------------------------------------------------------------
# Language / voice selection (feature: language option)
# ---------------------------------------------------------------------------
@dataclass
class LanguageConfig:
    """Which SAPI voice to ask for.

    ``"en"`` (default) keeps the pre-existing behaviour exactly: the engine
    default voice is used, unless ``speech.voice`` names one explicitly.
    Any other value (e.g. ``"ur"``) makes the app request a voice whose name
    or id contains :attr:`voice_hint` - the shipped Windows Urdu packs are
    named like "Microsoft Urdu". The sentences themselves stay English (COCO
    class names and fixed phrases), documented as a limitation: this feature
    selects the *voice*, it does not translate the text.

    When no matching voice is installed the app falls back to the English
    default voice and speaks :attr:`fallback_note` once at startup.
    """

    language: str = "en"
    voice_hint: str = "urdu"
    fallback_note: str = "No Urdu voice found; using the English voice."


# ---------------------------------------------------------------------------
# Scene summary (the D key "describe" command)
# ---------------------------------------------------------------------------
@dataclass
class SummaryConfig:
    """Full scene description spoken on demand (D key).

    Produces e.g. ``"I see 2 people, a chair and a bottle. The chair is on
    your left."`` Group sentences are capped at :attr:`max_groups` (sorted
    most-important / closest first) so the summary never becomes a speech
    wall; unmentioned groups are still counted in the overview.
    """

    enabled: bool = True
    max_groups: int = 3
    nothing_template: str = "I do not see anything."
    overview_template: str = "I see {items}."
    location_template: str = "The {what} {be} {direction}."
    # Words used for {be}: plural groups say "are", singular say "is".
    singular_be: str = "is"
    plural_be: str = "are"


# ---------------------------------------------------------------------------
# Low-light help (automatic CLAHE contrast enhancement)
# ---------------------------------------------------------------------------
@dataclass
class LightConfig:
    """Automatic contrast enhancement for a dark room, plus a spoken warning.

    When the frame's mean luminance drops below :attr:`dark_mean_threshold`
    for :attr:`dark_streak_frames` frames in a row, detection runs on a CLAHE-
    enhanced copy of the frame (the preview stays the true camera image) and
    the assistant warns the user at most once per :attr:`warn_cooldown_s`.
    """

    enabled: bool = True
    # Mean 0-255 luminance under which a frame counts as "dark". A dim indoor
    # webcam frame is typically 40-80; a normal office one is 100-160.
    dark_mean_threshold: float = 60.0
    dark_streak_frames: int = 3  # avoid reacting to a single dark frame
    clip_limit: float = 2.0  # CLAHE clipLimit
    tile_grid: tuple[int, int] = field(default_factory=lambda: (8, 8))
    warn_cooldown_s: float = 60.0
    warn_message: str = "It is too dark, detection may be poor."


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
@dataclass
class TrackerConfig:
    """Simple IoU tracker parameters (label + position stability)."""

    iou_match_threshold: float = 0.30
    max_lost_frames: int = 8
    history_len: int = 8
    # Announced boxes/distances are smoothed as the mean of the last N raw
    # boxes (the *history* stays raw so the motion trend is unaffected).
    smoothing_window: int = 5
    # An object must persist this many frames before it is announced or drawn,
    # which filters one-frame detection glitches (false alarms).
    min_confirm_frames: int = 3


# ---------------------------------------------------------------------------
# Root config bundle
# ---------------------------------------------------------------------------
@dataclass
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    danger: DangerConfig = field(default_factory=DangerConfig)
    beep: BeepConfig = field(default_factory=BeepConfig)
    light: LightConfig = field(default_factory=LightConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    session: SessionLogConfig = field(default_factory=SessionLogConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    model_path: Path = DEFAULT_MODEL_PATH
    class_whitelist: set[int] = field(
        default_factory=lambda: set(DEFAULT_CLASS_WHITELIST)
    )


def load_config() -> Config:
    """Return the default configuration bundle."""
    return Config()
