"""BlindAssist real-time application (OpenCV window UI).

Every frame follows the same pipeline:

    webcam -> YOLO detector -> IoU tracker -> position/distance -> announcer -> TTS

Inference runs on a **background worker thread** that owns a single "latest
frame" slot: while the worker is busy the newest frame is *dropped* rather than
queued, so the preview window never stutters and memory never grows. The UI
thread only ever draws the most recent results.

Run it with::

    python -m blindassist.app                          # live webcam + speech
    python -m blindassist.app --no-speech              # live, print only
    python -m blindassist.app --video clip.mp4         # run on a video file
    python -m blindassist.app --video img.jpg --headless-test   # still image
    python -m blindassist.app --headless-test --max-frames 60   # smoke test

On-screen keyboard controls:

    SPACE      start / pause          +/-   raise / lower confidence
    [ ]        slower / faster speech  M     mute / unmute
    S          save screenshot         Q/ESC quit

Anything not covered by the flags above is logged to ``logs/blindassist.log``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import settings as settings_store
from . import speech as speech_mod
from .announcer import Announcer, summarize_scene
from .beep import make_beep_controller
from .config import (
    DEFAULT_INFERENCE_SIZE,
    EVIDENCE_DIR,
    INPUT_SIZES,
    MAX_FRAME_SKIP,
    Config,
    load_config,
)
from .danger import LEVEL_CRITICAL
from .detector import Detector
from .lighting import LightWatcher, enhance, mean_luma
from .position import Position, Positioner
from .sessionlog import SessionLog
from .speech import BACKEND_ORDER, make_speaker
from .tracker import Tracker, confirmed_tracks

IS_WINDOWS = sys.platform.startswith("win")

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "blindassist.log"
SCREENSHOT_DIR = EVIDENCE_DIR / "screenshots"

WINDOW_NAME = "BlindAssist"

# BGR colours per urgency band (the distance is an area-ratio approximation).
URGENCY_COLOR: dict[str, tuple[int, int, int]] = {
    "very close": (0, 0, 255),  # red    - stop / right in front of you
    "near": (0, 165, 255),  # orange - close
    "far": (0, 200, 0),  # green  - across the room
}
ZONE_COLOR = (95, 95, 95)  # dashed left/ahead/right guide lines
ZONE_TEXT = (180, 180, 180)  # zone captions (outlined for readability)
BAR_BG = (35, 35, 35)

CONF_STEP = 0.05
CONF_MIN, CONF_MAX = 0.05, 0.95
RATE_STEP = 10
RATE_MIN, RATE_MAX = 80, 300
# TTS_BACKEND / --tts-backend: "auto" walks the whole fallback chain.
TTS_BACKENDS = ("auto", *BACKEND_ORDER)

WELCOME = (
    "Assistant started. Press space to pause, plus or minus to change "
    "sensitivity, M to mute, D to describe the whole scene, H for help, "
    "S to save a picture, and Q to quit."
)

# Frame-grab failure policy: retry this many grabs before trying to reopen the
# camera, and give up after this many so a dead camera cannot spin forever.
REOPEN_AFTER_FAILURES = 5
GIVE_UP_AFTER_FAILURES = 60
# A smaller dt than this is timing noise (the very first frame), not a real
# frame interval; feeding it into the FPS average would invent a huge number.
MIN_FRAME_DT = 0.001
# cv2.waitKey() returns this when no key was pressed.
KEY_NONE = 0xFF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_LOG = logging.getLogger("blindassist")


# ---------------------------------------------------------------------------
# Console + file logging helpers
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    """Log to ``logs/blindassist.log``. Console output stays on plain prints."""
    if _LOG.handlers:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _LOG.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOG.addHandler(handler)


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)
    _LOG.info(msg)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)
    _LOG.warning(msg)


def error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)
    _LOG.error(msg)


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
class FrameSource:
    """Anything the app can pull frames from (camera, video file, still image)."""

    def read(self):  # -> np.ndarray | None
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - trivial
        return

    def reopen(self) -> bool:
        """Try to restore a lost live source. Default: not reconnectable."""
        return False

    @property
    def description(self) -> str:
        return "source"

    @property
    def is_live(self) -> bool:
        return False


class CameraSource(FrameSource):
    """Webcam capture with DirectShow-first backend selection on Windows.

    Opening is confirmed with a real frame read, because a ``VideoCapture`` can
    report ``isOpened()`` yet still deliver nothing.
    """

    def __init__(self, index: int, width: int, height: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None
        self.ok = self._open()

    def _open(self) -> bool:
        backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if IS_WINDOWS else (cv2.CAP_ANY,)
        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                self.cap = cap
                return True
            cap.release()
        self.cap = None
        return False

    @property
    def description(self) -> str:
        return f"camera {self.index}"

    @property
    def is_live(self) -> bool:
        return True

    def read(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reopen(self) -> bool:
        self.release()
        return self._open()


class VideoSource(FrameSource):
    """A video file, or a still image (looped) for deterministic testing."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such file: {self.path}")
        self._image = None
        self.cap: cv2.VideoCapture | None = None
        if self.path.suffix.lower() in IMAGE_EXTENSIONS:
            img = cv2.imread(str(self.path))
            if img is None or img.size == 0:
                raise ValueError(f"could not decode image: {self.path}")
            self._image = img
        else:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise ValueError(f"could not open video: {self.path}")

    @property
    def description(self) -> str:
        kind = "image" if self._image is not None else "video"
        return f"{kind} {self.path.name}"

    def read(self):
        if self._image is not None:
            return self._image.copy()
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            return None  # end of file
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------
class BlindAssistApp:
    """Owns the capture loop, the detection worker and the OpenCV window."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Precedence: config.py defaults < settings.json < command line.
        # settings.py applies the stored values first; _apply_overrides then
        # lets explicit CLI flags win for this session.
        self.cfg: Config = load_config()
        if self.cfg.settings.enabled:
            stored, issues = settings_store.load()
            settings_store.apply_to_config(self.cfg, stored, issues)
            for issue in issues:
                warn(f"settings.json: {issue}")
        self.cfg = _apply_overrides(self.cfg, args)
        self.settings_path = settings_store.SETTINGS_PATH

        self.source: FrameSource | None = None
        self.detector: Detector | None = None
        self.tracker: Tracker | None = None
        self.positioner: Positioner | None = None
        self.announcer: Announcer | None = None
        self.speaker = None
        self.beep = None  # proximity beeper; created with the pipeline
        # Session CSV log: independent of the camera pipeline, so _announce
        # can record rows even before the pipeline exists (tests, early keys).
        self.sessionlog = SessionLog(self.cfg.session)
        self._light: LightWatcher | None = None

        # --- shared state (worker writes, UI reads) ------------------------
        self._pending_lock = threading.Lock()
        self._pending_frame = None
        self._frame_ready = threading.Event()
        self._stop_evt = threading.Event()
        self._settings_dirty = False  # a keyboard change happened this session
        self._results_lock = threading.Lock()
        self._positions: list[Position] = []
        self._last_message = ""
        self._worker: threading.Thread | None = None
        self._dark_mode = False  # low-light CLAHE currently active
        # Headless testing aid (--demo-keys): frame index -> simulated key.
        self._demo_presses = _parse_demo_keys(getattr(args, "demo_keys", ""))

        # --- UI state ------------------------------------------------------
        self.paused = False
        # Help overlay state (H key): drawn on top of everything else.
        self.show_help = False
        # NOTE: --no-speech is handled by NullSpeaker (a no-op that still logs
        # the utterance). `muted` is only the runtime M key, so the app keeps
        # printing [SPEAK] lines in print-only mode.
        self.muted = False
        # Set at start-up when the OS master volume is muted/0, so the overlay
        # can show the warning instead of the app looking silently broken.
        self.volume_warning = ""
        self.confidence = float(self.cfg.vision.confidence)
        self.rate_wpm = int(self.cfg.speech.rate_wpm)
        self.fps = 0.0
        self.dropped = 0
        self.processed = 0
        self._frames_seen = 0  # captured frames (includes skipped ones)
        self._consecutive_failures = 0
        # Last frame drawn; read by the S (screenshot) key. Must exist from the
        # start so _handle_key is safe to call before the capture loop runs.
        self._last_frame = None

    # -- setup -------------------------------------------------------------
    def _open_source(self) -> bool:
        if self.args.video:
            try:
                self.source = VideoSource(Path(self.args.video))
            except (FileNotFoundError, ValueError) as exc:
                error(str(exc))
                return False
            info(f"Opened {self.source.description}.")
            return True

        requested = self.cfg.vision.camera_index
        tried = [requested] + [i for i in range(4) if i != requested]
        for idx in tried:
            cam = CameraSource(
                idx, self.cfg.vision.frame_width, self.cfg.vision.frame_height
            )
            if cam.ok:
                if idx != requested:
                    warn(f"Camera {requested} unavailable; fell back to camera {idx}.")
                self.source = cam
                info(f"Opened {cam.description}.")
                return True
            cam.release()
        error(
            f"No usable camera found (tried indices {tried}). "
            "Plug in a webcam, or run with --video <file>."
        )
        return False

    def _build_pipeline(self, width: int, height: int) -> None:
        info("Loading YOLO model (first run may download weights)...")
        if self.args.model is not None and not self.cfg.model_path.exists():
            # An explicit --model is a user expectation, not a cache miss. Do not
            # silently download weights to some other path and keep going.
            raise FileNotFoundError(f"--model {self.args.model} does not exist")
        whitelist = None if self.args.all_classes else self.cfg.class_whitelist
        self.detector = Detector(self.cfg.vision, self.cfg.model_path, whitelist)
        self.detector.load()
        self.detector.cfg.confidence = self.confidence
        n_classes = (
            "all 80 COCO classes" if whitelist is None else f"{len(whitelist)} classes"
        )
        info(f"Model ready ({n_classes}). Frame {width}x{height}.")

        self.tracker = Tracker(self.cfg.tracker)
        self.positioner = Positioner(self.cfg.position, width, height)
        self.announcer = Announcer(self.cfg.speech, danger=self.cfg.danger)
        # Language feature: a non-English language rewrites the requested
        # *voice* (never the sentences); an explicit speech.voice always wins.
        language = self.cfg.language
        voice = self.cfg.speech.voice
        fallback_note = ""
        if voice is None and language.language != "en":
            voice = language.voice_hint
            fallback_note = language.fallback_note
            info(
                f"Language {language.language!r}: requesting a voice matching "
                f"'{language.voice_hint}'."
            )
        self.speaker = make_speaker(
            self.cfg.speech.enabled,
            self.rate_wpm,
            self.cfg.speech.volume,
            voice,
            allow_fallback=bool(fallback_note),
            fallback_note=fallback_note,
            backend=self.cfg.speech.backend,
            self_test=True,
            debug=speech_mod.debug_enabled(),
        )
        self.speaker.start()
        self._startup_voice_checks()
        if fallback_note:
            # The language note is spoken directly by the worker; wait a
            # moment for the engine to report, then log it like any other
            # spoken line so the transcript stays complete.
            ready = getattr(self.speaker, "_ready", None)
            if ready is not None:
                ready.wait(2.0)
            actual_note = getattr(self.speaker, "voice_fallback_note", None)
            error = getattr(self.speaker, "init_error", None)
            if actual_note:
                print(f"[SPEAK] {actual_note}", flush=True)
                _LOG.info("SPEAK %s", actual_note)
            elif error:
                warn(f"Speech unavailable: {error}")

        # Proximity beeper. Headless/automated runs stay silent on purpose,
        # so no unattended test ever fills the room with beeps.
        self.beep = make_beep_controller(
            self.cfg.beep, headless=bool(self.args.headless_test)
        )
        self.beep.start()
        if self.beep.available:
            info("Proximity beep enabled (B key toggles; gets faster near objects).")

        # Low-light watcher: as soon as the room gets too dark the detector
        # receives CLAHE-enhanced frames and the user hears the warning.
        self._light = LightWatcher(self.cfg.light)
        if self.cfg.light.enabled:
            info(
                "Low-light mode on (auto CLAHE under mean luma "
                f"{self.cfg.light.dark_mean_threshold:.0f})."
            )

    # -- detection worker --------------------------------------------------
    def _startup_voice_checks(self) -> None:
        """OS volume state + TTS self-test result, both logged (and on screen).

        This is the answer to "the boxes move but I hear nothing": the app now
        verifies, before the first frame, that a backend really produced audio.
        """
        volume = speech_mod.read_system_volume()
        if volume is None:
            info("System volume: not readable (audio stack not queryable).")
        else:
            level, muted = volume
            if muted or level <= 0.0:
                self.volume_warning = "System volume is muted"
                warn(
                    "System volume is muted (or 0) - speech will be inaudible. "
                    "Unmute the Windows speaker / check the output device."
                )
            else:
                info(f"System volume {level * 100:.0f}%, not muted.")

        if not self.cfg.speech.enabled:
            info("Speech disabled (--no-speech): nothing will be spoken.")
            return
        result = getattr(self.speaker, "self_test_result", None)
        if not result:
            warn("Voice self-test did not run.")
            return
        if result.get("ok"):
            peak = result.get("peak")
            peak_txt = "n/a" if peak is None else f"{peak:.2f}"
            info(
                f"Voice self-test OK on backend {result['backend']} "
                f"(peak {peak_txt}): heard {result['text']!r}."
            )
        else:
            error(f"Voice self-test FAILED: {result.get('error')}")

    # -- detection worker --------------------------------------------------
    def _submit(self, frame) -> None:
        """Hand a frame to the worker, dropping it if the worker is busy."""
        with self._pending_lock:
            if self._pending_frame is not None:
                self.dropped += 1
                return  # never queue: keep only the newest frame
            self._pending_frame = frame
            self._frame_ready.set()

    def _worker_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._frame_ready.wait(timeout=0.25)
            with self._pending_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._frame_ready.clear()
            if frame is None:
                continue
            # Frame skipping (feature: performance): the detector only sees
            # every Nth captured frame. The overlay keeps the last known
            # positions in between, so the UI loop runs at full speed.
            self._frames_seen += 1
            if _skip_detection(self._frames_seen, self.cfg.vision.frame_skip):
                continue
            try:
                # Low-light: brighten the frame handed to YOLO when the room
                # is genuinely dark (the preview keeps the true camera image).
                enhance_now = False
                warn_now = False
                if self._light is not None:
                    enhance_now, warn_now = self._light.update(mean_luma(frame))
                detect_frame = enhance(frame, self.cfg.light) if enhance_now else frame
                detections = self.detector.detect(detect_frame)
                tracks = self.tracker.update(detections)
                # Confirmation gate: only objects that persisted long enough.
                confirmed = confirmed_tracks(
                    tracks, self.cfg.tracker.min_confirm_frames
                )
                placements = [
                    self.positioner.place(t.label, t.confidence, t.box)
                    for t in confirmed
                ]
                message = self.announcer.update(placements)
            except Exception:  # keep the worker alive through a bad frame
                _LOG.exception("detection failed on a frame")
                continue

            # Pace the beep from the nearest object in view (None = silence).
            if self.beep is not None:
                self.beep.set_rank(
                    min((p.distance_rank for p in placements), default=None)
                )

            with self._results_lock:
                self._positions = placements
                if message:
                    self._last_message = message
                self._dark_mode = enhance_now
            self.processed += 1

            if message:
                self._announce(
                    message, urgent=self.announcer.last_level == LEVEL_CRITICAL
                )
            if warn_now:
                # Rate-limited by LightWatcher; respects the M key like any
                # other spoken line.
                self._announce(self.cfg.light.warn_message)

    def _announce(
        self, message: str, urgent: bool = False, kind: str | None = None
    ) -> None:
        """Log and speak one announcement (or just log it while muted).

        Every call is also appended to the daily session CSV log.
        """
        if self.muted:
            print(f"[MUTED] {message}", flush=True)
            _LOG.info("MUTED %s", message)
            self.sessionlog.write(kind or "muted", message)
            return
        print(f"[SPEAK] {message}", flush=True)
        _LOG.info("SPEAK %s", message)
        self.sessionlog.write(kind or ("urgent" if urgent else "speak"), message)
        if urgent:
            self.speaker.speak(message, urgent=True)
        else:
            self.speaker.speak(message)

    def _snapshot(self) -> tuple[list[Position], str]:
        with self._results_lock:
            return list(self._positions), self._last_message

    def _wait_for_first_result(self, timeout: float = 8.0) -> None:
        """Block until the worker has produced its first frame of results."""
        deadline = time.time() + timeout
        while self.processed < 1 and time.time() < deadline:
            time.sleep(0.05)

    # -- drawing -----------------------------------------------------------
    @staticmethod
    def _dashed_vline(frame, x: int, y0: int, y1: int, color, dash: int = 8) -> None:
        y = y0
        while y < y1:
            cv2.line(frame, (x, y), (x, min(y + dash, y1)), color, 1, cv2.LINE_AA)
            y += dash * 2

    @staticmethod
    def _outlined_text(frame, text, org, font, scale, color, thick) -> None:
        """Text with a dark halo so it stays readable over any background."""
        cv2.putText(frame, text, org, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _label(frame, text: str, x: int, y: int, color) -> None:
        """Draw a large, high-contrast label anchored at the box's top-left."""
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (tw, th), _base = cv2.getTextSize(text, font, scale, thick)
        top = y - th - 14
        if top < 0:  # no room above the box -> label inside it
            top = y + 4
        if x + tw + 8 > frame.shape[1]:  # keep the label on screen
            x = max(0, frame.shape[1] - tw - 8)
        cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 8), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 4, top + th + 5),
            font,
            scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )

    @staticmethod
    def _panel(frame, x0: int, y0: int, x1: int, y1: int, color, alpha: float) -> None:
        """Blend a semi-transparent colour panel over ``frame`` in place.

        ``alpha`` 0 leaves the pixels untouched (invisible panel), 1 paints
        them solid (the pre-feature behaviour); in between gives the cleaner
        see-through status panels of the improved overlay.
        """
        h, w = frame.shape[:2]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            return
        region = frame[y0:y1, x0:x1]
        overlay = np.empty_like(region)
        overlay[:] = color
        alpha = min(max(float(alpha), 0.0), 1.0)
        blended = cv2.addWeighted(overlay, alpha, region, 1.0 - alpha, 0.0)
        frame[y0:y1, x0:x1] = blended

    def _voice_status_text(self) -> str:
        """``VOICE: ON (pyttsx3)`` / ``VOICE: MUTED`` / ``VOICE: FAILED``.

        Drawn in the status bar so a sighted helper can see instantly whether
        the silence is a mute, a broken backend, or speech simply being off.
        """
        if self.muted:
            return "VOICE: MUTED"
        speaker = self.speaker
        if speaker is None:
            return "VOICE: OFF"
        status = str(getattr(speaker, "status", "OFF"))
        backend = str(getattr(speaker, "backend_name", "") or "")
        if status == "ON" and backend:
            return f"VOICE: ON ({backend})"
        return f"VOICE: {status}"

    def _draw(self, frame, positions, message: str):
        h, w = frame.shape[:2]
        panel_alpha = self.cfg.ui.panel_alpha

        # Left / ahead / right zone guides.
        if self.cfg.ui.show_zone_guides:
            for frac in (1 / 3, 2 / 3):
                self._dashed_vline(frame, int(w * frac), 0, h, ZONE_COLOR)
        caption_h = 46
        for name, cx in (("LEFT", w // 6), ("AHEAD", w // 2), ("RIGHT", 5 * w // 6)):
            self._outlined_text(
                frame,
                name,
                (cx - 32, h - caption_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                ZONE_TEXT,
                1,
            )

        # Detection boxes, coloured by urgency.
        for p in positions:
            color = URGENCY_COLOR.get(p.distance_band, (0, 255, 0))
            x1, y1 = int(p.box.x1), int(p.box.y1)
            x2, y2 = int(p.box.x2), int(p.box.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{p.label} {p.confidence * 100:.0f}% {p.distance_band}"
            self._label(frame, text, x1, y1, color)

        # Top status bar: semi-transparent panel over the live image.
        bar_h = 44
        self._panel(frame, 0, 0, w, bar_h, BAR_BG, panel_alpha)
        state = "PAUSED" if self.paused else "RUNNING"
        state_color = (0, 0, 255) if self.paused else (0, 255, 0)
        cv2.putText(
            frame,
            state,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            state_color,
            2,
            cv2.LINE_AA,
        )
        info_txt = (
            f"FPS {self.fps:.0f}  CONF {self.confidence:.2f}  RATE {self.rate_wpm}"
        )
        # Voice indicator: ON / MUTED / FAILED with the backend that is live.
        info_txt += f"  {self._voice_status_text()}"
        if self.beep is not None and self.beep.available and self.cfg.beep.enabled:
            info_txt += "  BEEP"
        if self._dark_mode:
            info_txt += "  DARK"
        if self.cfg.vision.inference_size != DEFAULT_INFERENCE_SIZE:
            info_txt += f"  SIZE {self.cfg.vision.inference_size}"
        if self.cfg.vision.frame_skip > 1:
            info_txt += f"  SKIP {self.cfg.vision.frame_skip}"
        cv2.putText(
            frame,
            info_txt,
            (150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        # Colour legend (top right): what the box colours mean. This is the
        # panel a sighted helper reads first when checking the app at a glance.
        if self.cfg.ui.show_legend:
            lw, lh = 152, 86
            x0, y0 = w - lw - 6, bar_h + 6
            self._panel(frame, x0, y0, x0 + lw, y0 + lh, BAR_BG, panel_alpha)
            for i, (band, color) in enumerate(URGENCY_COLOR.items()):
                y = y0 + 22 + i * 20
                cv2.rectangle(frame, (x0 + 8, y - 10), (x0 + 26, y), color, -1)
                cv2.putText(
                    frame,
                    band.upper(),
                    (x0 + 32, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

        # On-screen keyboard help.
        cv2.putText(
            frame,
            "SPACE pause   +/- conf   [ ] rate   T test   M mute   B beep   D describe   S save   H help   Q quit",
            (10, bar_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Audio-level warning: a muted/zero OS volume looks exactly like a bug.
        if self.volume_warning:
            cv2.putText(
                frame,
                f"! {self.volume_warning} - check the Windows volume",
                (10, bar_h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Bottom caption: the last spoken sentence, for sighted helpers.
        self._panel(frame, 0, h - caption_h, w, h, BAR_BG, panel_alpha)
        cv2.putText(
            frame,
            "SAID:",
            (10, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 200, 255),
            1,
            cv2.LINE_AA,
        )
        if message:
            cv2.putText(
                frame,
                message[:80],
                (72, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Full key reference, drawn last so it sits on top of everything.
        if self.show_help:
            self._draw_help(frame)
        return frame

    def _draw_help(self, frame) -> None:
        """Large semi-transparent key reference (toggled with the H key)."""
        h, w = frame.shape[:2]
        lines = [
            ("BLINDASSIST - KEY HELP", (0, 255, 255)),
            ("SPACE    pause / resume", (255, 255, 255)),
            ("+ / -    raise / lower confidence", (255, 255, 255)),
            ("[  ]     slower / faster speech", (255, 255, 255)),
            ("M        mute / unmute speech", (255, 255, 255)),
            ("T        test the voice now", (255, 255, 255)),
            ("B        proximity beep on / off", (255, 255, 255)),
            ("D        describe the whole scene", (255, 255, 255)),
            ("S        save a screenshot", (255, 255, 255)),
            ("H        show / hide this help", (255, 255, 255)),
            ("Q / ESC  quit", (255, 255, 255)),
        ]
        line_h = 26
        hw = min(w - 40, 760)
        hh = line_h * len(lines) + 24
        x0 = max(20, (w - hw) // 2)
        y0 = max(20, (h - hh) // 2)
        self._panel(frame, x0, y0, x0 + hw, y0 + hh, (20, 20, 20), 0.80)
        cv2.rectangle(frame, (x0, y0), (x0 + hw, y0 + hh), (0, 255, 255), 2)
        for i, (text, color) in enumerate(lines):
            scale = 0.7 if i == 0 else 0.6
            cv2.putText(
                frame,
                text,
                (x0 + 16, y0 + 24 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                2 if i == 0 else 1,
                cv2.LINE_AA,
            )

    # -- actions -----------------------------------------------------------
    def _save_screenshot(self, frame, tag: str | None = None) -> Path:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stem = _safe_tag(tag) if tag else time.strftime("%Y%m%d_%H%M%S")
        out = SCREENSHOT_DIR / f"{stem}.png"
        if tag is None:
            # Interactive S presses: the timestamp only has 1-second resolution,
            # so make the name unique instead of overwriting a just-saved frame.
            n = 2
            while out.exists():
                out = SCREENSHOT_DIR / f"{stem}_{n}.png"
                n += 1
        cv2.imwrite(str(out), frame)
        info(f"Screenshot saved -> {out}")
        return out

    def _handle_key(self, key: int) -> bool:
        """Apply a key press. Returns True when the app should quit.

        Safe to call before the pipeline exists: the confidence/rate keys simply
        record the new value and it is pushed to the model once one is loaded.
        """
        if key in (ord("q"), ord("Q"), 27):
            return True
        if key == ord(" "):
            self.paused = not self.paused
            info(f"{'Paused' if self.paused else 'Resumed'}.")
        elif key in (ord("+"), ord("=")):
            self.confidence = round(min(self.confidence + CONF_STEP, CONF_MAX), 2)
            if self.detector is not None:
                self.detector.cfg.confidence = self.confidence
            info(f"Confidence threshold -> {self.confidence:.2f}")
            self._persist_settings()
        elif key in (ord("-"), ord("_")):
            self.confidence = round(max(self.confidence - CONF_STEP, CONF_MIN), 2)
            if self.detector is not None:
                self.detector.cfg.confidence = self.confidence
            info(f"Confidence threshold -> {self.confidence:.2f}")
            self._persist_settings()
        elif key == ord("]"):
            self.rate_wpm = min(self.rate_wpm + RATE_STEP, RATE_MAX)
            if self.speaker is not None:
                self.speaker.set_rate(self.rate_wpm)
            info(f"Speech rate -> {self.rate_wpm} wpm")
            self._persist_settings()
        elif key == ord("["):
            self.rate_wpm = max(self.rate_wpm - RATE_STEP, RATE_MIN)
            if self.speaker is not None:
                self.speaker.set_rate(self.rate_wpm)
            info(f"Speech rate -> {self.rate_wpm} wpm")
            self._persist_settings()
        elif key in (ord("m"), ord("M")):
            self.muted = not self.muted
            # The speaker drops text too, so a muted session never reaches the
            # audio device even if a caller forgets to check self.muted.
            setter = getattr(self.speaker, "set_muted", None)
            if setter is not None:
                setter(self.muted)
            info(f"{'Muted' if self.muted else 'Unmuted'}.")
        elif key in (ord("t"), ord("T")):
            self._test_voice_now()
        elif key in (ord("b"), ord("B")):
            if self.beep is None or not self.beep.available:
                # Distinguish the two reasons: headless runs silence the
                # beeper on purpose; otherwise the OS has no winsound.
                if self.args.headless_test:
                    warn("Proximity beep is off in headless mode.")
                else:
                    warn("Proximity beep unavailable on this machine.")
            else:
                new = not self.cfg.beep.enabled
                self.cfg.beep.enabled = new
                self.beep.set_enabled(new)
                info(f"Proximity beep {'on' if new else 'off'}.")
                self._persist_settings()
        elif key in (ord("d"), ord("D")):
            # On-demand scene summary. Not rate-limited: the user asked for it.
            if self.cfg.summary.enabled:
                positions, _msg = self._snapshot()
                self._announce(
                    summarize_scene(positions, self.cfg.summary), kind="describe"
                )
            else:
                warn("Scene summary is disabled in the configuration.")
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
            info(f"Help {'shown' if self.show_help else 'hidden'}.")
        elif key in (ord("s"), ord("S")):
            self._save_current_screenshot()
        return False

    def _test_voice_now(self) -> None:
        """T key: prove the audio path in one second, right where you stand."""
        speaker = self.speaker
        if speaker is None:
            warn("Voice test: the speech pipeline is not built yet.")
            return
        if self.muted:
            warn("Voice test: muted (press M to unmute).")
            return
        if not getattr(speaker, "available", False):
            error = getattr(speaker, "init_error", None) or "no backend"
            warn(f"Voice test: speech is FAILED ({error}).")
            return
        test = getattr(speaker, "test_voice", None)
        if test is None:  # pragma: no cover - NullSpeaker-style fallback
            speaker.speak("Voice test.", urgent=True)
            return
        test()
        info(
            f"Voice test sent via {getattr(speaker, 'backend_name', 'backend')} "
            "(T again to repeat; see logs/speech.log for the timestamps)."
        )

    # -- settings persistence ----------------------------------------------
    def _persist_settings(self) -> None:
        """Write current settings.json (after keyboard-initiated changes)."""
        self._settings_dirty = True
        if not self.cfg.settings.enabled:
            return
        # The key handlers update UI state first (self.confidence, etc.) and
        # push it to the detector/speaker; sync cfg so the persisted file
        # reflects the values this session really runs with.
        self.cfg.vision.confidence = float(self.confidence)
        self.cfg.speech.rate_wpm = int(self.rate_wpm)
        if self.cfg.settings.autosave:
            self._save_settings()

    def _save_settings(self) -> None:
        try:
            settings_store.save(settings_store.snapshot(self.cfg), self.settings_path)
        except OSError as exc:
            warn(f"Could not save settings.json: {exc}")

    def _save_current_screenshot(self) -> None:
        """Save the last drawn frame with the overlay (the S key)."""
        frame = self._last_frame
        if frame is None:
            return
        positions, message = self._snapshot()
        self._save_screenshot(self._draw(frame.copy(), positions, message))

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        _setup_logging()
        info("BlindAssist starting.")

        if not self._open_source():
            return 2

        first = self.source.read()
        if first is None:
            error("Could not read a frame from the source.")
            self.source.release()
            return 2
        height, width = first.shape[:2]

        try:
            self._build_pipeline(width, height)
        except Exception as exc:  # model download / load failure
            error(f"Could not load the detector: {exc}")
            if self.speaker is not None:
                self.speaker.shutdown()
            self.source.release()
            return 3

        headless = bool(self.args.headless_test)
        max_frames = int(self.args.max_frames or 0)
        if headless and max_frames <= 0:
            max_frames = 150  # never hang an unattended run

        self._stop_evt.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name="blindassist-detect", daemon=True
        )
        self._worker.start()

        # A broken TTS engine must not kill the app: the video feed is the
        # primary output for a blind user, speech is secondary.
        try:
            self.speaker.speak(WELCOME)
        except Exception as exc:
            warn(f"Welcome announcement failed: {exc}")
        print(f"[SPEAK] {WELCOME}", flush=True)
        _LOG.info("SPEAK %s", WELCOME)
        self.sessionlog.write("welcome", WELCOME)
        info(
            "Controls: SPACE pause | +/- conf | [ ] rate | M mute | B beep | "
            "D describe | H help | S save | Q quit"
        )

        frame_idx = 0
        last_t = time.time()
        saved = False
        exit_reason = "quit"

        try:
            while True:
                frame = first if frame_idx == 0 else self.source.read()
                if frame is None:
                    if not self.source.is_live:
                        exit_reason = "end of source"
                        info("Reached the end of the input.")
                        break
                    self._consecutive_failures += 1
                    if self._consecutive_failures == 1:
                        warn("Frame grab failed; retrying...")
                    if (
                        self._consecutive_failures == REOPEN_AFTER_FAILURES
                        and self.source.reopen()
                    ):
                        warn("Camera reconnected.")
                        self._consecutive_failures = 0
                        continue
                    if self._consecutive_failures > GIVE_UP_AFTER_FAILURES:
                        error("Camera disconnected; shutting down cleanly.")
                        exit_reason = "camera lost"
                        break
                    time.sleep(0.05)
                    continue
                self._consecutive_failures = 0
                self._last_frame = frame

                if not self.paused:
                    self._submit(frame)

                now = time.time()
                dt = now - last_t
                # dt > 1 ms guards the very first frame (a near-zero dt would
                # seed the EMA with a bogus multi-thousand fps reading).
                if dt > MIN_FRAME_DT:
                    self.fps = (
                        0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
                    )
                last_t = now

                positions, message = self._snapshot()
                annotated = self._draw(frame, positions, message)

                if headless:
                    press = self._demo_presses.get(frame_idx)
                    if press is not None and self._handle_key(press):
                        # Same as a real key press: Q/ESC ends the run.
                        exit_reason = "quit"
                        break
                    if positions and not saved:
                        self._save_screenshot(annotated, self.args.tag)
                        saved = True
                        for p in positions:
                            print(f"[DETECT] {p.describe()}", flush=True)
                    # Pace the loop like a real camera; otherwise a still-image
                    # source would burn through every frame before YOLO finishes.
                    time.sleep(0.03)
                else:
                    cv2.imshow(WINDOW_NAME, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key != KEY_NONE and self._handle_key(key):
                        exit_reason = "quit"
                        break

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    exit_reason = f"max frames ({max_frames})"
                    info(f"Reached --max-frames={max_frames}; exiting.")
                    break
        except KeyboardInterrupt:
            exit_reason = "interrupted"
            print("\n[INFO] Interrupted by user.")
        finally:
            if headless and not saved and self._last_frame is not None:
                # The worker can still be mid-inference when the loop ends
                # (first YOLO call is slow to warm up). Wait for it so an
                # automated run still produces a detection + screenshot.
                self._wait_for_first_result(8.0)
                positions, message = self._snapshot()
                if positions:
                    self._save_screenshot(
                        self._draw(self._last_frame.copy(), positions, message),
                        self.args.tag,
                    )
                    for p in positions:
                        print(f"[DETECT] {p.describe()}", flush=True)
            self._shutdown()

        info(
            f"Stopped ({exit_reason}). Shown {frame_idx} frames, "
            f"detected on {self.processed}, dropped {self.dropped}, fps ~{self.fps:.1f}."
        )
        info(f"Announcements: {len(self.announcer.history)}")
        if self.sessionlog.rows:
            info(f"Session log: {self.sessionlog.rows} rows -> {self.sessionlog.path}")
        info(f"Log written to {LOG_FILE}")
        return 0

    def _shutdown(self) -> None:
        self._stop_evt.set()
        self._frame_ready.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            if self._worker.is_alive():
                warn("Detection worker did not stop in 3s; leaving daemon thread.")
            else:
                self._worker = None
        if self.source is not None:
            self.source.release()
        if self.speaker is not None:
            self.speaker.shutdown()
        if self.beep is not None:
            self.beep.stop()
        self.sessionlog.close()
        if self.cfg.settings.enabled and not self.cfg.settings.autosave:
            self._save_settings()  # clean exit persists without autosave too
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        info("Released camera, stopped speech, closed windows.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Fold CLI flags into the config, clamping anything out of range.

    A silent ``--conf 5.0`` would make the detector report nothing at all, which
    looks like a broken camera to the user. Clamping with a warning is kinder.
    """
    if getattr(args, "camera", None) is not None:
        cfg.vision.camera_index = max(0, int(args.camera))
    if getattr(args, "model", None) is not None:
        cfg.model_path = Path(args.model)
    if getattr(args, "conf", None) is not None:
        raw = float(args.conf)
        cfg.vision.confidence = min(max(raw, CONF_MIN), CONF_MAX)
        if cfg.vision.confidence != raw:
            warn(f"--conf {raw} out of range; clamped to {cfg.vision.confidence:.2f}")
    if getattr(args, "rate", None) is not None:
        raw = int(args.rate)
        cfg.speech.rate_wpm = min(max(raw, RATE_MIN), RATE_MAX)
        if cfg.speech.rate_wpm != raw:
            warn(f"--rate {raw} out of range; clamped to {cfg.speech.rate_wpm} wpm")
    if getattr(args, "language", None) is not None:
        cfg.language.language = str(args.language).strip().lower()
    if getattr(args, "size", None) is not None:
        raw = int(args.size)
        # Snap to the nearest supported size instead of failing: a slightly
        # odd value (e.g. --size 450) should still run, just with a warning.
        snapped = min(INPUT_SIZES, key=lambda s: abs(s - raw))
        cfg.vision.inference_size = snapped
        if snapped != raw:
            warn(f"--size {raw} not supported; using {snapped} px")
    if getattr(args, "skip", None) is not None:
        raw = int(args.skip)
        clamped = min(max(raw, 1), MAX_FRAME_SKIP)
        cfg.vision.frame_skip = clamped
        if clamped != raw:
            warn(f"--skip {raw} out of range; clamped to {clamped}")
    if getattr(args, "no_speech", False):
        cfg.speech.enabled = False
    if getattr(args, "tts_backend", None) is not None:
        requested = str(args.tts_backend).strip().lower()
        if requested not in TTS_BACKENDS:
            warn(
                f"--tts-backend {requested!r} unknown; "
                f"using one of {', '.join(TTS_BACKENDS)}"
            )
        else:
            cfg.speech.backend = requested
    if getattr(args, "debug_speech", False):
        speech_mod.set_debug(True)
        info("Speech stage tracing on (logs/speech.log + console).")
    return cfg


def _skip_detection(frames_seen: int, frame_skip: int) -> bool:
    """True when the detector should ignore the Nth captured frame.

    ``frame_skip`` 1 (the default) analyzes every frame; 3 analyzes every
    3rd frame (1st, 4th, 7th, ...) so the UI loop is not blocked by YOLO.
    """
    if frame_skip <= 1:
        return False
    return (frames_seen - 1) % frame_skip != 0


def _safe_tag(tag: str) -> str:
    """Strip anything that could turn the screenshot stem into a path.

    ``--tag ../../evil`` must not write outside Evidence/screenshots/.
    """
    cleaned = "".join(ch for ch in str(tag) if ch.isalnum() or ch in "-_. ")
    return cleaned.strip(" ._") or "frame"


_NAMED_KEYS: dict[str, int] = {"SPACE": 32}


def _parse_demo_keys(text: str) -> dict[int, int]:
    """``"12:D,30:SPACE"`` -> ``{12: ord('D'), 30: 32}`` (junk items skipped).

    Used only by headless runs to exercise key behaviour without a GUI.
    """
    presses: dict[int, int] = {}
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        frame_txt, sep, key_txt = item.partition(":")
        if not sep:
            warn(f"ignoring malformed demo key item {item!r} (want N:KEY)")
            continue
        try:
            frame = int(frame_txt)
        except ValueError:
            warn(f"ignoring demo key with bad frame number {item!r}")
            continue
        key_txt = key_txt.strip().upper()
        key = _NAMED_KEYS.get(key_txt)
        if (
            key is None
            and len(key_txt) == 1
            and (key_txt.isalnum() or key_txt in "+=-_[]")
        ):
            key = ord(key_txt)
        if key is None:
            warn(f"ignoring demo key with unknown key name {item!r}")
            continue
        presses[frame] = key
    return presses


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BlindAssist - real-time object detection assistant for blind users"
    )
    p.add_argument(
        "--camera", type=int, default=None, help="camera index (default: from config)"
    )
    p.add_argument("--conf", type=float, default=None, help="confidence threshold 0..1")
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="run on a video file instead of the webcam (an image works too)",
    )
    p.add_argument("--no-speech", action="store_true", help="disable TTS (print only)")
    p.add_argument(
        "--tts-backend",
        dest="tts_backend",
        default=None,
        choices=list(TTS_BACKENDS),
        help="speech backend: auto (default) tries pyttsx3, then PowerShell, "
        "then a beep, verifying each one with a startup self-test",
    )
    p.add_argument(
        "--debug-speech",
        dest="debug_speech",
        action="store_true",
        help="trace every stage of the speech pipeline to logs/speech.log "
        "and the console (queue -> backend -> start -> end)",
    )
    p.add_argument(
        "--headless-test",
        "--headless",
        dest="headless_test",
        action="store_true",
        help="no GUI window; for automated runs",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after N frames (0 = unlimited; headless defaults to 150)",
    )
    p.add_argument(
        "--rate", type=int, default=None, help="speech rate in words per minute"
    )
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="voice language, e.g. en (default) or ur - see LanguageConfig",
    )
    p.add_argument(
        "--size",
        type=int,
        default=None,
        metavar="PX",
        help=f"detector input size in px ({', '.join(map(str, INPUT_SIZES))}; "
        "smaller = faster, larger = more accurate; default from config)",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=None,
        metavar="N",
        help="analyze every Nth frame (1 = every frame, default; "
        f"larger = faster UI, {MAX_FRAME_SKIP} max)",
    )
    p.add_argument(
        "--all-classes",
        action="store_true",
        help="detect all 80 COCO classes (default: navigation whitelist)",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="frame",
        help="filename stem for the headless screenshot",
    )
    p.add_argument("--model", type=str, default=None, help="path to YOLO weights")
    p.add_argument(
        "--demo-keys",
        type=str,
        default="",
        help='headless testing aid: simulate key presses, e.g. "12:D,30:SPACE"',
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="(deprecated) the preview window is on by default",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return BlindAssistApp(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
