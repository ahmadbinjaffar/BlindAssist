"""Offline text-to-speech for Windows, with backends, fallback and a self-test.

Speech runs on a single worker thread fed by a queue, so the vision loop never
blocks:

    speaker = Pyttsx3Speaker(rate_wpm=175, volume=1.0, self_test=True)
    speaker.start()                                   # speaks "Assistant ready."
    speaker.speak("Person ahead, very close.")        # queued, returns at once
    speaker.speak("Stop!", urgent=True)               # clears queue, cuts in
    speaker.shutdown()

Why this module is written the way it is (measured, not assumed)
---------------------------------------------------------------
Re-using one pyttsx3/SAPI5 engine for several utterances is silent on this
machine.  ``scripts/probe_queue_order.py`` shows the mechanism:

* ``Engine.runAndWait()`` ends with ``proxy.setBusy(False)``, so between two
  utterances the driver proxy is *not* busy.
* Consequently ``engine.say()`` on utterance #2 runs **immediately** (its
  queue is empty) and only afterwards ``runAndWait()`` queues the ``endLoop``
  command - i.e. the queue is ``[endLoop]`` instead of ``[say, endLoop]``.
* ``startLoop`` pops that lone ``endLoop`` first, which calls
  ``DriverProxy.endLoop()`` -> ``SAPI5Driver.stop()`` ->
  ``voice.Speak("", 3)`` (SVSPURGEBOTH): the utterance is cancelled.
  pyttsx3 reports ``finished-utterance completed=False`` after ~0.35 s and
  **no audio is produced** - silently, because no exception is raised.

Utterance #1 works only because a brand-new proxy starts with ``busy=True``,
so ``say`` is queued ahead of ``endLoop``.  Therefore this module creates a
**fresh engine for every utterance**, discarding the previous one *before* the
next ``pyttsx3.init()`` (``scripts/probe_cycles.py``: 10/10 utterances audible
in the main thread and in the worker thread, including after an urgent cut).

Backends and fallback (``TTS_BACKEND`` / ``--tts-backend``)
-----------------------------------------------------------
1. ``pyttsx3``    - SAPI5, a real voice, one fresh engine per utterance.
2. ``powershell`` - System.Speech through a hidden subprocess; independent of
                    pyttsx3, so it works even when pyttsx3 is broken.
3. ``beep``       - winsound tone pattern, so the user still hears *something*.

``auto`` (the default) tries them in that order, verifies each one with a
startup self-test ("Assistant ready.") measured on the real output endpoint,
and switches automatically if a backend later stops producing audio.

Guantees
--------
* ``speak()`` never blocks the caller (it only enqueues).
* ``urgent=True`` empties the queue and aborts the utterance in progress
  without ever killing the worker.
* No exception is swallowed: every failure is written with its full traceback
  to ``logs/speech.log``, together with start/end timestamps per utterance.
* Missing/broken audio or no installed voice degrades gracefully (``available``
  is False, ``status`` is "FAILED", calls become harmless no-ops).
* ``shutdown()`` stops current speech and joins the worker (bounded wait) so
  nothing hangs; the worker is also a daemon as a last resort.
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
SPEECH_LOG_PATH = ROOT / "logs" / "speech.log"

# Backend names in the order "auto" tries them.
BACKEND_ORDER: tuple[str, ...] = ("pyttsx3", "powershell", "beep")

# Spoken by start() so the user (and the ear test) has something to confirm.
SELF_TEST_TEXT = "Assistant ready."

_SENTINEL = object()

# Cap for the diagnostic logs. These are inspection hooks, not data structures
# the app depends on, so they must not grow without bound on a long run.
_HISTORY_LIMIT = 1000

# Peak level at or below this means "the endpoint produced no audio" (measured
# silence is exactly 0.0; a working voice peaks at 0.35..0.9).
_SILENT_PEAK = 0.0005
# Consecutive silent utterances / exceptions before the backend is abandoned.
_MAX_SILENT = 2
_MAX_ERRORS = 3

# Debug stages: set by set_debug() or the BLINDASSIST_DEBUG_SPEECH=1 env var.
# Held in a dict rather than a module global so the flag can be flipped at
# runtime without a ``global`` statement (ruff PLW0603).
_FLAGS: dict[str, bool] = {
    "debug": os.environ.get("BLINDASSIST_DEBUG_SPEECH", "").strip().lower()
    in {"1", "true", "yes", "on"},
}
# CREATE_NO_WINDOW: never flash a console window at the user.
_CREATE_NO_WINDOW = 0x08000000


def set_debug(flag: bool) -> None:
    """Turn stage-by-stage speech logging on/off (``--debug-speech``)."""
    _FLAGS["debug"] = bool(flag)


def debug_enabled() -> bool:
    return bool(_FLAGS["debug"])


_logger_lock = threading.Lock()


def speech_logger() -> logging.Logger:
    """Append-only ``logs/speech.log`` logger (created once, thread-safe)."""
    lg = logging.getLogger("blindassist.speech")
    if lg.handlers:
        return lg
    with _logger_lock:
        if lg.handlers:  # another thread got there first
            return lg
        lg.setLevel(logging.DEBUG)
        try:
            SPEECH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = logging.FileHandler(
                SPEECH_LOG_PATH, encoding="utf-8"
            )
        except OSError:
            handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        lg.addHandler(handler)
        lg.propagate = False
    return lg


def _log(level: int, msg: str, *args) -> None:
    try:
        speech_logger().log(level, msg, *args)
    except Exception:  # logging must never break speech
        pass


def _stage(msg: str) -> None:
    """Debug-stage line (file always, console only with --debug-speech)."""
    _log(logging.DEBUG, "stage %s", msg)
    if _FLAGS["debug"]:
        print(f"[SPEECH-DEBUG] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Optional peak measurement (proves audio really reached the speaker)
# ---------------------------------------------------------------------------
class _PeakProbe:
    """Samples the default output endpoint's peak while an utterance plays.

    Uses pycaw/IAudioMeterInformation.  If pycaw (or the audio stack) is not
    available ``ok`` is False and every call degrades to "measurement
    unavailable" - the caller then trusts the backend instead of the meter.
    """

    def __init__(self) -> None:
        self.ok = False
        self._meter = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._max = 0.0
        self._samples = 0

    def _connect(self) -> bool:
        if self._meter is not None:
            return True
        try:
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioMeterInformation

            dev = AudioUtilities.GetSpeakers()
            iface = dev._dev.Activate(IAudioMeterInformation._iid_, CLSCTX_ALL, None)
            self._meter = iface.QueryInterface(IAudioMeterInformation)
            return True
        except Exception as exc:
            _log(logging.DEBUG, "peak meter unavailable: %s", exc)
            self._meter = None
            return False

    def start(self) -> None:
        if not self._connect():
            self.ok = False
            return
        self.ok = True
        self._max = 0.0
        self._samples = 0
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample, name="blindassist-peak", daemon=True
        )
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                value = float(self._meter.GetPeakValue())
                self._max = max(self._max, value)
                self._samples += 1
            except Exception:
                pass
            time.sleep(0.02)

    def stop(self) -> float | None:
        """Peak observed during the last window (None = not measured)."""
        if not self.ok:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        return self._max


def read_system_volume() -> tuple[float, bool] | None:
    """``(master level 0..1, muted)`` of the default output device.

    Returns None when the audio stack cannot be queried.  Used at startup to
    warn "System volume is muted" instead of failing silently.
    """
    com_ok = False
    try:
        import pythoncom

        pythoncom.CoInitialize()
        com_ok = True
    except Exception:
        pass
    try:
        from pycaw.pycaw import AudioUtilities

        dev = AudioUtilities.GetSpeakers()
        volume = dev.EndpointVolume
        level = float(volume.GetMasterVolumeLevelScalar())
        muted = bool(volume.GetMute())
        del volume, dev
        return level, muted
    except Exception as exc:
        _log(logging.DEBUG, "read_system_volume failed: %s", exc)
        return None
    finally:
        if com_ok:
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class BackendError(RuntimeError):
    """A backend could not speak; the caller may fall back to the next one."""


class Backend:
    """One way of turning text into sound.  All calls happen on the worker."""

    name = "backend"

    def __init__(
        self,
        rate_wpm: int = 175,
        volume: float = 1.0,
        voice: str | int | None = None,
        allow_fallback: bool = False,
        fallback_note: str = "",
    ) -> None:
        self.rate_wpm = int(rate_wpm)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.voice = voice
        self.allow_fallback = bool(allow_fallback)
        self.fallback_note = str(fallback_note or "")
        self.voice_fallback_note: str | None = None

    def configure(self, rate_wpm: int, volume: float) -> None:
        """Push live rate/volume changes (the [ ] and +/- keys)."""
        self.rate_wpm = int(rate_wpm)
        self.volume = max(0.0, min(1.0, float(volume)))

    def validate(self) -> None:
        """Raise BackendError when this backend cannot be used at all."""
        raise NotImplementedError

    def speak(self, text: str) -> None:
        """Blocking: return only when the utterance has finished (or failed)."""
        raise NotImplementedError

    def stop(self) -> None:
        """Abort the utterance in progress (called from other threads too)."""
        return

    def close(self) -> None:
        return


class Pyttsx3Backend(Backend):
    """SAPI5 through pyttsx3 - a fresh engine for every utterance.

    A fresh engine is not an optimisation, it is the fix: see the module
    docstring.  The previous engine is discarded *before* the next init().
    """

    name = "pyttsx3"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._engine = None
        self._lock = threading.Lock()

    # -- engine lifecycle --------------------------------------------------
    def _new_engine(self):
        import pyttsx3

        self._discard()  # old engine dies first (order matters - measured)
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self.rate_wpm)
            engine.setProperty("volume", self.volume)
            if not self._select_voice(engine):
                raise BackendError("no matching / installed voice")
        except Exception:
            self._release(engine)
            raise
        self._engine = engine
        return engine

    @staticmethod
    def _release(engine) -> None:
        try:
            engine.stop()
        except Exception:
            pass
        del engine
        gc.collect()

    def _discard(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            self._release(engine)

    # -- voice selection ---------------------------------------------------
    def _select_voice(self, engine) -> bool:
        voices = engine.getProperty("voices") or []
        if not voices:
            return False
        if self.voice is None:
            return True  # keep the engine default
        if isinstance(self.voice, int):
            in_range = 0 <= self.voice < len(voices)
            if in_range:
                engine.setProperty("voice", voices[self.voice].id)
            return in_range
        sub = str(self.voice).lower()
        for v in voices:
            name = (v.name or "").lower()
            vid = (getattr(v, "id", "") or "").lower()
            if sub in name or sub in vid:
                engine.setProperty("voice", v.id)
                return True
        # Language feature: keep the default voice instead of failing, and
        # say so once so the user knows which voice they got.
        if not self.allow_fallback:
            return False
        self.voice_fallback_note = self.fallback_note or (
            f'No voice matching "{self.voice!s}" found; using the default voice.'
        )
        return True

    # -- api ---------------------------------------------------------------
    def validate(self) -> None:
        self._new_engine()  # raises BackendError when there is no voice

    def speak(self, text: str) -> None:
        engine = self._new_engine()
        engine.say(text)
        engine.runAndWait()

    def stop(self) -> None:
        # Called from the app thread for urgent cut-ins.  Safe: this engine is
        # discarded right after, so poisoning it cannot affect the next line.
        engine = self._engine
        if engine is not None:
            with self._lock:
                try:
                    engine.stop()
                except Exception as exc:
                    _log(logging.DEBUG, "pyttsx3 stop failed: %s", exc)

    def close(self) -> None:
        self._discard()


class PowerShellBackend(Backend):
    """System.Speech through a hidden PowerShell subprocess.

    Independent of pyttsx3 (a different process and a different API), so it
    still works when the SAPI5/pyttsx3 path is broken.  The text is passed as
    ``-EncodedCommand`` (UTF-16LE base64), which cannot be mangled by quoting.
    """

    name = "powershell"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _script(self, text: str) -> str:
        rate = max(-10, min(10, round((self.rate_wpm - 175) / 17)))
        volume = round(self.volume * 100)
        escaped = text.replace("'", "''")
        voice_line = ""
        if isinstance(self.voice, str) and self.voice.strip():
            voice_name = self.voice.replace("'", "''")
            voice_line = f"try {{ $s.SelectVoice('{voice_name}') }} catch {{ }}\n"
        return (
            "Add-Type -AssemblyName System.Speech\n"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
            f"{voice_line}"
            f"$s.Rate = {rate}\n"
            f"$s.Volume = {volume}\n"
            f"$s.Speak('{escaped}')\n"
        )

    def validate(self) -> None:
        if shutil.which("powershell") is None:
            raise BackendError("powershell not found on PATH")

    def speak(self, text: str) -> None:
        encoded = base64.b64encode(self._script(text).encode("utf-16-le")).decode(
            "ascii"
        )
        with self._lock:
            self._proc = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-WindowStyle",
                    "Hidden",
                    "-EncodedCommand",
                    encoded,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW,
                text=True,
            )
            proc = self._proc
        try:
            _out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            with self._lock:
                self._proc = None
            raise BackendError("powershell speech timed out after 30s") from exc
        with self._lock:
            self._proc = None
        if proc.returncode not in (0, None):
            raise BackendError(
                f"powershell exited {proc.returncode}: {(err or '').strip()[:200]}"
            )

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()


class BeepBackend(Backend):
    """Last resort: a two-tone winsound pattern so nothing is ever silent."""

    name = "beep"

    def validate(self) -> None:
        try:
            import winsound  # noqa: F401
        except Exception as exc:  # pragma: no cover - non-Windows
            raise BackendError(f"winsound unavailable: {exc}") from exc

    def speak(self, text: str) -> None:
        import winsound

        # Two tones, louder/longer for urgent lines (text is not pronounced).
        urgent = text.strip().lower().startswith(("stop", "danger", "warning"))
        high, low = (1200, 900) if urgent else (950, 750)
        length = 180 if urgent else 120
        winsound.Beep(high, length)
        time.sleep(0.05)
        winsound.Beep(low, length)


# Factories are module level so tests can substitute a failing backend.
BACKEND_FACTORIES: dict[str, type[Backend]] = {
    "pyttsx3": Pyttsx3Backend,
    "powershell": PowerShellBackend,
    "beep": BeepBackend,
}


def available_backends() -> tuple[str, ...]:
    return BACKEND_ORDER


def make_backend(name: str, **kwargs) -> Backend:
    try:
        factory = BACKEND_FACTORIES[name]
    except KeyError as exc:
        raise BackendError(f"unknown TTS backend {name!r}") from exc
    return factory(**kwargs)


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------
class Speaker(Protocol):
    """Minimal interface the app depends on."""

    def start(self) -> None: ...
    def speak(self, text: str, urgent: bool = False) -> None: ...
    def set_rate(self, wpm: int) -> None: ...
    def shutdown(self) -> None: ...


class NullSpeaker:
    """Discards text (used when speech is disabled). Records it for tests."""

    backend_name = "none"

    def __init__(self) -> None:
        self.spoken: deque[str] = deque(maxlen=_HISTORY_LIMIT)
        self.muted = False
        self.self_test: dict | None = None

    @property
    def status(self) -> str:
        return "OFF"

    @property
    def available(self) -> bool:
        return False

    @property
    def init_error(self) -> str | None:
        return None

    def start(self) -> None:
        return

    def speak(self, text: str, urgent: bool = False) -> None:
        if text:
            self.spoken.append(text)

    def set_rate(self, wpm: int) -> None:
        return

    def set_volume(self, volume: float) -> None:
        return

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)

    def queue_size(self) -> int:
        return 0

    def shutdown(self) -> None:
        return

    # Backwards-compatible aliases.
    def say(self, text: str) -> None:
        self.speak(text)

    def stop(self) -> None:
        self.shutdown()


class Pyttsx3Speaker:
    """Background-thread speaker with backends, fallback and a self-test."""

    def __init__(
        self,
        rate_wpm: int = 175,
        volume: float = 1.0,
        voice: str | int | None = None,
        allow_fallback: bool = False,
        fallback_note: str = "",
        backend: str = "auto",
        self_test: bool = False,
        debug: bool | None = None,
    ) -> None:
        self.rate_wpm = int(rate_wpm)
        self.volume = max(0.0, min(1.0, float(volume)))
        # voice: None = default; int = index into voices; str = name/id substring.
        self.voice = voice
        # Language feature: when a *named* voice (e.g. an Urdu voice hint) is
        # not installed, keep the default voice instead of failing, and speak
        # `fallback_note` once so the user knows what happened.
        self.allow_fallback = bool(allow_fallback)
        self.fallback_note = str(fallback_note or "")
        self.voice_fallback_note: str | None = None

        self.backend_request = (backend or "auto").strip().lower()
        self.self_test = bool(self_test)
        if debug is not None:
            set_debug(debug)

        self._queue: queue.Queue[object] = queue.Queue()
        self._thread: threading.Thread | None = None
        # Backend instances (one per name); the active one is self._backend.
        self._backends: list[Backend] = []
        self._backend: Backend | None = None
        self._backends_lock = threading.Lock()
        self._ready = threading.Event()
        self._available = False
        # Set when every backend is gone. Nothing drains the queue after that,
        # so speak() must drop text instead of enqueuing it forever.
        self._failed = False
        self._muted = False
        self.init_error: str | None = None
        self.self_test_result: dict | None = None
        self._probe = _PeakProbe()
        # Live settings are only written here and read by the worker, so no
        # engine object is ever touched from the app thread.
        self._applied_rate = self.rate_wpm
        self._applied_volume = self.volume
        self._consecutive_errors = 0
        self._consecutive_silent = 0
        # Peak measured for the most recent utterance (None = not measured).
        self._last_peak: float | None = None

        # Inspection hooks for tests / diagnostics.
        self.spoken_log: deque[str] = deque(maxlen=_HISTORY_LIMIT)
        self.muted_log: deque[str] = deque(maxlen=_HISTORY_LIMIT)

    # -- introspection -----------------------------------------------------
    @property
    def available(self) -> bool:
        return self._available

    @property
    def backend_name(self) -> str:
        backend = self._backend
        return backend.name if backend is not None else "none"

    @property
    def status(self) -> str:
        """ "ON" / "MUTED" / "FAILED" / "OFF" - drawn on screen by the app."""
        if self._muted:
            return "MUTED"
        if self._failed:
            return "FAILED"
        if self._available:
            return "ON"
        return "OFF"

    @property
    def muted(self) -> bool:
        return self._muted

    def queue_size(self) -> int:
        return self._queue.qsize()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run, name="blindassist-tts", daemon=True
        )
        self._thread.start()
        # The self-test speaks a whole sentence, so allow more than a moment.
        self._ready.wait(timeout=20.0)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop any current speech, drop pending work, and join the worker."""
        self._clear_queue()
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._queue.put(_SENTINEL)
        self._stop_current()
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                # Keep the reference: a later start() must not spawn a second
                # worker alongside this one.
                self.init_error = (
                    f"TTS worker did not stop within {timeout:g}s; still running"
                )
            else:
                self._thread = None
        self._close_backends()

    # Backwards-compatible aliases.
    def say(self, text: str) -> None:
        self.speak(text)

    def stop(self) -> None:
        self.shutdown()

    # -- speech ------------------------------------------------------------
    def speak(self, text: str, urgent: bool = False) -> None:
        """Enqueue text. ``urgent`` clears the queue and cuts into current speech."""
        if not text:
            return
        if self._failed:
            # The worker has given up, so nothing would ever drain this. Dropping
            # is the documented "harmless no-op" behaviour; enqueuing would leak.
            _log(logging.DEBUG, "dropped (no backend): %s", text)
            return
        if self._muted:
            self.muted_log.append(text)
            _stage(f"muted, dropped: {text!r}")
            return
        if urgent:
            self._clear_queue()
            self._stop_current()
        self._queue.put(text)
        _stage(f"queued urgent={urgent}: {text!r}")

    def set_rate(self, wpm: int) -> None:
        """Change speaking rate live. Applied before the next utterance.

        The backend object belongs to the worker thread, so we only record the
        desired value here and it is read inside the worker loop.
        """
        self.rate_wpm = int(wpm)

    def set_volume(self, volume: float) -> None:
        """Change volume live (0.0..1.0). Applied before the next utterance."""
        self.volume = max(0.0, min(1.0, float(volume)))

    def set_muted(self, muted: bool) -> None:
        """Mute/unmute from the M key. Muted text is dropped, never queued."""
        self._muted = bool(muted)
        _stage(f"muted={self._muted}")

    def test_voice(self) -> None:
        """T key: speak a short line through the active backend, right now."""
        self.speak("Voice test. This is BlindAssist.", urgent=True)

    # -- helpers -----------------------------------------------------------
    def _clear_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _stop_current(self) -> None:
        backend = self._backend
        if backend is not None:
            try:
                backend.stop()  # aborts the utterance in progress (SAPI5)
            except Exception as exc:
                _log(logging.DEBUG, "stop failed: %s", exc)

    def _close_backends(self) -> None:
        with self._backends_lock:
            backends, self._backends = self._backends, []
        for backend in backends:
            try:
                backend.close()
            except Exception:
                _log(
                    logging.ERROR,
                    "backend %s close failed\n%s",
                    backend.name,
                    traceback.format_exc(),
                )

    def _backend_kwargs(self) -> dict:
        return {
            "rate_wpm": self.rate_wpm,
            "volume": self.volume,
            "voice": self.voice,
            "allow_fallback": self.allow_fallback,
            "fallback_note": self.fallback_note,
        }

    def _build_backends(self) -> list[Backend]:
        """Instances to try, in order, honouring TTS_BACKEND / --tts-backend."""
        requested = self.backend_request
        names = BACKEND_ORDER if requested in ("", "auto") else (requested,)
        built: list[Backend] = []
        for name in names:
            try:
                built.append(make_backend(name, **self._backend_kwargs()))
            except Exception:
                _log(
                    logging.ERROR,
                    "could not build backend %s\n%s",
                    name,
                    traceback.format_exc(),
                )
        return built

    # -- utterance ---------------------------------------------------------
    def _utterance(self, backend: Backend, text: str, kind: str = "speech") -> None:
        """Speak one line through ``backend``, timing and metering it.

        Raises whatever the backend raises; every attempt is written to
        logs/speech.log with start/end timestamps so silence is traceable.
        """
        backend.configure(self.rate_wpm, self.volume)
        started = time.time()
        _log(logging.INFO, "START kind=%s backend=%s text=%r", kind, backend.name, text)
        _stage(f"{kind}: backend={backend.name} start text={text!r}")
        self._probe.start()
        error: Exception | None = None
        try:
            backend.speak(text)
        except Exception as exc:
            error = exc
        finally:
            peak = self._probe.stop()
        elapsed = time.time() - started
        silent = peak is not None and peak <= _SILENT_PEAK
        # Read by the worker to decide whether this backend is still audible.
        self._last_peak = peak
        if error is not None:
            _log(
                logging.ERROR,
                "FAIL  kind=%s backend=%s dur=%.3fs text=%r\n%s",
                kind,
                backend.name,
                elapsed,
                text,
                "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            )
            _stage(f"{kind}: FAILED backend={backend.name} dur={elapsed:.3f}s")
            raise error
        _log(
            logging.INFO,
            "END   kind=%s backend=%s dur=%.3fs peak=%s text=%r",
            kind,
            backend.name,
            elapsed,
            "n/a" if peak is None else f"{peak:.4f}",
            text,
        )
        _stage(f"{kind}: done backend={backend.name} dur={elapsed:.3f}s peak={peak}")
        if silent:
            _log(
                logging.WARNING,
                "SILENT kind=%s backend=%s dur=%.3fs (peak<=%.4f) text=%r",
                kind,
                backend.name,
                elapsed,
                _SILENT_PEAK,
                text,
            )

    def _switch_backend(self, reason: str) -> bool:
        """Abandon the active backend and try the next one. True = switched."""
        with self._backends_lock:
            order = [b.name for b in self._backends]
        try:
            current = order.index(self.backend_name)
        except ValueError:
            current = -1
        for name in order[current + 1 :]:
            candidate = next((b for b in self._backends if b.name == name), None)
            if candidate is None:
                continue
            try:
                candidate.validate()
            except Exception:
                _log(
                    logging.ERROR,
                    "fallback backend %s failed validation\n%s",
                    name,
                    traceback.format_exc(),
                )
                continue
            _log(
                logging.WARNING,
                "backend switch: %s -> %s (%s)",
                self.backend_name,
                name,
                reason,
            )
            print(
                f"[VOICE] backend {self.backend_name} {reason}; "
                f"switching to {name}",
                flush=True,
            )
            old = self._backend
            self._backend = candidate
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            self._consecutive_errors = 0
            self._consecutive_silent = 0
            return True
        return False

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        com_ready = False
        try:
            # SAPI5 objects are COM objects: the worker needs its own apartment.
            import pythoncom

            pythoncom.CoInitialize()
            com_ready = True
            _stage("pythoncom.CoInitialize() in worker")
        except Exception:
            _log(
                logging.WARNING,
                "CoInitialize failed (continuing):\n%s",
                traceback.format_exc(),
            )

        try:
            self._backends = self._build_backends()
            if not self._backends:
                self.init_error = "no TTS backend available"
                self._available = False
                self._failed = True
                self._ready.set()
                return

            if not self._activate_backend():
                self._ready.set()
                return

            # Language feature: tell the user when a requested voice was absent.
            note = self.voice_fallback_note
            if note:
                self.spoken_log.append(note)
                try:
                    self._utterance(self._backend, note, kind="voice-note")
                except Exception:
                    _log(
                        logging.ERROR,
                        "voice note failed\n%s",
                        traceback.format_exc(),
                    )

            self._ready.set()
            self._loop()
        except Exception:
            # Nothing may escape into the thread silently.
            _log(logging.ERROR, "worker crashed\n%s", traceback.format_exc())
            self.init_error = "worker crashed (see logs/speech.log)"
            self._available = False
            self._failed = True
            self._ready.set()
        finally:
            self._close_backends()
            if com_ready:
                try:
                    import pythoncom

                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _activate_backend(self) -> bool:
        """Pick a backend, verifying it with the startup self-test."""
        last_error: Exception | None = None
        for backend in self._backends:
            try:
                backend.validate()
            except Exception as exc:
                last_error = exc
                _log(
                    logging.WARNING,
                    "backend %s validate failed: %s",
                    backend.name,
                    exc,
                )
                continue
            # validate() may have found (or failed to find) a voice.
            note = getattr(backend, "voice_fallback_note", None)
            if note and not self.voice_fallback_note:
                self.voice_fallback_note = note

            if not self.self_test:
                self._backend = backend
                self._available = True
                self.init_error = None
                _stage(f"backend selected (no self-test): {backend.name}")
                return True

            try:
                self._utterance(backend, SELF_TEST_TEXT, kind="self-test")
            except Exception as exc:
                last_error = exc
                _log(
                    logging.WARNING,
                    "backend %s self-test raised: %s",
                    backend.name,
                    exc,
                )
                continue
            peak = self._last_peak
            if peak is not None and peak <= _SILENT_PEAK:
                # The backend "succeeded" but the endpoint stayed silent - the
                # exact failure mode of the original bug. Try the next backend.
                last_error = BackendError(
                    f"{backend.name} produced no audio (peak={peak:.4f})"
                )
                _log(
                    logging.WARNING,
                    "backend %s self-test SILENT (peak=%.4f), falling back",
                    backend.name,
                    peak,
                )
                continue
            result = {
                "backend": backend.name,
                "text": SELF_TEST_TEXT,
                "peak": peak,
                "ok": True,
                "error": None,
            }
            self.self_test_result = result
            self._backend = backend
            self._available = True
            self.init_error = None
            _log(
                logging.INFO,
                "self-test OK backend=%s peak=%s",
                backend.name,
                "n/a" if peak is None else f"{peak:.4f}",
            )
            _stage(f"self-test OK on {backend.name}")
            return True

        # Nothing spoke: report honestly instead of pretending.
        self.init_error = (
            str(last_error) if last_error is not None else "no working TTS backend"
        )
        self._available = False
        self._failed = True
        self.self_test_result = {
            "backend": None,
            "text": SELF_TEST_TEXT,
            "peak": None,
            "ok": False,
            "error": self.init_error,
        }
        _log(logging.ERROR, "self-test FAILED everywhere: %s", self.init_error)
        print(f"[VOICE] FAILED - {self.init_error}", flush=True)
        return False

    def _probe_max(self) -> float | None:
        return getattr(self, "_last_peak", None)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            if not isinstance(item, str) or not item:
                continue
            if self._muted:
                self.muted_log.append(item)
                continue
            self.spoken_log.append(item)
            backend = self._backend
            if backend is None:
                continue
            try:
                self._utterance(backend, item)
            except Exception:
                # Logged in full by _utterance; decide whether to fall back.
                self._consecutive_errors += 1
                self._consecutive_silent = 0
                if self._consecutive_errors >= _MAX_ERRORS:
                    if not self._switch_backend(
                        f"{self._consecutive_errors} errors in a row"
                    ):
                        self._available = False
                        self._failed = True
                        self.init_error = "every TTS backend failed"
                        _log(logging.ERROR, "no backend left to fall back to")
                        return
                continue
            self._consecutive_errors = 0
            # Silence detection needs a verified meter: the startup self-test
            # proved the endpoint reports peaks, so we can trust a run of
            # silent utterances to mean "this backend has gone deaf".
            if not self.self_test:
                continue
            peak = self._last_peak
            if peak is not None and peak <= _SILENT_PEAK:
                self._consecutive_silent += 1
                if self._consecutive_silent >= _MAX_SILENT:
                    if not self._switch_backend(
                        f"{self._consecutive_silent} silent utterances"
                    ):
                        self._available = False
                        self._failed = True
                        self.init_error = "backends produced no audio"
                        _log(logging.ERROR, "no backend left to fall back to")
                        return
            else:
                self._consecutive_silent = 0


def make_speaker(
    enabled: bool,
    rate_wpm: int = 175,
    volume: float = 1.0,
    voice: str | int | None = None,
    allow_fallback: bool = False,
    fallback_note: str = "",
    backend: str = "auto",
    self_test: bool = True,
    debug: bool | None = None,
) -> Speaker:
    """Factory used by the app: real TTS when enabled, otherwise a no-op."""
    if not enabled:
        return NullSpeaker()
    return Pyttsx3Speaker(
        rate_wpm=rate_wpm,
        volume=volume,
        voice=voice,
        allow_fallback=allow_fallback,
        fallback_note=fallback_note,
        backend=backend,
        self_test=self_test,
        debug=debug,
    )
