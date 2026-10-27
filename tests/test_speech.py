"""Unit tests for the speech layer (hermetic - no audio device required)."""

from __future__ import annotations

import time
import types

from blindassist.speech import (
    BACKEND_FACTORIES,
    SELF_TEST_TEXT,
    BackendError,
    NullSpeaker,
    Pyttsx3Speaker,
    make_speaker,
)


def test_null_speaker_records_text():
    s = NullSpeaker()
    s.start()
    s.speak("hello")
    s.speak("")  # ignored
    s.shutdown()
    # `spoken` is a bounded deque so a long run cannot grow it without limit.
    assert list(s.spoken) == ["hello"]


def test_make_speaker_disabled_returns_null():
    assert isinstance(make_speaker(enabled=False), NullSpeaker)


def test_make_speaker_enabled_returns_pyttsx3():
    assert isinstance(make_speaker(enabled=True), Pyttsx3Speaker)


def test_speak_empty_is_ignored():
    s = Pyttsx3Speaker()
    s.speak("")
    assert s.queue_size() == 0


def test_speak_enqueues_before_start():
    s = Pyttsx3Speaker()
    s.speak("a")
    s.speak("b")
    assert s.queue_size() == 2


def test_urgent_clears_queue():
    s = Pyttsx3Speaker()
    s.speak("a")
    s.speak("b")
    assert s.queue_size() == 2
    s.speak("urgent", urgent=True)
    # queue was emptied, then only the urgent text remains
    assert s.queue_size() == 1


def test_urgent_is_safe_without_engine():
    s = Pyttsx3Speaker()
    s.speak("urgent", urgent=True)  # engine is None; must not raise
    assert s.queue_size() == 1


def test_configurable_rate_volume_voice():
    s = Pyttsx3Speaker(rate_wpm=200, volume=0.5, voice="female")
    assert s.rate_wpm == 200
    assert s.volume == 0.5
    assert s.voice == "female"


def test_volume_clamped_to_unit_range():
    s = Pyttsx3Speaker(volume=3.0)
    assert s.volume == 1.0
    s2 = Pyttsx3Speaker(volume=-1.0)
    assert s2.volume == 0.0


def test_say_and_stop_aliases():
    s = Pyttsx3Speaker()
    s.say("x")
    assert s.queue_size() == 1
    s.stop()  # alias for shutdown(); must not raise on a dead thread


def test_set_rate_changes_target_rate():
    """set_rate must record the new rate for the worker to apply."""
    s = Pyttsx3Speaker(rate_wpm=175)
    s.set_rate(240)
    assert s.rate_wpm == 240
    # It must NOT touch the engine directly (none exists here) nor enqueue
    # anything that could be lost/ordered ahead of speech.
    assert s.queue_size() == 0


def test_set_volume_clamped_to_unit_range():
    s = Pyttsx3Speaker(volume=1.0)
    s.set_volume(5.0)
    assert s.volume == 1.0
    s.set_volume(-2.0)
    assert s.volume == 0.0
    s.set_volume(0.4)
    assert abs(s.volume - 0.4) < 1e-9


def test_null_speaker_accepts_live_controls():
    """The app calls set_rate unconditionally, so NullSpeaker must too."""
    s = NullSpeaker()
    s.set_rate(300)
    s.set_volume(0.2)
    s.speak("still discarded")
    assert list(s.spoken) == ["still discarded"]


# -- leak / failure-path regressions ---------------------------------------
def _fake_pyttsx3(monkeypatch, init):
    """Install a stub pyttsx3 module so the worker's init path can be driven."""
    import sys
    import types

    mod = types.ModuleType("pyttsx3")
    mod.init = init
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)


def test_backend_failure_marks_speaker_unavailable(monkeypatch):
    """A broken audio backend must degrade to a no-op, not an exception.

    Pinned to the pyttsx3 backend: with TTS_BACKEND="auto" (the default the
    app uses) a broken pyttsx3 now falls back to PowerShell instead of giving
    up - that new behaviour is covered in the fallback tests below.
    """
    _fake_pyttsx3(monkeypatch, lambda: (_ for _ in ()).throw(RuntimeError("no audio")))
    s = Pyttsx3Speaker(backend="pyttsx3")
    s.start()
    assert s.available is False
    assert s.init_error is not None
    s.shutdown()


def test_speech_is_dropped_rather_than_queued_forever_when_tts_is_dead(monkeypatch):
    """Regression: with no working backend the worker exits and nothing drains
    the queue, so every announcement used to pile up for the whole run."""
    _fake_pyttsx3(monkeypatch, lambda: (_ for _ in ()).throw(RuntimeError("no audio")))
    s = Pyttsx3Speaker(backend="pyttsx3")
    s.start()
    for i in range(500):
        s.speak(f"announcement {i}")
    assert s.queue_size() == 0  # dropped, not accumulated
    s.shutdown()


def test_missing_voice_marks_speaker_unavailable(monkeypatch):
    """Zero installed voices is a real machine configuration, not a crash."""

    class _Engine:
        def getProperty(self, _name):
            return []

        def setProperty(self, *_a):
            return None

    _fake_pyttsx3(monkeypatch, _Engine)
    s = Pyttsx3Speaker(backend="pyttsx3")  # voice config is a pyttsx3 concern
    s.start()
    assert s.available is False
    assert s.init_error == "no matching / installed voice"
    s.speak("nobody can hear this")
    assert s.queue_size() == 0
    s.shutdown()


def test_out_of_range_voice_index_is_rejected(monkeypatch):
    class _Engine:
        def getProperty(self, _name):
            return [types.SimpleNamespace(name="v", id="id")]

        def setProperty(self, *_a):
            return None

    import types

    _fake_pyttsx3(monkeypatch, _Engine)
    s = Pyttsx3Speaker(voice=99, backend="pyttsx3")
    s.start()
    assert s.available is False
    s.shutdown()


def test_spoken_log_is_bounded(monkeypatch):
    """The diagnostic log must not grow without limit on a long run."""
    s = NullSpeaker()
    for i in range(5000):
        s.speak(f"m{i}")
    assert len(s.spoken) == s.spoken.maxlen
    assert s.spoken.maxlen == 1000


def test_shutdown_twice_is_safe(monkeypatch):
    s = Pyttsx3Speaker()
    s.shutdown()
    s.shutdown()  # must not raise or block
    assert s.queue_size() == 0


# -- the real worker path, driven by a fake engine --------------------------
class _RecEngine:
    """Records every pyttsx3 call the worker makes."""

    def __init__(self, voices=None):
        self.props: dict[str, object] = {}
        self.said: list[str] = []
        self.stops = 0
        self._voices = (
            voices
            if voices is not None
            else [types.SimpleNamespace(name="Zira Desktop", id="TTS_ZIRA")]
        )

    def getProperty(self, name):
        return self._voices if name == "voices" else None

    def setProperty(self, key, value):
        self.props[key] = value

    def say(self, text):
        self.said.append(text)

    def runAndWait(self):
        return

    def stop(self):
        self.stops += 1


def _wait(predicate, timeout=3.0):

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _speaker_with(monkeypatch, engine, voice=None):
    _fake_pyttsx3(monkeypatch, lambda: engine)
    s = Pyttsx3Speaker(rate_wpm=175, volume=1.0, voice=voice)
    s.start()
    return s


class _NoPeakProbe:
    """Stands in for the endpoint meter when a test uses fake backends.

    A fake backend makes no sound, so the real meter would (correctly) report
    silence and the speaker would fall back to a real backend mid-test.
    """

    ok = False

    def start(self) -> None:
        return None

    def stop(self) -> float | None:
        return None


def _disable_peak_probe(monkeypatch) -> None:
    import blindassist.speech as speech_mod

    monkeypatch.setattr(speech_mod, "_PeakProbe", _NoPeakProbe)


def test_healthy_backend_speaks_through_the_worker(monkeypatch):
    engine = _RecEngine()
    s = _speaker_with(monkeypatch, engine)
    assert s.available is True
    s.speak("person ahead.")
    assert _wait(lambda: engine.said == ["person ahead."])
    assert list(s.spoken_log) == ["person ahead."]
    s.shutdown()


def test_shutdown_stops_the_current_utterance(monkeypatch):
    engine = _RecEngine()
    s = _speaker_with(monkeypatch, engine)
    s.speak("one")
    s.speak("two")
    s.shutdown()
    assert engine.stops >= 1
    assert s._thread is None or not s._thread.is_alive()


def test_voice_selected_by_index(monkeypatch):
    engine = _RecEngine(
        voices=[
            types.SimpleNamespace(name="A", id="id_A"),
            types.SimpleNamespace(name="B", id="id_B"),
        ]
    )
    s = _speaker_with(monkeypatch, engine, voice=0)
    assert s.available is True
    assert engine.props["voice"] == "id_A"
    s.shutdown()


def test_voice_default_keeps_the_engine_choice(monkeypatch):
    """voice=None must not override whatever the engine already selected."""
    engine = _RecEngine(voices=[types.SimpleNamespace(name="A", id="id_A")])
    s = _speaker_with(monkeypatch, engine)  # voice defaults to None
    assert s.available is True
    assert "voice" not in engine.props
    s.shutdown()


def test_voice_selected_by_substring(monkeypatch):
    engine = _RecEngine(
        voices=[
            types.SimpleNamespace(name="David Desktop", id="TTS_DAVID"),
            types.SimpleNamespace(name="Zira Desktop", id="TTS_ZIRA"),
        ]
    )
    _fake_pyttsx3(monkeypatch, lambda: engine)
    s = Pyttsx3Speaker(voice="zira")
    s.start()
    assert s.available is True
    assert engine.props["voice"] == "TTS_ZIRA"
    s.shutdown()


def test_voice_substring_with_no_match_is_unavailable(monkeypatch):
    engine = _RecEngine()
    _fake_pyttsx3(monkeypatch, lambda: engine)
    s = Pyttsx3Speaker(voice="nonexistent-voice", backend="pyttsx3")
    s.start()
    assert s.available is False
    assert s.init_error == "no matching / installed voice"
    s.shutdown()


def test_set_rate_and_volume_reach_the_engine(monkeypatch):
    engine = _RecEngine()
    s = _speaker_with(monkeypatch, engine)
    s.set_rate(250)
    s.set_volume(0.25)
    s.speak("hello")
    assert _wait(lambda: "hello" in engine.said)
    assert engine.props["rate"] == 250
    assert engine.props["volume"] == 0.25
    s.shutdown()


def test_urgent_stops_the_current_utterance(monkeypatch):
    engine = _RecEngine()
    s = _speaker_with(monkeypatch, engine)
    s.speak("hello", urgent=True)
    assert _wait(lambda: engine.said == ["hello"])
    assert engine.stops >= 1  # _stop_current() ran before the urgent line
    s.shutdown()


# ---------------------------------------------------------------------------
# Language option / voice fallback (feature: language option)
# ---------------------------------------------------------------------------
def test_language_voice_is_selected_when_installed(monkeypatch):
    engine = _RecEngine(
        voices=[
            types.SimpleNamespace(name="Zira Desktop", id="TTS_ZIRA"),
            types.SimpleNamespace(name="Microsoft Urdu", id="TTS_URDU"),
        ]
    )
    s = _speaker_with(monkeypatch, engine, voice="urdu")
    assert s.available is True
    assert engine.props["voice"] == "TTS_URDU"
    assert s.voice_fallback_note is None  # no fallback message needed
    s.shutdown()


def test_missing_language_voice_falls_back_and_speaks_the_note(monkeypatch):
    engine = _RecEngine(voices=[types.SimpleNamespace(name="David", id="TTS_DAVID")])
    _fake_pyttsx3(monkeypatch, lambda: engine)
    s = Pyttsx3Speaker(
        voice="urdu",
        allow_fallback=True,
        fallback_note="No Urdu voice found; using the English voice.",
    )
    s.start()
    assert s.available is True
    assert s.voice_fallback_note == "No Urdu voice found; using the English voice."
    # The engine keeps its default voice, and the user hears the note first.
    assert "voice" not in engine.props
    assert _wait(
        lambda: engine.said[:1] == ["No Urdu voice found; using the English voice."]
    )
    s.speak("hello")
    assert _wait(lambda: engine.said[-1] == "hello")
    s.shutdown()


def test_missing_language_voice_with_fallback_off_still_fails(monkeypatch):
    """Pre-existing behaviour when nothing matched must not change."""
    engine = _RecEngine()
    _fake_pyttsx3(monkeypatch, lambda: engine)
    s = Pyttsx3Speaker(
        voice="urdu", backend="pyttsx3"
    )  # allow_fallback defaults to False
    s.start()
    assert s.available is False
    assert s.voice_fallback_note is None
    s.shutdown()


def test_missing_note_is_auto_generated(monkeypatch):
    """Even with no configured note the fallback is never silent (clear message)."""
    engine = _RecEngine()
    _fake_pyttsx3(monkeypatch, lambda: engine)
    s = Pyttsx3Speaker(voice="urdu", allow_fallback=True, fallback_note="")
    s.start()
    assert s.available is True
    assert _wait(
        lambda: engine.said[:1]
        == ['No voice matching "urdu" found; using the default voice.']
    )
    s.shutdown()


def test_urgent_drops_everything_already_queued():
    """Deterministic variant: with no worker running the queue is observable."""
    s = Pyttsx3Speaker()
    s.speak("a")
    s.speak("b")
    s.speak("c")
    assert s.queue_size() == 3
    s.speak("STOP", urgent=True)
    assert s.queue_size() == 1
    assert list(s.spoken_log) == []


def test_bad_utterance_does_not_kill_the_worker(monkeypatch):
    """One utterance raising must not stop the next one from being spoken."""

    class _FlakyEngine(_RecEngine):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def say(self, text):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("bad utterance")
            super().say(text)

    engine = _FlakyEngine()
    s = _speaker_with(monkeypatch, engine)
    s.speak("first")  # raises inside say()
    s.speak("second")  # must still get through
    assert _wait(lambda: engine.said == ["second"])
    assert s._thread is not None and s._thread.is_alive()
    s.shutdown()


# ---------------------------------------------------------------------------
# The "no voice" bug: root-cause regression + fallback + mute + shutdown
# ---------------------------------------------------------------------------
def test_every_utterance_gets_a_fresh_engine(monkeypatch):
    """ROOT-CAUSE regression.

    Re-using one pyttsx3 engine silences every utterance after the first:
    pyttsx3 queues the ``endLoop`` command ahead of the text (the proxy is not
    busy between utterances), so ``DriverProxy.endLoop()`` purges the speech it
    just started (``finished-utterance completed=False``, ~0.35 s, no audio).
    See scripts/probe_queue_order.py.  The worker therefore has to build a new
    engine per utterance and discard the previous one first.
    """
    created: list[_RecEngine] = []

    def factory():
        engine = _RecEngine()
        created.append(engine)
        return engine

    _fake_pyttsx3(monkeypatch, factory)
    s = Pyttsx3Speaker()
    s.start()
    for word in ("one", "two", "three"):
        s.speak(word)
    assert _wait(lambda: sum(len(e.said) for e in created) == 3)

    # One engine per utterance (plus the voice validation at start-up) ...
    assert len(created) >= 4
    # ... and no engine is ever asked to say two lines: that is the bug.
    assert all(len(e.said) <= 1 for e in created)
    assert len({id(e) for e in created}) == len(created)  # all distinct objects
    s.shutdown()


def test_queue_order_is_preserved(monkeypatch):
    """Announcements must come out in the order they were enqueued."""
    engine = _RecEngine()
    s = _speaker_with(monkeypatch, engine)
    for word in ("first", "second", "third", "fourth"):
        s.speak(word)
    assert _wait(lambda: len(engine.said) == 4)
    assert engine.said == ["first", "second", "third", "fourth"]
    s.shutdown()


def test_urgent_interrupt_keeps_the_worker_alive(monkeypatch):
    """urgent=True cuts in, and the worker still speaks the next line."""
    engine = _RecEngine()
    s = _speaker_with(monkeypatch, engine)
    s.speak("a long announcement that is about to be interrupted")
    s.speak("STOP", urgent=True)
    s.speak("and a normal line afterwards")
    assert _wait(lambda: "STOP" in engine.said)
    assert _wait(lambda: "and a normal line afterwards" in engine.said)
    assert s._thread is not None and s._thread.is_alive()
    s.shutdown()


def test_fallback_switches_away_from_broken_pyttsx3(monkeypatch):
    """TTS_BACKEND=auto: a pyttsx3 that cannot speak must not mean silence."""
    _disable_peak_probe(monkeypatch)
    _fake_pyttsx3(monkeypatch, lambda: (_ for _ in ()).throw(RuntimeError("no audio")))
    spoken: list[str] = []

    class _FakePowerShell:
        name = "powershell"

        def __init__(self, **_kwargs):
            pass

        def validate(self):
            return None

        def speak(self, text):
            spoken.append(text)

        def stop(self):
            return None

        def close(self):
            return None

        def configure(self, rate_wpm, volume):
            return None

    monkeypatch.setitem(BACKEND_FACTORIES, "powershell", _FakePowerShell)
    s = Pyttsx3Speaker(self_test=True)
    s.start()
    assert s.available is True
    assert s.backend_name == "powershell"
    assert s.self_test_result is not None and s.self_test_result["ok"] is True
    assert spoken[:1] == [SELF_TEST_TEXT]  # the self-test really ran
    s.speak("person ahead")
    assert _wait(lambda: "person ahead" in spoken)
    s.shutdown()


def test_fallback_gives_up_honestly_when_no_backend_works(monkeypatch):
    """No backend at all -> status FAILED, nothing is silently queued."""
    _disable_peak_probe(monkeypatch)
    _fake_pyttsx3(monkeypatch, lambda: (_ for _ in ()).throw(RuntimeError("no audio")))

    class _DeafPowerShell:
        name = "powershell"

        def __init__(self, **_kwargs):
            pass

        def validate(self):
            raise BackendError("cannot speak either")

        def speak(self, text):  # pragma: no cover - never reached
            raise BackendError("nope")

        def stop(self):
            return None

        def close(self):
            return None

        def configure(self, rate_wpm, volume):
            return None

    monkeypatch.setitem(BACKEND_FACTORIES, "powershell", _DeafPowerShell)
    monkeypatch.setitem(BACKEND_FACTORIES, "beep", _DeafPowerShell)
    s = Pyttsx3Speaker(self_test=True)
    s.start()
    assert s.available is False
    assert s.status == "FAILED"
    assert s.init_error is not None
    s.speak("nobody can hear this")
    assert s.queue_size() == 0  # dropped, never queued forever
    s.shutdown()


def test_mute_drops_text_and_reports_the_status():
    """M key: muted lines are dropped (and recorded), never queued."""
    s = Pyttsx3Speaker()
    s.speak("kept")
    s.set_muted(True)
    s.speak("dropped while muted")
    s.speak("even urgent is dropped", urgent=True)
    assert s.status == "MUTED"
    assert s.queue_size() == 1  # only the pre-mute line is still queued
    assert list(s.muted_log) == ["dropped while muted", "even urgent is dropped"]
    s.set_muted(False)
    s.speak("speaking again")
    assert s.queue_size() == 2
    assert s.status in {"OFF", "ON", "FAILED"}  # no longer MUTED
    s.shutdown()


def test_status_reports_failed_after_a_backend_failure(monkeypatch):
    """A backend that dies at start-up must be reported as FAILED, not ON."""
    _fake_pyttsx3(monkeypatch, lambda: (_ for _ in ()).throw(RuntimeError("no audio")))
    s = Pyttsx3Speaker(backend="pyttsx3", self_test=True)
    assert s.status == "OFF"  # nothing has run yet
    s.start()
    assert s.status == "FAILED"
    assert s.self_test_result is not None
    assert s.self_test_result["ok"] is False
    s.shutdown()


def test_shutdown_returns_quickly_and_leaves_no_thread():
    """shutdown() must never hang the caller, even with work in flight."""
    s = Pyttsx3Speaker()
    s.start()
    for i in range(5):
        s.speak(f"line {i}")
    t0 = time.time()
    s.shutdown(timeout=5.0)
    elapsed = time.time() - t0
    assert elapsed < 5.0
    assert s._thread is None or not s._thread.is_alive()
    # Queue is empty: nothing can leak out after shutdown.
    assert s.queue_size() == 0


def test_shutdown_is_bounded_when_the_backend_blocks(monkeypatch):
    """A stuck backend costs one bounded wait, not a hung process."""
    from blindassist.speech import Backend

    class _SlowBackend:
        name = "powershell"

        def __init__(self, **_kwargs):
            pass

        def validate(self):
            return None

        def speak(self, text):
            time.sleep(3.0)

        def stop(self):
            return None

        def close(self):
            return None

        def configure(self, rate_wpm, volume):
            return None

    assert Backend is not None  # imported for readability of the stub above
    monkeypatch.setitem(BACKEND_FACTORIES, "powershell", _SlowBackend)
    monkeypatch.setitem(BACKEND_FACTORIES, "pyttsx3", _SlowBackend)
    monkeypatch.setitem(BACKEND_FACTORIES, "beep", _SlowBackend)
    s = Pyttsx3Speaker(backend="powershell")
    s.start()
    s.speak("this line takes three seconds")
    t0 = time.time()
    s.shutdown(timeout=1.0)
    assert time.time() - t0 < 2.0
    assert s.init_error is not None and "did not stop" in s.init_error


def test_speech_log_records_start_and_end_timestamps(monkeypatch):
    """logs/speech.log must show when each utterance started and finished."""
    from blindassist import speech as speech_mod

    engine = _RecEngine()
    s = _speaker_with(monkeypatch, engine)
    s.speak("logged utterance")
    assert _wait(lambda: "logged utterance" in engine.said)
    time.sleep(0.2)  # let the END line land
    text = speech_mod.SPEECH_LOG_PATH.read_text("utf-8")
    assert "START kind=speech backend=pyttsx3 text='logged utterance'" in text
    assert "END   kind=speech backend=pyttsx3 dur=" in text
    assert "logged utterance" in text
    s.shutdown()


def test_debug_stages_can_be_enabled(monkeypatch):
    """--debug-speech / BLINDASSIST_DEBUG_SPEECH switches the stage trace on."""
    from blindassist import speech as speech_mod

    assert speech_mod.debug_enabled() in (True, False)
    speech_mod.set_debug(True)
    try:
        assert speech_mod.debug_enabled() is True
        s = Pyttsx3Speaker(debug=False)
        assert speech_mod.debug_enabled() is False
        s.shutdown()
    finally:
        speech_mod.set_debug(False)
