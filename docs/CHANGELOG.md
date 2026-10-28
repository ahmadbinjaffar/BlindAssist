# Changelog

All notable changes to BlindAssist. The original release (v1:
detection pipeline, evaluation, packaging) is described in
[`../README.md`](../README.md) and [`../Evidence/test_report.md`](../Evidence/test_report.md);
this file focuses on the v2 improvement pass.

## [2.1.0] - 2026-10-26 — the silent voice, found and fixed

**Reported symptom:** the detection boxes appeared on screen, but no object
name was heard from the speakers after the welcome line.

### Fixed

- **Root cause (measured, not guessed):** one pyttsx3/SAPI5 engine was reused
  for the whole session. `Engine.runAndWait()` ends with `proxy.setBusy(False)`,
  so from utterance #2 `engine.say()` ran while the driver queue was empty and
  `endLoop` was queued *after* it; `startLoop` popped that lone `endLoop`
  first → `DriverProxy.endLoop()` → `SAPI5Driver.stop()` →
  `voice.Speak("", 3)` (SVSPURGEBOTH). The line was purged after **0.35 s**,
  pyttsx3 reported `finished-utterance completed=False`, and there was **no
  audio and no exception** — `speech.py` checked nothing, so the app believed
  it was speaking. Evidence: `Evidence/probe_queue_order.txt`,
  `Evidence/probe_second.txt` (utterance #2: `runAndWait=0.35 s`, peak
  `0.0000`).
- **The fix:** `blindassist/speech.py` rewritten around backends
  (`Backend` / `Pyttsx3Backend` / `PowerShellBackend` / `BeepBackend`) with
  **a fresh engine per utterance**, the previous engine discarded *before* the
  next `pyttsx3.init()` (order measured in `Evidence/probe_cycles.txt`:
  10/10 utterances audible, peaks 0.83–0.89, main and worker thread).
- `pythoncom.CoInitialize()` in the speech worker (SAPI5 objects are COM
  objects; pywin32 pinned at `pywin32==312`).

### Added

- **Backend fallback chain** `pyttsx3 → powershell (System.Speech, hidden) →
  beep`, selected by `TTS_BACKEND` / `--tts-backend` (default `auto`).
- **Start-up self-test:** the active backend must speak `"Assistant ready."`
  while a pycaw peak meter (`pycaw==20260927`) watches the default output
  endpoint; a raise or a silent peak (≤ 0.0005) switches backend. Result
  shown as `VOICE: ON (backend)` / `VOICE: MUTED` / `VOICE: FAILED`.
- **Speech log:** every utterance is one `START` / `END` pair with wall-clock
  timestamps, duration, backend and measured peak in `logs/speech.log`;
  failures are logged with the full traceback — nothing is swallowed.
- **Keys and flags:** `T` speaks a test line, `M` really mutes, `--tts-backend`
  and `--debug-speech` trace every stage (queue → backend → start → end);
  a muted or zero Windows master volume shows a red on-screen warning.
- **`scripts/diagnose_audio.py`** (checks A–H, all PASS, `--ask` ear test),
  **`scripts/soak_speech.py`** (5-minute no-stall soak).

### Verification

| Check | Result | Evidence |
|---|---|---|
| Diagnose A–H | 8 PASS, exit 0 | `Evidence/audio_diagnosis.txt` |
| Speech tests | **44 passed** (suite total **234 passed**) | `Evidence/test_speech_verbose.txt` |
| End-to-end run on `assets/scene60.avi` | self-test peak 0.7434, welcome 0.8913, announcements 0.8824, **11 audio bursts** for 11 utterances | `logs/speech.log`, `Evidence/e2e_peak_after_fix.txt` |
| 5-minute soak | 64 cycles, 61 utterances, **0 silent, 0 timeouts**, worker alive at the end → **PASS** | `Evidence/speech_soak_5min.txt` |
| Human ear test (`--ask`) | user answered **"y"**: *“Did you hear 'Assistant ready'? (pyttsx3)”* — object names clearly audible | `Evidence/audio_diagnosis.txt` |
| Quality gate | 234 passed, 86 % coverage, ruff clean, black clean | `Evidence/final/summary.md` §6 |

## [2.0.0] - 2026-10-25 — nine improvements

Every feature below is configurable in `blindassist/config.py`, covered by
pytest, and was proven with a real app run (exit code 0) in addition to the
unit tests. Nothing in v1 changed behaviour: defaults reproduce the original
experience unless a feature is explicitly enabled/used.

### Added

#### 1. Danger priority levels + proximity beep
- `blindassist/danger.py`: objects are classified `CRITICAL` (people + the five
  vehicle classes), `HIGH` or `NORMAL`. Only `CRITICAL` interrupts: the
  announcer cuts in with **"Stop! Car ahead."** ("Stop! 2 people ahead." when
  grouped) on a shorter cooldown. `HIGH` sorts first but speaks normally.
- `blindassist/beep.py`: optional proximity beep (Windows `winsound`) on its
  own thread — pulses get faster/nearer (`450/800/1200 ms` at
  `1500/1000/700 Hz` for near/mid/far ranks). `B` toggles it; headless and
  non-Windows runs get a harmless null controller with a clear notice.
- Config: `DangerConfig` (labels, cooldown, warning template), `BeepConfig`
  (intervals, frequencies, duration, enabled).
- Proof: `[SPEAK] Stop! Bus ahead.` on `person.jpg`; real `winsound.Beep`
  exercised in `tests/test_beep.py` (2 audible beeps).

#### 2. Stable boxes and announcements
- `TrackerConfig.smoothing_window` (5): the published box is the mean of the
  last N raw boxes — labels stop jittering between frames.
- `TrackerConfig.min_confirm_frames` (3) + `confirmed_tracks()`: an object is
  announced *and drawn* only after 3 consecutive detections — no one-frame
  phantom labels, no announcement flicker.
- Proof: `chair.jpg` 500-frame run announces only after 3 confirmed frames.

#### 3. Automatic low-light handling (CLAHE)
- `blindassist/lighting.py`: `mean_luma()` + `LightWatcher` (3-frame streak
  below luma 60) + `enhance()` (CLAHE clip 2.0, 8×8 grid). Only the frame fed
  to YOLO is brightened; the preview stays true. Status bar shows `DARK`.
- The user hears **"It is too dark, detection may be poor."** (60 s cooldown).
- Proof: `scripts/make_dark_sample.py` → `dark_*.jpg` (luma 15–25); dark run
  speaks the warning exactly once and still detects objects.

#### 4. Scene summary on the `D` key
- `announcer.summarize_scene()` + `SummaryConfig` (max items/groups): one
  overview sentence ("I see 2 people, a chair and a bottle.") followed by up
  to 3 location sentences ("The chair is ahead."). Not rate-limited — the user
  asked for it.
- Proof: `[SPEAK] I see a bed. The bed is on your left.`

#### 5. Language option
- `LanguageConfig` + `speech.py` fallback chain: `--language ur` requests a
  voice matching `voice_hint` ("urdu"); if only the default voices exist the
  app announces **"No Urdu voice found; using the English voice."** in the
  default voice and continues (never a crash, never silence).
- Limitation (documented in README): the option selects a *voice*, sentences
  stay English.

#### 6. Persistent settings (`settings.json`)
- `blindassist/settings.py`: keyboard changes (+/-, [/], `B`, language) are
  saved to `settings.json` (schema v1, clamped: conf 0.05–0.95, rate 60–300,
  volume 0.05–1.0) and reloaded on the next run.
- Precedence: `config.py` < `settings.json` < command-line flags.
- Corrupt files, unknown keys and out-of-range values produce warnings, never
  crashes; `SettingsConfig.autosave` saves on clean exit when off-key saving.
- Proof: run 1 pressed `+` → `"conf": 0.5`; the next run started from the
  stored 0.50 and saved 0.55 (chain observed 0.45 → 0.50 → 0.55 → 0.60).

#### 7. Cleaner on-screen UI
- `UIConfig.panel_alpha` (0.6): status bar and caption are semi-transparent
  panels blended over the live image (`_panel()`).
- Colour legend (top right): `VERY CLOSE` / `NEAR` / `FAR` swatches.
- Help screen with the full key reference, toggled with `H`, drawn over the
  scene at 80 % opacity.
- Screenshots: `Evidence/screenshots/improved/{improved_overlay,improved_help,improved_dark}.png`.

#### 8. Performance controls (frame skip + input size)
- `VisionConfig.frame_skip` + `--skip N` (clamped 1–30): the detector only
  sees every Nth captured frame (`_skip_detection()`); the overlay keeps the
  last known positions meanwhile. The status bar shows `SKIP N`.
- `--size {320,480,640}` (`INPUT_SIZES`, odd values snap to the nearest):
  `SIZE 320` shown in the status bar; detector `imgsz` follows.
- Measured on this machine (CPU, full `Detector.detect()` pipeline,
  [`Evidence/benchmark_sizes.md`](../Evidence/benchmark_sizes.md)):

  | Input size | Avg per detection | Detections/s |
  |---|---|---|
  | 320 px | 71.6 ms | **14.0** |
  | 480 px | 95.0 ms | 10.5 |
  | 640 px | 162.4 ms | 6.2 |

  In a 400-frame identical still-image run, `--size 320` analyzed **57** frames
  vs **38** at 640 (+50 %) and `--skip 3` consumed frames 3× faster (361 → 284
  dropped while the worker kept up).

#### 9. Session announcement log
- `blindassist/sessionlog.py` + `SessionLogConfig`: every announcement is one
  CSV row — `timestamp,kind,message` — in `logs/session_YYYYMMDD.csv`
  (kinds: `welcome` / `speak` / `muted` / `urgent` / `describe`). Appends
  across runs on the same day, rolls over at midnight, thread-safe, and a
  broken log can never take the assistant down.
- Proof: 4-row CSV written during a live run (welcome + describe + muted +
  speak rows, RFC-4180 quoting for the comma-heavy welcome sentence).

### Changed
- Welcome announcement and startup `Controls:` line now mention `D` (describe)
  and `H` (help); the on-screen key line includes `B D H`.
- The `B` key distinguishes "off in headless mode" from "unavailable on this
  machine".
- `--demo-keys` accepts `+ - = _ [ ]` in addition to letters, digits and SPACE
  (headless proof aid used throughout this pass).
- Tests isolate `settings.json`, the session CSV **and** `logs/blindassist.log`
  into per-test temp dirs — pytest's deliberate failure-path errors (no camera,
  corrupt frame, missing video) no longer land in the production log, keeping
  the `Manual_Check.md` "no traceback in the log" criterion true.

### Fixed
- **Scripted quit**: in headless runs `--demo-keys "260:Q"` was ignored because
  the demo-press branch dropped `_handle_key()`'s return value; Q/ESC now end
  the run exactly like a real key press (regression test
  `test_demo_key_q_quits_a_headless_run`).
- `_persist_settings()` now syncs the live `confidence`/`rate_wpm` UI state
  into the config before snapshotting (key handlers update UI state first, so
  an earlier version could have stored stale values).

### Verification (final gate)
- `pytest`: **223 passed**, coverage **92 %** overall — every logic module
  ≥ 86 % (`announcer` 97, `app` 88, `beep` 90, `config` 100, `danger` 100,
  `detector` 86, `lighting` 100, `position` 100, `sessionlog` 93,
  `settings` 100, `speech` 94, `tracker` 92).
- `ruff check` and `black --check`: clean over `blindassist/`, `tests/`,
  `scripts/`.
- Live app runs for every feature, all exit code 0; all-keys smoke run
  exercised `+ - [ ] B D H M S Q` in one session with a clean
  `logs/blindassist.log`.
