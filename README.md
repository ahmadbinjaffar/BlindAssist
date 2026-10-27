# BlindAssist — Real-time Object Detection Assistant for Blind Users

A webcam watches the scene, detects objects with **YOLOv8n (nano)**, works out
*where* each object is (on your left / ahead / on your right) and *how close* it
is (very close / near / far), then **speaks short phrases aloud** with an offline
text-to-speech engine.

Example console output (real, from a live run):

```
[INFO] Opened camera 0.
[SPEAK] Assistant started. Press space to pause, plus or minus to change sensitivity, M to mute, S to save a picture, and Q to quit.
[SPEAK] Person on your left.
[SPEAK] Person ahead, very close.
```

Everything is **free / open-source** and runs **fully offline after setup**
(the only network step is the one-time install and model download).

---

## Deliverables

Everything this project produced and how to run or open each item. Commands are
run from the project folder (`BlindAssist_Project/`).

| # | Deliverable | Path | How to run / open |
|---|---|---|---|
| 1 | **The app** — real-time detector + offline voice | `blindassist/` | `.\run.bat` (or `.\.venv\Scripts\python.exe -m blindassist.app`) |
| 2 | **Setup script** — venv + pinned dependencies | `setup.bat` | double-click, or `cmd /c setup.bat` |
| 3 | **Tests** — 234 pytest checks, 86 % coverage | `tests/`, `pytest.ini` | `.\.venv\Scripts\python.exe -m pytest -q --cov=blindassist` |
| 4 | **This README** — usage, commands, numbers | `README.md` | open in any editor or on GitHub |
| 5 | **Architecture document** | `docs/ARCHITECTURE.md` | open as Markdown |
| 6 | **Change log (v1 → v2)** | `docs/CHANGELOG.md` | open as Markdown |
| 7 | **Project report** — 20 pages | `Docs/Project_Report_BlindAssist.docx` | double-click; PDF copy: `Docs/Project_Report_BlindAssist.pdf`; legacy Word copy: `Docs/Project_Report_BlindAssist.doc` |
| 8 | **Slide deck** — 17 slides, 16:9 | `Presentation/BlindAssist_Presentation.pptx` | PowerPoint / LibreOffice Impress; PDF copy: `Presentation/BlindAssist_Presentation.pdf` |
| 9 | **Presentation guide** — 41 pages, English + Roman Urdu | `Docs/Presentation_Guide.docx` | double-click; PDF copy: `Docs/Presentation_Guide.pdf` |
| 10 | **Analysis notebook** | `ML_Analysis.ipynb` | open in VS Code, or re-run: `.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace ML_Analysis.ipynb` |
| 11 | **The single source of every number** | `Evidence/final/summary.md` | open as text — report, deck and guide are checked against it |
| 12 | **Report previews** (one PNG per page) | `Evidence/report_preview/` | open the PNGs |
| 13 | **Deck previews** (17 slide PNGs + contact sheet) | `Evidence/ppt_preview/` | open the PNGs |
| 14 | **Guide previews** (41 page PNGs + contact sheets) | `Evidence/guide_preview/` | open the PNGs |
| 15 | **The two result charts** | `Evidence/final/chart_fps_before_after.png`, `chart_stability_lowlight.png` | open the PNGs |
| 16 | **Screenshots** (live run, help screen, dark mode) | `Evidence/screenshots/` | open the PNGs |
| 17 | **Test report** | `Evidence/test_report.md` | open as text |
| 18 | **Document audit report** (placeholders, images, numbers, slide order) | `Evidence/audit/audit_report.txt` | open as text, or re-run `.\.venv\Scripts\python.exe scripts\audit_docs.py` |
| 19 | **Rebuild scripts** for report / deck / guide / proposal | `scripts/make_report.py`, `scripts/make_pptx.py`, `scripts/make_guide.py`, `scripts/build_proposal.py` | `.\.venv\Scripts\python.exe scripts\make_report.py` (same for the others) |
| 20 | **Project proposal** — one page, text in the top half (244 words) | `Docs/Project_Proposal_BlindAssist.docx` | double-click; PDF copy: `Docs/Project_Proposal_BlindAssist.pdf`; rebuild: `.\.venv\Scripts\python.exe scripts\build_proposal.py` |

> **Not in this repository:** the written report, the slide deck, the
> presentation guide, their previews and the whole `Evidence/` folder of raw
> measurements stay on the author's machine. The code, the tests, the
> analysis scripts and this README are all here.

Every `.docx` / `.pptx` above has a PDF copy next to it so it opens on any
machine, and every page or slide was rendered to PNG in `Evidence/` for visual
inspection.

---

## Screenshots

**Live webcam, speech active** — two people at different distances; the box colour
follows urgency (red = very close, orange = near, green = far):

![Live webcam run with two detected people](Evidence/screenshots/live.png)

**No TTS voice available (fallback)** — the status bar shows `NO TTS`, detection
and the `SAID` caption keep working, nothing crashes:

![Fallback with no text-to-speech engine](Evidence/screenshots/fallback.png)

**Still image `assets/samples/chair.jpg`** — grouping and pluralisation:
*"Person on your right. Tv on your left. 3 chairs ahead."*

![Run on a still image with three chairs grouped](Evidence/screenshots/chair.png)

**Still image `assets/samples/bottle.jpg`** — five objects announced, nearest
band first:

![Run on a still image with five detected objects](Evidence/screenshots/bottle.png)

**Improved overlay** — semi-transparent status/caption panels and a colour
legend (feature 7):

![Improved overlay with legend and semi-transparent panels](Evidence/screenshots/improved/improved_overlay.png)

**Help screen (`H` key)** — the full key reference over a live scene:

![Help screen toggled with the H key](Evidence/screenshots/improved/improved_help.png)

**Auto low-light mode** — dark scenes are CLAHE-brightened for the detector
only, the preview stays true, and the status bar shows `DARK` (feature 3):

![Dark mode with DARK indicator](Evidence/screenshots/improved/improved_dark.png)

More screenshots (saved with the `S` key and by the automated QA runs) are in
[`Evidence/screenshots/`](Evidence/screenshots/).

---

## Features

- **Real-time detection** with YOLOv8n on the CPU — the preview is never blocked
  by inference (frames are dropped, never queued).
- **Direction words** — "on your left" / "ahead" / "on your right" from three
  dashed zone guides drawn on screen.
- **Approximate distance bands** — "very close" / "near" / "far" from the box's
  share of the frame. No depth sensor, no lie about metres.
- **Cooldown & grouping** — objects are grouped ("3 chairs ahead"), prioritised
  nearest-first, and rate-limited, so the app never talks continuously.
- **Large readable overlays** — coloured boxes, confidence %, zone labels, a top
  status bar (RUNNING/PAUSED, FPS, confidence, rate) and a bottom caption showing
  the last sentence actually spoken.
- **Offline TTS on a background thread** — a queue keeps speech from freezing the
  video; urgent phrases cut in and clear the backlog. A **fresh engine per
  sentence**, a **start-up self-test** that measures the real speaker peak and a
  `pyttsx3 → PowerShell → beep` fallback keep the voice from going quiet.
- **Automatic camera fallback** — tries index 0, falls back to other indices if
  busy, and exits with a clear error if there is no camera at all.
- **Works from a file too** — `--video` accepts a video file *or* a still image,
  so you can test with no webcam.
- **Everything tunable from the keyboard** — pause, sensitivity, speech rate,
  mute, screenshot, quit.
- **Fully offline after setup**, file logging to `logs/blindassist.log`, clean
  shutdown on every exit path.

### Improvements (v2 — see [`docs/CHANGELOG.md`](docs/CHANGELOG.md))

- **Danger priority** — moving vehicles and people get `CRITICAL`/`HIGH`
  priority; a critical object cuts in with *"Stop! Car ahead."* and (optionally)
  a proximity beep whose pulses speed up as the object gets closer (`B` key).
- **Stable announcements** — boxes are smoothed over 5 frames and an object is
  only announced/drawn after 3 consecutive detections: no more flicker or
  one-frame phantom labels.
- **Auto low-light (CLAHE)** — genuinely dark frames are brightened *for the
  detector only* and the user hears *"It is too dark, detection may be poor."*
- **Scene summary** — press `D` for a spoken overview: *"I see 2 people and a
  chair. The people are on your right."*
- **Language option** — `--language ur` picks a matching voice when one is
  installed (clear English fallback note otherwise).
- **Settings persist** — sensitivity, rate, volume, beep and language changes
  from the keyboard are saved to `settings.json` and reloaded on the next run
  (config.py < settings.json < command line).
- **Cleaner overlay** — semi-transparent status/caption panels, a colour
  legend, and a full help screen on `H`.
- **Faster on slow CPUs** — `--skip N` analyzes every Nth frame and
  `--size {320,480,640}` picks the detector input size; the status bar shows
  `SIZE`/`SKIP` and the live FPS.
- **Session log** — every announcement (spoken, muted, urgent, describe) is
  appended with a timestamp to `logs/session_YYYYMMDD.csv`.
- **Reliable voice (bug fix)** — one engine per sentence, a start-up
  self-test that measures the real speaker peak, and a
  `pyttsx3 → PowerShell → beep` fallback chain; every utterance is timestamped
  in `logs/speech.log` and the status bar shows `VOICE: ON / MUTED / FAILED`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Windows 10/11 (developed and tested on Windows; the capture path uses DirectShow) |
| **Python** | **3.13.7 64-bit** (pinned — the versions in `requirements.txt` were installed and tested against it) |
| **Webcam** | Any DirectShow camera; tested at 1280×720 |
| **Speakers / headphones** | For speech (optional — use `--no-speech` if you have none) |
| **Speech libraries** | `pywin32==312` (COM init in the speech thread) and `pycaw==20260927` (reads the Windows volume and measures the output peak) — both in `requirements.txt` |
| **Disk** | ~2 GB free (the venv alone is ~1.2 GB, torch is the bulk) |
| **Internet** | **Only** for the one-time install and the one-time ~6.25 MB model download |
| **GPU** | *Not required.* CPU-only torch is installed; the model auto-uses CUDA if you ever provide it |
| **Optional** | VS Code for F5 debugging; `ffmpeg` is **not** required |

> **No admin rights needed.** Nothing is installed outside your user folder.

---

## Setup from scratch

### Option A — the scripts (recommended)

```powershell
cd BlindAssist

# first time only: create the venv + install everything
.\setup.bat

# every time after that: start the app
.\run.bat
```

`setup.bat` is idempotent — re-running it only installs what is missing.

### Option B — by hand (what the script does)

**1. Create the virtual environment at a SHORT path.**

This matters on Windows: torch ships very deep header paths and the default
`MAX_PATH` is 260 characters, so a deeply nested project path can break the
install. Keep the venv short:

```powershell
python -m venv C:\dev\ba_venv
```

**2. Link it into the project as `.venv`:**

```powershell
cmd /c mklink /J ".venv" "C:\dev\ba_venv"
```

(If your own project path is short, a normal in-project venv is fine — skip the
junction and use `.venv\Scripts\python.exe` directly.)

**3. Install the pinned dependencies:**

```powershell
C:\dev\ba_venv\Scripts\python.exe -m pip install --upgrade pip
C:\dev\ba_venv\Scripts\python.exe -m pip install -r requirements.txt
```

> `torch`/`torchvision` are the **CPU-only** builds from the official PyTorch
> wheel index. `requirements.txt` already carries the `--extra-index-url` line
> for that; everything else resolves from PyPI. Every version is pinned with `==`.
> `pip_freeze.txt` records the exact full transitive environment of a known-good
> setup.
>
> **Honest caveat about transitive dependencies:** `requirements.txt` pins the
> *direct* packages — torch, ultralytics, opencv, numpy, pyttsx3, pytest, … — and
> those reinstall exactly every time. *Transitive* ones are not pinned, so a
> fresh install can resolve newer ones. A verified delete-and-reinstall of this
> project moved **8 of 97** packages, including **Pillow 12.3.0 → 13.0.0**; the
> test suite, ruff, black and the app all still passed on them. For a
> byte-for-byte rebuild, install `pip_freeze.txt` instead of `requirements.txt`.

**4. First run downloads the model (one time, needs internet).**
On the very first launch the app fetches `yolov8n.pt` (~6.25 MB) and keeps it in
`assets/`. After that, **no network is used**.

**5. Verify:**

```powershell
.\.venv\Scripts\python.exe -m pytest -q     # offline unit tests
.\run.bat                                   # start the app
```

---

## Run it

The app must be run as a **module** (it uses relative imports), from this folder:

```powershell
cd BlindAssist
.\.venv\Scripts\python.exe -m blindassist.app
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `SPACE` | start / pause |
| `+` / `-` | raise / lower the confidence threshold |
| `]` / `[` | faster / slower speech |
| `M` | mute / unmute (`VOICE: ON` / `VOICE: MUTED` in the status bar) |
| `T` | speak a test line — *"Voice test. This is BlindAssist."* |
| `B` | proximity beep on / off |
| `D` | describe the whole scene now |
| `H` | show / hide the on-screen help |
| `S` | save a screenshot to `Evidence/screenshots/` |
| `Q` or `ESC` | quit |

### Command-line flags

| Flag | Meaning |
|------|---------|
| `--camera N` | camera index (default 0; falls back automatically if busy) |
| `--conf 0.5` | confidence threshold |
| `--video FILE` | run on a video file **or a still image** instead of the webcam |
| `--no-speech` | disable TTS, just print `[SPEAK]` lines |
| `--tts-backend {auto,pyttsx3,powershell,beep}` | speech backend (default `auto`: each one is verified by the start-up self-test, then the chain falls back) |
| `--debug-speech` | trace every stage of the speech pipeline (queue → backend → start → end) to the console and `logs/speech.log` |
| `--headless-test` | no GUI window (for automated runs; `--headless` also works) |
| `--max-frames N` | stop after N frames (headless defaults to 150) |
| `--rate N` | speech rate in words per minute |
| `--language CODE` | voice language, e.g. `en` (default) or `ur` |
| `--size PX` | detector input size: `320`, `480` or `640` (smaller = faster) |
| `--skip N` | analyze every Nth frame (1 = every frame; larger = faster UI) |
| `--all-classes` | detect all 80 COCO classes (default: indoor-navigation set) |
| `--model FILE` | path to YOLO weights (default: auto-download once) |
| `--tag NAME` | filename stem for the headless screenshot |
| `--demo-keys "N:KEY,..."` | scripted key presses for headless runs, e.g. `"60:D,120:Q"` |

Examples:

```powershell
# live webcam, with speech (the normal way to use it)
.\.venv\Scripts\python.exe -m blindassist.app

# silent live preview
.\.venv\Scripts\python.exe -m blindassist.app --no-speech

# automated smoke test: no window, no audio, saves Evidence/screenshots/test.png
.\.venv\Scripts\python.exe -m blindassist.app --headless-test --no-speech --max-frames 60 --tag test

# run on a still image (handy for testing / producing evidence)
.\.venv\Scripts\python.exe -m blindassist.app --video assets\samples\chair.jpg --headless-test --tag chair
```

Exit codes: `0` = clean stop, `2` = no camera / unreadable `--video` file,
`3` = the detector could not be loaded.

### Quick checks

```powershell
# camera alone, no ML, instant
.\.venv\Scripts\python.exe scripts\probe_webcam.py --camera 0 --frames 15 --save Evidence\probe.jpg

# detector on 3 free sample images
.\.venv\Scripts\python.exe scripts\validate_detector.py
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No camera available` and **exit code 2** | Another app (Zoom/Teams/Chrome) holds the camera, or no camera is plugged in | Close the other app, or try `--camera 1` / `--camera 2` |
| Camera opens, then the preview is black | Driver/permission issue with that index | Try another index: `--camera 1`. Check Windows ▸ Privacy ▸ Camera allows desktop apps |
| `pip` fails with *path too long* / `ENOENT` during torch install | Windows `MAX_PATH` (260) limit | Recreate the venv at a **short** path (`C:\dev\ba_venv`) and re-link `.venv` — see [Setup](#option-b--by-hand-what-the-script-does) |
| `ModuleNotFoundError: No module named 'blindassist'` | Started from the wrong folder, or ran `app.py` directly | `cd` into `BlindAssist_Project` and run `python -m blindassist.app` |
| `FileNotFoundError` / **exit code 3** on first start | Model weights missing and no internet to fetch them | Connect once and re-run, or drop a `yolov8n.pt` into `assets/` |
| No speech, console shows `NO TTS` | No TTS voice installed, audio device missing, or `M` was pressed | Press `M`; install a Windows voice (Settings ▸ Time & Language ▸ Speech); or run with `--no-speech` — `[SPEAK]` lines still print |
| Speech is too fast / too slow | Rate setting | Press `]` / `[`, or start with `--rate 150` |
| `UnicodeEncodeError` when printing | Console codepage is cp1252 | Run `python -X utf8 -m blindassist.app` |
| Video lags or boxes trail the image | CPU-only inference (≈5 inferences/s on this laptop) | Lower `inference_size` in `blindassist/config.py`, or raise `--conf` to cut the box count |
| Small / distant objects are missed | Inherent limit of a 640 px nano model | Move closer; raise `--conf` down slightly (e.g. `--conf 0.3`) if you want more, noisier detections |
| Detection is wrong in the dark | Low light | Add light — measured weakness, see [Known limitations](#known-limitations) |
| `pytest` collects 0 tests | Run from the project root | `cd BlindAssist_Project` then `python -m pytest` (collection is restricted to `tests/` by `pytest.ini`) |
| `ruff`/`black` report failures | Style regressions | `python -m ruff check --fix blindassist tests scripts` then `python -m black blindassist tests scripts` |
| Venv behaves oddly after moving the folder | The `.venv` junction points at `C:\dev\ba_venv` | Re-run `.\setup.bat`; it repairs the link |
| Windows says `Access is denied` for `mklink /J` | Junction needs an elevated prompt in some setups | Use an **elevated** PowerShell, or skip the junction and use the absolute venv path |

Still stuck? The file log always has the detail: [`logs/blindassist.log`](logs/blindassist.log).

### No sound? Troubleshooting

The app **tests its own voice before the first frame**: it speaks
*"Assistant ready."* while a peak meter watches the real output device, so a
silent backend is replaced before you see anything. Work through this list:

1. **Read the status bar.** `VOICE: ON (pyttsx3)` = the start-up test was
   measured (peak 0.74 on this machine). `VOICE: MUTED` → press `M` or `T`.
   `VOICE: FAILED` → every backend failed, go to step 3.
2. **Run the diagnosis:** `.\.venv\Scripts\python.exe scripts\diagnose_audio.py`
   — checks **A to H** must all say `PASS` (playback device, master volume and
   mute, installed voices, main-thread speak, worker-thread speak, queue
   design, PowerShell fallback, speakers). With `--ask` it speaks and asks
   *"Did you hear 'Assistant ready'? (y/n)"* — your ear is the final test.
3. **Check the Windows volume.** If the master volume is 0 % or muted the app
   shows a red **System volume is muted** warning on the screen; unmute from
   the taskbar. `diagnose_audio.py` check **B** reads the same value.
4. **Force a backend and trace it:**
   `.\.venv\Scripts\python.exe -m blindassist.app --tts-backend powershell --debug-speech`
   (`beep` always makes a sound if anything does). `--debug-speech` prints the
   queue → backend → start → end stages to the console.
5. **Read `logs/speech.log`.** Every utterance is one `START` / `END` pair with
   wall-clock timestamps, duration, backend and measured peak — silence is
   traceable after the fact. Failures are logged with the full traceback.

---

## How it works

```
webcam frame (UI thread, shown immediately)
   └─> detection worker thread          (drop-never-queue single frame slot)
          └─> YOLOv8n detector          -> boxes + labels + confidence
                 └─> IoU tracker        -> stable track IDs
                        └─> positioner  -> left/ahead/right + very close/near/far
                               └─> announcer -> group, prioritise, cooldown
                                      └─> speech -> offline TTS (background thread)
```

The UI thread never waits for inference: it hands each frame to the worker
through a **single slot**, and if the worker is still busy the frame is
**dropped rather than queued**. That keeps the preview smooth and memory flat —
on this machine the camera itself is the bottleneck (below), not the detection.

**Direction** comes from splitting the frame into thirds: a box whose centre is
in the left third is "on your left", middle third "ahead", right third "on your
right". The same thirds are drawn on screen as dashed guides.

**Distance is an approximation — there is no depth sensor.** It is estimated only
from how much of the frame the box covers:

| Box area / frame area | Spoken as |
|-----------------------|-----------|
| ≥ 20 % | very close |
| ≥ 5 %  | near |
| < 5 %  | far |

This assumes objects are roughly comparable in real size, so it reliably
separates "right in front of you" from "across the room" but is **not** a
measurement in metres.

**What gets said** is decided by `announcer.py`: detections are grouped by label
("3 chairs ahead"), sorted nearest-first, limited to a few per cycle, and
suppressed for a cooldown per object so the same thing is not repeated forever.

The full picture, including the thread layout, is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Evaluate the model (real numbers)

Everything below writes its results into `Evidence/` and only needs the internet
on the first run (to fetch the COCO subset); afterwards it is fully offline.

```powershell
# 1. Detection quality on coco128 (128 images from COCO val2017):
#    precision, recall, mAP@0.5, mAP@0.5:0.95 + per-class AP -> Evidence/evaluation.md
.\.venv\Scripts\python.exe scripts\evaluate.py

# 2. Inference speed: 100 frames at 320 px and 640 px, with CPU/GPU recorded
#    -> Evidence/benchmark.md + Evidence/benchmark.png
.\.venv\Scripts\python.exe scripts\benchmark.py --frames 100

# 3. Direction + distance accuracy on 10 hand-annotated images
#    -> Evidence/position_accuracy.md + Evidence/eval_position/
.\.venv\Scripts\python.exe scripts\evaluate_position.py

# 4. Analysis notebook: loads all of the above, shows detection examples,
#    per-class AP charts and an interpretation. Rebuild + run top to bottom.
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace ML_Analysis.ipynb
```

Headline numbers from those runs are in
[`Evidence/evaluation.md`](Evidence/evaluation.md) and summarised in
[`ML_Analysis.ipynb`](ML_Analysis.ipynb).

---

## Run the tests (offline, no camera/model needed)

```powershell
# full suite + coverage
.\.venv\Scripts\python.exe -m pytest -v --cov=blindassist --cov-report=term

# static checks (rule set lives in pyproject.toml)
.\.venv\Scripts\python.exe -m ruff check blindassist tests scripts
.\.venv\Scripts\python.exe -m black --check blindassist tests scripts

# real-app runtime check: --video run + 60 s live run, watching for
# exceptions, memory growth and thread leaks -> Evidence/qa_runtime.md
.\.venv\Scripts\python.exe scripts\qa_runtime_check.py --seconds 60
```

Last run: **234 passed, 86% coverage** (every logic module ≥ 76%), ruff clean,
black clean, runtime check PASS. Full details, including every bug found and
fixed, are in [`Evidence/test_report.md`](Evidence/test_report.md).

A manual, by-hand verification sheet is in
[`Evidence/Manual_Check.md`](Evidence/Manual_Check.md).

`pytest.ini` restricts collection to `tests/`; `scripts/test_speech.py` is a
manual demo, run it directly with `python scripts\test_speech.py`.

---

## VS Code

Open the `BlindAssist_Project` folder. `.vscode/settings.json` already points the
Python interpreter at `C:\dev\ba_venv`. Press **F5** (Run ▸ Start
Debugging) with the first config selected:

- **BlindAssist: Run (live webcam + speech)** ← default
- **BlindAssist: Live (no speech, print only)**
- **BlindAssist: Run on a video / image**
- **BlindAssist: Headless test (auto-screenshot)**
- **BlindAssist: All 80 COCO classes**
- **Pytest (unit tests)**
- **Probe webcam (no model)**

---

## Measured performance (real runs on this machine)

Hardware: **Intel Core i5-6300U @ 2.40 GHz** (CPU only — the integrated
Intel HD Graphics 520 is not CUDA-capable, so torch runs on CPU).

### Speed

| What | Measured |
|------|----------|
| Camera (1280×720, no model) | **10.2 FPS** (`scripts/probe_webcam.py`) |
| Live app, detection active | **~10.1 FPS shown**, ≈5 inferences/s |
| Detector validation | **3/3** expected classes (person, chair, bottle) |
| Unit tests | **234 passed**, 86 % coverage |
| Benchmark, 320 px (100 frames) | **75.6 ms** → **13.2 FPS** (median 67.5, stdev 20.1) |
| Benchmark, 640 px (100 frames) | **148.0 ms** → **6.8 FPS** (median 145.0, stdev 19.0) |

Because the app sustains the camera's full rate while detecting, **inference does
not slow the video down** — it runs concurrently on a worker thread.

### Detection quality (coco128: 128 images, 929 instances, 71 of 80 classes present)

| Metric | Value |
|--------|-------|
| Precision | **0.6498** |
| Recall | **0.5420** |
| mAP@0.5 | **0.6124** |
| mAP@0.5:0.95 | **0.4453** |

### Direction & distance accuracy (10 hand-annotated images, 38 objects)

| Measure | Value |
|---------|-------|
| Objects detected | **27 / 38 (71 %)** |
| Direction correct, given a detection | **26 / 27 (96 %)** |
| Distance band correct, given a detection | **27 / 27 (100 %)** |
| Both correct, end-to-end | **26 / 38 (68 %)** |

So the positioner itself is reliable once it has a box; the end-to-end gap is
entirely **missed detections** of small, dim or cropped objects.

### Stability (60 s live run, `scripts/qa_runtime_check.py`)

RSS grew from 44 MB to 429 MB **during model warm-up and then went flat**
(steady-state slope −0.62 MB/s, i.e. no growth); threads settled at 25; zero
tracebacks, zero `[ERROR]` lines. See [`Evidence/qa_runtime.md`](Evidence/qa_runtime.md).

---

## Known limitations

- **This is a demo/aid, not a safety device.** Do not rely on it to keep you out
  of harm's way — it can miss objects entirely.
- **Distance is approximate.** Area-ratio bands (very close / near / far) only —
  no depth sensor, so no metres. A small nearby object and a large distant one
  can look similar.
- **Missed detections dominate the error.** On 10 hand-annotated images only
  27 of 38 objects were found (71 %) — small, dim or cropped objects are the
  main weakness. When a box *is* found, direction was right 96 % of the time.
- **Low light hurts.** Confidence drops in dark scenes; expect fewer and later
  detections at night.
- **CPU speed.** YOLOv8n runs at ≈5 inferences/s on this CPU while the camera
  delivers ~10 FPS, so boxes lag the video by a frame or two. Options: run with
  `--size 320` (≈14 detections/s measured on this machine, see
  [`Evidence/benchmark_sizes.md`](Evidence/benchmark_sizes.md)), `--skip 2` to
  analyze every other frame, lower `inference_size` in `config.py`, or run on
  a machine with a GPU.
- **The language option picks a *voice*, it does not translate.** `--language ur`
  selects a voice matching that language when one is installed; the sentences
  themselves stay English. This machine has only English (David/Zira) voices, so
  the app announces *"No Urdu voice found; using the English voice."* and
  carries on — clear fallback, but not an Urdu interface.
- **The proximity beep needs Windows `winsound`.** On other platforms (or in
  headless runs) the beep stays off with a clear notice; every other feature
  works unchanged.
- **COCO-only vocabulary.** The model recognises the 80 COCO classes, filtered by
  default to an indoor-navigation set (person, car, dog, chair, bottle, table,
  phone, …). Use `--all-classes` for the full set. It will call an unfamiliar
  object by its nearest COCO class — occasionally wrong.
- **No "stairs" class in COCO**, so stairs-like obstacles cannot be prioritised
  by label (documented in `config.py`).
- **Single, forward-facing camera.** No depth sensor, no stereo; direction words
  come from where the box sits in the frame.
- **The tracker's `trend()` ("approaching") helper is implemented and unit-tested
  but not currently spoken** — announcements are position + distance only.
- **First run needs internet** to fetch the ~6.25 MB weights. After that it is
  offline.

---

## Project structure

```
BlindAssist_Project/
├── blindassist/             # the Python package
│   ├── config.py            # every tunable in one place (incl. all v2 features)
│   ├── detector.py          # YOLOv8n wrapper (ultralytics)
│   ├── tracker.py           # lightweight IoU tracker (stable IDs) + smoothing
│   ├── position.py          # bbox -> direction words + approximate distance band
│   ├── announcer.py         # grouping + prioritisation + cooldown ("what & when")
│   ├── danger.py            # CRITICAL/HIGH priority + "Stop! Car ahead." interrupt
│   ├── beep.py              # non-blocking proximity beep (winsound / null fallback)
│   ├── lighting.py          # mean-luma watcher + CLAHE enhancement
│   ├── speech.py            # offline pyttsx3 TTS on a background thread
│   ├── settings.py          # settings.json load/save/apply
│   ├── sessionlog.py        # daily logs/session_YYYYMMDD.csv announcement log
│   └── app.py               # real-time app: capture loop, worker thread, OpenCV UI
├── tests/                   # 234 offline unit tests (pytest)
├── scripts/
│   ├── probe_webcam.py      # check camera access without any ML
│   ├── validate_detector.py # detector validation on 3 free sample images
│   ├── test_speech.py       # live audio check (responsiveness + urgent cut-in)
│   ├── diagnose_audio.py    # audio checks A-H + the --ask ear test
│   ├── soak_speech.py       # 5-minute no-stall speech soak -> Evidence/
│   ├── evaluate.py          # COCO metrics -> Evidence/evaluation.md
│   ├── benchmark.py         # speed at 320/640 px -> Evidence/benchmark.md/.png
│   ├── benchmark_sizes.py   # 320/480/640 sweep -> Evidence/benchmark_sizes.md
│   ├── make_dark_sample.py  # dark test images for the low-light feature
│   ├── evaluate_position.py # direction/distance accuracy -> Evidence/position_accuracy.md
│   ├── build_notebook.py    # regenerates ML_Analysis.ipynb from Evidence/
│   ├── qa_runtime_check.py  # 60 s live + video run, memory/thread watch
│   ├── capture_best.py      # save the best (most populated) frame over a few sec
│   ├── capture_detection.py # save one annotated frame
│   ├── capture_eval_frames.py # frames for the position-accuracy evaluation
│   └── proc_sample.ps1      # process-tree memory/thread sampler (used by QA)
├── assets/
│   ├── yolov8n.pt           # YOLOv8n weights (auto-downloaded once, then offline)
│   ├── samples/             # person / chair / bottle + dark_* test images
│   ├── eval_frames/         # hand-annotated frames + ground_truth.json
│   └── qa_clip.avi          # short clip used by the runtime check
├── Evidence/                # evaluation.md, benchmark, screenshots, test_report.md
│   ├── screenshots/         # saved with S / by headless runs
│   ├── screenshots/improved/ # v2 overlay / help / dark-mode shots
│   ├── eval/  eval_position/  model_validation/
│   └── Manual_Check.md      # by-hand verification sheet
├── ML_Analysis.ipynb        # analysis notebook (executed, zero errors)
├── logs/blindassist.log     # application log (written on every run)
├── logs/session_YYYYMMDD.csv # every announcement, one row per line (feature 9)
├── settings.json            # persisted keyboard settings (created on first change)
├── docs/                    # ARCHITECTURE.md, CHANGELOG.md, model_notes.md
├── .vscode/                 # interpreter + launch configs for VS Code
├── setup.bat / run.bat      # first-time install / start the app
├── requirements.txt         # pinned direct dependencies
├── pip_freeze.txt           # exact full `pip freeze` of a known-good setup
├── pyproject.toml           # black + ruff configuration
└── pytest.ini               # restricts test collection to tests/
```

---

## Credits & licenses

### This project

BlindAssist itself is an independent open-source project built on top of
**ultralytics**, which is
licensed **AGPL-3.0**. Because the app links that library, any public distribution
of this project must also be under **AGPL-3.0** (add a `LICENSE` file if you
publish it). Everything used is free / open-source; nothing here is proprietary.

### Third-party components

Versions below are the ones actually installed (read from each package's own
`METADATA`), not guesses.

| Component | Version | License |
|---|---|---|
| [Ultralytics / YOLOv8](https://github.com/ultralytics/ultralytics) — detector + `yolov8n.pt` weights | 8.4.184 | **AGPL-3.0** |
| [OpenCV](https://opencv.org/) (`opencv-python`) | 5.0.0.93 | Apache-2.0 |
| [PyTorch](https://pytorch.org/) (`torch`, CPU build) | 2.14.1+cpu | Apache-2.0 (plus BSD/MIT/BSL for bundled components) |
| [NumPy](https://numpy.org/) | 2.5.2 | BSD-3-Clause |
| [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — offline TTS | 2.99 | MPL-2.0 |
| [Matplotlib](https://matplotlib.org/) | 3.11.2 | Matplotlib License (PSF-based) |
| [Pillow](https://python-pillow.org/) | 13.0.0 | MIT-CMU |
| [PyYAML](https://github.com/yaml/pyyaml) | 6.0.3 | MIT |
| [pytest](https://pytest.org/) / [pytest-cov](https://pytest-cov.readthedocs.io/) | 9.1.1 / 7.1.0 | MIT |
| [Ruff](https://docs.astral.sh/ruff/) / [Black](https://github.com/psf/black) | 0.17.2 / 26.10.1 | MIT |
| [nbconvert](https://nbconvert.readthedocs.io/) / [nbformat](https://nbformat.readthedocs.io/) | 7.17.2 / 5.11.1 | BSD-3-Clause |

### Datasets & model

- **COCO** — used for evaluation (`coco128`) and for the model's class
  vocabulary. Per [cocodataset.org/terms of use](https://cocodataset.org/dataset/termsofuse.htm):
  the **annotations and website are CC BY 4.0**; the **images are not owned by the
  COCO Consortium** and remain subject to their original Flickr licences; the COCO
  **software** is BSD-style. If you redistribute COCO images, keep the attribution
  and check each image's Flickr licence.
- **YOLOv8n weights** (`assets/yolov8n.pt`) are distributed by Ultralytics under
  **AGPL-3.0**. They are auto-downloaded from Ultralytics on first run and are
  never redistributed by this repository.
- COCO class names are used for labels. COCO: *Microsoft COCO: Common Objects in
  Context*, Lin et al., ECCV 2014.

### Speech

Text-to-speech uses Windows' own voices through **pyttsx3** — no cloud service,
no audio ever leaves the machine.
