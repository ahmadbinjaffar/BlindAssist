"""Audio / TTS diagnostics for BlindAssist. Prints a PASS/FAIL table.

Checks (the letters match the audit brief):
  A  Windows playback device(s) present
  B  master volume not muted and not 0
  C  pyttsx3 + SAPI5 initialises, list every installed voice
  D  minimal pyttsx3 test in the MAIN thread
  E  same test in a BACKGROUND thread
  F  same test through the queue + worker used by speech.py
  G  independent fallback: Windows System.Speech through PowerShell
  H  speaker hardware: winsound.Beep

Run:
    .venv\\Scripts\\python.exe scripts\\diagnose_audio.py            # report only
    .venv\\Scripts\\python.exe scripts\\diagnose_audio.py --ask      # + ear test
    .venv\\Scripts\\python.exe scripts\\diagnose_audio.py --speak "hello"

Every check writes its result into Evidence/audio_diagnosis.txt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = ROOT / "Evidence" / "audio_diagnosis.txt"

RESULTS: list[tuple[str, str, str, str]] = []  # id, name, status, detail
LINES: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    LINES.append(line)


def record(cid: str, name: str, ok: bool | None, detail: str) -> str:
    status = "PASS" if ok else ("FAIL" if ok is False else "INFO")
    RESULTS.append((cid, name, status, detail))
    say(f"  [{status}] {cid}  {name}")
    if detail:
        for ln in str(detail).splitlines():
            say(f"         {ln}")
    return status


def ps(script: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run a PowerShell command and return (rc, stdout, stderr)."""
    r = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        errors="replace",
        check=False,
    )
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def timed(fn, timeout: float = 20.0) -> tuple[bool, str]:
    """Run fn in the main thread with a watchdog. Returns (ok, detail)."""
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except Exception:
            box["error"] = traceback.format_exc()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, f"HUNG: still running after {timeout:g}s"
    if "error" in box:
        return False, box["error"].strip().splitlines()[-1]
    return True, str(box.get("value", ""))


# --------------------------------------------------------------------------- A
def check_devices() -> bool:
    say("")
    say("A. Windows playback devices")
    rc, out, err = ps(
        "$d = Get-CimInstance Win32_SoundDevice; if (-not $d) { 'NO_SOUND_DEVICE' } "
        "else { $d | ForEach-Object { 'DEV: ' + $_.Name + ' | status=' + $_.Status } }; "
        "$e = Get-CimInstance Win32_PnPEntity -Filter \"PNPClass='AudioEndpoint'\"; "
        "if (-not $e) { 'NO_ENDPOINT' } else { $e | ForEach-Object { "
        "'ENDPOINT: ' + $_.Name + ' | status=' + $_.Status + "
        "' | err=' + $_.ConfigManagerErrorCode } }"
    )
    lines = [ln for ln in (out or "").splitlines() if ln.strip()]
    ok = bool(lines) and not any(
        "NO_SOUND_DEVICE" in ln or "NO_ENDPOINT" in ln for ln in lines
    )
    if rc != 0:
        ok = False
        lines.append(f"powershell rc={rc} {err}")
    record("A", "playback device present", ok, "\n".join(lines) or err)
    # sounddevice is optional extra evidence
    try:
        import sounddevice as sd  # type: ignore

        outs = [
            f"{n['name']} (out={n['max_output_channels']})"
            for n in sd.query_devices()
            if n["max_output_channels"] > 0
        ]
        record("A2", "sounddevice listing", bool(outs), "; ".join(outs) or "none")
    except Exception as e:
        record("A2", "sounddevice listing", None, f"not available: {type(e).__name__}")
    return ok


# --------------------------------------------------------------------------- B
def check_volume() -> bool:
    say("")
    say("B. Master volume / mute state")
    detail = ""
    ok = False
    try:
        # Primary: exactly the call the app makes at start-up, so this check
        # failing means the app would show "System volume is muted" too.
        from blindassist import speech as speech_mod

        reading = speech_mod.read_system_volume()
        if reading is None:
            raise RuntimeError("read_system_volume() returned None")
        level, mute = reading
        ok = (not mute) and level > 0.0
        detail = f"master volume {level * 100:.0f}%, muted={mute}"
    except Exception as e:
        # Fallback: winmm.dll through PowerShell. A here-string header must be
        # the last thing on its line, so the C# source is built from an array
        # of lines instead of an inline @'...'@ block.
        _rc, out, err = ps(
            "$src = @('using System;',"
            " 'using System.Runtime.InteropServices;',"
            " 'public class BaVolume {',"
            ' \'  [DllImport("winmm.dll")] public static extern int'
            " waveOutGetVolume(IntPtr h, out uint v);',"
            " '}');"
            " Add-Type -TypeDefinition ($src -join [Environment]::NewLine);"
            " $v = 0; [BaVolume]::waveOutGetVolume([IntPtr]::Zero, [ref]$v);"
            " $l = ($v -band 0xFFFF)/65535.0;"
            " $r = (($v -shr 16) -band 0xFFFF)/65535.0;"
            " 'volume=' + [math]::Round(($l+$r)/2*100,1) + '%'"
        )
        detail = out or f"pycaw unavailable ({type(e).__name__}); ps: {err}"
        ok = False
        if "volume=" in (out or ""):
            pct = float(out.split("volume=")[1].split("%")[0])
            ok = pct > 0
            detail += " (waveOut API: per-app mixer volume, not the mute state)"
    record("B", "volume not muted / not 0", ok, detail)
    return ok


# --------------------------------------------------------------------------- C
def check_voices() -> bool:
    say("")
    say("C. pyttsx3 + SAPI5 init and installed voices")
    try:
        import pyttsx3

        eng = pyttsx3.init()
        voices = eng.getProperty("voices") or []
        rows = []
        for i, v in enumerate(voices):
            lang = getattr(v, "languages", None)
            lang = lang[0] if isinstance(lang, (list, tuple)) and lang else lang
            rows.append(f"{i}: id={v.id}")
            rows.append(f"     name={getattr(v, 'name', '?')} | lang={lang}")
        ok = len(voices) > 0
        record(
            "C",
            f"pyttsx3 init, voices found = {len(voices)}",
            ok,
            (
                "\n".join(rows)
                if rows
                else "ZERO VOICES - SAPI5 has no voice " "installed"
            ),
        )
        try:
            eng.stop()
        except Exception:
            pass
        return ok
    except Exception:
        record("C", "pyttsx3 init", False, traceback.format_exc().strip())
        return False


# --------------------------------------------------------------------------- D
def check_main_thread() -> bool:
    say("")
    say("D. pyttsx3.say + runAndWait() in the MAIN thread")
    try:
        import pyttsx3

        def go() -> str:
            e = pyttsx3.init()
            e.setProperty("rate", 175)
            e.say("Audio test")
            t0 = time.time()
            e.runAndWait()
            e.stop()
            return f"runAndWait returned after {time.time() - t0:.2f}s"

        ok, detail = timed(go, 25)
        record("D", "main-thread speak", ok, detail)
        return ok
    except Exception:
        record("D", "main-thread speak", False, traceback.format_exc().strip())
        return False


# --------------------------------------------------------------------------- E
def check_thread() -> bool:
    say("")
    say("E. pyttsx3.say + runAndWait() in a BACKGROUND thread")
    result: dict = {}

    def go() -> None:
        try:
            import pyttsx3

            e = pyttsx3.init()
            e.say("Audio test")
            t0 = time.time()
            e.runAndWait()
            e.stop()
            result["ok"] = True
            result["detail"] = f"returned after {time.time() - t0:.2f}s"
        except Exception:
            result["ok"] = False
            result["detail"] = traceback.format_exc().strip().splitlines()[-1]

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(25)
    if t.is_alive():
        record("E", "background-thread speak", False, "HUNG after 25s")
        return False
    record(
        "E",
        "background-thread speak",
        bool(result.get("ok")),
        result.get("detail", "no result"),
    )
    return bool(result.get("ok"))


# --------------------------------------------------------------------------- F
def check_queue_worker() -> bool:
    say("")
    say("F. queue + worker design (blindassist.speech.Pyttsx3Speaker)")
    try:
        from blindassist.speech import Pyttsx3Speaker

        sp = Pyttsx3Speaker(rate_wpm=175, volume=1.0)
        sp.start()
        sp.speak("One.")
        sp.speak("Two.")
        sp.speak("URGENT", urgent=True)
        deadline = time.time() + 20
        while time.time() < deadline:
            if list(sp.spoken_log)[-1:] == ["URGENT"]:
                break
            time.sleep(0.1)
        detail = (
            f"available={sp.available} init_error={sp.init_error} "
            f"spoken_log={list(sp.spoken_log)}"
        )
        sp.shutdown(timeout=10)
        alive = sp._thread is not None and sp._thread.is_alive()
        ok = sp.available and not alive
        record(
            "F",
            "speech.py queue + worker",
            ok,
            detail + f" | worker joined cleanly={not alive}",
        )
        return ok
    except Exception:
        record("F", "speech.py queue + worker", False, traceback.format_exc().strip())
        return False


# --------------------------------------------------------------------------- G
def check_powershell() -> bool:
    say("")
    say("G. Independent fallback - Windows System.Speech via PowerShell")
    rc, out, err = ps(
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "'voices=' + (($s.GetInstalledVoices() | ForEach-Object { "
        "$_.VoiceInfo.Name }) -join ','); "
        "$s.Speak('Audio test'); 'spoken ok'"
    )
    ok = rc == 0 and "spoken ok" in (out or "")
    record(
        "G",
        "System.Speech (PowerShell)",
        ok,
        (out or "").replace("\n", " | ") or f"rc={rc} {err}",
    )
    return ok


# --------------------------------------------------------------------------- H
def check_beep() -> bool:
    say("")
    say("H. Speaker hardware (winsound.Beep 1000 Hz, 500 ms)")
    try:
        import winsound

        t0 = time.time()
        winsound.Beep(1000, 500)
        record("H", "winsound.Beep", True, f"beep played in {time.time() - t0:.2f}s")
        return True
    except Exception:
        record("H", "winsound.Beep", False, traceback.format_exc().strip())
        return False


# --------------------------------------------------------------------- ear test
def backend_cycle(ask: bool) -> None:
    """Speak 'Assistant ready' on every backend and ask the user's ear."""
    say("")
    say("EAR TEST - which backend can you actually hear?")
    from blindassist import speech as sp_mod

    order = (
        sp_mod.available_backends()
        if hasattr(sp_mod, "available_backends")
        else ["pyttsx3", "powershell", "beep"]
    )
    for name in order:
        try:
            eng = sp_mod.make_backend(name)
            try:
                eng.configure(175, 1.0)
                eng.speak(sp_mod.SELF_TEST_TEXT)  # blocks until the line ends
            finally:
                eng.close()
            if not ask:
                say(
                    f"  [INFO] backend {name}: self-test sent "
                    f"(run with --ask to confirm by ear)"
                )
                continue
            ans = (
                input(f"Did you hear 'Assistant ready'? ({name}) (y/n) ")
                .strip()
                .lower()
            )
            say(f"  ear answer for {name}: {ans!r}")
            if ans == "y":
                LOG.write_text("\n".join(LINES) + "\n", "utf-8")
                say(f"  -> user confirmed backend: {name}")
                return
        except Exception:
            say(
                f"  [FAIL] backend {name}: "
                + traceback.format_exc().strip().splitlines()[-1]
            )
    say("  -> no backend confirmed by ear")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ask", action="store_true", help="ask the user by ear after the report"
    )
    ap.add_argument(
        "--speak", default="", help="speak an arbitrary line with the active backend"
    )
    args = ap.parse_args()

    say("BlindAssist audio diagnosis")
    say(f"time   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"python : {sys.version.split()[0]}")
    say(f"cwd    : {Path.cwd()}")

    a = check_devices()
    b = check_volume()
    c = check_voices()
    d = check_main_thread()
    e = check_thread()
    f = check_queue_worker()
    g = check_powershell()
    h = check_beep()

    say("")
    say("=" * 78)
    say("SUMMARY")
    say("=" * 78)
    for cid, name, status, _detail in RESULTS:
        say(f"  {status:<5} {cid:<3} {name}")
    say("")
    say(
        f"device={a} volume={b} voices={c} main={d} thread={e} "
        f"queue={f} powershell={g} beep={h}"
    )

    if args.speak:
        say("")
        say(f"Speaking with the app's own backend: {args.speak!r}")
        try:
            from blindassist.speech import make_speaker

            sp = make_speaker(True, rate_wpm=175, volume=1.0)
            sp.start()
            sp.speak(args.speak)
            time.sleep(4)
            sp.shutdown()
            say("  done")
        except Exception:
            say(traceback.format_exc())

    if args.ask:
        backend_cycle(True)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n", "utf-8")
    say("")
    say(f"log written -> {LOG}")
    return 0 if all(s != "FAIL" for _, _, s, _ in RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
