"""Objective proof: does audio actually reach the Windows render endpoint?

Samples the default output device's peak meter (IAudioMeterInformation) while
playing three sources in turn:
  1. winsound.Beep                 (known-good hardware path)
  2. pyttsx3.say + runAndWait()    (main thread - what speech.py uses)
  3. PowerShell System.Speech      (independent fallback)
  4. blindassist.speech queue+worker (the app's real path)

Peak > 0 during a window = real sound left the machine.
Peak == 0 the whole time  = silence, even though the call "succeeded".

Run:  .venv\\Scripts\\python.exe scripts\\audio_peak_probe.py
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Peak level that counts as "the endpoint is actually producing sound"
# (measured silence is exactly 0.0; a working voice peaks at 0.35..0.9).
_AUDIBLE = 0.001


def make_meter():
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation

    dev = AudioUtilities.GetSpeakers()
    iface = dev._dev.Activate(IAudioMeterInformation._iid_, CLSCTX_ALL, None)
    return iface.QueryInterface(IAudioMeterInformation)


class PeakSampler:
    def __init__(self) -> None:
        self.meter = make_meter()
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(float(self.meter.GetPeakValue()))
            except Exception:
                self.samples.append(-1.0)
            time.sleep(0.02)

    def report(self) -> str:
        vals = [v for v in self.samples if v >= 0]
        if not vals:
            return "NO SAMPLES"
        return (
            f"samples={len(vals)} max_peak={max(vals):.4f} "
            f"nonzero={sum(1 for v in vals if v > _AUDIBLE)} "
            f"({100 * sum(1 for v in vals if v > _AUDIBLE) / len(vals):.0f}%)"
        )


def baseline() -> str:
    with PeakSampler() as p:
        time.sleep(1.5)
    return p.report()


def test_beep() -> str:
    with PeakSampler() as p:
        time.sleep(0.4)
        import winsound

        winsound.Beep(1000, 700)
        time.sleep(0.4)
    return p.report()


def test_pyttsx3_main() -> str:
    with PeakSampler() as p:
        time.sleep(0.3)
        import pyttsx3

        e = pyttsx3.init()
        e.say("This is a peak probe of the text to speech engine.")
        t0 = time.time()
        e.runAndWait()
        dur = time.time() - t0
        e.stop()
        time.sleep(0.3)
    return p.report() + f" | runAndWait={dur:.2f}s"


def test_powershell() -> str:
    with PeakSampler() as p:
        time.sleep(0.3)
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                ".Speak('Peak probe from powershell')",
            ],
            timeout=60,
            check=False,
        )
        time.sleep(0.3)
    return p.report()


def test_app_speaker() -> str:
    from blindassist.speech import Pyttsx3Speaker

    sp = Pyttsx3Speaker(rate_wpm=175, volume=1.0)
    sp.start()
    with PeakSampler() as p:
        time.sleep(0.3)
        sp.speak("Blind assist queue worker peak probe sentence.")
        time.sleep(4.0)
    detail = f"spoken_log={list(sp.spoken_log)}"
    sp.shutdown(timeout=8)
    return p.report() + " | " + detail


def main() -> int:
    print("Peak probe - is sound really leaving the machine?\n")
    print(f"1  silence baseline      : {baseline()}")
    print(f"2  winsound.Beep         : {test_beep()}")
    print(f"3  pyttsx3 MAIN thread   : {test_pyttsx3_main()}")
    print(f"4  PowerShell System.Speech: {test_powershell()}")
    print(f"5  speech.py queue+worker: {test_app_speaker()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
