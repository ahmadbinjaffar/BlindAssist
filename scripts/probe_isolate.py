"""Isolate the exact trigger: what makes the process go silent?

All windows are separate 6-second peak measurements.

  C1  fresh engine #1, say + runAndWait, NO stop() anywhere
  C2  fresh engine #2, say + runAndWait, NO stop() anywhere   -> is reuse OK?
  C3  stop() engine #2, then fresh engine #3 says something    -> is stop() the trigger?
  D   PowerShell System.Speech after the trigger               -> is fallback viable?
  E   winsound.Beep after the trigger                          -> is hardware still OK?

Run:  .venv\\Scripts\\python.exe scripts\\probe_isolate.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_peak_probe import PeakSampler  # noqa: E402


def window(label: str, fn, hold: float = 6.0) -> str:
    with PeakSampler() as p:
        time.sleep(0.3)
        fn()
        time.sleep(hold)
    print(f"  {label:<44} {p.report()}", flush=True)
    return p.report()


def fresh_say(text: str):
    import pyttsx3

    eng = pyttsx3.init()
    eng.setProperty("rate", 175)
    box["eng"] = eng
    eng.say(text)
    eng.runAndWait()
    return eng


box: dict = {"eng": None}


def main() -> int:
    print("Trigger isolation\n")

    window(
        "C1 fresh engine #1, NO stop()",
        lambda: fresh_say("First sentence in this process."),
    )
    time.sleep(1.0)
    window(
        "C2 fresh engine #2, NO stop()",
        lambda: fresh_say("Second sentence in this process."),
    )
    time.sleep(1.0)
    eng2 = box["eng"]

    def stop_then_speak() -> None:
        print("      -> calling stop() on engine #2 now", flush=True)
        try:
            eng2.stop()
        except Exception as exc:
            print(f"      -> stop() raised {type(exc).__name__}: {exc}", flush=True)
        time.sleep(1.0)
        fresh_say("Third sentence after a stop call.")

    window("C3 stop() then fresh engine #3", stop_then_speak)
    time.sleep(1.0)

    def ps_speak() -> None:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                ".Speak('Fallback sentence after the trigger')",
            ],
            timeout=60,
            check=False,
        )

    window("D  PowerShell System.Speech after trigger", ps_speak)
    time.sleep(1.0)

    def beep() -> None:
        import winsound

        winsound.Beep(1000, 700)

    window("E  winsound.Beep after trigger", beep, hold=2.0)

    print("\nVerdict rules:")
    print("  C2 == 0  -> mere engine re-use is fatal")
    print("  C2 > 0 and C3 == 0 -> stop() is the trigger")
    print("  D > 0 -> the PowerShell fallback still works after the trigger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
