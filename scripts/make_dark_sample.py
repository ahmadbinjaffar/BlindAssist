"""Create a darkened copy of a sample image for low-light (CLAHE) testing.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\make_dark_sample.py [factor]

Writes ``assets/samples/dark_<name>.jpg`` from every ``assets/samples/*.jpg``
that does not already start with "dark_", and prints each copy's mean luma so
the dark threshold (default 60) is verifiable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "assets" / "samples"


def main() -> None:
    factor = float(sys.argv[1]) if len(sys.argv) > 1 else 0.22
    for src in sorted(SAMPLES.glob("*.jpg")):
        if src.stem.startswith("dark_"):
            continue
        img = cv2.imread(str(src))
        if img is None:
            print(f"[WARN] could not read {src}")
            continue
        dark = (img.astype(np.float32) * factor).astype(np.uint8)
        out = SAMPLES / f"dark_{src.name}"
        cv2.imwrite(str(out), dark)
        luma = float(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).mean())
        print(f"{out.name}: mean luma {luma:.1f} (factor {factor})")


if __name__ == "__main__":
    main()
