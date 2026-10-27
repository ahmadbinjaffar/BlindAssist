"""Persistent user settings (settings.json in the project root).

The keyboard controls (+/- confidence, [ ] speech rate, B beep) change values
for the *current* session. This module makes such changes survive restarts:

* :func:`apply_to_config` - fold the stored file into a freshly loaded
  :class:`~blindassist.config.Config` **before** CLI parsing, so an explicit
  command-line flag always wins: ``config defaults < settings.json < CLI``.
* :func:`save` / :func:`snapshot` - write the current values back after every
  keyboard change (tiny file, near-zero cost) and at app shutdown.

Only a whitelist of keys is understood; unknown keys, corrupt JSON, bad types
and out-of-range numbers are coerced, clamped or dropped *with a warning*
string instead of crashing - the assistant must still start for a blind user
even if a hand-edited settings file is broken.

The file location can be overridden for tests by passing ``path=`` or, for
whole-app tests, by monkeypatching :data:`SETTINGS_PATH`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config

# Default location (project root next to run.bat).
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"

SCHEMA_VERSION = 1

# Sane bounds; intentionally a subset of the app's keyboard clamp ranges so a
# hand-edited value can never put the assistant into a useless state.
_CONF_BOUNDS = (0.05, 0.95)
_RATE_BOUNDS = (60, 300)
_VOLUME_BOUNDS = (0.05, 1.0)

# The keys we persist. Anything else in the file is ignored.
_KEYS = ("conf", "rate", "volume", "beep_enabled", "danger_labels", "language")


def load(path: Path | None = None) -> tuple[dict, list[str]]:
    """Read the settings file. Returns ``(values, warnings)``.

    A missing file is normal (first run) and returns an empty dict. A corrupt
    file or a file that is not a JSON object yields warnings and defaults -
    never an exception and never a refusal to start.
    """
    file = Path(path) if path is not None else SETTINGS_PATH
    warnings: list[str] = []
    if not file.exists():
        return {}, warnings
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, [f"settings file unreadable, using defaults: {exc}"]
    if not isinstance(raw, dict):
        return {}, [
            f"settings file ignored: expected an object, got {type(raw).__name__}"
        ]
    stored: dict = {}
    for key in (*_KEYS, "schema_version"):
        if key in raw:
            stored[key] = raw[key]
    unknown = set(raw) - set(_KEYS) - {"schema_version"}
    if unknown:
        warnings.append("ignored unknown settings keys: " + ", ".join(sorted(unknown)))
    return stored, warnings


def save(values: dict, path: Path | None = None) -> None:
    """Write current settings (creating the file; overwrites atomically-ish)."""
    file = Path(path) if path is not None else SETTINGS_PATH
    payload = dict(values)
    payload["schema_version"] = SCHEMA_VERSION
    tmp = file.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    tmp.replace(file)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return min(max(value, low), high)


def _number(
    stored: dict, key: str, warnings: list[str], bounds: tuple[float, float]
) -> float | None:
    raw = stored.get(key)
    try:
        value = float(raw)  # accepts "0.3" as well as 0.3
    except (TypeError, ValueError):
        warnings.append(f"ignored non-numeric setting {key}={raw!r}")
        return None
    clamped = _clamp(value, bounds)
    if clamped != value:
        warnings.append(f"{key}={value!r} out of range; clamped to {clamped}")
    return clamped


def apply_to_config(cfg: Config, stored: dict, warnings: list[str] | None = None):
    """Fold stored settings into ``cfg``. Returns the warnings list."""
    warns = warnings if warnings is not None else []
    if not stored:
        return warns

    conf = _number(stored, "conf", warns, _CONF_BOUNDS)
    if conf is not None:
        cfg.vision.confidence = conf
    rate = _number(stored, "rate", warns, _RATE_BOUNDS)
    if rate is not None:
        cfg.speech.rate_wpm = int(rate)
    volume = _number(stored, "volume", warns, _VOLUME_BOUNDS)
    if volume is not None:
        cfg.speech.volume = volume

    beep = stored.get("beep_enabled")
    if isinstance(beep, bool):
        cfg.beep.enabled = beep
    elif beep is not None:
        warns.append(f"ignored non-boolean setting beep_enabled={beep!r}")

    labels = stored.get("danger_labels")
    if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
        cfg.danger.critical_labels = tuple(labels)
    elif labels is not None:
        warns.append("ignored setting danger_labels: expected a list of class names")

    language = stored.get("language")
    if isinstance(language, str) and language.strip():
        cfg.language.language = language.strip().lower()
    elif language is not None:
        warns.append(f"ignored non-string setting language={language!r}")

    return warns


def snapshot(cfg: Config) -> dict:
    """The persisted image of the values the user can change."""
    return {
        "conf": float(cfg.vision.confidence),
        "rate": int(cfg.speech.rate_wpm),
        "volume": float(cfg.speech.volume),
        "beep_enabled": bool(cfg.beep.enabled),
        "danger_labels": list(cfg.danger.critical_labels),
        "language": cfg.language.language,
    }
