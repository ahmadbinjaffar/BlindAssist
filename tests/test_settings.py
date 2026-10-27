"""settings.json persistence: load/save, coercion, clamping, config merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blindassist import settings as settings_store
from blindassist.config import load_config


def write_raw(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def test_missing_file_is_empty_and_warning_free(tmp_path):
    stored, warns = settings_store.load(tmp_path / "nope.json")
    assert stored == {}
    assert warns == []


def test_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    values = {
        "conf": 0.31,
        "rate": 150,
        "volume": 0.8,
        "beep_enabled": False,
        "danger_labels": ["person", "dog"],
        "language": "ur",
    }
    settings_store.save(values, path)
    stored, warns = settings_store.load(path)
    assert warns == []
    for key, value in values.items():
        assert stored[key] == value


def test_schema_version_is_written(tmp_path):
    path = tmp_path / "settings.json"
    settings_store.save({"conf": 0.5}, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == settings_store.SCHEMA_VERSION


def test_corrupt_file_gives_defaults_with_warning(tmp_path):
    path = tmp_path / "settings.json"
    write_raw(path, "{not json")
    stored, warns = settings_store.load(path)
    assert stored == {}
    assert warns and "unreadable" in warns[0]


def test_non_object_file_is_ignored_with_warning(tmp_path):
    path = tmp_path / "settings.json"
    write_raw(path, "[1, 2, 3]")
    stored, warns = settings_store.load(path)
    assert stored == {}
    assert warns and "expected an object" in warns[0]


def test_unknown_keys_are_dropped_with_warning(tmp_path):
    path = tmp_path / "settings.json"
    write_raw(path, {"conf": 0.5, "wave_speed": 88})
    stored, warns = settings_store.load(path)
    assert stored == {"conf": 0.5}
    assert any("wave_speed" in w for w in warns)


# ---------------------------------------------------------------------------
# apply_to_config
# ---------------------------------------------------------------------------
def test_apply_sets_every_documented_value():
    cfg = load_config()
    settings_store.apply_to_config(
        cfg,
        {
            "conf": 0.31,
            "rate": 150,
            "volume": 0.8,
            "beep_enabled": False,
            "danger_labels": ["person", "dog"],
            "language": "ur",
        },
    )
    assert cfg.vision.confidence == 0.31
    assert cfg.speech.rate_wpm == 150
    assert cfg.speech.volume == 0.8
    assert cfg.beep.enabled is False
    assert cfg.danger.critical_labels == ("person", "dog")
    assert cfg.language.language == "ur"


def test_numeric_strings_are_coerced():
    cfg = load_config()
    settings_store.apply_to_config(cfg, {"conf": "0.40", "rate": "180"})
    assert cfg.vision.confidence == 0.40
    assert cfg.speech.rate_wpm == 180


def test_out_of_range_values_are_clamped():
    cfg = load_config()
    warns: list[str] = []
    settings_store.apply_to_config(
        cfg,
        {"conf": 5.0, "rate": 10000, "volume": -3},
        warns,
    )
    assert cfg.vision.confidence == 0.95
    assert cfg.speech.rate_wpm == 300
    assert cfg.speech.volume == 0.05
    assert any("clamped" in w for w in warns)


def test_bad_types_are_ignored_with_warning():
    cfg = load_config()
    warns: list[str] = []
    settings_store.apply_to_config(
        cfg,
        {
            "conf": {"a": 1},
            "beep_enabled": "maybe",
            "danger_labels": [1, 2],
            "language": 7,
        },
        warns,
    )
    assert any("conf" in w for w in warns)
    assert any("beep_enabled" in w for w in warns)
    assert any("danger_labels" in w for w in warns)
    assert any("language" in w for w in warns)
    # Everything keeps its default.
    assert cfg.vision.confidence == pytest.approx(0.45)
    assert cfg.beep.enabled is True
    assert cfg.danger.critical_labels != (1, 2)
    assert cfg.language.language == "en"


def test_empty_stored_leaves_defaults():
    cfg = load_config()
    warns: list[str] = []
    settings_store.apply_to_config(cfg, {}, warns)
    assert cfg.vision.confidence == pytest.approx(0.45)
    assert warns == []


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------
def test_snapshot_contains_all_keys():
    cfg = load_config()
    snap = settings_store.snapshot(cfg)
    assert set(snap) == {
        "conf",
        "rate",
        "volume",
        "beep_enabled",
        "danger_labels",
        "language",
    }
    assert snap["danger_labels"] == list(cfg.danger.critical_labels)


# ---------------------------------------------------------------------------
# App integration: keyboard changes are saved and reloaded (the core promise)
# ---------------------------------------------------------------------------
def make_app(tmp_path, monkeypatch, argv):
    from blindassist.app import BlindAssistApp, build_arg_parser

    monkeypatch.setattr(settings_store, "SETTINGS_PATH", tmp_path / "settings.json")
    app = BlindAssistApp(build_arg_parser().parse_args(argv))
    app.settings_path = tmp_path / "settings.json"  # persistence target
    return app


def test_keyboard_changes_are_saved_and_reloaded(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, [])
    app._handle_key(ord("+"))  # 0.45 -> 0.50
    assert (tmp_path / "settings.json").exists()
    app2 = make_app(tmp_path, monkeypatch, [])
    # ...and reloaded at next start (CLI wins over stored settings).
    assert app2.confidence == pytest.approx(0.50)


def test_cli_flag_wins_over_stored_setting(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, [])
    app._handle_key(ord("+"))
    app3 = make_app(tmp_path, monkeypatch, ["--conf", "0.30"])
    assert app3.confidence == pytest.approx(0.30)  # explicit flag overrides
    stored, _ = settings_store.load(tmp_path / "settings.json")
    assert stored["conf"] == pytest.approx(0.50)  # flag was NOT persisted


def test_disabled_settings_store_writes_nothing(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, [])
    app.cfg.settings.enabled = False
    app._handle_key(ord("+"))
    assert list(tmp_path.glob("*.json*")) == []
