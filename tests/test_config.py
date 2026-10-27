"""Tests for configuration defaults."""

from __future__ import annotations

from blindassist.config import (
    DEFAULT_PRIORITY_LABELS,
    PositionConfig,
    SpeechConfig,
    load_config,
)


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.vision.camera_index == 0
    assert cfg.vision.confidence > 0
    assert cfg.model_path.name == "yolov8n.pt"
    assert 0 in cfg.class_whitelist  # person


def test_position_distance_thresholds_ordered():
    pc = PositionConfig()
    assert pc.very_close_area_ratio > pc.near_area_ratio > 0


def test_speech_pacing_defaults():
    sc = SpeechConfig()
    assert sc.cooldown_s == 3.0
    assert sc.max_items_per_message == 3
    assert sc.priority_labels[0] == "person"


def test_priority_labels_start_with_most_urgent():
    assert DEFAULT_PRIORITY_LABELS[0] == "person"
    # vehicles and dog are present
    for label in ("car", "bus", "truck", "bicycle", "motorcycle", "dog"):
        assert label in DEFAULT_PRIORITY_LABELS
