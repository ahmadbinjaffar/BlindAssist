"""Tests for the detector wrapper (no model download / inference)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blindassist.config import VisionConfig
from blindassist.detector import YOLOV8N_WEIGHTS, Detection, Detector
from blindassist.position import Box


def test_detection_flat_construction():
    # Detection(label, confidence, x1, y1, x2, y2)
    det = Detection("person", 0.9, 10.0, 20.0, 110.0, 220.0, class_id=0)
    assert det.label == "person"
    assert det.confidence == 0.9
    assert det.xyxy == (10.0, 20.0, 110.0, 220.0)


def test_detection_box_property():
    det = Detection("dog", 0.8, 0.0, 0.0, 10.0, 10.0)
    b = det.box
    assert isinstance(b, Box)
    assert b.width == 10.0 and b.height == 10.0 and b.area == 100.0


def test_detection_defaults_class_id():
    det = Detection("x", 0.5, 0, 0, 1, 1)
    assert det.class_id == -1


def test_detector_starts_unloaded():
    d = Detector(VisionConfig(), Path("assets/does_not_exist.pt"))
    assert d._model is None
    assert d.names == {}


def test_default_weights_name_is_yolov8n():
    assert YOLOV8N_WEIGHTS == "yolov8n.pt"


def test_device_defaults_to_cpu_without_gpu():
    d = Detector(VisionConfig(), Path("x.pt"))
    # _pick_device with no request should return a valid cpu/cuda string.
    dev = d._pick_device()
    assert dev in ("cpu",) or dev.startswith("cuda")


def test_whitelist_is_stored():
    d = Detector(VisionConfig(), Path("x.pt"), class_whitelist={0, 2})
    assert d.class_whitelist == {0, 2}


# -- inference (fake ultralytics model: no weights, no network) -------------
class _FakeBox:
    """Mimics the sliceable ultralytics box tensor: ``b.cls[0]``, ``b.xyxy[0].tolist()``."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = np.array([xyxy], dtype=float)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, results, names):
        self._results = results
        self.names = names
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        return self._results


NAMES = {0: "person", 56: "chair", 39: "bottle"}


def _detector_with(model, whitelist=None, cfg=None):
    d = Detector(cfg or VisionConfig(), Path("x.pt"), class_whitelist=whitelist)
    d._model = model
    d.names = NAMES
    return d


def test_detect_converts_boxes_to_detections():
    model = _FakeModel(
        [
            _FakeResult(
                [
                    _FakeBox(0, 0.91, [10.0, 20.0, 110.0, 220.0]),
                    _FakeBox(56, 0.55, [0.0, 0.0, 50.0, 50.0]),
                ]
            )
        ],
        NAMES,
    )
    d = _detector_with(model)
    out = d.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert [x.label for x in out] == ["person", "chair"]
    assert out[0].confidence == 0.91
    assert out[0].class_id == 0
    assert out[0].xyxy == (10.0, 20.0, 110.0, 220.0)
    assert out[1].box.area == 2500.0


def test_detect_passes_config_thresholds_to_the_model():
    cfg = VisionConfig()
    cfg.confidence, cfg.iou, cfg.inference_size = 0.33, 0.41, 320
    model = _FakeModel([_FakeResult([])], NAMES)
    _detector_with(model, cfg=cfg).detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert model.last_kwargs["conf"] == 0.33
    assert model.last_kwargs["iou"] == 0.41
    assert model.last_kwargs["imgsz"] == 320


def test_detect_filters_classes_outside_the_whitelist():
    model = _FakeModel(
        [
            _FakeResult(
                [
                    _FakeBox(0, 0.9, [0, 0, 10, 10]),
                    _FakeBox(39, 0.8, [0, 0, 10, 10]),
                ]
            )
        ],
        NAMES,
    )
    d = _detector_with(model, whitelist={0})  # bottles are not announced
    out = d.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert [x.label for x in out] == ["person"]


def test_detect_with_no_whitelist_keeps_everything():
    model = _FakeModel(
        [
            _FakeResult(
                [
                    _FakeBox(0, 0.9, [0, 0, 10, 10]),
                    _FakeBox(39, 0.8, [0, 0, 10, 10]),
                ]
            )
        ],
        NAMES,
    )
    d = _detector_with(model, whitelist=None)
    assert len(d.detect(np.zeros((10, 10, 3), dtype=np.uint8))) == 2


def test_detect_returns_empty_for_no_results():
    assert (
        _detector_with(_FakeModel([], NAMES)).detect(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )
        == []
    )


def test_detect_returns_empty_when_boxes_is_none():
    assert (
        _detector_with(_FakeModel([_FakeResult(None)], NAMES)).detect(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )
        == []
    )


def test_detect_many_processes_every_frame():
    model = _FakeModel([_FakeResult([_FakeBox(0, 0.9, [0, 0, 10, 10])])], NAMES)
    d = _detector_with(model)
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    out = d.detect_many(frames)
    assert len(out) == 3
    assert all(len(batch) == 1 for batch in out)


def test_explicit_device_request_is_honoured():
    d = Detector(VisionConfig(), Path("x.pt"), device="cpu")
    assert d._pick_device() == "cpu"


# -- load() with a stubbed ultralytics module --------------------------------
def test_load_uses_existing_weights_without_downloading(tmp_path, monkeypatch):
    import sys
    import types

    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"stub")

    calls = {"download": 0}
    fake_model = _FakeModel([], NAMES)

    def _YOLO(_path):
        return fake_model

    stub = types.ModuleType("ultralytics")
    stub.YOLO = _YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", stub)

    d = Detector(VisionConfig(), weights, device="cpu")

    def _no_download():  # pragma: no cover - must never run
        calls["download"] += 1

    d._download_weights = _no_download
    d.load()
    assert calls["download"] == 0
    assert d.names == NAMES
    assert d.device == "cpu"
    d.load()  # second call is a no-op
    assert d.model is fake_model
