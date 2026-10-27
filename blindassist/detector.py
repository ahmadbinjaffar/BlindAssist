"""YOLOv8n object detector wrapper.

Loads an Ultralytics YOLOv8-nano checkpoint **once** and runs inference on BGR
frames (the format OpenCV returns). Results are converted into lightweight
:class:`Detection` objects (plain floats, no ultralytics types) so the rest of
the pipeline stays dependency-light and easy to test.

Design notes
------------
* **Model**: ``yolov8n.pt`` - the nano variant, pretrained on the 80-class COCO
  dataset. Chosen because it is the smallest and fastest YOLOv8 model and runs
  in real time on a CPU (see ``docs/model_notes.md``).
* **Weights location**: downloaded once into ``assets/yolov8n.pt``; every later
  load is fully offline.
* **Device**: CPU by default, automatically switches to CUDA GPU when available.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import VisionConfig
from .position import Box

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from ultralytics import YOLO

# Canonical asset name published by ultralytics (COCO-pretrained).
YOLOV8N_WEIGHTS = "yolov8n.pt"


@dataclass(frozen=True)
class Detection:
    """A single detected object (pixel coordinates, xyxy format).

    Constructed as ``Detection(label, confidence, x1, y1, x2, y2)``. A ``box``
    property exposes the same data as a :class:`blindassist.position.Box` for
    the tracker / positioner.
    """

    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = -1

    @property
    def box(self) -> Box:
        return Box(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detector:
    """Thin wrapper around an Ultralytics YOLOv8n model."""

    def __init__(
        self,
        cfg: VisionConfig,
        model_path: Path,
        class_whitelist: set[int] | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_path = Path(model_path)
        self.class_whitelist = class_whitelist
        # None => decide automatically (CPU, or CUDA GPU if present).
        self._requested_device = device
        self.device: str = "cpu"
        self._model: YOLO | None = None
        self.names: dict[int, str] = {}

    # -- lifecycle ---------------------------------------------------------
    def _pick_device(self) -> str:
        if self._requested_device:
            return self._requested_device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _download_weights(self) -> None:
        """Fetch the COCO-pretrained weights into ``self.model_path`` (once)."""
        from ultralytics import YOLO

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        # Instantiating by bare name makes ultralytics download the asset into
        # the current working directory (confirmed behaviour).
        YOLO(YOLOV8N_WEIGHTS)
        src = Path.cwd() / YOLOV8N_WEIGHTS
        if src.exists() and src.resolve() != self.model_path.resolve():
            shutil.move(str(src), str(self.model_path))

    def load(self) -> None:
        """Load the model weights and resolve the compute device."""
        if self._model is not None:
            return
        from ultralytics import YOLO

        self.device = self._pick_device()
        if not self.model_path.exists():
            self._download_weights()
        self._model = YOLO(str(self.model_path))
        self.names = dict(self._model.names)

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self.load()
        if self._model is None:
            # A real exception, not `assert`: assertions are stripped under
            # `python -O`, which would turn this into a confusing AttributeError.
            raise RuntimeError(
                f"model weights could not be loaded from {self.model_path}"
            )
        return self._model

    # -- inference ---------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame and return filtered detections."""
        results = self.model.predict(
            source=frame,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            imgsz=self.cfg.inference_size,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        r = results[0]
        if r.boxes is None:
            return detections
        for b in r.boxes:
            cls_id = int(b.cls[0])
            if self.class_whitelist is not None and cls_id not in self.class_whitelist:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            detections.append(
                Detection(
                    label=str(self.names.get(cls_id, str(cls_id))),
                    confidence=float(b.conf[0]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_id=cls_id,
                )
            )
        return detections

    def detect_many(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        return [self.detect(f) for f in frames]
