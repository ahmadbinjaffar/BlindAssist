"""BlindAssist - real-time object detection assistant for blind users.

Pipeline: webcam frame -> YOLO detection -> IoU tracking -> position/distance
-> rate-limited text-to-speech.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config

__all__ = ["Config", "__version__", "load_config"]
