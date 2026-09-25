from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

from vi.schemas import Box


# Classes that become tubes. Furniture, fixtures and appliances are scene-card assets handled by
# heartbeats and zones, never tracked as moving objects (E-DET-09).
TUBE_CLASSES = {"person", "dog", "cat", "bicycle", "car", "motorcycle", "bus", "truck",
                "backpack", "handbag", "suitcase", "umbrella"}


def is_tube_class(label: str) -> bool:
    return label in TUBE_CLASSES


class Detection(BaseModel):
    box: Box
    class_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: list[float] | None = None   # R14: ReID alongside the box
    truncated: bool = False                # E-DET-02: box touches the frame border
    roi_truncated: bool = False            # E-DET-10: box touches its crop border (partial view of the object)
    origin: Literal["roi", "full", "unknown"] = "unknown"   # which crop produced it (heartbeat = full)


class Detector(Protocol):
    """Ring 1 interface. Batched over ROI crops from many cameras (R12)."""

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]: ...
