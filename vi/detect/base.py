from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from vi.schemas import Box


class Detection(BaseModel):
    box: Box
    class_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: list[float] | None = None   # R14: ReID alongside the box


class Detector(Protocol):
    """Ring 1 interface. Batched over ROI crops from many cameras (R12)."""

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]: ...
