from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from vi.schemas import Box


class MotionBlob(BaseModel):
    box: Box
    energy: float = Field(ge=0.0)
    coherence: float = Field(0.0, ge=0.0, le=1.0)   # MV-field agreement; 0 for frame-diff


class GateResult(BaseModel):
    camera_id: str
    t_ms: int
    blobs: list[MotionBlob]
    global_motion: bool = False       # E-GATE-04: PTZ/shake => suppress blobs
    luma_step: bool = False           # E-GATE-02: scene-state change, not object motion
    mean_luma: float = 0.0
    noise_floor: float = 0.0
    active_fraction: float = 0.0


class Gate(Protocol):
    """Ring 0 interface. Implementations: FrameDiffGate (slice), MVGate (week 3)."""

    def update(self, frame_gray: np.ndarray, t_ms: int) -> GateResult: ...
