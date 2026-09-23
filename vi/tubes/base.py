from __future__ import annotations

from typing import Protocol

from vi.detect import Detection
from vi.schemas import Tube


class Tracker(Protocol):
    """Ring 2 per-camera interface. update() returns the full set of live tubes plus the
    tubes that closed on this tick (state exited/lost/dead)."""

    def update(self, detections: list[Detection], t_ms: int,
               det_source: str = "detector") -> tuple[list[Tube], list[Tube]]: ...
