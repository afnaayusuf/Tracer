from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import CamTime, Modality, Provenance
from .tube import TubeSnapshot


class Tick(BaseModel):
    """Fast-path record. Committed within the latency budget with geometry + event ids
    only; semantic attributes arrive later as EnrichmentPatch (R4)."""

    camera_id: str
    tile_id: str | None = None
    tick_index: int = Field(ge=0)
    t_start: CamTime
    t_end: CamTime
    modality: Modality = Modality.rgb
    tubes: list[TubeSnapshot] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    scene_state: dict[str, Any] = Field(default_factory=dict)   # lit/dark, door states
    gate_energy: float = Field(0.0, ge=0.0)
    provenance: Provenance
