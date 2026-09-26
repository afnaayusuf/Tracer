from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .common import Box, CamTime, FloorPoint, Modality


class TubeState(str, Enum):
    born = "born"
    active = "active"
    occluded = "occluded"   # E-TUBE-02: bounded; becomes lost after max_occluded_ms
    exited = "exited"       # left through a known exit zone
    lost = "lost"           # vanished without an exit; candidate for fusion re-link
    dead = "dead"           # closed, no further deltas


class Color(str, Enum):
    black = "black"
    white = "white"
    gray = "gray"
    red = "red"
    orange = "orange"
    yellow = "yellow"
    green = "green"
    blue = "blue"
    purple = "purple"
    pink = "pink"
    brown = "brown"
    multicolor = "multicolor"


class Tone(str, Enum):
    light = "light"
    mid = "mid"
    dark = "dark"


class Attributes(BaseModel):
    """Writer-VLM output for one crop. E-FOV-05: in non-color modalities every color
    field must be None with color_reason='ir_mode'; the grammar and this validator
    both enforce it so a monochrome crop can never yield 'red shirt'."""

    modality: Modality
    top_color: Color | None = None
    bottom_color: Color | None = None
    top_tone: Tone | None = None
    bottom_tone: Tone | None = None
    color_reason: Literal["ir_mode", "not_visible", "low_confidence"] | None = None
    carried_item: str | None = Field(None, max_length=60)
    carried_item_confidence: float = Field(0.0, ge=0.0, le=1.0)
    description: str = Field("", max_length=240)
    enhanced: bool = False            # E-FOV-01: SR or heavy enhancement applied
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ir_rule(self) -> "Attributes":
        if not self.modality.has_color:
            if self.top_color is not None or self.bottom_color is not None:
                raise ValueError("color attributes are not allowed in non-color modality")
            if self.color_reason != "ir_mode":
                raise ValueError("color_reason must be 'ir_mode' in non-color modality")
        return self


class TubeSnapshot(BaseModel):
    """Per-tick view of a tube (what a Tick carries)."""

    tube_id: str
    class_label: str
    state: TubeState
    box: Box
    foot: FloorPoint | None = None
    zone_ids: list[str] = Field(default_factory=list)
    det_confidence: float = Field(0.0, ge=0.0, le=1.0)
    det_source: Literal["detector", "heartbeat", "predicted"] = "detector"


class Tube(BaseModel):
    """Lifecycle record for one object in one camera. entity_id is assigned by fusion."""

    tube_id: str
    camera_id: str
    tile_id: str | None = None
    entity_id: str | None = None
    named: str | None = None          # gallery match, e.g. "Jay"; None => anonymous
    class_label: str
    state: TubeState = TubeState.born
    born: CamTime
    last_seen: CamTime
    box: Box
    foot: FloorPoint | None = None
    zone_ids: list[str] = Field(default_factory=list)
    modality: Modality = Modality.rgb
    keyframe_refs: list[str] = Field(default_factory=list)   # E-FOV-06: persisted at birth
    attributes: Attributes | None = None
    occluded_since_ms: int | None = None
    merge_candidates: list[str] = Field(default_factory=list)  # E-TUBE-01: prefer flag over wrong merge
    reflection_suspect: bool = False                            # E-DET-04
    quality: Literal["ok", "low"] = "ok"   # low: brief, tiny or border-hugging (E-DET-01); counted apart from confirmed people
    quality_reason: str | None = None

    @model_validator(mode="after")
    def _times(self) -> "Tube":
        if self.last_seen.corrected_ms() < self.born.corrected_ms():
            raise ValueError("last_seen precedes born")
        return self


class EnrichmentPatch(BaseModel):
    """Slow-path result applied to a tube after the tick was already committed (R4)."""

    patch_id: str
    tube_id: str
    produced_at_ms: int
    source: Literal["vlm", "pose", "ocr", "siglip", "reid", "sr"]
    payload: dict[str, Any]
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    modality: Modality = Modality.rgb
    enhanced: bool = False
    failed: bool = False               # E-FOV-07: null-fill rather than block


ENHANCED_DISCOUNT = 0.7


def effective_confidence(attr: Attributes) -> float:
    """E-FOV-01: attributes read through SR/enhancement are discounted so that a
    rule needing >=0.8 confidence never fires on an upscaled 60-px crop."""
    return attr.confidence * (ENHANCED_DISCOUNT if attr.enhanced else 1.0)
