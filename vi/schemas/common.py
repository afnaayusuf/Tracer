from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "0.1.0"


class Modality(str, Enum):
    rgb = "rgb"
    lowlight_rgb = "lowlight_rgb"
    ir = "ir"          # IR-cut filter off, monochrome near-infrared
    thermal = "thermal"

    @property
    def has_color(self) -> bool:
        return self in (Modality.rgb, Modality.lowlight_rgb)


class CamTime(BaseModel):
    """E-ING-01 / E-ING-06: a timestamp is the camera's own UTC reading plus the
    estimated offset to the site reference clock. Fusion and the fact miner use
    corrected_ms(); display converts to local time at the edge of the system only."""

    cam_utc_ms: int
    offset_ms: int = 0
    offset_confidence: float = Field(0.0, ge=0.0, le=1.0)

    def corrected_ms(self) -> int:
        return self.cam_utc_ms + self.offset_ms


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have x2>x1 and y2>y1")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def short_side(self) -> float:
        return min(self.width, self.height)

    def foot_point(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: "Box") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (self.area + other.area - inter)


class FloorPoint(BaseModel):
    """World coordinates on the ground plane. E-TUBE-12: when feet are occluded the
    source is 'fallback_bbox_bottom' and uncertainty_m widens; fusion must read it."""

    x_m: float
    y_m: float
    tile_id: str | None = None
    uncertainty_m: float = Field(0.0, ge=0.0)
    source: Literal["homography", "fallback_bbox_bottom", "manual"] = "homography"


class Provenance(BaseModel):
    """E-STO-04 / E-STO-05: every persisted record knows which schema and which KB
    version compiled it, so a hand-off decision can be reproduced later."""

    schema_version: str = SCHEMA_VERSION
    kb_version: int = Field(ge=0)
    pipeline_git: str | None = None
