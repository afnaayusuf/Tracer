from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .common import Box, Modality


class Asset(BaseModel):
    name: str
    box: Box
    mask_ref: str | None = None
    is_asset_home_candidate: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class Actuator(BaseModel):
    """controls starts as None: the KB records the unknown explicitly (E-KB-03 seed)."""

    kind: Literal["switch_panel", "door", "window", "appliance", "tap", "other"]
    box: Box
    controls: list[str] | None = None
    controls_status: Literal["unknown", "hypothesis", "fact"] = "unknown"
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ReflectiveSurface(BaseModel):
    """E-GATE-06 / E-DET-04: masks the tracker uses to suppress ghost tubes; also the
    designed path for reflection-based inspect."""

    box: Box
    kind: Literal["mirror", "glass", "metal", "screen", "water", "other"]
    reflects_zone_hint: str | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ExitZone(BaseModel):
    box: Box
    direction_hint: str | None = None
    leads_to_tile_hint: str | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class CameraPose(BaseModel):
    height_m: float | None = None
    tilt_deg: float | None = None
    fov_class: Literal["narrow", "normal", "wide", "fisheye"] | None = None  # E-DET-07
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class SceneCard(BaseModel):
    """Calibration output for one camera in one lighting regime. Nothing here is a
    fact; every item is a hypothesis until confirmed by the walk or observation."""

    camera_id: str
    regime: Modality
    tile_hypothesis: str
    tile_name_proposal: str
    tile_confidence: float = Field(0.0, ge=0.0, le=1.0)
    assets: list[Asset] = Field(default_factory=list)
    actuators: list[Actuator] = Field(default_factory=list)
    light_sources: list[Box] = Field(default_factory=list)
    exits: list[ExitZone] = Field(default_factory=list)
    reflective_surfaces: list[ReflectiveSurface] = Field(default_factory=list)
    blind_regions: list[Box] = Field(default_factory=list)
    media_zones: list[Box] = Field(default_factory=list)   # E-DET-05: TVs, posters, photos
    floor_polygon: list[tuple[float, float]] = Field(default_factory=list)
    ground_points: list[tuple[float, float]] = Field(default_factory=list)
    camera_pose: CameraPose = Field(default_factory=CameraPose)
    model_used: str
    produced_at_ms: int
