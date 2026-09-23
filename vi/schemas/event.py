from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import CamTime


class EventType(str, Enum):
    # tube events (Ring 3a, deterministic)
    enter_zone = "enter_zone"
    exit_zone = "exit_zone"
    dwell = "dwell"
    loiter = "loiter"
    approach = "approach"
    meet = "meet"
    pickup = "pickup"
    drop = "drop"
    left_behind = "left_behind"
    asset_missing_from_home = "asset_missing_from_home"
    fall = "fall"
    run = "run"
    crowd = "crowd"
    handoff = "handoff"
    impossible_transition = "impossible_transition"
    # scene-state events (Ring 3a, from I-frames / gate)
    illumination_change = "illumination_change"
    door_state_change = "door_state_change"
    appliance_state_change = "appliance_state_change"
    modality_switch = "modality_switch"
    # ingest events
    signal_lost = "signal_lost"
    signal_restored = "signal_restored"
    camera_moved_suspect = "camera_moved_suspect"   # E-KB-01


class Event(BaseModel):
    event_id: str
    type: EventType
    t: CamTime
    camera_id: str
    tile_id: str | None = None
    zone_id: str | None = None
    subject_tube_ids: list[str] = Field(default_factory=list)
    subject_entity_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)      # assets, actuators, other tubes
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Literal["deterministic", "walk", "observed", "user"] = "deterministic"
    episode_id: str | None = None
    dedupe_key: str | None = None    # E-EVT-06: entity+type+time bucket across overlapping cams
