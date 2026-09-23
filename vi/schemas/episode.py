from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from .common import CamTime, Provenance
from .event import Event
from .tick import Tick
from .tube import EnrichmentPatch


class EpisodeStatus(str, Enum):
    open = "open"
    closed = "closed"        # tile went quiet
    soft_cut = "soft_cut"    # E-EVT-07: cast churn or max duration
    truncated = "truncated"  # E-ING-02: signal lost


class CastMember(BaseModel):
    entity_id: str | None = None
    tube_ids: list[str]
    class_label: str
    named: str | None = None
    best_keyframe_ref: str | None = None


class EpisodeHeader(BaseModel):
    kind: Literal["header"] = "header"
    episode_id: str
    tile_id: str
    camera_ids: list[str]
    t0: CamTime
    provenance: Provenance


class TickRecord(BaseModel):
    kind: Literal["tick"] = "tick"
    episode_id: str
    tick: Tick


class EventRecord(BaseModel):
    kind: Literal["event"] = "event"
    episode_id: str
    event: Event


class PatchRecord(BaseModel):
    """E-STO-01: patches may arrive after close; the file is append-only."""

    kind: Literal["patch"] = "patch"
    episode_id: str
    patch: EnrichmentPatch


class EpisodeClose(BaseModel):
    kind: Literal["close"] = "close"
    episode_id: str
    t1: CamTime
    status: EpisodeStatus
    cast: list[CastMember] = Field(default_factory=list)


EpisodeRecord = Annotated[
    Union[EpisodeHeader, TickRecord, EventRecord, PatchRecord, EpisodeClose],
    Field(discriminator="kind"),
]
episode_record_adapter: TypeAdapter = TypeAdapter(EpisodeRecord)
