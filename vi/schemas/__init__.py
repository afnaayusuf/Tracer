"""Schema contract. Every ring reads and writes only these models.
Rule: a field that a model could hallucinate is either an enum, nullable with a reason, or carries a confidence."""
from .common import SCHEMA_VERSION, Box, CamTime, FloorPoint, Modality, Provenance
from .tube import Attributes, Color, EnrichmentPatch, Tone, Tube, TubeSnapshot, TubeState, effective_confidence
from .event import Event, EventType
from .tick import Tick
from .episode import CastMember, EpisodeClose, EpisodeHeader, EpisodeRecord, EpisodeStatus
from .scene_card import SceneCard
from .fact import Fact, FactStatus
from .contact_sheet import CellResult, ContactSheetResult, parse_or_fail

__all__ = [n for n in dir() if not n.startswith("_")]
