from __future__ import annotations

import hashlib
from pathlib import Path

from vi.schemas import CamTime, EnrichmentPatch, Event, Provenance, Tick, Tube
from vi.schemas.episode import (CastMember, EpisodeClose, EpisodeHeader, EpisodeStatus,
                                EventRecord, PatchRecord, TickRecord, TubeRecord, episode_record_adapter)


def episode_id_for(tile_id: str, t0_corrected_ms: int) -> str:
    """E-STO-02: deterministic id => a retried open() never creates a second episode."""
    return "ep_" + hashlib.sha1(f"{tile_id}|{t0_corrected_ms}".encode()).hexdigest()[:16]


class EpisodeWriter:
    """Append-only JSONL, one file per episode. Records are idempotent by id (E-STO-02);
    patches after close are accepted and appended (E-STO-01)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, EpisodeHeader] = {}
        self._closed: set[str] = set()
        self._seen: dict[str, set[str]] = {}

    def path(self, episode_id: str) -> Path:
        return self.root / f"{episode_id}.jsonl"

    def _append(self, episode_id: str, record_key: str, record) -> bool:
        seen = self._seen.setdefault(episode_id, set())
        if record_key in seen:
            return False
        seen.add(record_key)
        with self.path(episode_id).open("a") as f:
            f.write(record.model_dump_json() + "\n")
        return True

    def _resume(self, episode_id: str) -> None:
        """A writer restarted after a crash must not append a second header or duplicate ticks:
        rebuild the idempotency keys from what is already on disk."""
        path = self.path(episode_id)
        if not path.exists() or episode_id in self._seen:
            return
        seen = self._seen.setdefault(episode_id, set())
        for rec in self.read(path):
            k = rec.kind
            if k == "header": seen.add("header")
            elif k == "tick": seen.add(f"tick:{rec.tick.camera_id}:{rec.tick.tick_index}")
            elif k == "event": seen.add(f"event:{rec.event.event_id}")
            elif k == "patch": seen.add(f"patch:{rec.patch.patch_id}")
            elif k == "tube": seen.add(f"tube:{rec.tube.tube_id}")
            elif k == "close": seen.add("close"); self._closed.add(episode_id)

    def open(self, tile_id: str, camera_ids: list[str], t0: CamTime, provenance: Provenance) -> str:
        eid = episode_id_for(tile_id, t0.corrected_ms())
        self._resume(eid)
        if eid in self._open or eid in self._closed:
            return eid
        hdr = EpisodeHeader(episode_id=eid, tile_id=tile_id, camera_ids=camera_ids, t0=t0, provenance=provenance)
        self._open[eid] = hdr
        self._append(eid, "header", hdr)
        return eid

    def write_tick(self, episode_id: str, tick: Tick) -> bool:
        return self._append(episode_id, f"tick:{tick.camera_id}:{tick.tick_index}", TickRecord(episode_id=episode_id, tick=tick))

    def write_event(self, episode_id: str, event: Event) -> bool:
        event.episode_id = episode_id
        return self._append(episode_id, f"event:{event.event_id}", EventRecord(episode_id=episode_id, event=event))

    def write_patch(self, episode_id: str, patch: EnrichmentPatch) -> bool:
        return self._append(episode_id, f"patch:{patch.patch_id}", PatchRecord(episode_id=episode_id, patch=patch))

    def write_tube(self, episode_id: str, tube: Tube) -> bool:
        return self._append(episode_id, f"tube:{tube.tube_id}", TubeRecord(episode_id=episode_id, tube=tube))

    def close(self, episode_id: str, t1: CamTime, status: EpisodeStatus, cast: list[CastMember]) -> bool:
        ok = self._append(episode_id, "close", EpisodeClose(episode_id=episode_id, t1=t1, status=status, cast=cast))
        self._open.pop(episode_id, None)
        self._closed.add(episode_id)
        return ok

    @staticmethod
    def read(path: str | Path):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                yield episode_record_adapter.validate_json(line)


def should_soft_cut(cast_prev: set[str], cast_now: set[str], duration_ms: int,
                    max_duration_ms: int = 30 * 60_000, churn_thr: float = 0.6) -> bool:
    """E-EVT-07: a busy tile never goes quiet, so an episode is cut when the cast has
    mostly turned over or the episode exceeds max duration."""
    if duration_ms >= max_duration_ms:
        return True
    if not cast_prev:
        return False
    churn = 1.0 - len(cast_prev & cast_now) / len(cast_prev | cast_now)
    return churn >= churn_thr
