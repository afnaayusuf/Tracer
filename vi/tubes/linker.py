"""Appearance-based re-linking of tubes into entities within one camera (E-TUBE-04), with a
drifting gallery per entity (E-TUBE-08). Cross-camera fusion reuses the same gallery format.

Rules:
  * a newborn tube is compared against entities whose last tube closed within max_gap_ms and
    whose last position is within max_jump_px; tubes that closed through an exit zone need a
    stricter similarity (exit zones may be placeholders, and people step out and back);
  * cosine similarity >= sim_thr links it to that entity, else it starts a new one;
  * each entity keeps an EMA embedding plus up to `exemplars` raw embeddings; refreshes happen
    on confident active ticks so the gallery follows jacket-on/jacket-off;
  * a link is emitted as an Event(relink) with both tube ids and the similarity, so the agent
    can cite it and a wrong link can be audited.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from vi.schemas import CamTime, Event, EventType, Tube, TubeState


@dataclass
class _Entity:
    entity_id: str
    tube_ids: list[str]
    ema: np.ndarray
    exemplars: list[np.ndarray] = field(default_factory=list)
    aux: np.ndarray | None = None
    last_box_center: tuple[float, float] = (0.0, 0.0)
    last_box_h: float = 1.0
    born_ms: int = 0
    last_active_ms: int = 0
    lost_at_ms: int | None = None      # set when its tube closed (lost/exited) OR went occluded; None while seen
    closed_as_exit: bool = False
    live_tube_id: str | None = None    # an occluded-but-live tube: linking a newborn to this entity absorbs it


class TubeLinker:
    def __init__(self, camera_id: str, sim_thr: float = 0.88, max_gap_ms: int = 30_000,
                 max_jump_px: float = 400.0, ema_alpha: float = 0.3, exemplars: int = 5,
                 exited_sim_thr: float = 0.90, near_sim_thr: float = 0.85, near_gap_ms: int = 5000,
                 near_jump_px: float = 200.0, aux_thr: float = 0.80):
        self.camera_id = camera_id
        self.sim_thr = sim_thr
        # a tube reappearing within a few seconds and a couple of body-widths of where one vanished
        # is the same person unless appearance says otherwise: the bar drops to near_sim_thr there
        self.near_sim_thr, self.near_gap_ms, self.near_jump_px = near_sim_thr, near_gap_ms, near_jump_px
        # a second, cheap appearance signal (colour histogram) must agree; it stops a generic image
        # embedding from joining a green hi-vis vest to an orange one at the same spot
        self.aux_thr = aux_thr
        self.exited_sim_thr = exited_sim_thr   # placeholder exit zones misclassify lost as exited; allow with more evidence
        self.max_gap_ms = max_gap_ms
        self.max_jump_px = max_jump_px
        self.ema_alpha = ema_alpha
        self.n_exemplars = exemplars
        self._entities: dict[str, _Entity] = {}
        self._tube_entity: dict[str, str] = {}
        self._seq = 0
        self.relinks = 0
        self.merges = 0
        self.max_coexist_ms = 1000
        self.absorbed: list[str] = []      # ghost tubes the tracker should drop (read and clear each tick)

    # ---------------------------------------------------------------- helpers
    def _new_entity(self, tube: Tube, emb: np.ndarray, aux: np.ndarray | None = None) -> _Entity:
        self._seq += 1
        ent = _Entity(entity_id=f"{self.camera_id}:E{self._seq}", tube_ids=[tube.tube_id], ema=emb.copy(),
                      exemplars=[emb.copy()], last_box_center=_center(tube), last_box_h=tube.box.height,
                      aux=None if aux is None else aux.copy(), born_ms=tube.born.corrected_ms(),
                      last_active_ms=tube.last_seen.corrected_ms())
        self._entities[ent.entity_id] = ent
        self._tube_entity[tube.tube_id] = ent.entity_id
        return ent

    @staticmethod
    def _sim(ent: _Entity, emb: np.ndarray) -> float:
        best = float(ent.ema @ emb)
        for x in ent.exemplars:
            best = max(best, float(x @ emb))
        return best

    def _candidates(self, tube: Tube, t_ms: int) -> list[_Entity]:
        cx, cy = _center(tube)
        out = []
        for e in self._entities.values():
            if e.lost_at_ms is None or t_ms - e.lost_at_ms > self.max_gap_ms:
                continue
            if ((cx - e.last_box_center[0]) ** 2 + (cy - e.last_box_center[1]) ** 2) ** 0.5 > self.max_jump_px:
                continue
            out.append(e)
        return out

    # ---------------------------------------------------------------- API
    def _thr(self, ent: _Entity, tube: Tube, t_ms: int) -> float:
        if ent.closed_as_exit:
            return self.exited_sim_thr
        cx, cy = _center(tube)
        dist = ((cx - ent.last_box_center[0]) ** 2 + (cy - ent.last_box_center[1]) ** 2) ** 0.5
        gap = t_ms - (ent.lost_at_ms or t_ms)
        return self.near_sim_thr if (gap <= self.near_gap_ms and dist <= self.near_jump_px) else self.sim_thr

    def on_birth(self, tube: Tube, emb: np.ndarray, t_ms: int, aux: np.ndarray | None = None) -> Event | None:
        cands = self._candidates(tube, t_ms)
        if aux is not None:   # second-signal gate first: candidates whose colour disagrees are out
            cands = [e for e in cands if e.aux is None or float(e.aux @ aux) >= self.aux_thr]
        if cands:
            best = max(cands, key=lambda e: self._sim(e, emb))
            sim = self._sim(best, emb)
            thr = self._thr(best, tube, t_ms)
            if sim >= thr:
                prev = best.tube_ids[-1]
                absorbed = best.live_tube_id                 # the occluded ghost this newborn replaces
                best.tube_ids.append(tube.tube_id)
                best.lost_at_ms, best.live_tube_id = None, None
                self._tube_entity[tube.tube_id] = best.entity_id
                self.on_refresh(tube, emb, aux)
                self.relinks += 1
                self.absorbed.append(absorbed) if absorbed else None
                return Event(event_id="ev_" + hashlib.sha1(f"relink|{prev}|{tube.tube_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[prev, tube.tube_id], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(sim, 3), "threshold": round(thr, 2),
                                      "gap_ms": t_ms - (tube.born.corrected_ms()), "absorbed_tube": absorbed},
                             confidence=min(1.0, sim))
        self._new_entity(tube, emb, aux)
        return None

    def on_refresh(self, tube: Tube, emb: np.ndarray, aux: np.ndarray | None = None) -> None:
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        e.ema = e.ema * (1 - self.ema_alpha) + emb * self.ema_alpha
        e.ema /= max(1e-8, np.linalg.norm(e.ema))
        if aux is not None:
            e.aux = aux.copy() if e.aux is None else (e.aux * 0.7 + aux * 0.3)
            e.aux /= max(1e-8, np.linalg.norm(e.aux))
        e.exemplars.append(emb.copy())
        if len(e.exemplars) > self.n_exemplars:
            e.exemplars.pop(0)
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height

    def on_state(self, tube: Tube, t_ms: int) -> None:
        """Called every tick for live tubes. The moment a tube goes occluded its entity becomes a
        relink candidate: the same person re-detected nearby is a newborn tube the tracker could
        not associate with the drifted prediction (the hard-hat man, session 23)."""
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        if tube.state == TubeState.occluded:
            if e.lost_at_ms is None:
                e.lost_at_ms = tube.occluded_since_ms if tube.occluded_since_ms is not None else t_ms
            e.live_tube_id = tube.tube_id
            e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        elif tube.state == TubeState.active:
            e.last_active_ms = t_ms
            if e.live_tube_id == tube.tube_id:
                e.lost_at_ms, e.live_tube_id = None, None   # seen again: no longer a candidate

    def on_close(self, tube: Tube, t_ms: int) -> Event | None:
        """A tube that dies `lost` next to a live entity it overlapped with for under a second,
        and that looks like it, was a second box on the same body: merge it (mirror of on_birth).
        Long co-existence means two people, whatever the embeddings say."""
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return None
        e = self._entities[eid]
        if tube.state == TubeState.lost and len(e.tube_ids) >= 1:
            cx, cy = _center(tube)
            best, best_sim = None, 0.0
            for o in self._entities.values():
                if o is e or o.lost_at_ms is not None:
                    continue                                              # only live, currently seen entities
                overlap = min(tube.last_seen.corrected_ms(), o.last_active_ms) - max(tube.born.corrected_ms(), o.born_ms)
                dist = ((cx - o.last_box_center[0]) ** 2 + (cy - o.last_box_center[1]) ** 2) ** 0.5
                if overlap > self.max_coexist_ms or dist > self.near_jump_px:
                    continue
                sim = self._sim(o, e.ema)
                if sim > best_sim:
                    best, best_sim = o, sim
            if best is not None and best_sim >= self.near_sim_thr:
                for tid in e.tube_ids:
                    self._tube_entity[tid] = best.entity_id
                best.tube_ids = sorted(set(best.tube_ids + e.tube_ids))
                best.exemplars = (best.exemplars + e.exemplars)[-self.n_exemplars:]
                del self._entities[eid]
                self.merges += 1
                return Event(event_id="ev_" + hashlib.sha1(f"merge|{eid}|{best.entity_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[tube.tube_id, best.tube_ids[-1]], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(best_sim, 3), "kind": "merge_on_death", "merged_entity": eid},
                             confidence=min(1.0, best_sim))
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        e.lost_at_ms = t_ms if tube.state in (TubeState.lost, TubeState.occluded, TubeState.exited) else None
        e.closed_as_exit = tube.state == TubeState.exited
        e.live_tube_id = None
        return None

    def entity_of(self, tube_id: str) -> str | None:
        return self._tube_entity.get(tube_id)

    @property
    def entities(self) -> int:
        return len(self._entities)


def _center(t: Tube) -> tuple[float, float]:
    return ((t.box.x1 + t.box.x2) / 2.0, (t.box.y1 + t.box.y2) / 2.0)
