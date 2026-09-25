from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from vi.gate.base import GateResult
from vi.schemas import CamTime, Event, EventType, TubeSnapshot

from .zones import Zone

PERSON_CLASSES = {"person"}


def _eid(*parts: object) -> str:
    return "ev_" + hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclass
class _Membership:
    inside_run: int = 0
    outside_run: int = 0
    is_inside: bool = False
    entered_at_ms: int | None = None
    dwell_fired: bool = False


@dataclass
class _AssetState:
    present: bool | None = None
    last_present_ms: int | None = None
    visitors: dict[str, int] = field(default_factory=dict)   # tube_id -> last ms seen inside


class EventCompiler:
    """Ring 3a. Deterministic predicates over tube snapshots, zones and gate results.
    No model calls. One instance per camera.

    E-EVT-01 hysteresis: enter after `enter_ticks` consecutive inside ticks, exit after
             `exit_ticks` consecutive outside ticks.
    E-EVT-03 pickup: fires only on a heartbeat that (a) reports the asset absent, (b) is
             taken while no person is inside the asset-home zone, and (c) follows a
             heartbeat that reported it present. The subject is whoever was inside the
             zone in between; if nobody was, the event degrades to
             asset_missing_from_home with no subject rather than blaming someone.
    E-GATE-02 luma step -> illumination_change (scene-state), never object motion.
    E-KB-01 sustained global motion -> camera_moved_suspect.
    """

    def __init__(self, camera_id: str, zones: list[Zone], enter_ticks: int = 2, exit_ticks: int = 2,
                 dwell_ms: int = 10_000, tile_id: str | None = None, offset_ms: int = 0,
                 moved_after_updates: int = 10):
        self.camera_id = camera_id
        self.tile_id = tile_id
        self.zones = {z.zone_id: z for z in zones if z.camera_id == camera_id}
        self.enter_ticks = enter_ticks
        self.exit_ticks = exit_ticks
        self.dwell_ms = dwell_ms
        self.offset_ms = offset_ms
        self.moved_after_updates = moved_after_updates
        self._mem: dict[tuple[str, str], _Membership] = {}
        self._assets: dict[str, _AssetState] = {
            z.zone_id: _AssetState() for z in self.zones.values() if z.kind == "asset_home"}
        self._global_run = 0
        self._moved_fired = False
        self._ticks_seen = 0

    def _t(self, t_ms: int) -> CamTime:
        return CamTime(cam_utc_ms=t_ms, offset_ms=self.offset_ms)

    def _event(self, type_: EventType, t_ms: int, zone_id: str | None, subjects: list[str],
               objects: list[str] | None = None, payload: dict | None = None,
               confidence: float = 1.0) -> Event:
        t_bucket = t_ms // 1000
        return Event(event_id=_eid(self.camera_id, type_.value, zone_id, ",".join(sorted(subjects)), t_bucket),
                     type=type_, t=self._t(t_ms), camera_id=self.camera_id, tile_id=self.tile_id,
                     zone_id=zone_id, subject_tube_ids=subjects, object_ids=objects or [],
                     payload=payload or {}, confidence=confidence,
                     dedupe_key=f"{type_.value}|{zone_id}|{t_bucket}")

    # ---------------- tube predicates ----------------
    def on_tick(self, tubes: list[TubeSnapshot], t_ms: int) -> list[Event]:
        events: list[Event] = []
        live = {s.tube_id for s in tubes}
        first_tick = self._ticks_seen == 0
        self._ticks_seen += 1
        if first_tick:
            # E-EVT-10: whoever is already in frame when the episode opens did not "enter";
            # seed zone memberships silently so enter_zone means a boundary was crossed.
            for snap in tubes:
                foot = snap.box.foot_point()
                for z in self.zones.values():
                    if z.contains(foot):
                        m = self._mem.setdefault((snap.tube_id, z.zone_id), _Membership())
                        m.is_inside, m.inside_run, m.entered_at_ms = True, self.enter_ticks, t_ms
                        if z.kind == "asset_home" and snap.class_label in PERSON_CLASSES:
                            self._assets[z.zone_id].visitors[snap.tube_id] = t_ms
            return events
        for snap in tubes:
            foot = snap.box.foot_point()
            for z in self.zones.values():
                key = (snap.tube_id, z.zone_id)
                m = self._mem.setdefault(key, _Membership())
                inside = z.contains(foot)
                if inside:
                    m.inside_run += 1
                    m.outside_run = 0
                    if z.kind == "asset_home" and snap.class_label in PERSON_CLASSES:
                        self._assets[z.zone_id].visitors[snap.tube_id] = t_ms
                    if not m.is_inside and m.inside_run >= self.enter_ticks:
                        m.is_inside = True
                        m.entered_at_ms = t_ms
                        m.dwell_fired = False
                        events.append(self._event(EventType.enter_zone, t_ms, z.zone_id, [snap.tube_id]))
                    elif m.is_inside and not m.dwell_fired and m.entered_at_ms is not None \
                            and t_ms - m.entered_at_ms >= self.dwell_ms:
                        m.dwell_fired = True
                        events.append(self._event(EventType.dwell, t_ms, z.zone_id, [snap.tube_id],
                                                  payload={"dwell_ms": t_ms - m.entered_at_ms}))
                else:
                    m.outside_run += 1
                    m.inside_run = 0
                    if m.is_inside and m.outside_run >= self.exit_ticks:
                        m.is_inside = False
                        events.append(self._event(EventType.exit_zone, t_ms, z.zone_id, [snap.tube_id],
                                                  payload={"inside_ms": t_ms - (m.entered_at_ms or t_ms)}))
        # tubes that vanished while inside a zone: close their membership silently (exit is a
        # tube-lifecycle matter, handled by fusion/lost state, not a zone exit)
        for key in [k for k in self._mem if k[0] not in live]:
            del self._mem[key]
        return events

    # ---------------- asset heartbeat ----------------
    def on_heartbeat(self, zone_id: str, asset_present: bool, t_ms: int,
                     persons_inside: list[str]) -> list[Event]:
        z = self.zones[zone_id]
        st = self._assets[zone_id]
        events: list[Event] = []
        if asset_present:
            st.present = True
            st.last_present_ms = t_ms
            st.visitors.clear()
            return events
        # absent
        if persons_inside:
            return events  # E-EVT-03(b): someone may be occluding the shelf; wait
        if st.present is True:
            since = st.last_present_ms or 0
            suspects = [tid for tid, last in st.visitors.items() if last >= since]
            if suspects:
                events.append(self._event(EventType.pickup, t_ms, zone_id, suspects, objects=[z.asset_id or zone_id],
                                          payload={"window_ms": [since, t_ms]}, confidence=0.8 if len(suspects) == 1 else 0.5))
            else:
                events.append(self._event(EventType.asset_missing_from_home, t_ms, zone_id, [],
                                          objects=[z.asset_id or zone_id], payload={"window_ms": [since, t_ms]},
                                          confidence=0.9))
        st.present = False
        return events

    # ---------------- ingest ----------------
    def on_size_change(self, t_ms: int, old: tuple[int, int] | None, new: tuple[int, int]) -> list[Event]:
        """E-ING-05: resolution change => homography and zones invalid; same flag as a moved camera."""
        return [self._event(EventType.camera_moved_suspect, t_ms, None, [],
                            payload={"reason": "resolution_change", "old": list(old or ()), "new": list(new)},
                            confidence=0.95)]

    # ---------------- gate / scene state ----------------
    def on_gate(self, g: GateResult) -> list[Event]:
        events: list[Event] = []
        if g.luma_step:
            events.append(self._event(EventType.illumination_change, g.t_ms, None, [],
                                      payload={"mean_luma": g.mean_luma}))
        if g.global_motion:
            self._global_run += 1
            if self._global_run >= self.moved_after_updates and not self._moved_fired:
                self._moved_fired = True
                events.append(self._event(EventType.camera_moved_suspect, g.t_ms, None, [],
                                          payload={"consecutive_global_updates": self._global_run}, confidence=0.7))
        else:
            self._global_run = 0
        return events
