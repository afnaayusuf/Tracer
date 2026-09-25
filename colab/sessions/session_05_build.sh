#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 05 build: ByteTracker (dt-aware Kalman, two-stage association,
#  buffered IoU), tracker contract tests parametrized over both trackers, first-tick zone
#  seeding, Ring 2 bench for both trackers, real-footage slice re-run with ByteTracker.
#
#  Colab: paste into one %%bash cell after the env cell (GH_TOKEN, GH_REPO), or upload and run
#         bash /content/build_session_05.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 05"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }

step "0. repository"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch -q origin
  git -C "$REPO_DIR" pull -q --ff-only || die "local branch diverged from origin; resolve manually"
else
  git clone -q "https://github.com/$GH_REPO.git" "$REPO_DIR"
fi
cd "$REPO_DIR"
[ -n "${GH_TOKEN:-}" ] && git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"

step "1. audit"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present: $(git status --short | tr '\n' ' ')"
[ -n "$(git log --grep='^session 04:' --format=%h)" ] || die "session 04 commit not found; run build_session_04.sh first"
if grep -rn "import cv2\|from cv2\|opencv" vi bench tests pyproject.toml 2>/dev/null; then
  [ "${FORCE:-0}" = "1" ] || die "cv2/opencv found in code (bypasses vi.ingest)"
fi
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
edge_cases.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/test_detect_roi.py
tests/test_episode.py
tests/test_eval_mot.py
tests/test_events.py
tests/test_fact.py
tests/test_gate.py
tests/test_harness.py
tests/test_ingest.py
tests/test_schemas.py
tests/test_tubes.py
vi/__init__.py
vi/detect/__init__.py
vi/detect/base.py
vi/detect/rfdetr.py
vi/detect/roi.py
vi/episode/__init__.py
vi/episode/keyframes.py
vi/episode/writer.py
vi/eval/__init__.py
vi/eval/mot.py
vi/events/__init__.py
vi/events/compiler.py
vi/events/zones.py
vi/gate/__init__.py
vi/gate/base.py
vi/gate/framediff.py
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/reader.py
vi/ingest/synthetic.py
vi/schemas/__init__.py
vi/schemas/common.py
vi/schemas/contact_sheet.py
vi/schemas/episode.py
vi/schemas/event.py
vi/schemas/export.py
vi/schemas/fact.py
vi/schemas/scene_card.py
vi/schemas/tick.py
vi/schemas/tube.py
vi/tubes/__init__.py
vi/tubes/base.py
vi/tubes/bytetrack.py
vi/tubes/geometry.py
vi/tubes/kalman.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p colab/sessions data/bench
cat > vi/tubes/kalman.py << 'EOF_VI'
"""Constant-velocity Kalman filter on a bounding box, written from the equations (no Deep SORT
lineage; see SPEC C2). State x = [cx, cy, a, h, vx, vy, va, vh] with velocities in units per
second, so predict(dt) is correct at any sampled frame rate (E-ING-03 / E-TUBE-13)."""
from __future__ import annotations

import numpy as np

from vi.schemas import Box

STD_POS = 1.0 / 20.0     # position noise as a fraction of box height (SORT convention)
STD_VEL = 1.0 / 160.0    # velocity noise as a fraction of box height per reference frame
REF_FPS = 30.0           # the per-frame noise constants above were tuned at ~30 fps


def box_to_z(b: Box) -> np.ndarray:
    w, h = b.width, b.height
    return np.array([b.x1 + w / 2.0, b.y1 + h / 2.0, w / max(h, 1e-6), h])


def z_to_box(z: np.ndarray) -> Box:
    cx, cy, a, h = float(z[0]), float(z[1]), max(float(z[2]), 1e-3), max(float(z[3]), 1.0)
    w = a * h
    return Box(x1=cx - w / 2.0, y1=cy - h / 2.0, x2=cx + w / 2.0, y2=cy + h / 2.0)


class KalmanBoxFilter:
    def __init__(self, box: Box):
        z = box_to_z(box)
        self.x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        std = np.array([2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h,
                        10 * STD_VEL * h * REF_FPS, 10 * STD_VEL * h * REF_FPS, 1e-5 * REF_FPS,
                        10 * STD_VEL * h * REF_FPS])
        self.P = np.diag(std ** 2)
        self.age_s = 0.0
        self.hits = 1
        self.misses = 0

    def predict(self, dt_s: float) -> Box:
        dt = max(dt_s, 1e-3)
        F = np.eye(8)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = dt
        h = max(self.x[3], 1.0)
        # process noise: per-frame constants scaled to the elapsed time
        scale = dt * REF_FPS
        std = np.array([STD_POS * h, STD_POS * h, 1e-2, STD_POS * h,
                        STD_VEL * h * REF_FPS, STD_VEL * h * REF_FPS, 1e-5 * REF_FPS, STD_VEL * h * REF_FPS])
        Q = np.diag((std ** 2) * scale)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age_s += dt
        return z_to_box(self.x[:4])

    def update(self, box: Box) -> Box:
        z = box_to_z(box)
        h = max(z[3], 1.0)
        R = np.diag(np.array([STD_POS * h, STD_POS * h, 1e-1, STD_POS * h]) ** 2)
        H = np.zeros((4, 8))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0
        return z_to_box(self.x[:4])

    @property
    def box(self) -> Box:
        return z_to_box(self.x[:4])

    @property
    def speed_px_s(self) -> float:
        return float(np.hypot(self.x[4], self.x[5]))
EOF_VI
cat > vi/tubes/bytetrack.py << 'EOF_VI'
"""ByteTrack-style two-stage association with a dt-aware Kalman filter and buffered IoU,
implemented from the papers (Zhang et al. 2022; Yang et al. 2023 for the buffered matching
space), not copied from any repository. Keeps SimpleIoUTracker's lifecycle contract exactly:
born -> active -> occluded -> lost|exited, merge_candidates on ambiguity, per-camera only.

Why each piece exists:
  Kalman(dt)      E-TUBE-13: at 2-5 fps a walker moves most of a box width per tick; the
                  prediction closes the gap so IoU association survives sampled decode.
  buffered IoU    same case on the first tick, before a velocity estimate exists.
  two stages      low-confidence detections (partially occluded people) re-attach to
                  existing tracks but never create new ones -> fewer FP tubes.
  confirm_ticks   a track reports `born` until it has been seen twice; unconfirmed tracks
                  that miss are dropped silently -> flickering static objects don't become tubes.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from vi.detect import Detection
from vi.schemas import Box, CamTime, Modality, Tube, TubeState

from .kalman import KalmanBoxFilter


def _centre(b: Box) -> tuple[float, float]:
    return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)


def buffered(b: Box, r: float) -> Box:
    if r <= 0:
        return b
    dx, dy = b.width * r / 2.0, b.height * r / 2.0
    return Box(x1=b.x1 - dx, y1=b.y1 - dy, x2=b.x2 + dx, y2=b.y2 + dy)


@dataclass
class _Track:
    tube: Tube
    kf: KalmanBoxFilter
    last_t_ms: int
    confirmed: bool = False


class ByteTracker:
    def __init__(self, camera_id: str, high_thr: float = 0.5, low_thr: float = 0.1,
                 match_iou: float = 0.2, low_match_iou: float = 0.4, buffer: float = 0.4,
                 max_occluded_ms: int = 3000, confirm_ticks: int = 2, ambig_margin: float = 0.1,
                 first_tick_gate: float = 1.0,
                 exit_boxes: list[Box] | None = None, modality: Modality = Modality.rgb,
                 offset_ms: int = 0, keyframe_sink: Callable[[str, int, Box], str] | None = None):
        self.camera_id = camera_id
        self.high_thr, self.low_thr = high_thr, low_thr
        self.match_iou, self.low_match_iou = match_iou, low_match_iou
        self.buffer = buffer
        self.max_occluded_ms = max_occluded_ms
        self.confirm_ticks = confirm_ticks
        self.ambig_margin = ambig_margin
        self.first_tick_gate = first_tick_gate
        self.exit_boxes = exit_boxes or []
        self.modality = modality
        self.offset_ms = offset_ms
        self.keyframe_sink = keyframe_sink
        self._tracks: dict[str, _Track] = {}
        self._seq = 0
        self._last_t_ms: int | None = None

    # ------------------------------------------------------------------ helpers
    def _t(self, t_ms: int) -> CamTime:
        return CamTime(cam_utc_ms=t_ms, offset_ms=self.offset_ms)

    def _new_id(self, t_ms: int) -> str:
        self._seq += 1
        return f"{self.camera_id}:{t_ms}:{self._seq}"

    def _touches_exit(self, box: Box) -> bool:
        return any(box.iou(e) > 0.0 for e in self.exit_boxes)

    def _cost(self, tracks: list[_Track], dets: list[Detection]) -> np.ndarray:
        """1 - buffered IoU. A track seen only once has no velocity yet, so its prediction is
        just its last box; for those the cost falls back to a normalised centre distance
        (gate = `first_tick_gate` box heights) so a fast walker's second detection still
        attaches on the very first tick (E-TUBE-13)."""
        m = np.ones((len(tracks), len(dets)))
        for i, tr in enumerate(tracks):
            pb = buffered(tr.kf.box, self.buffer)
            for j, d in enumerate(dets):
                c = 1.0 - pb.iou(buffered(d.box, self.buffer))
                if tr.kf.hits < 2 and c >= 1.0 - self.match_iou:
                    (px, py), (dx, dy) = _centre(pb), _centre(d.box)
                    dist = float(np.hypot(px - dx, py - dy)) / (self.first_tick_gate * max(pb.height, 1.0))
                    if dist < 1.0:
                        c = min(c, 1.0 - self.match_iou * (1.0 - dist) - 1e-6)  # just inside the IoU gate
                m[i, j] = c
        return m

    def _assign(self, tracks: list[_Track], dets: list[Detection], min_iou: float
                ) -> tuple[list[tuple[int, int]], list[int], list[int], np.ndarray]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets))), np.zeros((len(tracks), len(dets)))
        cost = self._cost(tracks, dets)
        r, c = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(r, c) if 1.0 - cost[i, j] >= min_iou]
        ut = [i for i in range(len(tracks)) if i not in {p[0] for p in pairs}]
        ud = [j for j in range(len(dets)) if j not in {p[1] for p in pairs}]
        return pairs, ut, ud, cost

    # ------------------------------------------------------------------ main
    def update(self, detections: list[Detection], t_ms: int,
               det_source: str = "detector") -> tuple[list[Tube], list[Tube]]:
        dt_s = 0.0 if self._last_t_ms is None else max(0.0, (t_ms - self._last_t_ms) / 1000.0)
        self._last_t_ms = t_ms
        for tr in self._tracks.values():
            if dt_s > 0:
                tr.kf.predict(dt_s)
                tr.tube.box = tr.kf.box
        high = [d for d in detections if d.confidence >= self.high_thr or det_source == "heartbeat"]
        low = [d for d in detections if self.low_thr <= d.confidence < self.high_thr and det_source != "heartbeat"]
        ids = list(self._tracks.keys())
        tracks = [self._tracks[i] for i in ids]

        # stage 1: all tracks vs high-confidence detections
        pairs, ut, ud_high, cost = self._assign(tracks, high, self.match_iou)
        # E-TUBE-01: ambiguity -> merge_candidates, never a silent swap
        for i, j in pairs:
            for j2 in range(len(high)):
                if j2 != j and (1.0 - cost[i, j2]) >= self.match_iou and abs(cost[i, j] - cost[i, j2]) < self.ambig_margin:
                    other = next((tracks[i2].tube.tube_id for i2, jj in pairs if jj == j2), None)
                    if other and other not in tracks[i].tube.merge_candidates:
                        tracks[i].tube.merge_candidates.append(other)
        matched: dict[int, Detection] = {i: high[j] for i, j in pairs}
        # stage 2: leftover tracks vs low-confidence detections (re-attach only)
        if ut and low:
            pairs2, ut2, _, _ = self._assign([tracks[i] for i in ut], low, self.low_match_iou)
            for a, j in pairs2:
                matched[ut[a]] = low[j]
            ut = [ut[a] for a in ut2]

        closed: list[Tube] = []
        for i, d in matched.items():
            tr = tracks[i]
            tr.tube.box = tr.kf.update(d.box)
            tr.tube.last_seen = self._t(t_ms)
            tr.tube.occluded_since_ms = None
            tr.last_t_ms = t_ms
            if not tr.confirmed and tr.kf.hits >= self.confirm_ticks:
                tr.confirmed = True
            tr.tube.state = TubeState.active if tr.confirmed else TubeState.born
        for i in ut:
            tr = tracks[i]
            if not tr.confirmed and det_source != "heartbeat":
                del self._tracks[tr.tube.tube_id]          # unconfirmed and gone: never a tube
                continue
            tr.kf.misses += 1
            if tr.tube.occluded_since_ms is None:
                tr.tube.occluded_since_ms = t_ms
                tr.tube.state = TubeState.occluded
            elif t_ms - tr.tube.occluded_since_ms > self.max_occluded_ms:
                tr.tube.state = TubeState.exited if self._touches_exit(tr.tube.box) else TubeState.lost
                closed.append(tr.tube)
                del self._tracks[tr.tube.tube_id]
        for j in ud_high:                                     # births only from high-confidence
            d = high[j]
            tid = self._new_id(t_ms)
            tube = Tube(tube_id=tid, camera_id=self.camera_id, class_label=d.class_label, state=TubeState.born,
                        born=self._t(t_ms), last_seen=self._t(t_ms), box=d.box, modality=self.modality)
            if self.keyframe_sink is not None:
                tube.keyframe_refs.append(self.keyframe_sink(self.camera_id, t_ms, d.box))
            tr = _Track(tube=tube, kf=KalmanBoxFilter(d.box), last_t_ms=t_ms, confirmed=self.confirm_ticks <= 1)
            if tr.confirmed:
                tube.state = TubeState.active
            self._tracks[tid] = tr
        return [tr.tube for tr in self._tracks.values()], closed
EOF_VI
cat > vi/tubes/__init__.py << 'EOF_VI'
from .base import Tracker
from .bytetrack import ByteTracker
from .kalman import KalmanBoxFilter
from .simple_iou import SimpleIoUTracker

TRACKERS = {"simple": SimpleIoUTracker, "byte": ByteTracker}
EOF_VI
cat > vi/events/compiler.py << 'EOF_VI'
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
EOF_VI
cat > bench/slice_gpu.py << 'EOF_VI'
"""Real-footage slice (Day 4): reader -> gate -> ROIs -> RF-DETR -> tubes -> events -> episode file,
with a native-res keyframe saved at every tube birth. One row to data/bench/slice_gpu.jsonl.

  python bench/slice_gpu.py --source /content/HI_DEF_VIDEO.mp4 --fps 4 --model nano            # ByteTracker
  python bench/slice_gpu.py --source /content/HI_DEF_VIDEO.mp4 --fps 4 --tracker simple --threshold 0.5
  python bench/slice_gpu.py --source clip.mp4 --zones data/zones/cam1.json --tile lobby
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vi.detect import blobs_to_rois, crop_roi, is_tube_class, pad_batch, remap_detections
from vi.detect.rfdetr import RFDETRDetector
from vi.episode import EpisodeWriter, KeyframeStore
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate
from vi.ingest import VideoReader
from vi.schemas import CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.tubes import TRACKERS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--tile", default="tile1")
    ap.add_argument("--model", default="nano")
    ap.add_argument("--fps", type=float, default=4.0, help="R11 range 2-5; IoU-only tracking collapses below ~4")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="detector floor; ByteTracker re-attaches 0.1-0.5 detections and births only >=0.5")
    ap.add_argument("--tracker", choices=sorted(TRACKERS), default="byte")
    ap.add_argument("--warmup", type=int, default=3, help="frames excluded from timing (JIT profiling runs)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default = edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=640, want_rgb=True)
    det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=a.batch)
    gate = FrameDiffGate(a.camera)
    kf = KeyframeStore(Path(a.out).parent / "keyframes")
    current = {"frame": None}
    zones = None
    tracker = compiler = writer = ep = None
    stage = Counter()
    detect_ms: list[float] = []
    ev_types: Counter = Counter()
    births = 0
    closed_all = []
    tick_ms = int(1000 / a.fps)
    frames = 0

    for fr in reader.frames():
        if frames >= a.max_frames:
            break
        h, w = fr.rgb.shape[:2]
        current["frame"] = fr.rgb
        if zones is None:   # first frame: zones need the native size
            zones = load_zones(a.zones, a.camera) if a.zones else default_zones(a.camera, w, h, tile_id=a.tile)
            exit_boxes = [z.polygon for z in zones if z.kind == "exit"]
            from vi.schemas import Box
            exits = [Box(x1=min(p[0] for p in poly), y1=min(p[1] for p in poly),
                         x2=max(p[0] for p in poly), y2=max(p[1] for p in poly)) for poly in exit_boxes]
            tkw = {"iou_thr": a.iou_thr} if a.tracker == "simple" else {}
            tracker = TRACKERS[a.tracker](a.camera, max_occluded_ms=a.max_occluded_ms, exit_boxes=exits,
                                          keyframe_sink=kf.make_sink(lambda: current["frame"]), **tkw)
            compiler = EventCompiler(a.camera, zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=a.tile)
            writer = EpisodeWriter(a.out)
            ep = writer.open(a.tile, [a.camera], CamTime(cam_utc_ms=fr.pts_ms), prov)
        t0 = time.perf_counter()
        g = gate.update(fr.gray, fr.pts_ms)
        stage["gate"] += time.perf_counter() - t0
        events = compiler.on_gate(g)
        t0 = time.perf_counter()
        rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride)
        dets = []
        for i in range(0, len(rois), a.batch):
            chunk = rois[i:i + a.batch]
            crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], a.batch)
            for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                dets += remap_detections(d, r, w, h)
        dets = [d for d in dets if is_tube_class(d.class_label)]
        dt_detect = time.perf_counter() - t0
        if frames >= a.warmup:
            stage["detect"] += dt_detect
            detect_ms.append(dt_detect * 1000)
        t0 = time.perf_counter()
        before = len(tracker._tracks)
        live, closed = tracker.update(dets, fr.pts_ms)
        births += max(0, len(live) + len(closed) - before) if closed else max(0, len(live) - before)
        closed_all += closed
        stage["track"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        snaps = [TubeSnapshot(tube_id=t.tube_id, class_label=t.class_label, state=t.state, box=t.box,
                              det_source="detector" if t.state.value == "active" else "predicted") for t in live]
        events += compiler.on_tick(snaps, fr.pts_ms)
        tick = Tick(camera_id=a.camera, tile_id=a.tile, tick_index=frames, t_start=CamTime(cam_utc_ms=fr.pts_ms),
                    t_end=CamTime(cam_utc_ms=fr.pts_ms + tick_ms), tubes=snaps,
                    event_ids=[e.event_id for e in events], gate_energy=sum(b.energy for b in g.blobs), provenance=prov)
        writer.write_tick(ep, tick)
        for e in events:
            writer.write_event(ep, e)
            ev_types[e.type.value] += 1
            print(f"t={fr.pts_ms:7d}  {e.type.value:22s} zone={e.zone_id} subjects={e.subject_tube_ids}")
        stage["compile"] += time.perf_counter() - t0
        frames += 1

    if frames == 0:
        raise SystemExit("no frames read; check --source")
    tubes = [tr.tube for tr in tracker._tracks.values()] + closed_all
    cast = [CastMember(tube_ids=[t.tube_id], class_label=t.class_label, best_keyframe_ref=(t.keyframe_refs or [None])[0])
            for t in tubes]
    writer.close(ep, CamTime(cam_utc_ms=fr.pts_ms + tick_ms), EpisodeStatus.closed, cast)
    st = reader.stats
    row = {
        "ring": "slice", "source": Path(a.source).name, "model": f"rf-detr-{a.model}", "sampled_fps": a.fps,
        "tracker": a.tracker, "det_threshold": a.threshold,
        "frames": frames, "decode_ms_per_frame": round(st.decode_ms_total / frames, 2),
        **{f"{k}_ms_per_frame": round(v * 1000 / frames, 2) for k, v in stage.items() if k != "detect"},
        "detect_ms_p50": round(float(np.median(detect_ms)), 2) if detect_ms else None,
        "detect_ms_p95": round(float(np.percentile(detect_ms, 95)), 2) if detect_ms else None,
        "tubes_total": len(tubes), "tubes_live_at_end": len(tracker._tracks),
        "person_tubes": sum(1 for t in tubes if t.class_label == "person"),
        "merge_candidates_flagged": sum(1 for t in tubes if t.merge_candidates),
        "tubes_by_final_state": dict(Counter(t.state.value for t in tubes)),
        "classes": dict(Counter(t.class_label for t in tubes)),
        "events": dict(ev_types), "keyframes_saved": kf.saved, "episode_id": ep,
        "episode_records": sum(1 for _ in EpisodeWriter.read(writer.path(ep))),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "slice_gpu.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print(f"\nepisode -> {writer.path(ep)}\nkeyframes -> {kf.root}")


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/ring2_tubes.py << 'EOF_VI'
"""Ring 2 bench: tracker vs ground truth -> MOTA / IDF1 / IDSW / fragmentation. Baseline for the
ByteTrack rewrite. Rows to data/bench/ring2.jsonl.

  python bench/ring2_tubes.py --synthetic --tracker byte --sample-every 5
  python bench/ring2_tubes.py --mot17 /content/drive/MyDrive/MOT17/train/MOT17-02-FRCNN --fps 30
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vi.detect import Detection
from vi.eval import evaluate_mot, load_mot_txt
from vi.schemas import Box
from vi.tubes import TRACKERS


def synthetic_gt(frames: int = 120, seed: int = 0):
    """Two walkers crossing paths plus one that stops for 20 frames (stationary + crossing cases)."""
    rng = np.random.default_rng(seed)
    gt: dict[int, list[tuple[int, Box]]] = {}
    for f in range(frames):
        xa = 20 + f * 5
        xb = 620 - f * 5
        xc = 300 if 40 <= f < 60 else 300 + (f - 60) * 3 if f >= 60 else 300 - (40 - f) * 3
        items = []
        for tid, x in ((1, xa), (2, xb), (3, xc)):
            j = rng.normal(0, 1.5, 4)
            items.append((tid, Box(x1=x + j[0], y1=100 + j[1], x2=x + 40 + j[2], y2=220 + j[3])))
        gt[f] = items
    return gt


def run_tracker(dets_by_frame, fps: float, iou_thr: float, max_occluded_ms: int, drop_rate: float = 0.0,
                seed: int = 0, tracker: str = "simple"):
    rng = np.random.default_rng(seed)
    kw = {"iou_thr": iou_thr} if tracker == "simple" else {}
    tr = TRACKERS[tracker]("cam1", max_occluded_ms=max_occluded_ms, **kw)
    pred: dict[int, list] = {}
    id_map: dict[str, int] = {}
    for f in sorted(dets_by_frame):
        t_ms = int(f * 1000 / fps)
        dets = [Detection(box=b, class_label="person", confidence=0.9)
                for _, b in dets_by_frame[f] if rng.random() >= drop_rate]
        live, _ = tr.update(dets, t_ms)
        pred[f] = [(id_map.setdefault(t.tube_id, len(id_map) + 1), t.box) for t in live if t.state.value == "active"]
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mot17", help="sequence dir containing gt/gt.txt and det/det.txt")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the sequence")
    ap.add_argument("--sample-every", type=int, default=1, help="use every Nth frame (simulate 2-5 fps decode)")
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--max-occluded-ms", type=int, default=2000)
    ap.add_argument("--det-min-conf", type=float, default=0.5)
    ap.add_argument("--drop-rate", type=float, default=0.0, help="synthetic: random missed detections")
    ap.add_argument("--tracker", choices=sorted(TRACKERS), default="simple")
    a = ap.parse_args()
    if a.synthetic:
        gt = synthetic_gt()
        dets = gt
        name = "synthetic_crossing"
    elif a.mot17:
        seq = Path(a.mot17)
        gt = load_mot_txt(seq / "gt" / "gt.txt", gt=True)
        dets = load_mot_txt(seq / "det" / "det.txt", min_conf=a.det_min_conf)
        name = seq.name
    else:
        ap.error("--synthetic or --mot17 required")
    if a.sample_every > 1:
        gt = {f: v for f, v in gt.items() if f % a.sample_every == 0}
        dets = {f: v for f, v in dets.items() if f % a.sample_every == 0}
    pred = run_tracker(dets, a.fps, a.iou_thr, a.max_occluded_ms, a.drop_rate, tracker=a.tracker)
    res = evaluate_mot(gt, pred)
    row = {"ring": 2, "tracker": TRACKERS[a.tracker].__name__, "sequence": name, "fps": a.fps, "sample_every": a.sample_every,
           "effective_fps": round(a.fps / a.sample_every, 2), "iou_thr": a.iou_thr, "max_occluded_ms": a.max_occluded_ms,
           "drop_rate": a.drop_rate, **res.as_row(), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "ring2.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/README.md << 'EOF_VI'
# bench/

One script per ring. Each prints exactly one row of the benchmark table (see PLAN.md) and
checkpoints it to `data/bench/<ring>.jsonl` as it goes, so a killed Colab session loses nothing.

| script | ring | GPU | dataset | row |
|---|---|---|---|---|
| slice_cpu.py | all (stubs) | none | synthetic | proves plumbing; run first, every day |
| slice_gpu.py | 0–3a on real footage | L4 | own clip | reader → gate → ROIs → RF-DETR → tubes → events → episode file + birth keyframes; stage ms, tubes by state |
| ring0_gate.py | 0 | CPU | VIRAT | gate FN rate, ms per GOP per stream, MV vs framediff |
| ring1_detect.py | 1 | L4 | own clip / VIRAT | `--mode frame`: ms p50/p95 per frame at sampled fps; `--mode roi`: gate → packed ROI batches (nano/medium) |
| ring2_tubes.py | 2 | CPU | synthetic, MOT17 | MOTA/IDF1/IDSW/fragmentation (vi/eval/mot.py); `--tracker simple\|byte`, `--sample-every` simulates 2–5 fps decode |
| ring3b_sheet.py | 3b | A100-40 | crops from ring2 | attribute accuracy, bleed rate (12 vs 16 cells), ms per sheet |
| ring3a_events.py | 3a | CPU | MEVA subset | event precision/recall per type |
| agent_replay.py | block 2 | A100-40 -> G4 | WILDTRACK episodes | both scenarios replay; tool calls, latency |

Colab: GPU runtime, `pip install rfdetr==1.7.0`, then run from the repo root (see colab/README.md).
EOF_VI
cat > tests/test_tubes.py << 'EOF_VI'
import numpy as np
import pytest

from vi.detect import Detection
from vi.schemas import TubeState
from vi.tubes import ByteTracker, SimpleIoUTracker

TRACKERS = [
    pytest.param(lambda **kw: SimpleIoUTracker(iou_thr=0.2, **kw), id="simple"),
    pytest.param(lambda **kw: ByteTracker(confirm_ticks=1, **kw), id="byte"),
]
from vi.tubes.geometry import FOOT_UNCERTAINTY_M, OCCLUDED_FEET_UNCERTAINTY_M, project_foot

from conftest import box


def det(x1, y1, x2, y2, label="person"):
    return Detection(box=box(x1, y1, x2, y2), class_label=label, confidence=0.9)


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-TUBE-02")
def test_occlusion_then_lost_or_exited(make):
    tr = make(camera_id="c1", max_occluded_ms=1000, exit_boxes=[box(300, 0, 320, 240)])
    tr.update([det(10, 10, 50, 100)], 0)
    live, closed = tr.update([], 500)
    assert live[0].state == TubeState.occluded and live[0].occluded_since_ms == 500 and not closed
    live, closed = tr.update([], 1600)
    assert live == [] and closed[0].state == TubeState.lost
    # same story but the last box touched an exit region
    tr2 = make(camera_id="c1", max_occluded_ms=1000, exit_boxes=[box(300, 0, 320, 240)])
    tr2.update([det(290, 10, 320, 100)], 0)
    tr2.update([], 500)
    _, closed = tr2.update([], 1600)
    assert closed[0].state == TubeState.exited


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-TUBE-01")
def test_crossing_paths_flag_merge_candidates_instead_of_guessing(make):
    tr = make(camera_id="c1", ambig_margin=0.2)
    tr.update([det(0, 0, 40, 100), det(30, 0, 70, 100)], 0)
    # two people shoulder to shoulder: each detection overlaps both tracks with similar IoU
    live, _ = tr.update([det(10, 0, 50, 100), det(15, 0, 55, 100)], 100)
    assert len(live) == 2                              # no new ids invented
    assert any(t.merge_candidates for t in live)       # ambiguity recorded, not resolved


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-TUBE-03")
@pytest.mark.edge("E-GATE-01")
def test_heartbeat_detection_keeps_stationary_track_active(make):
    tr = make(camera_id="c1", max_occluded_ms=1000)
    tr.update([det(10, 10, 50, 100)], 0)
    for i in range(1, 30):
        live, closed = tr.update([det(10, 10, 50, 100)], i * 1000, det_source="heartbeat")
    assert live[0].state == TubeState.active and closed == []


@pytest.mark.parametrize("make", TRACKERS)
@pytest.mark.edge("E-FOV-06")
def test_keyframe_ref_persisted_at_birth(make):
    refs = []
    def sink(cam, t_ms, b):
        ref = f"r2://site/{cam}/{t_ms}.jpg"
        refs.append(ref)
        return ref
    tr = make(camera_id="c1", keyframe_sink=sink)
    live, _ = tr.update([det(10, 10, 50, 100)], 42)
    assert live[0].keyframe_refs == ["r2://site/c1/42.jpg"] and refs == live[0].keyframe_refs


@pytest.mark.edge("E-TUBE-12")
def test_occluded_feet_widen_uncertainty_and_tag_source():
    H = np.eye(3) * np.array([0.01, 0.01, 1.0])[:, None]  # trivial scale homography
    b = box(100, 50, 140, 200)
    ok = project_foot(b, H, feet_visible=True)
    bad = project_foot(b, H, feet_visible=False)
    assert ok.source == "homography" and ok.uncertainty_m == FOOT_UNCERTAINTY_M
    assert bad.source == "fallback_bbox_bottom" and bad.uncertainty_m == OCCLUDED_FEET_UNCERTAINTY_M
    assert (ok.x_m, ok.y_m) == (bad.x_m, bad.y_m)
    assert project_foot(b, None) is None


@pytest.mark.edge("E-TUBE-13")
def test_bytetracker_survives_sampled_frame_rate_where_iou_only_fragments():
    import sys
    sys.path.insert(0, "bench")
    from ring2_tubes import synthetic_gt
    from vi.eval import evaluate_mot

    gt = {f: v for f, v in synthetic_gt().items() if f % 5 == 0}          # 6 fps effective
    def run(tr):
        pred, ids = {}, {}
        for f in sorted(gt):
            live, _ = tr.update([det(b.x1, b.y1, b.x2, b.y2) for _, b in gt[f]], int(f * 1000 / 30))
            pred[f] = [(ids.setdefault(t.tube_id, len(ids) + 1), t.box) for t in live if t.state == TubeState.active]
        return evaluate_mot(gt, pred)
    simple = run(SimpleIoUTracker("c1", iou_thr=0.3, max_occluded_ms=2000))
    byte = run(ByteTracker("c1", max_occluded_ms=2000))
    assert simple.fragmentation_ratio > 3 and simple.idsw > 10        # the measured baseline
    assert byte.fragmentation_ratio == 1.0 and byte.idsw == 0 and byte.idf1 > 0.95


def test_kalman_predicts_constant_velocity_in_seconds():
    from vi.tubes import KalmanBoxFilter
    kf = KalmanBoxFilter(box(0, 0, 40, 120))
    for i in range(1, 6):                       # 100 px/s to the right, sampled irregularly
        kf.predict(0.1 if i % 2 else 0.3)
        t = 0.1 * ((i + 1) // 2) + 0.3 * (i // 2)
        kf.update(box(100 * t, 0, 100 * t + 40, 120))
    assert abs(kf.speed_px_s - 100) < 15
    predicted = kf.predict(1.0)
    assert abs(predicted.x1 - (kf.x[0] - 20)) < 1e-6 and 150 < predicted.x1 < 260
EOF_VI
cat > tests/test_events.py << 'EOF_VI'
import numpy as np
import pytest

from vi.events import EventCompiler, Zone
from vi.gate.base import GateResult
from vi.schemas import EventType, TubeSnapshot, TubeState

from conftest import box


def zone(zid, cam="c1", kind="generic", asset=None, poly=((100, 100), (200, 100), (200, 200), (100, 200))):
    return Zone(zone_id=zid, camera_id=cam, polygon=list(poly), kind=kind, asset_id=asset)


def snap(tid, x1, y1, x2, y2, label="person"):
    return TubeSnapshot(tube_id=tid, class_label=label, state=TubeState.active, box=box(x1, y1, x2, y2))


def inside(tid):   # foot point (150, 150)
    return snap(tid, 140, 100, 160, 150)


def outside(tid):  # foot point (150, 250)
    return snap(tid, 140, 200, 160, 250)


@pytest.mark.edge("E-EVT-01")
def test_zone_hysteresis_filters_boundary_jitter():
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=2, exit_ticks=2)
    ec.on_tick([], -500)                        # episode opens on an empty tile
    evs = []
    for i in range(6):   # jitter: in, out, in, out, ...
        evs += ec.on_tick([inside("a") if i % 2 == 0 else outside("a")], i * 500)
    assert evs == []
    evs = ec.on_tick([inside("a")], 3000) + ec.on_tick([inside("a")], 3500)
    assert [e.type for e in evs] == [EventType.enter_zone]
    evs = ec.on_tick([outside("a")], 4000) + ec.on_tick([outside("a")], 4500)
    assert [e.type for e in evs] == [EventType.exit_zone]


@pytest.mark.edge("E-EVT-03")
def test_pickup_requires_absence_with_nobody_in_zone():
    ec = EventCompiler("c1", [zone("shelf", kind="asset_home", asset="bike_keys")], enter_ticks=1, exit_ticks=1)
    ec.on_tick([], 0)
    assert ec.on_heartbeat("shelf", True, 0, persons_inside=[]) == []
    ec.on_tick([inside("emma")], 1000)
    # shelf hidden behind Emma: heartbeat says absent while she is inside -> no event
    assert ec.on_heartbeat("shelf", False, 1500, persons_inside=["emma"]) == []
    ec.on_tick([outside("emma")], 2000)
    evs = ec.on_heartbeat("shelf", False, 2500, persons_inside=[])
    assert len(evs) == 1 and evs[0].type == EventType.pickup
    assert evs[0].subject_tube_ids == ["emma"] and evs[0].object_ids == ["bike_keys"]
    # a second absent heartbeat must not fire again
    assert ec.on_heartbeat("shelf", False, 3000, persons_inside=[]) == []


@pytest.mark.edge("E-EVT-03")
def test_missing_asset_with_no_visitor_does_not_blame_anyone():
    ec = EventCompiler("c1", [zone("shelf", kind="asset_home", asset="bike_keys")])
    ec.on_heartbeat("shelf", True, 0, persons_inside=[])
    evs = ec.on_heartbeat("shelf", False, 10_000, persons_inside=[])
    assert evs[0].type == EventType.asset_missing_from_home and evs[0].subject_tube_ids == []


@pytest.mark.edge("E-EVT-06")
def test_overlapping_cameras_share_dedupe_key_but_not_event_id():
    a = EventCompiler("camA", [zone("door", cam="camA")], enter_ticks=1)
    b = EventCompiler("camB", [zone("door", cam="camB")], enter_ticks=1)
    a.on_tick([], 0)
    b.on_tick([], 0)
    ea = a.on_tick([inside("tA")], 1000)[0]
    eb = b.on_tick([inside("tB")], 1300)[0]
    assert ea.dedupe_key == eb.dedupe_key and ea.event_id != eb.event_id


@pytest.mark.edge("E-GATE-02")
@pytest.mark.edge("E-KB-01")
def test_gate_results_become_scene_state_events():
    ec = EventCompiler("c1", [], moved_after_updates=3)
    evs = ec.on_gate(GateResult(camera_id="c1", t_ms=0, blobs=[], luma_step=True, mean_luma=180))
    assert [e.type for e in evs] == [EventType.illumination_change]
    moved = []
    for i in range(5):
        moved += ec.on_gate(GateResult(camera_id="c1", t_ms=i * 100, blobs=[], global_motion=True))
    assert [e.type for e in moved] == [EventType.camera_moved_suspect]   # fires once


def test_dwell_fires_once():
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=1, dwell_ms=2000)
    ec.on_tick([], -500)
    evs = []
    for i in range(6):
        evs += ec.on_tick([inside("a")], i * 1000)
    assert [e.type for e in evs] == [EventType.enter_zone, EventType.dwell]


@pytest.mark.edge("E-EVT-10")
def test_tubes_present_at_episode_open_do_not_enter():
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=1, exit_ticks=1)
    assert ec.on_tick([inside("a"), inside("b")], 0) == []                 # already inside: no enter
    assert ec.on_tick([inside("a"), inside("b")], 500) == []
    evs = ec.on_tick([outside("a"), inside("b"), inside("c")], 1000)      # a leaves, c arrives
    assert sorted(e.type for e in evs) == sorted([EventType.exit_zone, EventType.enter_zone])
    assert [e.subject_tube_ids for e in evs if e.type == EventType.enter_zone] == [["c"]]
EOF_VI
cat > edge_cases.yaml << 'EOF_VI'
# Edge-case registry. Source of truth for `make coverage`.
# status: implemented (code exists; MUST have a test marked @pytest.mark.edge("ID"))
#         planned     (in scope for month 1; coverage prints it as TODO)
#         deferred    (out of month-1 scope; listed so it is not forgotten)
# Adding a case: append here first, then code, then test. Never the other order.
version: 1
cases:
# ---------------- ingest / time ----------------
- id: E-ING-01
  ring: ingest
  title: Camera clocks disagree
  trigger: Cameras drift seconds apart; fusion hand-offs and the ±1 s fact-miner window fail.
  handling: Every timestamp is CamTime{cam_utc_ms, offset_ms, offset_confidence}; offsets estimated in the calibration walk and re-estimated from hand-offs; fusion uses corrected_ms() only.
  status: implemented
- id: E-ING-02
  ring: ingest
  title: Stream drops or reconnects
  trigger: RTSP disconnect mid-episode.
  handling: signal_lost/signal_restored events; open episode on that camera closes with status=truncated; tubes go to lost, not exited.
  status: implemented
- id: E-ING-03
  ring: ingest
  title: Variable or dropped frame rate
  trigger: Camera throttles under load; fps changes.
  handling: Ticks are wall-clock windows (t_start,t_end), never frame indices; tube age computed from corrected time.
  status: implemented
- id: E-ING-04
  ring: ingest
  title: Codec without motion vectors
  trigger: MJPEG or unknown codec stream.
  handling: Gate interface is codec-agnostic; FrameDiffGate fallback at 1 fps; camera profile records gate_mode.
  status: implemented
- id: E-ING-05
  ring: ingest
  title: Resolution or aspect change mid-stream
  trigger: Camera reconfigured; homography and zones now invalid.
  handling: Reader SizeGuard flags the change; EventCompiler.on_size_change emits camera_moved_suspect; fusion freezes that camera until recalibration (week 3).
  status: implemented
- id: E-ING-06
  ring: ingest
  title: Timezone and DST
  trigger: '"Yesterday afternoon" across a DST switch or a site in another timezone.'
  handling: All storage in UTC ms; site timezone stored in KB; agent resolves local windows explicitly and echoes them in the answer.
  status: implemented
- id: E-ING-07
  ring: ingest
  title: Duplicate or out-of-order frames
  trigger: RTSP jitter delivers repeated or reordered PTS.
  handling: Dedupe by PTS per camera; drop frames older than the last processed tick.
  status: implemented
# ---------------- ring 0 gate ----------------
- id: E-GATE-01
  ring: ring0
  title: Stationary object is invisible to the gate
  trigger: Person stops moving; bag placed and left; fallen person still.
  handling: Heartbeat detections on I-frames (1–2 s active tiles, 10–30 s quiet, always on asset homes) keep tubes alive; not the gate's job.
  status: implemented
- id: E-GATE-02
  ring: ring0
  title: Lighting change looks like whole-frame motion
  trigger: Light switch, IR-cut toggle, cloud passes.
  handling: Global luminance step detected before differencing; emitted as illumination_change scene-state event; blobs suppressed for that update.
  status: implemented
- id: E-GATE-03
  ring: ring0
  title: Foliage, rain, insects, sensor noise
  trigger: Persistent low-level motion that is not an object.
  handling: Adaptive noise floor (median block energy over history) × k; MV-field coherence test once MVGate lands.
  status: implemented
- id: E-GATE-04
  ring: ring0
  title: Camera shake or PTZ move
  trigger: Most of the frame changes at once.
  handling: active_fraction > threshold => global_motion; blobs suppressed, background frozen; sustained => camera_moved_suspect.
  status: implemented
- id: E-GATE-05
  ring: ring0
  title: Very slow motion under threshold
  trigger: Someone creeping; object slid slowly.
  handling: Heartbeat detections catch state deltas the gate misses; gate FN rate is a tracked metric.
  status: planned
- id: E-GATE-06
  ring: ring0
  title: Motion in screens, mirrors, reflections
  trigger: TV playing; mirror shows a person in another zone; steel reflects motion.
  handling: Scene card media_zones and reflective_surfaces become gate masks; tubes born inside them are reflection_suspect.
  status: planned
- id: E-GATE-07
  ring: ring0
  title: Intra-only or all-I-frame streams
  trigger: Encoder configured with no P-frames; no motion vectors.
  handling: Same fallback as E-ING-04.
  status: planned
- id: E-GATE-08
  ring: ring0
  title: Night noise inflates motion energy
  trigger: High-gain low-light mode.
  handling: Per-modality threshold profile; coherence weighted higher at night.
  status: planned
# ---------------- ring 1 detect ----------------
- id: E-DET-01
  ring: ring1
  title: Tiny objects below detector floor
  trigger: Keys, phone, small tools at room-camera resolution.
  handling: Not detected as tubes; tracked via carried_item attribute and asset-home occupancy; inspect tool for on-demand native-res count/find; resolution ceiling stated in answers.
  status: planned
- id: E-DET-02
  ring: ring1
  title: Truncation at frame edge
  trigger: Half a person at the border.
  handling: remap_detections flags boxes within 2 px of the frame border as truncated; Ring 2 then uses fallback_bbox_bottom with wide uncertainty and keyframe scoring penalises them.
  status: implemented
- id: E-DET-03
  ring: ring1
  title: Class confusion
  trigger: Child vs small adult; dog vs bag; mannequin vs person.
  handling: Class is an attribute with confidence; enrichment can override; agent never asserts class below threshold.
  status: planned
- id: E-DET-04
  ring: ring1
  title: Ghost detections in reflections
  trigger: Mirror/glass shows a duplicate person.
  handling: Tubes inside reflective_surfaces are reflection_suspect; fusion never creates an entity from a suspect tube alone.
  status: planned
- id: E-DET-05
  ring: ring1
  title: People on screens, posters, photos
  trigger: TV shows a face; framed photo detected as person.
  handling: media_zones from scene card; detections fully inside a media zone with no tube motion are dropped.
  status: planned
- id: E-DET-06
  ring: ring1
  title: Dense crowds
  trigger: Heavy overlap; fragmentation.
  handling: crowd event on density; tube quality flag; enrichment skipped for low-quality tubes.
  status: deferred
- id: E-DET-07
  ring: ring1
  title: Fisheye or wide-lens distortion
  trigger: Homography from a distorted image.
  handling: fov_class from scene card; undistort before foot-point projection.
  status: deferred
- id: E-DET-08
  ring: ring1
  title: IR appearance shift lowers recall
  trigger: Night mode.
  handling: Measured on night eval set before any enhancer; enhancer only if gap is real and license clear.
  status: planned
- id: E-DET-09
  ring: ring1
  title: Static furniture and fixtures become tubes
  trigger: Detector emits dining table, tv, chair as objects; tracker births permanent tubes for them.
  handling: Only TUBE_CLASSES (people, animals, vehicles, carried bags) become tubes; furniture is a scene-card asset owned by zones and heartbeats.
  status: implemented
# ---------------- ring 2 tubes / fusion ----------------
- id: E-TUBE-01
  ring: ring2
  title: ID switch when paths cross
  trigger: Two people cross; IoU ambiguous.
  handling: Keep best match; record other as merge_candidate; never silently merge; fusion may later resolve with ReID.
  status: implemented
- id: E-TUBE-02
  ring: ring2
  title: Occlusion
  trigger: Person behind furniture or another person.
  handling: state=occluded with occluded_since; after max_occluded_ms becomes lost (or exited if last box touched an exit zone).
  status: implemented
- id: E-TUBE-03
  ring: ring2
  title: Long stationary dwell
  trigger: Sleeping person; parked object.
  handling: Heartbeat detections with det_source=heartbeat keep the track active.
  status: implemented
- id: E-TUBE-04
  ring: ring2
  title: Re-entry after leaving
  trigger: Same person returns minutes later.
  handling: New tube; fusion links to the same entity via ReID + gallery; the agent reports entity, not tube.
  status: planned
- id: E-TUBE-05
  ring: ring2
  title: Same person in two overlapping cameras
  trigger: Overlap region.
  handling: Per-camera tracker never merges; fusion merges by floor distance + ReID + time; enrichment once per entity, best view elected.
  status: planned
- id: E-TUBE-06
  ring: ring2
  title: Hand-off outside transit bounds
  trigger: Entity appears in a non-adjacent tile too fast.
  handling: impossible_transition event; no merge; anomaly surfaced.
  status: planned
- id: E-TUBE-07
  ring: ring2
  title: Blind-spot ambiguity
  trigger: Two enter a blind spot, one exits.
  handling: Delay-state keeps both candidates with probabilities; answer states ambiguity.
  status: deferred
- id: E-TUBE-08
  ring: ring2
  title: Appearance drift within a day
  trigger: Jacket on/off; bag picked up.
  handling: Multiple exemplars per entity; embedding refresh on confident matches; oldest expire.
  status: planned
- id: E-TUBE-09
  ring: ring2
  title: Carried object is not its own tube
  trigger: Bag on shoulder; keys in hand.
  handling: carried_item attribute on the person tube; object tube only when placed and stationary.
  status: planned
- id: E-TUBE-10
  ring: ring2
  title: Pets, strollers, wheelchairs
  trigger: Non-person moving classes; child inside stroller.
  handling: Class-specific lifecycle; stroller+child handled as one tube with carried_item=child hint.
  status: deferred
- id: E-TUBE-13
  ring: ring2
  title: Fragmentation at sampled frame rate
  trigger: Decode at 2 fps means a walking person moves half a box width between ticks; IoU association breaks the tube.
  handling: ByteTracker with dt-aware Kalman, buffered IoU and a first-tick centre gate keeps fragmentation at 1.0 down to ~4-6 fps (synthetic); below ~3 fps motion-only association is ambiguous by construction, so active tiles decode at >=4 fps and 2 fps is for quiet tiles only.
  status: implemented
- id: E-TUBE-12
  ring: ring2
  title: Feet occluded, foot point wrong
  trigger: Person behind counter.
  handling: FloorPoint.source=fallback_bbox_bottom and uncertainty widened; fusion tolerances read it.
  status: implemented
# ---------------- ring 3a events ----------------
- id: E-EVT-01
  ring: ring3a
  title: Zone boundary jitter
  trigger: Foot point oscillates on a zone edge.
  handling: 'Hysteresis: enter after N consecutive inside ticks, exit after N outside.'
  status: implemented
- id: E-EVT-10
  ring: ring3a
  title: Everyone in frame at episode open "enters"
  trigger: First tick births every visible tube inside a zone; enter_zone fires for all of them at once.
  handling: EventCompiler seeds zone memberships silently on its first tick; enter_zone means a boundary was crossed during the episode.
  status: implemented
- id: E-EVT-02
  ring: ring3a
  title: Cross-camera event order under clock offset
  trigger: Kitchen camera 2 s ahead of hallway.
  handling: Sort by corrected_ms; hand-off window tolerates offset_confidence.
  status: planned
- id: E-EVT-03
  ring: ring3a
  title: False pickup from occlusion of the shelf
  trigger: Person stands in front of the asset home; heartbeat cannot see the asset.
  handling: Pickup requires asset absent on a heartbeat taken with nobody inside the zone, after a present reading; subject = visitors in between; no visitors => asset_missing_from_home with no blame.
  status: implemented
- id: E-EVT-04
  ring: ring3a
  title: Fall vs lying down on purpose
  trigger: Person lies on sofa/bed.
  handling: rest zones from scene card suppress fall; fall requires vertical velocity + pose primitive outside rest zones.
  status: deferred
- id: E-EVT-05
  ring: ring3a
  title: Left-behind vs placed at home
  trigger: Bag set on its usual hook.
  handling: left_behind only outside the object's home zone.
  status: planned
- id: E-EVT-06
  ring: ring3a
  title: Duplicate events from overlapping cameras
  trigger: Both cameras see the same enter.
  handling: dedupe_key = type|entity|time bucket at fusion; per-camera events keep camera_id for evidence.
  status: implemented
- id: E-EVT-07
  ring: ring3a
  title: Episode never closes
  trigger: Busy lobby active for hours.
  handling: soft_cut on cast churn > threshold or max duration; events may span episodes.
  status: implemented
- id: E-EVT-08
  ring: ring3a
  title: Event spans an episode boundary
  trigger: Dwell starts in one episode and ends in the next.
  handling: Events carry episode_id of emission; queries join across episodes by entity and time, never by episode alone.
  status: planned
# ---------------- ring 3b foveation / enrichment ----------------
- id: E-FOV-01
  ring: ring3b
  title: Crop too small to describe
  trigger: Short side < 96 px.
  handling: Enhancement ladder; enhanced=true propagates to attributes; confidence discounted.
  status: implemented
- id: E-FOV-02
  ring: ring3b
  title: Motion blur
  trigger: Fast movement.
  handling: Keyframe score uses Laplacian sharpness; blurred frames skipped; if none sharp, attributes low confidence.
  status: planned
- id: E-FOV-03
  ring: ring3b
  title: Backlit or overexposed subject
  trigger: Doorway against daylight.
  handling: Exposure score in keyframe selection; try other keyframes; color_reason=low_confidence.
  status: planned
- id: E-FOV-04
  ring: ring3b
  title: Cross-cell attribute bleed in contact sheets
  trigger: VLM describes cell 3 with cell 4's jacket.
  handling: Hard borders + cell ids; fixed expected_cells; per-cell schema; bleed rate measured on eval set; fall back to 12 cells if >5%.
  status: implemented
- id: E-FOV-05
  ring: ring3b
  title: Hallucinated color in IR
  trigger: Monochrome crop.
  handling: Attributes validator and grammar forbid colors when modality is ir/thermal; color_reason=ir_mode; tone fields instead.
  status: implemented
- id: E-FOV-06
  ring: ring3b
  title: Ring buffer expired before enrichment
  trigger: Slow path lags past buffer window.
  handling: Best keyframe crop persisted to R2 at tube birth; keyframe_refs on the tube.
  status: implemented
- id: E-FOV-07
  ring: ring3b
  title: VLM output violates schema
  trigger: Grammar bug or truncation.
  handling: Retry once; then EnrichmentPatch{failed=true}; tick never blocked.
  status: implemented
- id: E-FOV-08
  ring: ring3b
  title: Attributes contradict across keyframes
  trigger: Red jacket in one, dark in another.
  handling: Majority vote over keyframes; keep history; confidence reflects agreement.
  status: planned
# ---------------- storage / handoff ----------------
- id: E-STO-01
  ring: storage
  title: Patch arrives after episode closed
  trigger: Slow path finishes late.
  handling: Episode file is append-only; PatchRecord accepted after close.
  status: implemented
- id: E-STO-02
  ring: storage
  title: Retries create duplicates
  trigger: Writer crashes and replays.
  handling: Deterministic episode/event ids; per-record idempotency keys.
  status: implemented
- id: E-STO-03
  ring: storage
  title: Retention expired for a clip
  trigger: Clip requested for footage past retention.
  handling: clip tool checks availability; degrades to keyframes; answer states retention.
  status: planned
- id: E-STO-04
  ring: storage
  title: Schema version bump
  trigger: New field added mid-deployment.
  handling: Provenance.schema_version on every record; readers tolerate unknown fields; migration script per bump.
  status: implemented
- id: E-STO-05
  ring: storage
  title: KB version drift
  trigger: Transit bounds changed after an episode was compiled.
  handling: Provenance.kb_version on every record; replays pin the version.
  status: implemented
# ---------------- block 2 agent ----------------
- id: E-AGT-01
  ring: agent
  title: Ambiguous anchor
  trigger: '"After Jay went out" matches two exits.'
  handling: One clarifying question; never a silent pick.
  status: planned
- id: E-AGT-02
  ring: agent
  title: No results
  trigger: Nothing matches the window.
  handling: Say so; propose widening; never fabricate.
  status: planned
- id: E-AGT-03
  ring: agent
  title: Unknown named entity
  trigger: '"Jay" not in gallery.'
  handling: Ask who Jay is; offer naming form.
  status: planned
- id: E-AGT-04
  ring: agent
  title: Fuzzy time expressions
  trigger: '"yesterday afternoon", "after lunch".'
  handling: Resolve to explicit local window; echo it in the answer.
  status: planned
- id: E-AGT-05
  ring: agent
  title: Color question on IR footage
  trigger: '"what color was his shirt" at night.'
  handling: State IR mode; give tone; never a color.
  status: planned
- id: E-AGT-06
  ring: agent
  title: Unverified claim
  trigger: Attribute from a low-confidence patch.
  handling: verify() before asserting; every claim cites record ids and timestamps.
  status: planned
- id: E-AGT-07
  ring: agent
  title: Broken custody chain
  trigger: carried_item missed; drop never fired.
  handling: 'Graceful degrade: last known custody + offer clip.'
  status: planned
- id: E-AGT-08
  ring: agent
  title: Duplicates from overlapping cameras in the answer
  trigger: Same person listed twice.
  handling: Answer over entities, not tubes.
  status: planned
# ---------------- KB / calibration ----------------
- id: E-KB-01
  ring: kb
  title: Camera moved after calibration
  trigger: Bumped or remounted.
  handling: Sustained global motion => camera_moved_suspect; landmark check on I-frames; freeze fusion; prompt recalibration.
  status: implemented
- id: E-KB-02
  ring: kb
  title: Scene card labels wrong
  trigger: '"Dining hall" is actually the kitchen.'
  handling: Everything on the card is a hypothesis; user correction bumps version and re-labels.
  status: planned
- id: E-KB-03
  ring: kb
  title: Causal fact contradicted
  trigger: Switch flipped, light did not change.
  handling: Fact.add_contradiction; retire at 2 unless user_confirmed.
  status: implemented
- id: E-KB-04
  ring: kb
  title: Furniture moved
  trigger: Shelf relocated; asset home zone stale.
  handling: Repeated asset_missing with asset detected elsewhere => propose new home.
  status: deferred
# ---------------- night ----------------
- id: E-NIGHT-01
  ring: night
  title: IR switch mid-tube
  trigger: Camera toggles IR while a person is in view.
  handling: modality_switch event; tube segments carry modality; attributes per segment.
  status: planned
- id: E-NIGHT-02
  ring: night
  title: ReID does not transfer day to IR
  trigger: Same person, different modality.
  handling: Separate exemplar sets per modality; tile/time continuity weighted higher at night.
  status: planned
- id: E-NIGHT-03
  ring: night
  title: IR blooming near camera
  trigger: Subject washed out by illuminator.
  handling: Exposure score rejects keyframe; low confidence attributes.
  status: deferred
# ---------------- privacy ----------------
- id: E-PRIV-01
  ring: privacy
  title: Gallery entry deletion
  trigger: User deletes a named person.
  handling: 'Cascade: exemplars, names on records, KB facts referencing the entity; anonymous ids remain.'
  status: planned
- id: E-PRIV-02
  ring: privacy
  title: Face embeddings without opt-in
  trigger: Face model enabled by default.
  handling: Site-level opt-in flag gates the face lane; default off.
  status: planned
EOF_VI
cat > SPEC.md << 'EOF_VI'
# vi-engine — frozen specification (v0.1, 2026-09-23)

This file is the canonical design. If code and this file disagree, fix one of them the same day.
Anything not in here is not settled. Sections marked **MEASURE** are open until a row in the
benchmark table (PLAN.md) fills them.

## 0. Purpose and hard constraints

A natural-language intelligence layer over video sources (live cameras, uploads, archives) that
answers open-ended questions with timestamped, evidence-backed results. Horizontal engine, not a
surveillance product.

- **C1 No training.** No fine-tuning, no LoRA, no distillation. Adaptation happens only through
  exemplar galleries, prompts, schemas, thresholds and the knowledge base.
- **C2 Open weights, commercial licenses.** Apache-2.0 / MIT / BSD / SAM License only in the
  serving path. AGPL/GPL (Ultralytics, BoxMOT, GPL YOLO forks) are reference-only.
- **C3 Compute scales with activity, not cameras × fps.** The unit of work is the object tube and
  the tube event; frames are transport.
- **C4 Everything a model could hallucinate is constrained.** Enum, nullable-with-reason, or
  confidence. Grammar-constrained decoding for every model output.
- **C5 Two-speed emission.** Geometry and events commit within ~100 ms of tick close; semantics
  arrive later as patches. No consumer blocks on the slow path.

## 1. Topology (Block 1)

```
cameras ─► Ring 0 bitstream gate (CPU, no decode)
           └► Ring 1 selective decode + batched ROI detection (GPU)
              └► Ring 2 tube assembly + world fusion (CPU)
                 ├► Ring 3a event compiler (CPU, fast path)  ─┐
                 └► Ring 3b foveation + enrichers (GPU, slow) ─┴► episode files ─► Block 2
```

**Ring 0 — bitstream gate.** Motion vectors + macroblock metadata parsed from H.264/H.265, no pixel
reconstruction on the CPU path (libavcodec `+export_mvs` with loop-filter/IDCT skipped, or
PyNvVideoCodec 2.1 decode statistics when the camera is already on NVDEC). Per-camera adaptive
noise floor, MV-field coherence to reject PTZ/shake, luminance-step detection for scene-state.
Stationary blindness is by design and is covered by **heartbeat detections** on I-frames
(1–2 s active tiles, 10–30 s quiet tiles, always on asset-home zones). Fallback for MJPEG /
intra-only streams: frame differencing at 1 fps behind the same interface (`vi/gate/base.py`).
Slice implementation: `FrameDiffGate`. Production: `MVGate` (week 3). **MEASURE:** gate FN rate,
ms per GOP per stream.

**Ring 1 — selective decode + detect.** Decode only gated cameras at 2–5 fps sampled on NVDEC
(PyNvVideoCodec, MIT). Pack motion ROIs from many cameras into one batch; one detector forward
per batch; TensorRT FP16 on server, ONNX Runtime on edge. Detector: RF-DETR 1.7.0 Nano (edge) /
Medium–Large (server), Apache-2.0 sizes only. Client nouns via SAM 3 concept + exemplar prompts
at tube-event cadence. Every detection carries a ReID embedding. **MEASURE:** ms per packed
batch, mAP on ROIs.

**Ring 2 — tubes + fusion.** Per-camera `ByteTracker` written from the papers (dt-aware
constant-velocity Kalman, two-stage association, buffered IoU, first-tick centre gate; no code
from the ByteTrack/BoT-SORT repos, whose Kalman file traces to GPL Deep SORT), motion only on
edge, appearance-assisted on server. Measured (synthetic crossings): fragmentation 1.0 and zero ID
switches down to 6 fps; below ~3 fps motion-only association is ambiguous by construction, so
active tiles decode at ≥4 fps and R11's 2 fps floor applies to quiet tiles. Explicit lifecycle
`born → active → occluded → exited|lost → dead`. Ambiguous association records
`merge_candidates`, never silently merges. Fusion lifts tubes to world entities via homography
foot points + tile-graph transit bounds + ReID cosine; overlapping cameras merge into one
entity; best view elected for enrichment. Gallery match stamps names; misses stay anonymous
with stable ids. Slice: `SimpleIoUTracker`. **MEASURE:** HOTA/IDF1, cross-camera merge accuracy.

**Ring 3a — event compiler.** Deterministic predicates over tube snapshots, zones, tile graph and
gate results; zero model calls. Tube events: enter_zone, exit_zone, dwell, loiter, approach,
meet, pickup, drop, left_behind, asset_missing_from_home, fall, run, crowd, handoff,
impossible_transition. Scene-state events: illumination_change, door_state_change,
appliance_state_change, modality_switch. Ingest events: signal_lost/restored,
camera_moved_suspect. Zone hysteresis on enter/exit. `pickup` requires the asset absent on a
heartbeat taken with nobody in the zone after a present reading; subject = visitors in between;
no visitors → `asset_missing_from_home` with no subject. Pickup/drop write the custody table.

**Ring 3b — foveation + enrichers.** Fires on tube events only (birth, quality-scored appearance
change, death). Keyframe score: area, Laplacian sharpness, low MV magnitude, occlusion-free,
exposure. Native-res crop from a per-camera ring buffer, +20% pad, macroblock-aligned; best crop
persisted at birth (`keyframe_refs`). Enhancement ladder: ≥224 px none; 96–224 px tracker-aligned
multi-frame denoise; <96 px single-image SR, `enhanced=true` propagated and confidence discounted
×0.7. Contact sheets: 12–16 crops, hard borders, cell ids, fixed `expected_cells`, one
Qwen3.5-4B call under grammar → `ContactSheetResult`. Specialists in parallel on the same crop:
pose (RTMPose-m), OCR (PP-OCRv6, vehicle/label tubes only), zero-shot tags (SigLIP 2), ReID
refresh. Each writes an `EnrichmentPatch`; schema violation → retry once → `failed=true`.
**MEASURE:** attribute accuracy, cross-cell bleed rate, ms per sheet.

**Episodes.** Episode = activity-bounded window per tile (first tube birth → tile quiet +
hysteresis), soft-cut on cast churn ≥0.6 or 30 min. File = append-only JSONL: header, ticks,
events, patches (may arrive after close), close-with-cast. Deterministic ids. Postgres 18 holds
ticks/events/entities/custody/KB; R2 holds crops and GOPs. Retrieval: BM25 (pg_search or
VectorChord-BM25) + pgvector 0.8.6, RRF fusion, rerank.

## 2. Block 2 — agent

Reasoning model (Qwen3.8-27B, fallback Qwen3.5-27B) with tools `search`, `get_script`,
`create_rule`, `run_check`, `clip`, `verify`, `inspect(hypothesis=…)`, `kb_lookup`. It never
receives video; pixels only through `verify`/`inspect` as native-res crops. Loop: ground the
question (entities, tiles, explicit time window) → resolve anchors against named-entity events
(SQL) → hybrid search → read script + verify → assemble crops/clips/naming form. Ambiguous
anchor → one clarifying question. Every claim cites entity ids, timestamps, cameras. Answers are
over world entities, never tubes. Naming form → gallery exemplars + retroactive relabel + relink.

## 3. Calibration and knowledge base

Per camera, per lighting regime: one deep read by the biggest available model → `SceneCard`
(tile hypothesis, assets → SAM 3 masks → zones, actuators with `controls: unknown`, light
sources, exits, reflective surfaces, blind regions, media zones, floor polygon, ground points,
camera pose). Everything is a hypothesis. **Movement walk** fits homography scale, derives
tile-graph edges and transit bounds, detects overlaps. **Actuation walk** toggles every switch
and door once to seed causal facts. KB = typed property graph in Postgres; every edge is a
`Fact{status, confidence, source, support, contradictions, evidence, version}`; promote at 2
supports or user confirmation, retire at 2 contradictions unless user-confirmed. Unknowns are
stored explicitly. Fact miner (backlog lane): scene-state event → actuation-like events site-wide
within ±1 s → batch adjudication by the reasoning model → hypothesis → user prompt. Every
episode records the KB version it was compiled under.

## 4. Night modality

Per-camera mode detection (chroma≈0 → IR; noise profile → low-light RGB). Mode switch is a
scene-state event. Ring 0: night threshold profile, coherence weighted higher. Ring 1: raw
frames, at most gamma/CLAHE; no per-frame learned enhancer unless the night eval shows a gap and
the license is clear. Ring 3b: enhancement per crop only. Writer schema: in non-color modalities
color fields are null with `color_reason=ir_mode`; tone fields instead; enforced by validator
and grammar. ReID: separate exemplar sets per modality; tile/time continuity weighted higher at
night. Agent states IR mode when colors are asked.

## 5. Supersedes

- Writer VLM as its own tracker → deterministic tube assembly; VLM only in Ring 3b on sheets.
- CPU frame-differencing gate → bitstream MV gate (frame-diff remains the codec fallback).
- Fixed time windows → activity episodes.
- Enhancement-before-detection → enhancement per crop, tagged.

## 6. Requirements (R1–R30, condensed)

Global: R1 no training; R2 no AGPL, BOM in CI; R3 grammar-constrained JSON everywhere; R4 two-speed
emission; R5 per-unit cost metrics per ring; R6 eval set before tuning.
Ring 0: R7 MV/decode-stats gate; R8 adaptive thresholds + coherence; R9 codec fallback; R10 heartbeats.
Ring 1: R11 NVDEC selective decode; R12 packed batches, Apache-only detector; R13 open-vocab at
event cadence; R14 ReID with every detection.
Ring 2: R15 tracker from source; R16 explicit lifecycle; R17 fusion by homography+graph+ReID;
R18 gallery match in fusion.
Ring 3a: R19 deterministic events + custody table.
Ring 3b: R20 keyframe scoring; R21 tagged enhancement ladder; R22 contact sheets with
`carried_item`; R23 parallel specialists as independent patches.
Storage: R24 episode JSONL + Postgres + R2; R25 hybrid retrieval.
Agent: R26 no video to the reasoning model; R27 clarify on ambiguity, cite everything.
Identity/edge: R28 gallery lifecycle + opt-in face + deletable; R29 rings 0–2 on edge;
R30 SAM License compliance reviewed.

## 7. Bill of materials

| Role | Pick | Version | License |
|---|---|---|---|
| Writer VLM | Qwen3.5-4B (9B if needed; 2B on edge) | HF Qwen/Qwen3.5-4B (Mar 2026) | Apache-2.0 |
| Reasoning agent | Qwen3.8-27B → fallback Qwen3.5-27B | HF Qwen/Qwen3.8-27B (Aug 2026) **verify** | Apache-2.0 (reported) |
| Detector | RF-DETR Nano (edge) / Medium–Large (server) | rfdetr 1.7.0 | Apache-2.0 |
| Open-vocab + inspect | SAM 3 / 3.1 | facebook/sam3 | SAM License |
| Counting | CountGD | HF nikigoli/CountGD | MIT |
| Tracker | BoT-SORT + ByteTrack from source | own | MIT originals |
| ReID | OSNet (torchreid); CLIP-ReID **verify license** | torchreid ckpts | MIT |
| Pose | RTMPose-m | mmpose | Apache-2.0 |
| OCR | PP-OCRv6 small/tiny | PaddleOCR 3.7.0 | Apache-2.0 |
| Tags / image emb. | SigLIP 2 base-patch16 | google/siglip2-base-patch16-224 | Apache-2.0 |
| Text emb. | Qwen3-Embedding-0.6B | HF | Apache-2.0 |
| Single-image SR | Real-ESRGAN x4plus | RealESRGAN_x4plus.pth | BSD-3 |
| GPU decode + MV stats | PyNvVideoCodec | 2.1 | MIT |
| CPU MV extraction | PyAV / libavcodec `+export_mvs` | FFmpeg 7.x | LGPL (dynamic) |
| LLM serving | vLLM + xgrammar (SGLang alternate) | ≥0.12 API, pin stable | Apache-2.0 |
| DB | PostgreSQL 18 + pgvector 0.8.6 + pg_search **verify license** / VectorChord-BM25 | — | PG / see note |
| Object storage | Cloudflare R2 | — | — |
| Eval | TrackEval | main | MIT |

Pins to verify before commit: Qwen3.8-27B card; MVTrack weights (else classical MV clustering);
pg_search license; SAM 3.1 video path in the loader; CLIP-ReID license.

## 8. Edge cases

`edge_cases.yaml` is the registry; `make coverage` fails when an implemented case has no test.
See EDGE_CASES.md (generated).
EOF_VI
cp "$0" colab/sessions/session_05_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. Ring 2: SimpleIoUTracker vs ByteTracker on synthetic crossings (CPU)"
cat > /tmp/_row.py << 'EOF_PY'
import json, sys
r = json.loads(sys.stdin.read())
print(f"  {r['tracker']:18s} {r['effective_fps']:5.1f} fps  mota={r['mota']:.3f} idf1={r['idf1']:.3f} idsw={r['idsw']:2d} frag={r['fragmentation_ratio']:.2f}")
EOF_PY
for every in 1 5 10; do
  for trk in simple byte; do
    python bench/ring2_tubes.py --synthetic --tracker $trk --sample-every $every --fps 30 | python /tmp/_row.py
  done
done

step "5. real-footage slice with ByteTracker (GPU)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  if [ -f "$SOURCE" ]; then
    python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --max-frames 400 2>&1 | grep -v TracerWarning | grep -v "^  \(assert\|if\|elif\|topk\)"
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — switch to L4 and rerun for the slice"
fi

step "6. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: ByteTracker from the papers (dt-aware Kalman, two-stage, buffered IoU), tracker contract parametrized, first-tick seeding (E-EVT-10), E-TUBE-13 measured"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "7. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "bench rows: $(cat data/bench/*.jsonl 2>/dev/null | wc -l)"
echo "compare person_tubes in the new slice row against session 04's 21, and lost vs exited."
