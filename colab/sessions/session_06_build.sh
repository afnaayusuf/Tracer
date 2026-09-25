#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 06 build: heartbeat scheduler (full-frame detection at open,
#  1 s active / 10 s quiet), hybrid ROI+heartbeat detection, open grace window for zone
#  seeding, tube continuity metrics, CPU fake detector + end-to-end smoke tests.
#  Experiment: the slice runs on your clip three times (roi | frame | hybrid) and prints the
#  rows side by side so the detection regime is chosen by measurement.
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_06.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 06"
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
[ -n "$(git log --grep='^session 05:' --format=%h)" ] || die "session 05 commit not found; run build_session_05.sh first"
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
tests/test_slice_gpu_smoke.py
tests/test_tubes.py
vi/__init__.py
vi/detect/__init__.py
vi/detect/base.py
vi/detect/fake.py
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
vi/gate/heartbeat.py
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
cat > vi/gate/heartbeat.py << 'EOF_VI'
from __future__ import annotations


class HeartbeatScheduler:
    """R10 / E-GATE-01 / E-GATE-05: the motion gate is blind to whatever is not moving, so a
    full-frame detection runs on a clock the gate cannot suppress: at episode open, every
    `active_ms` while the tile has live tubes, every `quiet_ms` otherwise. Asset-home zones are
    checked on every heartbeat. The scheduler is pure so it can be tested without video."""

    def __init__(self, active_ms: int = 1000, quiet_ms: int = 10_000, fire_at_open: bool = True):
        self.active_ms = active_ms
        self.quiet_ms = quiet_ms
        self.fire_at_open = fire_at_open
        self._last_ms: int | None = None
        self.fired = 0

    def due(self, t_ms: int, tile_active: bool) -> bool:
        if self._last_ms is None:
            if self.fire_at_open:
                self._last_ms = t_ms
                self.fired += 1
                return True
            self._last_ms = t_ms
            return False
        period = self.active_ms if tile_active else self.quiet_ms
        if t_ms - self._last_ms >= period:
            self._last_ms = t_ms
            self.fired += 1
            return True
        return False
EOF_VI
cat > vi/gate/__init__.py << 'EOF_VI'
from .base import Gate, GateResult, MotionBlob
from .framediff import FrameDiffGate
from .heartbeat import HeartbeatScheduler
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
        high = [d for d in detections if d.confidence >= self.high_thr]
        low = [d for d in detections if self.low_thr <= d.confidence < self.high_thr]
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
cat > vi/detect/roi.py << 'EOF_VI'
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from vi.gate.base import MotionBlob
from vi.schemas import Box

from .base import Detection

EDGE_PX = 2


class ROI(BaseModel):
    """A crop window in native frame coordinates (ints, inclusive-exclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int
    source_blobs: int = 1

    @property
    def box(self) -> Box:
        return Box(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)


def blobs_to_rois(blobs: list[MotionBlob], frame_w: int, frame_h: int, stride: int = 1,
                  pad: float = 0.25, min_side: int = 96, merge_iou: float = 0.05) -> list[ROI]:
    """R12: turn gate blobs (in gate coordinates, downscaled by `stride`) into padded,
    clamped, merged crop windows in native coordinates. Overlapping windows are merged so
    one object never yields two crops; tiny windows are grown to `min_side` so the
    detector sees enough context."""
    boxes: list[Box] = []
    for b in blobs:
        x1, y1, x2, y2 = b.box.x1 * stride, b.box.y1 * stride, b.box.x2 * stride, b.box.y2 * stride
        w, h = x2 - x1, y2 - y1
        px, py = max(w * pad, (min_side - w) / 2, 0), max(h * pad, (min_side - h) / 2, 0)
        boxes.append(Box(x1=max(0, x1 - px), y1=max(0, y1 - py),
                         x2=min(frame_w, x2 + px), y2=min(frame_h, y2 + py)))
    # greedy merge of overlapping windows
    merged: list[tuple[Box, int]] = []
    for bx in sorted(boxes, key=lambda b: -b.area):
        for i, (m, n) in enumerate(merged):
            if m.iou(bx) > merge_iou or _contains(m, bx):
                merged[i] = (Box(x1=min(m.x1, bx.x1), y1=min(m.y1, bx.y1),
                                 x2=max(m.x2, bx.x2), y2=max(m.y2, bx.y2)), n + 1)
                break
        else:
            merged.append((bx, 1))
    return [ROI(x1=int(m.x1), y1=int(m.y1), x2=int(np.ceil(m.x2)), y2=int(np.ceil(m.y2)), source_blobs=n)
            for m, n in merged if m.x2 - m.x1 >= 8 and m.y2 - m.y1 >= 8]


def _contains(outer: Box, inner: Box) -> bool:
    return outer.x1 <= inner.x1 and outer.y1 <= inner.y1 and outer.x2 >= inner.x2 and outer.y2 >= inner.y2


def crop_roi(frame_rgb: np.ndarray, roi: ROI) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[roi.y1:roi.y2, roi.x1:roi.x2])


def remap_detections(dets: list[Detection], roi: ROI, frame_w: int, frame_h: int) -> list[Detection]:
    """Shift crop-space boxes back to frame space and flag boxes touching the frame border
    (E-DET-02): a person cut off at the edge has an unreliable foot point."""
    out = []
    for d in dets:
        b = Box(x1=d.box.x1 + roi.x1, y1=d.box.y1 + roi.y1, x2=d.box.x2 + roi.x1, y2=d.box.y2 + roi.y1)
        truncated = b.x1 <= EDGE_PX or b.y1 <= EDGE_PX or b.x2 >= frame_w - EDGE_PX or b.y2 >= frame_h - EDGE_PX
        out.append(d.model_copy(update={"box": b, "truncated": truncated}))
    return out


def pad_batch(crops: list[np.ndarray], batch_size: int) -> tuple[list[np.ndarray], int]:
    """A traced model runs at one fixed batch size; pad with the last crop and report how
    many entries are real so the caller drops the padding."""
    if not crops:
        return [], 0
    real = len(crops)
    padded = list(crops[:batch_size])
    while len(padded) < batch_size:
        padded.append(padded[-1])
    return padded, min(real, batch_size)


def merge_detections(primary: list[Detection], secondary: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Union of two detection sets on the same frame; when boxes overlap above iou_thr the
    higher-confidence one is kept. Used to combine ROI detections with a full-frame heartbeat."""
    out = list(primary)
    for d in secondary:
        dup = None
        for i, p in enumerate(out):
            if p.class_label == d.class_label and p.box.iou(d.box) >= iou_thr:
                dup = i
                break
        if dup is None:
            out.append(d)
        elif d.confidence > out[dup].confidence:
            out[dup] = d
    return out
EOF_VI
cat > vi/detect/__init__.py << 'EOF_VI'
from .base import TUBE_CLASSES, Detection, Detector, is_tube_class
from .roi import ROI, blobs_to_rois, crop_roi, merge_detections, pad_batch, remap_detections
EOF_VI
cat > vi/detect/fake.py << 'EOF_VI'
from __future__ import annotations

import numpy as np

from vi.gate.framediff import FrameDiffGate
from vi.schemas import Box

from .base import Detection


class BrightBlobDetector:
    """CPU stand-in for RF-DETR: 'detects' bright rectangles (the synthetic walker) so the whole
    GPU slice, including ROI packing, heartbeats and merging, can run in tests without a GPU."""

    resolution = 0
    optimized = False

    def __init__(self, threshold: float = 0.1, luma: int = 200, min_side: int = 12, **_):
        self.threshold = threshold
        self.luma = luma
        self.min_side = min_side

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        gray = image_rgb.mean(axis=2) if image_rgb.ndim == 3 else image_rgb
        mask = gray > self.luma
        h, w = mask.shape
        b = 4
        mb = mask[: h // b * b, : w // b * b].reshape(h // b, b, w // b, b).mean(axis=(1, 3)) > 0.5
        dets = []
        for comp in FrameDiffGate._components(mb):
            ii = [c[0] for c in comp]
            jj = [c[1] for c in comp]
            box = Box(x1=min(jj) * b, y1=min(ii) * b, x2=(max(jj) + 1) * b, y2=(max(ii) + 1) * b)
            if box.short_side >= self.min_side:
                dets.append(Detection(box=box, class_label="person", confidence=0.9))
        return dets

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        return [self.detect(c) for c in crops]
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
                 moved_after_updates: int = 10, open_grace_ms: int = 1500):
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
        self.open_grace_ms = open_grace_ms
        self._open_t_ms: int | None = None

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
        if self._open_t_ms is None:
            self._open_t_ms = t_ms
        self._ticks_seen += 1
        if t_ms - self._open_t_ms <= self.open_grace_ms:
            # E-EVT-10: whoever is already in frame when the episode opens did not "enter";
            # seed zone memberships silently (for a short grace window, since the gate needs
            # a frame or two to warm up) so enter_zone means a boundary was crossed.
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

  python bench/slice_gpu.py --source clip.mp4 --detect hybrid      # ROIs + 1 Hz full-frame heartbeat (default)
  python bench/slice_gpu.py --source clip.mp4 --detect roi         # motion ROIs only (session 04/05 behaviour)
  python bench/slice_gpu.py --source clip.mp4 --detect frame       # full frame every tick (upper bound on recall)
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

from vi.detect import blobs_to_rois, crop_roi, is_tube_class, merge_detections, pad_batch, remap_detections
try:
    from vi.detect.rfdetr import RFDETRDetector
except ImportError:  # CPU runtime: only --model fake works
    RFDETRDetector = None  # type: ignore
from vi.episode import EpisodeWriter, KeyframeStore
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate, HeartbeatScheduler
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
    ap.add_argument("--detect", choices=["roi", "frame", "hybrid"], default="hybrid",
                    help="roi: motion ROIs only; frame: full frame every tick; hybrid: ROIs + full-frame heartbeat")
    ap.add_argument("--heartbeat-ms", type=int, default=1000, help="hybrid: full-frame detection period on active tiles")
    ap.add_argument("--quiet-heartbeat-ms", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default = edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=640, want_rgb=True)
    if a.model == "fake":
        from vi.detect.fake import BrightBlobDetector
        det = BrightBlobDetector(threshold=a.threshold)          # CPU smoke path (tests, no GPU)
    else:
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
    hb = HeartbeatScheduler(active_ms=a.heartbeat_ms, quiet_ms=a.quiet_heartbeat_ms)
    heartbeats = 0
    person_dets: list[int] = []
    concurrent_persons: list[int] = []
    state_ticks: Counter = Counter()

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
        dets = []
        det_source = "detector"
        if a.detect in ("roi", "hybrid"):
            rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride)
            for i in range(0, len(rois), a.batch):
                chunk = rois[i:i + a.batch]
                crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], a.batch)
                for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                    dets += remap_detections(d, r, w, h)
        tile_active = len(tracker._tracks) > 0
        if a.detect == "frame" or (a.detect == "hybrid" and hb.due(fr.pts_ms, tile_active)):
            full = det.detect(fr.rgb)
            dets = merge_detections(dets, full) if dets else full
            det_source = "heartbeat"
            heartbeats += 1
        dets = [d for d in dets if is_tube_class(d.class_label)]
        person_dets.append(sum(1 for d in dets if d.class_label == "person" and d.confidence >= 0.5))
        dt_detect = time.perf_counter() - t0
        if frames >= a.warmup:
            stage["detect"] += dt_detect
            detect_ms.append(dt_detect * 1000)
        t0 = time.perf_counter()
        before = len(tracker._tracks)
        live, closed = tracker.update(dets, fr.pts_ms, det_source=det_source)
        concurrent_persons.append(sum(1 for t in live if t.class_label == "person" and t.state.value == "active"))
        state_ticks.update(t.state.value for t in live if t.class_label == "person")
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
        "tracker": a.tracker, "detect": a.detect, "det_threshold": a.threshold, "heartbeats": heartbeats,
        "frames": frames, "decode_ms_per_frame": round(st.decode_ms_total / frames, 2),
        **{f"{k}_ms_per_frame": round(v * 1000 / frames, 2) for k, v in stage.items() if k != "detect"},
        "detect_ms_p50": round(float(np.median(detect_ms)), 2) if detect_ms else None,
        "detect_ms_p95": round(float(np.percentile(detect_ms, 95)), 2) if detect_ms else None,
        "tubes_total": len(tubes), "tubes_live_at_end": len(tracker._tracks),
        "person_tubes": sum(1 for t in tubes if t.class_label == "person"),
        "person_dets_per_frame": round(float(np.mean(person_dets)), 2) if person_dets else 0,
        "max_concurrent_persons": max(concurrent_persons) if concurrent_persons else 0,
        "person_visibility_duty": round(state_ticks["active"] / max(1, state_ticks["active"] + state_ticks["occluded"]), 3),
        "fragmentation_est": round(sum(1 for t in tubes if t.class_label == "person") / max(1.0, float(np.mean(person_dets))), 2) if person_dets else None,
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
cat > bench/README.md << 'EOF_VI'
# bench/

One script per ring. Each prints exactly one row of the benchmark table (see PLAN.md) and
checkpoints it to `data/bench/<ring>.jsonl` as it goes, so a killed Colab session loses nothing.

| script | ring | GPU | dataset | row |
|---|---|---|---|---|
| slice_cpu.py | all (stubs) | none | synthetic | proves plumbing; run first, every day |
| slice_gpu.py | 0–3a on real footage | L4 | own clip | reader → gate → ROIs (+ heartbeat) → RF-DETR → tubes → events → episode file + birth keyframes; `--detect roi\|frame\|hybrid`, person_tubes, visibility duty, fragmentation_est |
| ring0_gate.py | 0 | CPU | VIRAT | gate FN rate, ms per GOP per stream, MV vs framediff |
| ring1_detect.py | 1 | L4 | own clip / VIRAT | `--mode frame`: ms p50/p95 per frame at sampled fps; `--mode roi`: gate → packed ROI batches (nano/medium) |
| ring2_tubes.py | 2 | CPU | synthetic, MOT17 | MOTA/IDF1/IDSW/fragmentation (vi/eval/mot.py); `--tracker simple\|byte`, `--sample-every` simulates 2–5 fps decode |
| ring3b_sheet.py | 3b | A100-40 | crops from ring2 | attribute accuracy, bleed rate (12 vs 16 cells), ms per sheet |
| ring3a_events.py | 3a | CPU | MEVA subset | event precision/recall per type |
| agent_replay.py | block 2 | A100-40 -> G4 | WILDTRACK episodes | both scenarios replay; tool calls, latency |

Colab: GPU runtime, `pip install rfdetr==1.7.0`, then run from the repo root (see colab/README.md).
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
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=2, exit_ticks=2, open_grace_ms=0)
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
    ec = EventCompiler("c1", [zone("shelf", kind="asset_home", asset="bike_keys")], enter_ticks=1, exit_ticks=1,
                       open_grace_ms=0)
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
    a = EventCompiler("camA", [zone("door", cam="camA")], enter_ticks=1, open_grace_ms=0)
    b = EventCompiler("camB", [zone("door", cam="camB")], enter_ticks=1, open_grace_ms=0)
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
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=1, dwell_ms=2000, open_grace_ms=0)
    ec.on_tick([], -500)
    evs = []
    for i in range(6):
        evs += ec.on_tick([inside("a")], i * 1000)
    assert [e.type for e in evs] == [EventType.enter_zone, EventType.dwell]


@pytest.mark.edge("E-EVT-10")
def test_tubes_present_at_episode_open_do_not_enter():
    ec = EventCompiler("c1", [zone("z1")], enter_ticks=1, exit_ticks=1, open_grace_ms=800)
    assert ec.on_tick([], 0) == []                                         # gate warming up: empty tick
    assert ec.on_tick([inside("a"), inside("b")], 333) == []               # discovered inside grace: no enter
    assert ec.on_tick([inside("a"), inside("b")], 666) == []
    evs = ec.on_tick([outside("a"), inside("b"), inside("c")], 1000)      # a leaves, c arrives after grace
    assert sorted(e.type for e in evs) == sorted([EventType.exit_zone, EventType.enter_zone])
    assert [e.subject_tube_ids for e in evs if e.type == EventType.enter_zone] == [["c"]]
EOF_VI
cat > tests/test_gate.py << 'EOF_VI'
import numpy as np
import pytest

from vi.gate import FrameDiffGate
from vi.schemas import Box


def _warm(gate, frame, n=8, sigma=3.0, seed=7):
    rng = np.random.default_rng(seed)
    for i in range(n):   # real cameras never send identical frames: warm with sensor noise
        noisy = (frame.astype(float) + rng.normal(0, sigma, frame.shape)).clip(0, 255).astype(np.uint8)
        gate.update(noisy, i * 100)


@pytest.mark.edge("E-GATE-03")
def test_adaptive_noise_floor_ignores_sensor_noise_but_finds_object(base_frame):
    g = FrameDiffGate("c1")
    rng = np.random.default_rng(1)
    _warm(g, base_frame)
    noisy = (base_frame.astype(float) + rng.normal(0, 3, base_frame.shape)).clip(0, 255).astype(np.uint8)
    assert g.update(noisy, 600).blobs == []
    f = noisy.copy()
    f[100:160, 40:80] = 250
    r = g.update(f, 700)
    assert len(r.blobs) == 1 and r.blobs[0].box.iou(Box(x1=40, y1=100, x2=80, y2=160)) > 0.5


@pytest.mark.edge("E-GATE-02")
def test_light_switch_is_luma_step_not_motion(base_frame):
    g = FrameDiffGate("c1")
    _warm(g, base_frame)
    lit = (base_frame.astype(int) + 60).clip(0, 255).astype(np.uint8)
    r = g.update(lit, 600)
    assert r.luma_step is True and r.blobs == [] and r.global_motion is False


@pytest.mark.edge("E-GATE-04")
def test_camera_shake_is_global_motion_and_suppresses_blobs():
    rng = np.random.default_rng(2)
    textured = rng.integers(0, 255, (240, 320)).astype(np.uint8)
    g = FrameDiffGate("c1")
    _warm(g, textured)
    shifted = np.roll(textured, 12, axis=1)
    r = g.update(shifted, 600)
    assert r.global_motion is True and r.blobs == []


@pytest.mark.edge("E-GATE-01")
def test_stationary_object_fades_from_gate_by_design(base_frame):
    g = FrameDiffGate("c1", bg_alpha=0.3)
    _warm(g, base_frame)
    f = base_frame.copy()
    f[100:160, 40:80] = 250
    assert len(g.update(f, 600).blobs) == 1
    for i in range(1, 40):
        r = g.update(f, 600 + i * 100)
    assert r.blobs == []   # gate is blind to the stationary object; heartbeat detections own it


@pytest.mark.edge("E-GATE-01")
@pytest.mark.edge("E-GATE-05")
def test_heartbeat_fires_at_open_then_by_tile_state():
    from vi.gate import HeartbeatScheduler
    hb = HeartbeatScheduler(active_ms=1000, quiet_ms=5000)
    assert hb.due(0, tile_active=False) is True                 # episode open: always look once
    assert hb.due(500, tile_active=True) is False
    assert hb.due(1000, tile_active=True) is True               # active tile: every second
    assert hb.due(1999, tile_active=True) is False
    assert [hb.due(t, tile_active=False) for t in (2000, 4000, 6000, 7000)] == [False, False, True, False]
    assert hb.fired == 3
EOF_VI
cat > tests/test_detect_roi.py << 'EOF_VI'
import numpy as np
import pytest

from vi.detect import Detection, blobs_to_rois, crop_roi, pad_batch, remap_detections
from vi.gate.base import MotionBlob
from vi.schemas import Box


def blob(x1, y1, x2, y2):
    return MotionBlob(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), energy=1.0)


def test_rois_are_padded_clamped_and_scaled_by_stride():
    rois = blobs_to_rois([blob(0, 0, 32, 48)], frame_w=1280, frame_h=720, stride=2, pad=0.25, min_side=96)
    r = rois[0]
    assert (r.x1, r.y1) == (0, 0)                     # clamped at the frame origin
    assert r.x2 >= 64 * 1.25 and r.y2 >= 96 * 1.25    # scaled by stride, then padded
    assert r.x2 <= 1280 and r.y2 <= 720


def test_small_blobs_grow_to_min_side_and_overlaps_merge():
    rois = blobs_to_rois([blob(100, 100, 108, 108)], 640, 480, stride=1, min_side=96)
    assert rois[0].x2 - rois[0].x1 >= 96 and rois[0].y2 - rois[0].y1 >= 96
    two = blobs_to_rois([blob(100, 100, 160, 200), blob(150, 150, 220, 260)], 640, 480, stride=1)
    assert len(two) == 1 and two[0].source_blobs == 2
    far = blobs_to_rois([blob(0, 0, 40, 40), blob(500, 400, 540, 440)], 640, 480, stride=1)
    assert len(far) == 2


@pytest.mark.edge("E-DET-02")
def test_remap_shifts_boxes_and_flags_frame_border_truncation():
    from vi.detect.roi import ROI
    roi = ROI(x1=100, y1=50, x2=400, y2=350)
    inside = Detection(box=Box(x1=10, y1=10, x2=60, y2=120), class_label="person", confidence=0.9)
    at_edge = Detection(box=Box(x1=0, y1=10, x2=60, y2=120), class_label="person", confidence=0.9)  # crop x=0 -> frame x=100
    bottom = Detection(box=Box(x1=10, y1=200, x2=60, y2=300), class_label="person", confidence=0.9)  # frame y2=350
    out = remap_detections([inside, at_edge, bottom], roi, frame_w=640, frame_h=352)
    assert out[0].box.x1 == 110 and out[0].box.y1 == 60 and out[0].truncated is False
    assert out[1].truncated is False                # touches the ROI edge, not the frame edge
    assert out[2].truncated is True                 # y2=350 within 2 px of frame bottom 352


def test_pad_batch_fills_with_last_crop_and_reports_real_count():
    crops = [np.zeros((10, 10, 3), np.uint8), np.ones((10, 10, 3), np.uint8)]
    padded, real = pad_batch(crops, 4)
    assert len(padded) == 4 and real == 2 and padded[3] is crops[1]
    assert pad_batch([], 4) == ([], 0)
    frame = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    from vi.detect.roi import ROI
    c = crop_roi(frame, ROI(x1=5, y1=2, x2=15, y2=12))
    assert c.shape == (10, 10, 3) and c.flags["C_CONTIGUOUS"]


def test_merge_detections_keeps_higher_confidence_duplicate_and_unions_the_rest():
    from vi.detect import merge_detections
    a = Detection(box=Box(x1=0, y1=0, x2=40, y2=100), class_label="person", confidence=0.6)
    a2 = Detection(box=Box(x1=2, y1=1, x2=41, y2=101), class_label="person", confidence=0.9)
    b = Detection(box=Box(x1=300, y1=0, x2=340, y2=100), class_label="person", confidence=0.7)
    out = merge_detections([a], [a2, b])
    assert len(out) == 2 and out[0].confidence == 0.9 and out[1] is b
EOF_VI
cat > tests/test_slice_gpu_smoke.py << 'EOF_VI'
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("detect", ["roi", "frame", "hybrid"])
def test_slice_runs_end_to_end_with_fake_detector(tmp_path, detect):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip

    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/slice_gpu.py", "--source", str(clip), "--model", "fake",
                          "--detect", detect, "--fps", "5", "--out", str(tmp_path / "episodes")],
                         capture_output=True, text=True, cwd=".",
                         env={**os.environ, "PYTHONPATH": os.getcwd()})   # this checkout, not another editable install
    assert out.returncode == 0, out.stderr[-2000:]
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["detect"] == detect and row["frames"] > 0 and row["episode_records"] > row["frames"]
    assert row["person_tubes"] >= 1
    if detect == "hybrid":
        assert row["heartbeats"] >= 1
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
  handling: HeartbeatScheduler runs a full-frame detection at episode open, every 1 s on active tiles and every 10 s on quiet ones (merged with ROI detections); measured on real footage as person_tubes and lost-vs-exited in bench/slice_gpu.py --detect hybrid.
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
  handling: Same HeartbeatScheduler full-frame pass catches motion below the gate threshold; gate FN rate stays a tracked metric.
  status: implemented
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
  handling: EventCompiler seeds zone memberships silently for open_grace_ms (1.5 s) after its first tick, since the gate needs a frame to warm up; the open heartbeat makes that first tick non-empty.
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
Stationary blindness is by design and is covered by **heartbeat detections**: `HeartbeatScheduler`
runs a full-frame detection at episode open, every 1 s on active tiles and every 10 s on quiet
tiles, merged with the ROI detections (`--detect hybrid` in the slice). Measured on real footage
in session 05: ROI-only detection lost standing people within 2.5 s regardless of tracker. Fallback for MJPEG /
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
cp "$0" colab/sessions/session_06_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness (includes CPU end-to-end slice smoke tests with the fake detector)"
pipi -e ".[dev,ingest]"
make check

step "4. detection-regime experiment on real footage (GPU): roi vs frame vs hybrid"
cat > /tmp/_slice_row.py << 'EOF_PY'
import json, sys
txt = sys.stdin.read()
r = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
print(f"  {r['detect']:7s} person_tubes={r['person_tubes']:3d}  max_concurrent={r['max_concurrent_persons']:2d}  "
      f"dets/frame={r['person_dets_per_frame']:5.2f}  frag_est={r['fragmentation_est']}  duty={r['person_visibility_duty']}  "
      f"exited={r['tubes_by_final_state'].get('exited',0):2d} lost={r['tubes_by_final_state'].get('lost',0):2d}  "
      f"enter={r['events'].get('enter_zone',0):2d} exit={r['events'].get('exit_zone',0):2d}  "
      f"detect_p50={r['detect_ms_p50']}ms  heartbeats={r['heartbeats']}")
EOF_PY
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  if [ -f "$SOURCE" ]; then
    for mode in roi frame hybrid; do
      python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect $mode --max-frames 400 \
        2>/dev/null | python /tmp/_slice_row.py
    done
    echo "  (session 05 baseline was roi: person_tubes=21 exited=3 lost=12 enter=28 exit=11)"
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — switch to L4 and rerun for the experiment"
fi

step "5. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: heartbeat scheduler (R10), hybrid ROI+full-frame detection, open grace window, continuity metrics, fake detector + slice smoke tests"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "6. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "bench rows: $(cat data/bench/*.jsonl 2>/dev/null | wc -l)"
echo "the three rows in step 4 decide the detection regime; paste them back."
