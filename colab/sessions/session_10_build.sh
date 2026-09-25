#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 10 build:
#   * Kalman: predicted height/aspect clamped to 0.6-1.6x the last measurement (E-TUBE-14, the
#     ballooning orange box in the session-08 debug frame)
#   * ByteTracker birth gates: min body height 32 px, person>=0.5, other classes>=0.65 (E-DET-11,
#     the suitcase/bicycle/far-speck tubes from zoomed crops)
#   * privacy: data/keyframes and data/debug removed from git and ignored (face-visible crops of
#     real people were on a public repo). Make the repo private too.
#   * clones with GH_TOKEN so a private repo keeps working
#  Experiment: frame-nano vs frame-medium (same per-call cost at one camera; medium sees far
#  people at 576 px) and hybrid-tick with the new gates.
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_10.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 10"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"

step "0. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"

step "1. audit"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present: $(git status --short | tr '\n' ' ')"
git pull -q --ff-only || die "local branch diverged from origin; resolve manually"
[ -n "$(git log --grep='^session 09' --format=%h)" ] || die "session 09 commit not found; run build_session_09.sh first"
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
vi/episode/debug.py
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
        self.last_meas_h = z[3]
        self.last_meas_a = z[2]
        self.size_band = (0.6, 1.6)   # E-TUBE-14: predicted h and aspect may not drift outside this band

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
        # E-TUBE-14: a box predicted through a long occlusion must not balloon or collapse; a
        # person does not change size while unseen. Clamp size to a band around the last
        # measurement and zero the size velocities once the clamp engages.
        lo, hi = self.size_band
        h_min, h_max = self.last_meas_h * lo, self.last_meas_h * hi
        a_min, a_max = self.last_meas_a * lo, self.last_meas_a * hi
        if not (h_min <= self.x[3] <= h_max):
            self.x[3] = float(np.clip(self.x[3], h_min, h_max)); self.x[7] = 0.0
        if not (a_min <= self.x[2] <= a_max):
            self.x[2] = float(np.clip(self.x[2], a_min, a_max)); self.x[6] = 0.0
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
        self.last_meas_h = z[3]
        self.last_meas_a = z[2]
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
    origin: str = "unknown"     # "full": born from a heartbeat; ROI ticks cannot see it if it is static
    born_t_ms: int = 0


class ByteTracker:
    def __init__(self, camera_id: str, high_thr: float = 0.5, low_thr: float = 0.1,
                 match_iou: float = 0.2, low_match_iou: float = 0.4, buffer: float = 0.4,
                 max_occluded_ms: int = 3000, confirm_ticks: int = 2, ambig_margin: float = 0.1,
                 first_tick_gate: float = 1.0, static_grace_ms: int = 1500,
                 min_birth_height_px: float = 32.0, birth_thr: dict[str, float] | None = None,
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
        self.static_grace_ms = static_grace_ms   # E-GATE-01: unconfirmed heartbeat-born tracks wait for the next heartbeat
        # E-DET-11: births need a body big enough to track and describe, and non-person classes
        # need more confidence than people (zoomed crops read cardboard as "suitcase" at 0.5).
        self.min_birth_height_px = min_birth_height_px
        self.birth_thr = {"person": high_thr, "*": max(high_thr, 0.65)} | (birth_thr or {})
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
            if not tr.confirmed:
                # A heartbeat-born track missed on an ROI tick is not evidence of anything: ROIs are
                # blind to static objects. It survives (still `born`) until the next heartbeat tick or
                # the grace window, whichever comes first; a miss on a heartbeat tick is a real miss.
                waiting = tr.origin == "full" and det_source != "heartbeat" and t_ms - tr.born_t_ms <= self.static_grace_ms
                if not waiting:
                    del self._tracks[tr.tube.tube_id]      # unconfirmed and gone: never a tube
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
            if d.box.height < self.min_birth_height_px:
                continue
            if d.confidence < self.birth_thr.get(d.class_label, self.birth_thr["*"]):
                continue
            tid = self._new_id(t_ms)
            tube = Tube(tube_id=tid, camera_id=self.camera_id, class_label=d.class_label, state=TubeState.born,
                        born=self._t(t_ms), last_seen=self._t(t_ms), box=d.box, modality=self.modality)
            if self.keyframe_sink is not None:
                tube.keyframe_refs.append(self.keyframe_sink(self.camera_id, t_ms, d.box))
            tr = _Track(tube=tube, kf=KalmanBoxFilter(d.box), last_t_ms=t_ms, confirmed=self.confirm_ticks <= 1,
                        origin=getattr(d, "origin", "unknown"), born_t_ms=t_ms)
            if tr.confirmed:
                tube.state = TubeState.active
            self._tracks[tid] = tr
        return [tr.tube for tr in self._tracks.values()], closed
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


@pytest.mark.edge("E-GATE-01")
def test_heartbeat_born_static_track_survives_roi_ticks_and_confirms_on_next_heartbeat():
    """Hybrid detection: a static person is only visible to the 1 Hz full-frame heartbeat."""
    tr = ByteTracker("c1", confirm_ticks=2, static_grace_ms=1500, max_occluded_ms=3000)
    full = det(100, 100, 140, 220).model_copy(update={"origin": "full"})
    live, _ = tr.update([full], 0, det_source="heartbeat")          # born from the heartbeat
    assert live[0].state == TubeState.born
    for t in (250, 500, 750):                                       # ROI ticks: nothing, but not deleted
        live, _ = tr.update([], t, det_source="detector")
        assert len(live) == 1 and live[0].state == TubeState.born
    live, _ = tr.update([full], 1000, det_source="heartbeat")       # next heartbeat confirms it
    assert live[0].state == TubeState.active and tr._tracks[live[0].tube_id].confirmed
    # a crop-born unconfirmed track that misses is still deleted at once
    roi_born = det(400, 100, 440, 220).model_copy(update={"origin": "roi"})
    tr.update([full, roi_born], 1250)
    live, _ = tr.update([full], 1500)
    assert [t.tube_id for t in live] == [live[0].tube_id] and len(live) == 1
    # and a heartbeat-born track that misses ON a heartbeat tick is gone too
    tr2 = ByteTracker("c1", confirm_ticks=2)
    tr2.update([full], 0, det_source="heartbeat")
    live, _ = tr2.update([], 1000, det_source="heartbeat")
    assert live == []


@pytest.mark.edge("E-TUBE-14")
def test_predicted_box_does_not_balloon_through_long_occlusion():
    from vi.tubes import KalmanBoxFilter
    kf = KalmanBoxFilter(box(100, 100, 140, 220))
    # two noisy updates that suggest the box is growing fast
    kf.predict(0.25); kf.update(box(100, 96, 142, 228))
    kf.predict(0.25); kf.update(box(100, 92, 145, 236))
    for _ in range(40):                      # 10 s unseen
        b = kf.predict(0.25)
    assert 0.6 * 144 <= b.height <= 1.6 * 144 and 0.6 * (45 / 144) <= b.width / b.height <= 1.6 * (45 / 144)


@pytest.mark.edge("E-DET-11")
def test_birth_gates_reject_tiny_bodies_and_low_confidence_non_persons():
    tr = ByteTracker("c1", confirm_ticks=1)
    tiny = Detection(box=box(10, 10, 20, 38), class_label="person", confidence=0.9)     # 28 px tall
    bag = Detection(box=box(200, 10, 260, 100), class_label="suitcase", confidence=0.55)
    ok_bag = Detection(box=box(400, 10, 460, 100), class_label="suitcase", confidence=0.7)
    person = Detection(box=box(600, 10, 640, 130), class_label="person", confidence=0.55)
    live, _ = tr.update([tiny, bag, ok_bag, person], 0)
    assert sorted(t.class_label for t in live) == ["person", "suitcase"]
    assert all(t.box.height >= 32 for t in live)
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
- id: E-DET-10
  ring: ring1
  title: One object, several detections
  trigger: A person straddles two adjacent ROI crops, or is seen by both a crop and the full-frame heartbeat with boxes of different extent; each becomes a tube.
  handling: Every tick's detections pass through dedupe_detections (class-wise, IoU>=0.5 or intersection-over-smaller>=0.6); complete boxes beat crop-truncated ones. Measured in session 06 as max_concurrent 13 on a 9-person frame.
  status: implemented
- id: E-DET-11
  ring: ring1
  title: Zoomed crops promote background specks and boxes into confident detections
  trigger: In an ROI crop a 20-px figure at the far dock reads as person 0.6 and cardboard as suitcase 0.5; each becomes a tube (13 suitcase tubes in session 05).
  handling: 'Birth gates in the tracker: minimum box height (32 px) and class-specific confidence (person 0.5, other classes 0.65); re-attachment thresholds unchanged.'
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
- id: E-TUBE-14
  ring: ring2
  title: Predicted box balloons during long occlusion
  trigger: Kalman aspect/height velocities keep integrating while a track is unseen; the predicted box grows over the scene and re-attaches to junk (seen in the session-08 debug frames).
  handling: KalmanBoxFilter clamps predicted height and aspect to 0.6–1.6× the last measurement and zeroes the size velocities when the clamp engages.
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
cat > .gitignore << 'EOF_VI'
__pycache__/
*.egg-info/
.pytest_cache/
data/episodes/
data/synthetic/
data/keyframes/
data/debug/
EOF_VI
cp "$0" colab/sessions/session_10_build.sh 2>/dev/null || true
# privacy: stop tracking footage-derived images (files stay on disk)
git rm -r -q --cached data/keyframes data/debug 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files (incl. untracking keyframes/debug)"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. frame-nano vs frame-medium, and hybrid-tick with birth gates (GPU)"
cat > /tmp/_slice_row.py << 'EOF_PY'
import json, sys
label = sys.argv[1]
txt = sys.stdin.read()
if "{" not in txt:
    print(f"  {label:13s} (no row produced; see log tail below)"); sys.exit(1)
r = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
s = r['tubes_by_final_state']; cl = r['classes']
print(f"  {label:13s} tubes={r['person_tubes']:3d}/{r['max_concurrent_persons']:2d}  life={r['mean_person_tube_life_s']:5.2f}s  "
      f"exited={s.get('exited',0):2d} lost={s.get('lost',0):2d}  rebirths={r['rebirths']:2d}  dups={len(r['duplicate_pairs']):2d}  "
      f"non-person tubes={sum(v for k, v in cl.items() if k != 'person'):2d}  dets/frame={r['person_dets_per_frame']}  "
      f"p50={r['detect_ms_p50']}ms")
EOF_PY
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  if [ -f "$SOURCE" ]; then
    set +e
    run() { local label="$1"; shift
      python bench/slice_gpu.py --source "$SOURCE" --fps 4 --tracker byte --max-frames 400 "$@" > "/tmp/slice_$label.log" 2>&1
      python /tmp/_slice_row.py "$label" < "/tmp/slice_$label.log" || { warn "$label failed:"; grep -v TracerWarning "/tmp/slice_$label.log" | tail -8; }
    }
    run frame-nano    --detect frame  --model nano
    run frame-medium  --detect frame  --model medium
    run hybrid-tick   --detect hybrid --model nano --heartbeat-ms 250
    set -e
    echo "  (session 09: frame-nano 15/10 life 9.0s | hybrid-tick 22/12 life 7.4s, 13 suitcase tubes in s05)"
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — switch to L4 and rerun"
fi

step "5. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: Kalman size clamp (E-TUBE-14), birth gates (E-DET-11), footage-derived images untracked and ignored"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "6. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "then: make the GitHub repo private (Settings > General > Danger Zone). Paste the three rows back."
