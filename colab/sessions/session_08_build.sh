#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 08 build: instrumentation + one tracker fix.
#   * Detection.origin (roi | full): which crop produced each box.
#   * ByteTracker: a track born from the full-frame heartbeat survives ROI ticks (which cannot
#     see static objects) until the next heartbeat, instead of being deleted as unconfirmed.
#   * slice row: births_by_origin, confirmed_by_origin, rebirths (tube born within one box-height
#     and 1.5 s of one that just died = fragmentation, counted directly).
#   * --debug-frames N: annotated JPEGs (crops green/blue, detections yellow/cyan, tubes red/orange/grey).
#  Experiment: roi | frame | hybrid@1s | hybrid@every-tick, plus 12 debug frames for hybrid@1s.
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_08.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 08"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }

step "0. repository"
if [ -d "$REPO_DIR/.git" ]; then git -C "$REPO_DIR" fetch -q origin; else git clone -q "https://github.com/$GH_REPO.git" "$REPO_DIR"; fi
cd "$REPO_DIR"
[ -n "${GH_TOKEN:-}" ] && git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"

step "1. audit"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present: $(git status --short | tr '\n' ' ')"
git pull -q --ff-only || die "local branch diverged from origin; resolve manually"
[ -n "$(git log --grep='^session 07' --format=%h)" ] || die "session 07 commit not found; run build_session_07.sh first"
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
cat > vi/detect/base.py << 'EOF_VI'
from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

from vi.schemas import Box


# Classes that become tubes. Furniture, fixtures and appliances are scene-card assets handled by
# heartbeats and zones, never tracked as moving objects (E-DET-09).
TUBE_CLASSES = {"person", "dog", "cat", "bicycle", "car", "motorcycle", "bus", "truck",
                "backpack", "handbag", "suitcase", "umbrella"}


def is_tube_class(label: str) -> bool:
    return label in TUBE_CLASSES


class Detection(BaseModel):
    box: Box
    class_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: list[float] | None = None   # R14: ReID alongside the box
    truncated: bool = False                # E-DET-02: box touches the frame border
    roi_truncated: bool = False            # E-DET-10: box touches its crop border (partial view of the object)
    origin: Literal["roi", "full", "unknown"] = "unknown"   # which crop produced it (heartbeat = full)


class Detector(Protocol):
    """Ring 1 interface. Batched over ROI crops from many cameras (R12)."""

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]: ...
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
    """Shift crop-space boxes back to frame space. Flags: `truncated` when the box touches the
    frame border (E-DET-02, unreliable foot point); `roi_truncated` when it touches the crop
    border but not the frame border (E-DET-10, a partial view that a full-frame or neighbouring
    crop may have seen whole)."""
    out = []
    rw, rh = roi.x2 - roi.x1, roi.y2 - roi.y1
    origin = "full" if (roi.x1 == 0 and roi.y1 == 0 and roi.x2 >= frame_w and roi.y2 >= frame_h) else "roi"
    for d in dets:
        at_roi_edge = d.box.x1 <= EDGE_PX or d.box.y1 <= EDGE_PX or d.box.x2 >= rw - EDGE_PX or d.box.y2 >= rh - EDGE_PX
        b = Box(x1=d.box.x1 + roi.x1, y1=d.box.y1 + roi.y1, x2=d.box.x2 + roi.x1, y2=d.box.y2 + roi.y1)
        truncated = b.x1 <= EDGE_PX or b.y1 <= EDGE_PX or b.x2 >= frame_w - EDGE_PX or b.y2 >= frame_h - EDGE_PX
        out.append(d.model_copy(update={"box": b, "truncated": truncated, "origin": origin,
                                        "roi_truncated": bool(at_roi_edge and not truncated and origin == "roi")}))
    return out


def full_frame_roi(frame_w: int, frame_h: int) -> ROI:
    """The heartbeat's full-frame pass is just one more crop in the batch (R10 / R12)."""
    return ROI(x1=0, y1=0, x2=frame_w, y2=frame_h, source_blobs=0)


def _ios(a: Box, b: Box) -> float:
    """Intersection over the smaller box: catches a partial (crop-truncated) view sitting
    inside a complete detection, which plain IoU under-scores."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max(1e-6, min(a.area, b.area))


def dedupe_detections(dets: list[Detection], iou_thr: float = 0.5, ios_thr: float = 0.6) -> list[Detection]:
    """E-DET-10: one object, one detection per frame. Detections from overlapping crops and from
    the full-frame heartbeat are merged class-wise; complete boxes beat crop-truncated ones,
    then higher confidence wins."""
    ranked = sorted(dets, key=lambda d: (not d.roi_truncated, d.confidence), reverse=True)
    kept: list[Detection] = []
    for d in ranked:
        dup = any(k.class_label == d.class_label and (k.box.iou(d.box) >= iou_thr or _ios(k.box, d.box) >= ios_thr)
                  for k in kept)
        if not dup:
            kept.append(d)
    return kept


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
    """Union of two detection sets on the same frame, deduplicated (see dedupe_detections)."""
    return dedupe_detections(list(primary) + list(secondary), iou_thr=iou_thr)
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
cat > vi/episode/debug.py << 'EOF_VI'
from __future__ import annotations

from pathlib import Path

import numpy as np

from vi.detect import ROI, Detection
from vi.schemas import Tube

COL = {"roi": (60, 200, 60), "full": (60, 160, 255), "det_roi": (255, 220, 40), "det_full": (80, 230, 255),
       "active": (255, 60, 60), "occluded": (255, 140, 40), "born": (200, 200, 200)}


def annotate(frame_rgb: np.ndarray, rois: list[ROI], dets: list[Detection], tubes: list[Tube],
             title: str, path: str | Path) -> Path | None:
    """Debug frame: crops (green; full frame blue), detections (yellow from crops, cyan from the
    full frame, dashed-ish by confidence label), tubes (red active, orange predicted, grey born)
    with id suffix and state. Opens in Colab's file browser; upload one to the chat to review."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    im = Image.fromarray(np.ascontiguousarray(frame_rgb))
    dr = ImageDraw.Draw(im)
    for r in rois:
        c = COL["full"] if r.source_blobs == 0 else COL["roi"]
        dr.rectangle([r.x1, r.y1, r.x2 - 1, r.y2 - 1], outline=c, width=1)
    for d in dets:
        c = COL["det_full"] if d.origin == "full" else COL["det_roi"]
        b = d.box
        dr.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=1)
        dr.text((b.x1 + 2, b.y2 - 12), f"{d.class_label[:6]} {d.confidence:.2f}{'t' if d.roi_truncated else ''}", fill=c)
    for t in tubes:
        c = COL.get(t.state.value, COL["born"])
        b = t.box
        dr.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=3)
        dr.text((b.x1 + 2, max(0, b.y1 - 12)), f"#{t.tube_id.split(':')[-1]} {t.state.value[:3]}", fill=c)
    dr.text((6, 6), title, fill=(255, 255, 255))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=85)
    return path
EOF_VI
cat > vi/episode/__init__.py << 'EOF_VI'
from .debug import annotate
from .keyframes import KeyframeStore
from .writer import EpisodeWriter, episode_id_for, should_soft_cut
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

from vi.detect import (blobs_to_rois, crop_roi, dedupe_detections, full_frame_roi, is_tube_class, pad_batch,
                       remap_detections)
try:
    from vi.detect.rfdetr import RFDETRDetector
except ImportError:  # CPU runtime: only --model fake works
    RFDETRDetector = None  # type: ignore
from vi.episode import EpisodeWriter, KeyframeStore, annotate
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
    ap.add_argument("--debug-frames", type=int, default=0, help="save N annotated frames to data/debug/<episode>/")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default = edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=640, want_rgb=True)
    batch = 1 if a.detect == "frame" else a.batch          # frame mode: one image per tick, trace at 1
    if a.model == "fake":
        from vi.detect.fake import BrightBlobDetector
        det = BrightBlobDetector(threshold=a.threshold, batch_size=batch)   # CPU smoke path (tests, no GPU)
    else:
        det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=batch)
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
    births_by_origin: Counter = Counter()
    confirmed_by_origin: Counter = Counter()
    rebirths = 0                                   # tube born next to one that just died = fragmentation, counted directly
    recent_dead: list[tuple[int, float, float, float]] = []   # (t_ms, cx, cy, h)
    seen_ids: set[str] = set()
    confirmed_ids: set[str] = set()
    origin_of: dict[str, str] = {}
    debug_every = max(1, int(a.fps * 1.5)) if a.debug_frames else 0     # one frame every ~1.5 s until N saved
    debug_paths: list[str] = []
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
        det_source = "detector"
        rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride) if a.detect in ("roi", "hybrid") else []
        tile_active = len(tracker._tracks) > 0
        if a.detect == "frame" or (a.detect == "hybrid" and hb.due(fr.pts_ms, tile_active)):
            rois.append(full_frame_roi(w, h))                   # heartbeat = one more crop in the batch
            det_source = "heartbeat"
            heartbeats += 1
        dets = []
        for i in range(0, len(rois), batch):
            chunk = rois[i:i + batch]
            crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], batch)
            for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                dets += remap_detections(d, r, w, h)
        dets = dedupe_detections([d for d in dets if is_tube_class(d.class_label)])   # E-DET-10
        person_dets.append(sum(1 for d in dets if d.class_label == "person" and d.confidence >= 0.5))
        dt_detect = time.perf_counter() - t0
        if frames >= a.warmup:
            stage["detect"] += dt_detect
            detect_ms.append(dt_detect * 1000)
        t0 = time.perf_counter()
        before = len(tracker._tracks)
        live, closed = tracker.update(dets, fr.pts_ms, det_source=det_source)
        # provenance + rebirth accounting (person tubes only)
        for t in live:
            if t.class_label != "person":
                continue
            if t.tube_id not in seen_ids:
                seen_ids.add(t.tube_id)
                origin = next((d.origin for d in dets if d.box.iou(t.box) > 0.7), "unknown")
                births_by_origin[origin] += 1
                origin_of[t.tube_id] = origin
                cx, cy = (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2
                if any(fr.pts_ms - tm <= 1500 and ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 <= hh for tm, x, y, hh in recent_dead):
                    rebirths += 1
            if t.state.value == "active" and t.tube_id not in confirmed_ids:
                confirmed_ids.add(t.tube_id)
                confirmed_by_origin[origin_of.get(t.tube_id, "unknown")] += 1
        for t in closed:
            if t.class_label == "person":
                recent_dead.append((fr.pts_ms, (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2, t.box.height))
        recent_dead = [r for r in recent_dead if fr.pts_ms - r[0] <= 1500]
        if debug_every and frames % debug_every == 0 and len(debug_paths) < a.debug_frames:
            p = annotate(fr.rgb, rois, dets, live, f"t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                         Path(a.out).parent / "debug" / ep / f"tick_{frames:04d}.jpg")
            if p:
                debug_paths.append(str(p))
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
        "births_by_origin": dict(births_by_origin), "confirmed_by_origin": dict(confirmed_by_origin),
        "rebirths": rebirths,
        "debug_frames": debug_paths,
        "tubes_per_concurrent": round(sum(1 for t in tubes if t.class_label == "person") / max(1, max(concurrent_persons) if concurrent_persons else 1), 2),
        "mean_person_tube_life_s": round(float(np.mean([(t.last_seen.corrected_ms() - t.born.corrected_ms()) / 1000
                                                        for t in tubes if t.class_label == "person"] or [0])), 2),
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
    print(f"\nepisode -> {writer.path(ep)}\nkeyframes -> {kf.root}" + (f"\ndebug frames -> {Path(a.out).parent / 'debug' / ep}" if debug_paths else ""))


if __name__ == "__main__":
    main()
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
EOF_VI
cp "$0" colab/sessions/session_08_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. experiment with provenance (GPU): roi | frame | hybrid@1s | hybrid@tick"
cat > /tmp/_slice_row.py << 'EOF_PY'
import json, sys
label = sys.argv[1]
txt = sys.stdin.read()
if "{" not in txt:
    print(f"  {label:12s} (no row produced; see log tail below)"); sys.exit(1)
r = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
s = r['tubes_by_final_state']; b = r['births_by_origin']; c = r['confirmed_by_origin']
print(f"  {label:12s} tubes={r['person_tubes']:3d}/{r['max_concurrent_persons']:2d}  life={r['mean_person_tube_life_s']:5.2f}s  "
      f"exited={s.get('exited',0):2d} lost={s.get('lost',0):2d}  rebirths={r['rebirths']:2d}  "
      f"births roi/full={b.get('roi',0)}/{b.get('full',0)}  confirmed roi/full={c.get('roi',0)}/{c.get('full',0)}  "
      f"enter={r['events'].get('enter_zone',0):2d}  p50={r['detect_ms_p50']}ms  hb={r['heartbeats']}")
for p in r.get("debug_frames", [])[:12]:
    print(f"      debug: {p}")
EOF_PY
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  if [ -f "$SOURCE" ]; then
    set +e
    run() { # label, extra args...
      local label="$1"; shift
      python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --max-frames 400 "$@" > "/tmp/slice_$label.log" 2>&1
      python /tmp/_slice_row.py "$label" < "/tmp/slice_$label.log" || { warn "$label failed:"; grep -v TracerWarning "/tmp/slice_$label.log" | tail -8; }
    }
    run roi         --detect roi
    run frame       --detect frame
    run hybrid-1s   --detect hybrid --heartbeat-ms 1000 --debug-frames 12
    run hybrid-tick --detect hybrid --heartbeat-ms 250
    set -e
    echo "  (session 07: roi 17/7 life 5.5s | frame 15/10 life 9.0s | hybrid-1s 26/9 life 4.8s)"
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — switch to L4 and rerun for the experiment"
fi

step "5. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: detection origin, heartbeat-born static grace in ByteTracker, birth provenance + rebirth counters, annotated debug frames"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "6. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste the four rows back, and upload two debug frames from the hybrid-1s run: one with hb=Y and one with hb=n."
