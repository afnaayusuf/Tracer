#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 09 build:
#   * pad_batch pads with the smallest crop (hybrid-tick was carrying 4-5 full frames per call)
#   * duplicate trap: when two live person tubes overlap (IoU>=0.5) the slice records both tubes'
#     ids/origins/boxes and every person detection on that tick, and saves the annotated frame
#   * measured decision written into SPEC.md: single-camera regime = full frame every tick
#  Experiment: frame (new default) and hybrid-tick (with the duplicate trap armed).
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_09.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 09"
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
[ -n "$(git log --grep='^session 08' --format=%h)" ] || die "session 08 commit not found; run build_session_08.sh first"
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
    """A traced model runs at one fixed batch size; pad with the *smallest* crop (preprocessing
    cost scales with pixels, and the full-frame heartbeat crop is usually last) and report how
    many entries are real so the caller drops the padding."""
    if not crops:
        return [], 0
    real = len(crops)
    padded = list(crops[:batch_size])
    filler = min(padded, key=lambda c: c.shape[0] * c.shape[1])
    while len(padded) < batch_size:
        padded.append(filler)
    return padded, min(real, batch_size)


def merge_detections(primary: list[Detection], secondary: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Union of two detection sets on the same frame, deduplicated (see dedupe_detections)."""
    return dedupe_detections(list(primary) + list(secondary), iou_thr=iou_thr)
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
    ap.add_argument("--detect", choices=["roi", "frame", "hybrid"], default="frame",
                    help="frame (default, measured best at one camera): full frame every tick; roi: motion ROIs only; "
                         "hybrid: ROIs + full-frame heartbeat (multi-camera cost lever, under investigation)")
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
    duplicate_pairs: list[dict] = []                # two live person tubes on one body: the hybrid bug, caught in the act
    dup_frames: list[str] = []
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
        persons = [t for t in live if t.class_label == "person" and t.state.value in ("active", "born")]
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                ti, tj = persons[i], persons[j]
                iou = ti.box.iou(tj.box)
                if iou >= 0.5 and len(duplicate_pairs) < 12:
                    dets_here = [{"origin": d.origin, "conf": round(d.confidence, 2), "box": [round(v) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)],
                                  "iou_i": round(d.box.iou(ti.box), 2), "iou_j": round(d.box.iou(tj.box), 2)}
                                 for d in dets if d.class_label == "person" and (d.box.iou(ti.box) > 0.1 or d.box.iou(tj.box) > 0.1)]
                    duplicate_pairs.append({"t_ms": fr.pts_ms, "hb": det_source == "heartbeat", "iou": round(iou, 2),
                                            "a": {"id": ti.tube_id, "origin": origin_of.get(ti.tube_id), "state": ti.state.value,
                                                  "box": [round(v) for v in (ti.box.x1, ti.box.y1, ti.box.x2, ti.box.y2)]},
                                            "b": {"id": tj.tube_id, "origin": origin_of.get(tj.tube_id), "state": tj.state.value,
                                                  "box": [round(v) for v in (tj.box.x1, tj.box.y1, tj.box.x2, tj.box.y2)]},
                                            "dets_on_tick": dets_here})
                    if len(dup_frames) < 4:
                        p = annotate(fr.rgb, rois, dets, live, f"DUPLICATE t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                                     Path(a.out).parent / "debug" / ep / f"dup_{frames:04d}.jpg")
                        if p:
                            dup_frames.append(str(p))
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
        "rebirths": rebirths, "duplicate_pairs": duplicate_pairs, "duplicate_frames": dup_frames,
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
    assert len(padded) == 4 and real == 2 and padded[3] is crops[0]     # smallest crop is the filler
    big = np.zeros((720, 1280, 3), np.uint8)
    padded, _ = pad_batch([np.zeros((50, 50, 3), np.uint8), big], 8)
    assert all(p.shape == (50, 50, 3) for p in padded[2:])              # never pad with the full frame
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


def test_detector_pads_every_call_to_the_traced_batch():
    """Regression for the session-06 crash: a full-frame heartbeat is one image, the traced
    model wants exactly batch_size. Exercised on the RF-DETR wrapper via a stub model."""
    from vi.detect.rfdetr import RFDETRDetector

    class _Stub:
        def __init__(self):
            self.seen = []
        def predict(self, images, threshold, include_source_image):
            batch = images if isinstance(images, list) else [images]
            self.seen.append(len(batch))
            class R:  # minimal supervision-like result
                xyxy = np.array([[0, 0, 10, 20]]); confidence = np.array([0.9]); class_id = np.array([1])
                data = {"class_name": np.array(["person"])}
            return [R() for _ in batch] if isinstance(images, list) else R()

    det = RFDETRDetector.__new__(RFDETRDetector)
    det.model, det.optimized, det.batch_size, det.threshold, det.keep, det.size = _Stub(), True, 8, 0.5, {"person"}, "nano"
    single = det.detect(np.zeros((720, 1280, 3), np.uint8))
    assert len(single) == 1 and det.model.seen == [8]                 # padded to the traced batch
    eleven = det.detect_batch([np.zeros((50, 50, 3), np.uint8)] * 11)
    assert len(eleven) == 11 and det.model.seen[1:] == [8, 8]         # 8 + (3 padded to 8), 11 results back


@pytest.mark.edge("E-DET-10")
def test_dedupe_prefers_complete_box_over_crop_truncated_partials():
    from vi.detect import dedupe_detections, full_frame_roi
    from vi.detect.roi import ROI
    # one person at frame x 90..140 straddles two adjacent crops; each crop sees a partial view
    left = remap_detections([Detection(box=Box(x1=90, y1=50, x2=100, y2=170), class_label="person", confidence=0.7)],
                            ROI(x1=0, y1=0, x2=100, y2=200), 640, 480)
    right = remap_detections([Detection(box=Box(x1=0, y1=52, x2=40, y2=168), class_label="person", confidence=0.8)],
                             ROI(x1=100, y1=0, x2=300, y2=200), 640, 480)
    whole = remap_detections([Detection(box=Box(x1=90, y1=50, x2=140, y2=170), class_label="person", confidence=0.6)],
                             full_frame_roi(640, 480), 640, 480)
    assert left[0].roi_truncated and right[0].roi_truncated and not whole[0].roi_truncated
    out = dedupe_detections(left + right + whole)
    assert len(out) == 1 and out[0].box.x1 == 90 and out[0].box.x2 == 140     # the complete box wins
    # two genuinely different people survive
    other = Detection(box=Box(x1=400, y1=50, x2=450, y2=170), class_label="person", confidence=0.9)
    assert len(dedupe_detections(out + [other])) == 2
    # a suitcase overlapping a person is a different class: kept
    bag = Detection(box=Box(x1=100, y1=120, x2=130, y2=170), class_label="suitcase", confidence=0.9)
    assert len(dedupe_detections(out + [bag])) == 2
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
adds the full frame as one more crop in the ROI batch at episode open, every 1 s on active tiles
and every 10 s on quiet tiles (`--detect hybrid`); all detections of a tick are deduplicated
class-wise, complete boxes beating crop-truncated ones (E-DET-10). Measured on real footage
in session 05: ROI-only detection lost standing people within 2.5 s regardless of tracker. Fallback for MJPEG /
intra-only streams: frame differencing at 1 fps behind the same interface (`vi/gate/base.py`).
Slice implementation: `FrameDiffGate`. Production: `MVGate` (week 3). **MEASURE:** gate FN rate,
ms per GOP per stream.

**Measured decision (sessions 06–08, one camera, L4):** full-frame detection every tick costs the
same as ROI-only (~27 ms per call; per-call overhead dominates at batch size 1–8) and tracks best
(15 tubes / 10 concurrent, life 9.0 s, 0 rebirths, vs 17/7, 5.5 s for ROI-only). The single-camera
slice therefore detects on the full frame every tick. ROI gating and heartbeat hybrids remain the
multi-camera cost lever and are re-measured when the cross-camera batching bench exists; the
hybrid duplicate-tube defect is tracked under E-DET-10.

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
cp "$0" colab/sessions/session_09_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. frame (default) and hybrid-tick with the duplicate trap (GPU)"
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
      f"dups={len(r['duplicate_pairs']):2d}  p50={r['detect_ms_p50']}ms  hb={r['heartbeats']}")
for d in r["duplicate_pairs"][:4]:
    print(f"      DUP t={d['t_ms']} hb={d['hb']} iou={d['iou']}  A={d['a']['origin']}/{d['a']['state']} {d['a']['box']}  B={d['b']['origin']}/{d['b']['state']} {d['b']['box']}")
    for x in d["dets_on_tick"][:4]:
        print(f"          det {x['origin']:4s} conf={x['conf']} box={x['box']} iou(A)={x['iou_i']} iou(B)={x['iou_j']}")
for p in r.get("duplicate_frames", []):
    print(f"      frame: {p}")
EOF_PY
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  if [ -f "$SOURCE" ]; then
    set +e
    run() { local label="$1"; shift
      python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --max-frames 400 "$@" > "/tmp/slice_$label.log" 2>&1
      python /tmp/_slice_row.py "$label" < "/tmp/slice_$label.log" || { warn "$label failed:"; grep -v TracerWarning "/tmp/slice_$label.log" | tail -8; }
    }
    run frame       --detect frame --debug-frames 6
    run hybrid-tick --detect hybrid --heartbeat-ms 250
    set -e
    echo "  (session 08: frame 15/10 life 9.0s | hybrid-tick 22/12 life 7.4s p50 61ms)"
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — switch to L4 and rerun"
fi

step "5. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: pad with smallest crop, duplicate-tube trap with evidence + frames, full-frame regime as measured single-camera default"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "6. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste the rows and DUP lines back; upload one dup_*.jpg and one frame-mode tick_*.jpg from data/debug/."
