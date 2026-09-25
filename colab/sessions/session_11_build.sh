#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 11 build: appearance re-identification.
#   * vi/reid: HistogramEmbedder (CPU), SigLIPEmbedder (google/siglip2-base-patch16-224, Apache-2.0,
#     Hugging Face), OSNetEmbedder (torchreid, optional); all L2-normalised
#   * vi/tubes/linker.py: TubeLinker ties a newborn tube to an entity whose tube went lost within
#     30 s / 400 px when cosine >= 0.75; EMA + exemplar gallery; emits Event(relink)
#   * slice: --reid, entities / relinks / entities_per_concurrent / embed_ms in the row
#  Experiment: frame-nano without ReID, with SigLIP, and hybrid-tick with SigLIP.
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_11.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 11"
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
[ -n "$(git log --grep='^session 10' --format=%h)" ] || die "session 10 commit not found; run build_session_10.sh first"
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
tests/test_reid.py
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
vi/reid/__init__.py
vi/reid/base.py
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
vi/tubes/linker.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p vi/reid colab/sessions data/bench
cat > vi/reid/__init__.py << 'EOF_VI'
from .base import Embedder, HistogramEmbedder, crop_for_embedding, make_embedder
EOF_VI
cat > vi/reid/base.py << 'EOF_VI'
"""Appearance embeddings for re-identification (R14 / E-TUBE-04 / E-TUBE-08).

Backends, all open weights and no training:
  hist    CPU colour histogram (tests, fallback, weak baseline)
  siglip  google/siglip2-base-patch16-224 image tower (Apache-2.0, Hugging Face) — generic but
          reliable to download; the default GPU embedder until OSNet weight hosting is verified
  osnet   torchreid OSNet (MIT) — the dedicated ReID model in the BOM; weights come from the
          author's Google Drive, so this backend is optional and verified per runtime
Every backend returns L2-normalised float32 vectors so cosine similarity is a dot product.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from vi.schemas import Box


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


def crop_for_embedding(frame_rgb: np.ndarray, box: Box, pad: float = 0.08) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    px, py = box.width * pad, box.height * pad
    x1, y1 = int(max(0, box.x1 - px)), int(max(0, box.y1 - py))
    x2, y2 = int(min(w, box.x2 + px)), int(min(h, box.y2 + py))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((8, 4, 3), np.uint8)
    return np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)


class HistogramEmbedder:
    """Upper/lower body colour histograms (8 bins per RGB channel each) -> 48-D. Deliberately
    simple: clothing colour is what survives a 13-second occlusion in a warehouse."""

    name = "hist"
    dim = 48

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            h = c.shape[0]
            parts = (c[: h // 2], c[h // 2:])
            feats = []
            for part in parts:
                for ch in range(3):
                    hist, _ = np.histogram(part[..., ch], bins=8, range=(0, 256))
                    feats.append(hist.astype(np.float32) / max(1, part.size / 3))
            out[i] = np.concatenate(feats)
        return _l2(out)


class SigLIPEmbedder:
    name = "siglip"
    dim = 768

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        ims = [Image.fromarray(np.ascontiguousarray(c)) for c in crops]
        with self.torch.no_grad():
            inputs = self.processor(images=ims, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
        return _l2(feats.float().cpu().numpy())


class OSNetEmbedder:
    name = "osnet"
    dim = 512

    def __init__(self, model_name: str = "osnet_x0_25", model_path: str | None = None, device: str | None = None):
        import torch
        from torchreid.utils import FeatureExtractor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(model_name=model_name, model_path=model_path, device=self.device)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        feats = self.extractor(list(crops))
        return _l2(feats.cpu().numpy())


def make_embedder(kind: str = "auto") -> Embedder:
    """auto: osnet if importable and weights load, else siglip, else hist. Prints the choice."""
    order = ["osnet", "siglip", "hist"] if kind == "auto" else [kind]
    for k in order:
        try:
            if k == "hist":
                return HistogramEmbedder()
            if k == "siglip":
                e = SigLIPEmbedder()
            elif k == "osnet":
                e = OSNetEmbedder()
            else:
                raise ValueError(k)
            print(f"[reid] embedder: {e.name} ({e.dim}-d)")
            return e
        except Exception as ex:  # pragma: no cover
            print(f"[reid] {k} unavailable ({type(ex).__name__}: {str(ex)[:80]})")
    return HistogramEmbedder()
EOF_VI
cat > vi/tubes/linker.py << 'EOF_VI'
"""Appearance-based re-linking of tubes into entities within one camera (E-TUBE-04), with a
drifting gallery per entity (E-TUBE-08). Cross-camera fusion reuses the same gallery format.

Rules:
  * a newborn tube is compared against entities whose last tube went `lost` (never `exited`)
    within max_gap_ms and whose last position is within max_jump_px;
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
    last_box_center: tuple[float, float] = (0.0, 0.0)
    last_box_h: float = 1.0
    lost_at_ms: int | None = None      # set when its current tube is lost; None while live/exited


class TubeLinker:
    def __init__(self, camera_id: str, sim_thr: float = 0.75, max_gap_ms: int = 30_000,
                 max_jump_px: float = 400.0, ema_alpha: float = 0.3, exemplars: int = 5):
        self.camera_id = camera_id
        self.sim_thr = sim_thr
        self.max_gap_ms = max_gap_ms
        self.max_jump_px = max_jump_px
        self.ema_alpha = ema_alpha
        self.n_exemplars = exemplars
        self._entities: dict[str, _Entity] = {}
        self._tube_entity: dict[str, str] = {}
        self._seq = 0
        self.relinks = 0

    # ---------------------------------------------------------------- helpers
    def _new_entity(self, tube: Tube, emb: np.ndarray) -> _Entity:
        self._seq += 1
        ent = _Entity(entity_id=f"{self.camera_id}:E{self._seq}", tube_ids=[tube.tube_id], ema=emb.copy(),
                      exemplars=[emb.copy()], last_box_center=_center(tube), last_box_h=tube.box.height)
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
    def on_birth(self, tube: Tube, emb: np.ndarray, t_ms: int) -> Event | None:
        cands = self._candidates(tube, t_ms)
        if cands:
            best = max(cands, key=lambda e: self._sim(e, emb))
            sim = self._sim(best, emb)
            if sim >= self.sim_thr:
                prev = best.tube_ids[-1]
                best.tube_ids.append(tube.tube_id)
                best.lost_at_ms = None
                self._tube_entity[tube.tube_id] = best.entity_id
                self.on_refresh(tube, emb)
                self.relinks += 1
                return Event(event_id="ev_" + hashlib.sha1(f"relink|{prev}|{tube.tube_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[prev, tube.tube_id], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(sim, 3), "gap_ms": t_ms - (tube.born.corrected_ms())},
                             confidence=min(1.0, sim))
        self._new_entity(tube, emb)
        return None

    def on_refresh(self, tube: Tube, emb: np.ndarray) -> None:
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        e.ema = e.ema * (1 - self.ema_alpha) + emb * self.ema_alpha
        e.ema /= max(1e-8, np.linalg.norm(e.ema))
        e.exemplars.append(emb.copy())
        if len(e.exemplars) > self.n_exemplars:
            e.exemplars.pop(0)
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height

    def on_close(self, tube: Tube, t_ms: int) -> None:
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        e.lost_at_ms = t_ms if tube.state in (TubeState.lost, TubeState.occluded) else None

    def entity_of(self, tube_id: str) -> str | None:
        return self._tube_entity.get(tube_id)

    @property
    def entities(self) -> int:
        return len(self._entities)


def _center(t: Tube) -> tuple[float, float]:
    return ((t.box.x1 + t.box.x2) / 2.0, (t.box.y1 + t.box.y2) / 2.0)
EOF_VI
cat > vi/tubes/__init__.py << 'EOF_VI'
from .base import Tracker
from .bytetrack import ByteTracker
from .kalman import KalmanBoxFilter
from .linker import TubeLinker
from .simple_iou import SimpleIoUTracker

TRACKERS = {"simple": SimpleIoUTracker, "byte": ByteTracker}
EOF_VI
cat > vi/schemas/event.py << 'EOF_VI'
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import CamTime


class EventType(str, Enum):
    # tube events (Ring 3a, deterministic)
    enter_zone = "enter_zone"
    exit_zone = "exit_zone"
    dwell = "dwell"
    loiter = "loiter"
    approach = "approach"
    meet = "meet"
    pickup = "pickup"
    drop = "drop"
    left_behind = "left_behind"
    asset_missing_from_home = "asset_missing_from_home"
    fall = "fall"
    run = "run"
    crowd = "crowd"
    handoff = "handoff"
    relink = "relink"            # E-TUBE-04: newborn tube tied to a lost one by appearance (same camera)
    impossible_transition = "impossible_transition"
    # scene-state events (Ring 3a, from I-frames / gate)
    illumination_change = "illumination_change"
    door_state_change = "door_state_change"
    appliance_state_change = "appliance_state_change"
    modality_switch = "modality_switch"
    # ingest events
    signal_lost = "signal_lost"
    signal_restored = "signal_restored"
    camera_moved_suspect = "camera_moved_suspect"   # E-KB-01


class Event(BaseModel):
    event_id: str
    type: EventType
    t: CamTime
    camera_id: str
    tile_id: str | None = None
    zone_id: str | None = None
    subject_tube_ids: list[str] = Field(default_factory=list)
    subject_entity_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)      # assets, actuators, other tubes
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Literal["deterministic", "walk", "observed", "user"] = "deterministic"
    episode_id: str | None = None
    dedupe_key: str | None = None    # E-EVT-06: entity+type+time bucket across overlapping cams
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
from vi.reid import crop_for_embedding, make_embedder
from vi.tubes import TRACKERS, TubeLinker


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
    ap.add_argument("--reid", choices=["none", "auto", "hist", "siglip", "osnet"], default="none",
                    help="appearance embeddings + TubeLinker (E-TUBE-04); auto = osnet > siglip > hist")
    ap.add_argument("--reid-every-ticks", type=int, default=8, help="gallery refresh cadence for active tubes")
    ap.add_argument("--reid-sim", type=float, default=0.75)
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
    embedder = make_embedder(a.reid) if a.reid != "none" else None
    linker = TubeLinker(a.camera, sim_thr=a.reid_sim) if embedder else None
    embed_ms: list[float] = []
    last_embed_tick: dict[str, int] = {}
    relink_events = 0
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
        if linker is not None:
            due_birth = [t for t in live if t.class_label == "person" and t.tube_id not in last_embed_tick]
            due_refresh = [t for t in live if t.class_label == "person" and t.state.value == "active"
                           and t.tube_id in last_embed_tick and frames - last_embed_tick[t.tube_id] >= a.reid_every_ticks]
            todo = due_birth + due_refresh
            if todo:
                t0 = time.perf_counter()
                embs = embedder.embed([crop_for_embedding(fr.rgb, t.box) for t in todo])
                embed_ms.append((time.perf_counter() - t0) * 1000)
                for t, e in zip(todo, embs):
                    last_embed_tick[t.tube_id] = frames
                    if t.tube_id in {x.tube_id for x in due_birth}:
                        ev = linker.on_birth(t, e, fr.pts_ms)
                        if ev is not None:
                            events.append(ev)
                            relink_events += 1
                            print(f"t={fr.pts_ms:7d}  relink                 {ev.subject_tube_ids[0]} -> {ev.subject_tube_ids[1]} sim={ev.payload['similarity']}")
                    else:
                        linker.on_refresh(t, e)
            for t in live:
                if t.class_label == "person":
                    t.entity_id = linker.entity_of(t.tube_id)
            for t in closed:
                linker.on_close(t, fr.pts_ms)
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
        "reid": embedder.name if embedder else "none",
        "entities": linker.entities if linker else None, "relinks": linker.relinks if linker else None,
        "entities_per_concurrent": round(linker.entities / max(1, max(concurrent_persons) if concurrent_persons else 1), 2) if linker else None,
        "embed_ms_p50": round(float(np.median(embed_ms)), 2) if embed_ms else None,
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
cat > bench/README.md << 'EOF_VI'
# bench/

One script per ring. Each prints exactly one row of the benchmark table (see PLAN.md) and
checkpoints it to `data/bench/<ring>.jsonl` as it goes, so a killed Colab session loses nothing.

| script | ring | GPU | dataset | row |
|---|---|---|---|---|
| slice_cpu.py | all (stubs) | none | synthetic | proves plumbing; run first, every day |
| slice_gpu.py | 0–3a on real footage | L4 | own clip | reader → gate → ROIs (+ heartbeat) → RF-DETR → tubes → events → episode file + birth keyframes; `--detect roi\|frame\|hybrid`, `--reid hist\|siglip\|osnet\|auto` (entities, relinks), person_tubes, tube life, rebirths, duplicate trap, debug frames |
| ring0_gate.py | 0 | CPU | VIRAT | gate FN rate, ms per GOP per stream, MV vs framediff |
| ring1_detect.py | 1 | L4 | own clip / VIRAT | `--mode frame`: ms p50/p95 per frame at sampled fps; `--mode roi`: gate → packed ROI batches (nano/medium) |
| ring2_tubes.py | 2 | CPU | synthetic, MOT17 | MOTA/IDF1/IDSW/fragmentation (vi/eval/mot.py); `--tracker simple\|byte`, `--sample-every` simulates 2–5 fps decode |
| ring3b_sheet.py | 3b | A100-40 | crops from ring2 | attribute accuracy, bleed rate (12 vs 16 cells), ms per sheet |
| ring3a_events.py | 3a | CPU | MEVA subset | event precision/recall per type |
| agent_replay.py | block 2 | A100-40 -> G4 | WILDTRACK episodes | both scenarios replay; tool calls, latency |

Colab: GPU runtime, `pip install rfdetr==1.7.0`, then run from the repo root (see colab/README.md).
EOF_VI
cat > tests/test_reid.py << 'EOF_VI'
import numpy as np
import pytest

from vi.reid import HistogramEmbedder, crop_for_embedding
from vi.schemas import Box, CamTime, EventType, Tube, TubeState
from vi.tubes import TubeLinker


def tube(tid, x, state=TubeState.active, t=0, h=120):
    return Tube(tube_id=tid, camera_id="c1", class_label="person", state=state, born=CamTime(cam_utc_ms=t),
                last_seen=CamTime(cam_utc_ms=t), box=Box(x1=x, y1=100, x2=x + 40, y2=100 + h))


def unit(seed, dim=48):
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_histogram_embedder_is_normalised_and_separates_colours():
    e = HistogramEmbedder()
    red = np.zeros((120, 40, 3), np.uint8); red[..., 0] = 220
    blue = np.zeros((120, 40, 3), np.uint8); blue[..., 2] = 220
    v = e.embed([red, blue, red.copy()])
    assert v.shape == (3, 48) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[2] > 0.99 and v[0] @ v[1] < 0.6
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert crop_for_embedding(frame, Box(x1=100, y1=100, x2=140, y2=220)).shape[0] > 120  # padded


@pytest.mark.edge("E-TUBE-04")
def test_newborn_relinks_to_recently_lost_entity_with_matching_appearance():
    lk = TubeLinker("c1", sim_thr=0.75, max_gap_ms=30_000, max_jump_px=400)
    a = unit(1)
    t1 = tube("c1:0:1", 300)
    assert lk.on_birth(t1, a, 0) is None and lk.entities == 1
    t1.state = TubeState.lost
    lk.on_close(t1, 5000)                              # hidden behind a colleague
    t2 = tube("c1:9000:2", 340, t=9000)
    ev = lk.on_birth(t2, a * 0.95 + unit(2) * 0.05, 9000)
    assert ev is not None and ev.type == EventType.relink and ev.subject_tube_ids == ["c1:0:1", "c1:9000:2"]
    assert lk.entity_of("c1:9000:2") == lk.entity_of("c1:0:1") and lk.entities == 1 and lk.relinks == 1
    # a different-looking person at the same spot starts a new entity
    t3 = tube("c1:9500:3", 300, t=9500)
    assert lk.on_birth(t3, unit(7), 9500) is None and lk.entities == 2


@pytest.mark.edge("E-TUBE-04")
def test_relink_refuses_long_gaps_far_jumps_and_exited_tubes():
    a = unit(1)
    lk = TubeLinker("c1", max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, a, 0); t1.state = TubeState.lost; lk.on_close(t1, 1000)
    assert lk.on_birth(tube("c1:20000:2", 300, t=20000), a, 20000) is None            # too long ago
    lk2 = TubeLinker("c1", max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk2.on_birth(t1, a, 0); t1.state = TubeState.lost; lk2.on_close(t1, 1000)
    assert lk2.on_birth(tube("c1:2000:2", 900, t=2000), a, 2000) is None                # 600 px jump
    lk3 = TubeLinker("c1")
    t1 = tube("c1:0:1", 300); lk3.on_birth(t1, a, 0); t1.state = TubeState.exited; lk3.on_close(t1, 1000)
    assert lk3.on_birth(tube("c1:2000:2", 300, t=2000), a, 2000) is None                # left through an exit


@pytest.mark.edge("E-TUBE-08")
def test_gallery_follows_appearance_drift():
    lk = TubeLinker("c1", sim_thr=0.8, ema_alpha=0.5, exemplars=3)
    base = unit(3); drift = unit(4)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, base, 0)
    steps = [(base * (1 - k) + drift * k) for k in (0.2, 0.4, 0.6, 0.8)]
    for s in steps:
        lk.on_refresh(t1, s / np.linalg.norm(s))
    t1.state = TubeState.lost; lk.on_close(t1, 5000)
    final = drift * 0.9 + base * 0.1
    ev = lk.on_birth(tube("c1:6000:2", 300, t=6000), final / np.linalg.norm(final), 6000)
    assert ev is not None                        # linked to the drifted gallery, not the birth look
    assert float(base @ (final / np.linalg.norm(final))) < 0.8      # which plain birth-embedding matching would miss
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


def test_slice_runs_with_histogram_reid(tmp_path):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip

    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/slice_gpu.py", "--source", str(clip), "--model", "fake",
                          "--detect", "frame", "--fps", "5", "--reid", "hist", "--out", str(tmp_path / "episodes")],
                         capture_output=True, text=True, cwd=".", env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-2000:]
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["reid"] == "hist" and row["entities"] >= 1 and row["embed_ms_p50"] is not None
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
  handling: TubeLinker ties a newborn tube to an entity whose tube went lost within 30 s and 400 px when appearance cosine >= 0.75; emits Event(relink); the agent reports entities, not tubes.
  status: implemented
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
  handling: Per-entity EMA embedding plus up to 5 exemplars refreshed on active ticks; matching takes the best of EMA and exemplars so a jacket-off entity still links.
  status: implemented
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

**Intra-camera re-linking (measured need, session 10):** on the warehouse clip frame-nano left 15
tubes for 10–11 people with zero rebirths: fragmentation comes from occlusions longer than the
2.5 s limit at the packing table. `TubeLinker` ties a newborn tube to an entity whose tube went
`lost` within 30 s and 400 px when appearance cosine ≥ 0.75, keeps a per-entity EMA + 5 exemplars,
and emits `Event(relink)` for auditability. Embedders: SigLIP 2 image tower (Apache-2.0, reliable
download) by default on GPU; OSNet (MIT) once weight hosting is verified; colour histogram on CPU.

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
cp "$0" colab/sessions/session_11_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. ReID on real footage (GPU): frame-nano | frame-nano+siglip | hybrid-tick+siglip"
cat > /tmp/_slice_row.py << 'EOF_PY'
import json, sys
label = sys.argv[1]
txt = sys.stdin.read()
if "{" not in txt:
    print(f"  {label:18s} (no row produced; see log tail below)"); sys.exit(1)
r = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
s = r['tubes_by_final_state']
ent = f"entities={r['entities']:3d} ({r['entities_per_concurrent']}/concurrent) relinks={r['relinks']:2d} embed_p50={r['embed_ms_p50']}ms" if r.get('entities') is not None else "entities=—"
print(f"  {label:18s} tubes={r['person_tubes']:3d}/{r['max_concurrent_persons']:2d}  life={r['mean_person_tube_life_s']:5.2f}s  "
      f"lost={s.get('lost',0):2d}  {ent}  reid={r['reid']}  p50={r['detect_ms_p50']}ms")
EOF_PY
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  if [ -f "$SOURCE" ]; then
    set +e
    run() { local label="$1"; shift
      python bench/slice_gpu.py --source "$SOURCE" --fps 4 --tracker byte --model nano --max-frames 400 "$@" > "/tmp/slice_$label.log" 2>&1
      grep -E "^\[reid\]|relink " "/tmp/slice_$label.log" | head -12 | sed 's/^/      /'
      python /tmp/_slice_row.py "$label" < "/tmp/slice_$label.log" || { warn "$label failed:"; grep -v TracerWarning "/tmp/slice_$label.log" | tail -8; }
    }
    run frame              --detect frame
    run frame+siglip       --detect frame  --reid siglip
    run hybrid-tick+siglip --detect hybrid --heartbeat-ms 250 --reid siglip
    set -e
    echo "  (target: entities near the number of people, 10-11; session 10 frame-nano was 15 tubes / 11 concurrent)"
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — switch to L4 and rerun"
fi

step "5. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: appearance embedders (hist/siglip/osnet), TubeLinker intra-camera relinking (E-TUBE-04/08), entities in the slice row"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "6. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste the three rows and the relink lines back."
